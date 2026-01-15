import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
# import os
# from VQTokenizer import NeuralVQVAE
import math

# --- 1. 标签映射配置 ---
# 定义 fingerflex 子集的 label 到 文本的映射
fingerflexLABEL_MAP = {
    0: "Inter-stimulus interval", # -> GT Token ID: 3306
    1: "thumb", # -> GT Token ID: 25036
    2: "index finger", #  -> GT Token ID: 1252
    3: "middle finger", #  -> GT Token ID: 19656
    4: "ring finger", #  -> GT Token ID: 12640
    5: "little finger" # -> GT Token ID: 55392
}
BCILABEL_MAP = fingerflexLABEL_MAP
Gesture_MAP = fingerflexLABEL_MAP
Motor_MAP = {
    0: "blank screen",  # Global ID 10189 (blank screen)
    11: "tongue movement", # Global ID 83 (tongue movement)
    12: "hand movement" # Global ID 10661 (hand movement)
}
LABEL_MAP = fingerflexLABEL_MAP  # Motor_MAP # BCILABEL_MAP  # 


class TokenAlignmentTrainer:
    def __init__(self, model_path="../Qwen2.5-7B-Instruct", subset='fingerflex'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 1. 原始 Label (0-5) -> 文本
        self.label_to_text = LABEL_MAP
        self.subset = subset
        # 预计算 Ground Truth Tokens
        self.gt_token_map = {}
        for idx, text in LABEL_MAP.items():
            # 获取单个单词对应的 token id (取第一个有效token)
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if self.subset == 'motor_basic':
                self.gt_token_map[max(0, idx-10)] = tokens[0] 
            else:
                self.gt_token_map[idx] = tokens[0] 
            print('tokens ', len(tokens), tokens)
            print(f"Label {idx} ({text}) -> GT Token ID: {tokens[0]}")
            # Label 0 (Inter-stimulus interval) -> GT Token ID: 3306
        print('len(self.gt_token_map) = ', len(self.gt_token_map)) # 5 # exit()
        '''
tokens 5 [3306, 5477, 318, 19425, 9873]
Label 0 (Inter-stimulus interval) -> GT Token ID: 3306
tokens 1 [25036]
Label 1 (thumb) -> GT Token ID: 25036
tokens 2 [1252, 14317]
Label 2 (index finger) -> GT Token ID: 1252
tokens 2 [19656, 14317]
Label 3 (middle finger) -> GT Token ID: 19656
tokens 2 [12640, 14317]
Label 4 (ring finger) -> GT Token ID: 12640
tokens 2 [55392, 14317]
Label 5 (little finger) -> GT Token ID: 55392
        '''
        # 在 TokenAlignmentTrainer 中记录这 6 个具体的 ID
        self.valid_token_ids = list(self.gt_token_map.values()) # 只有 6 个 ID
        # 2. 局部索引 (0-5) -> Qwen Global Token ID
        # 这个映射用于以后推理时将结果还原回 LLM Token
        self.local_to_global_token = {}
        for idx, text in self.label_to_text.items():
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            self.local_to_global_token[idx] = tokens[0]
            print(f"Local {idx} -> Global ID {tokens[0]} ({text})")

    def get_gt_batch(self, labels_tensor):
        """将 batch 的原始 labels (0-5) 转换为 Qwen token IDs"""
        # for l in labels_tensor: print("Label in Batch: ", l.item())
        if self.subset == 'motor_basic':
            alst = []
            for l in labels_tensor:
                if l.item() -10 in self.gt_token_map:
                    # print('l.item()-10 = ', l.item()-10, 'len(self.gt_token_map) = ', len(self.gt_token_map))
                    alst.append(self.gt_token_map[l.item()-10])
                else:
                    alst.append(self.gt_token_map[0])
            a = torch.tensor(alst, device=labels_tensor.device) # 
        else:
            a = torch.tensor([self.gt_token_map[l.item()] for l in labels_tensor], 
                            device=labels_tensor.device)
        # print("GT Tokens Batch shape: ", a.shape)#  torch.Size([32])
        return a
    
    def get_local_labels(self, labels_tensor):
        """
        直接返回原始标签 (0-5)，因为它们已经是 6 分类所需的索引了。
        """
        return labels_tensor.to(torch.long)


# --- 2. 构建对齐模型 (VQ-to-Token Predictor) ---
class TransformerVQToTokenModel(nn.Module):
    def __init__(self, codebook_size, embed_dim=256, nhead=8, 
                 num_layers=4, num_classes=6, vocab_size=151643, max_seq_len=250, save_path=""):
        super().__init__()
        # 1. Embedding 层：将离散的 VQ indices (0-511) 映射为连续向量
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        # 2. 位置编码：赋予 Transformer 处理序列顺序的能力
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)
        # 3. Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4,
            # dropout=0.1, activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 分类头 (Mapping to LLM Vocab)
        self.ln = nn.LayerNorm(embed_dim)
        '''Qwen 词表达15 万，线性层占较大显存（128x 151643x 4 字节 = 77 MB）
        如果显存紧张，可减小embed_dim'''
        self.save_path = save_path
        if 'vocab6' in self.save_path:
            self.fc_out = nn.Linear(embed_dim, num_classes)
        else:
            if 'CtrLoss' in self.save_path:
                print("Contrastive_Loss output embed dim:", 3584) # 
                # self.fc_out = nn.Linear(embed_dim, 3584) # Qwen2.5-7B 的隐藏层维度是 3584
                self.projection = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(embed_dim * 2, 3584) )
            else:
                print("Vocab Size:", vocab_size)  # 151665
                self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, vq_indices):
        """
        vq_indices: [Batch, Seq_Len] (例如 [32, 250])
        """
        # [B, T] -> [B, T, E]
        x = self.embedding(vq_indices) 
        x = self.pos_encoder(x)
        
        # Transformer 编码
        # output shape: [B, T, E]
        x = self.transformer_encoder(x)
        
        # 全局池化：将序列维度压缩，提取整段信号的特征
        # 也可以使用 x[:, 0, :] 如果你在输入中添加了特殊的 [CLS] token
        x = torch.mean(x, dim=1)
        x = self.ln(x)
        
        if 'CtrLoss' in self.save_path:
            x = self.projection(x)
            # L2 归一化非常关键，但要确保它是在非线性映射之后
            return F.normalize(x, p=2, dim=1)
        else:
            return self.fc_out(x) # [B, Vocab_Size]
        

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
    
class VQToTokenModel(nn.Module):
    def __init__(self, codebook_size, embed_dim=1, vocab_size=151643): # Qwen2.5 vocab size
        super().__init__()
        # 假设输入是 VQVAE 产生的 indices 序列 [B, T_seq]
        # 我们使用一个简单的 Embedding + GRU/Transformer 或直接对每个 index 进行分类
        self.codebook_size = codebook_size
        
        # 映射层：将 VQ 的索引特征映射到 LLM 的词表空间
        # 如果是针对整个窗口做一个分类：
        self.classifier = nn.Sequential(
            nn.Embedding(codebook_size, 128),
            nn.Flatten(),
            nn.Linear(128 * 250, 512), # 250 是 VQVAE 输出的序列长度 (1000/2/2)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, vocab_size)
        )

    def forward(self, vq_indices):
        # vq_indices shape: [B, T_seq] (e.g., [32, 250])
        logits = self.classifier(vq_indices) # [B, vocab_size]
        return logits

