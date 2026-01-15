import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer # AutoModelForCausalLM,
import os
from VQTokenizer2 import NeuralVQVAE
# from torch.utils.data import DataLoader, random_split, Dataset
from data import MillerFingersDataset, BCIComp4Dataset4_single, prepare_dataloaders #  ClassBalancedBatchSampler, random,
import math
from torch.optim.lr_scheduler import LambdaLR
from collections import Counter
import shutil
import numpy as np


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

fingerflextoken2_FullToken = {
    3306:[3306, 5477, 318, 19425, 9873],
    25036:[25036],
    1252:[1252, 14317],
    19656:[19656, 14317],
    12640:[12640, 14317],
    55392:[55392, 14317]   
}
GT_token2_FullToken = fingerflextoken2_FullToken


# --- 2. 增强型模型组件 ---
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # 使用不同的空洞率 (dilation) 或 核大小，捕捉多尺度特征
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(embed_dim * 3)
        self.proj = nn.Linear(embed_dim * 3, embed_dim)

    def forward(self, x):
        # x: [B, T, E] -> [B, E, T]
        x = x.transpose(1, 2)
        c1 = F.gelu(self.conv1(x))
        c2 = F.gelu(self.conv2(x))
        c3 = F.gelu(self.conv3(x))
        out = torch.cat([c1, c2, c3], dim=1)
        out = self.bn(out).transpose(1, 2)
        return self.proj(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerVQToTokenModel6_Robust(nn.Module):
    def __init__(self, codebook_size, embed_dim=256, nhead=8, num_layers=3, num_classes=6, max_seq_len=64):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        
        # 引入多尺度卷积增强
        self.ms_extractor = MultiScaleFeatureExtractor(embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4,
            dropout=0.4, # 增加 Dropout 防止过拟合
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.ln_final = nn.LayerNorm(embed_dim)
        self.fc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_seq_len * embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, vq_indices):
        # [B, T] -> [B, T, E]
        x = self.embedding(vq_indices)
        
        # 1. 卷积提取局部特征
        x = self.ms_extractor(x)
        # 2. 位置编码
        x = self.pos_encoder(x)
        # 3. Transformer 全局建模
        x = self.transformer_encoder(x)
        # 4. 归一化与分类
        x = self.ln_final(x)
        logits = self.fc_head(x)
        return logits

# --- 3. 训练对齐器类 ---

class TokenAlignmentTrainer:
    def __init__(self, model_path="../Qwen2.5-7B-Instruct", subset='fingerflex'):
        # 仅在需要 Token 映射时初始化 Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except:
            self.tokenizer = None
        self.gt_token_map = {}
        if self.tokenizer:
            for idx, text in LABEL_MAP.items():
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                self.gt_token_map[idx] = tokens[0]

    def get_local_labels(self, labels_tensor):
        return labels_tensor.to(torch.long)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    创建带预热的余弦调度器
    num_warmup_steps: 预热的 Epoch 数
    num_training_steps: 总训练 Epoch 数
    """
    def lr_lambda(current_step):
        # 1. 线性预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 2. 余弦退火阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)


from torch.optim.lr_scheduler import ReduceLROnPlateau


# --- 权重平滑类 ---
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# --- 2. 训练流程优化 ---
def train_alignment():
    # 🌟 优化 A: 动态权重策略
    # 给类0(静息态)极低的权重，给其他手指更高的权重，强制模型跳出舒适区
    class_weights = torch.tensor([0.05, 1.2, 1.2, 1.2, 1.2, 1.2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(token_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    
    # 🌟 优化 B: 组合式学习率策略
    warmup_scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                      num_warmup_steps=WARMUP_EPOCHS, 
                                                      num_training_steps=EPOCHS)
    # Plateau 调度器监控验证集准确率，如果在 15 轮内没提升，则减半学习率
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, verbose=True)
    
    ema = EMA(token_model, 0.999)
    ema.register()

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        token_model.train()
        vqvae.eval() # 始终冻结 VQVAE
        
        all_preds, all_gts = [], []
        total_loss = 0

        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.no_grad():
                _, _, _, indices = vqvae(x)
                
                # 🌟 优化 C: 训练时增强 - 随机平移 (Temporal Jitter)
                # 这能有效防止 Transformer 过度背诵 VQ 序列的绝对位置
                if torch.rand(1) > 0.5:
                    shift = torch.randint(-4, 5, (1,)).item()
                    indices = torch.roll(indices, shifts=shift, dims=1)

            logits = token_model(indices)
            loss = criterion(logits, labels)
            loss.backward()
            
            # 🌟 优化 D: 严格梯度裁剪，防止 ECoG 异常波动
            torch.nn.utils.clip_grad_norm_(token_model.parameters(), max_norm=0.5)
            optimizer.step()
            ema.update()

            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())
            total_loss += loss.item()

        # 调整学习率
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()

        # 🌟 验证阶段：切换到 EMA 权重进行公平测试
        ema.apply_shadow()
        val_acc = test(epoch)
        
        # 更新 Plateau 调度器
        if epoch >= WARMUP_EPOCHS:
            plateau_scheduler.step(val_acc)
        
        # 打印详细报告
        train_acc = np.mean(np.array(all_preds) == np.array(all_gts)) * 100
        pred_dist = Counter(all_preds)
        print(f"\n--- Epoch {epoch} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc*100:.2f}% | LR: {optimizer.param_groups[0]['lr']:.8f}")
        print(f"预测分布: {[f'C{k}: {v/len(all_preds)*100:.1f}%' for k, v in sorted(pred_dist.items())]}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(token_model.state_dict(), tokenmodel_path.replace(".pth", "_best.pth"))
            print(f"⭐ New Best Model Saved (Acc: {best_acc*100:.2f}%)")
        else:
            patience_counter += 1

        ema.restore() # 恢复原始参数继续下一轮训练

        # 🌟 优化 E: 早停机制 (Early Stopping)
        if patience_counter > 60: # 如果 60 轮没提升则停止
            print("触发早停，训练结束。")
            break
        
# --- 4. 测试脚本 ---
from test_VQTokenizer import diagnose_vq_collapse, calculate_accuracy, save_confusion_matrix, analyze_token_distribution
def test(epoch):
    token_model.eval()
    vqvae.eval()
    all_preds, all_gts = [], []
    
    with torch.no_grad():
        for x, labels in test_loader:
            x, labels = x.to(device), labels.to(device)
            _, _, _, indices = vqvae(x)
            logits = token_model(indices)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())

    final_acc = np.mean(np.array(all_preds) == np.array(all_gts))
    save_confusion_matrix(all_gts, all_preds, epoch, final_acc, LABEL_MAP, save_path)
    
    print(f"\n✅ 验证完成! 平均准确率: {final_acc*100:.2f}%")
    return final_acc


def test0(epoch):
    print(">>> 开始测试 TransformerVQToTokenModel 前向传播...")
    # 形状为 [Batch, Seq_Len]，值在 0 到 CODEBOOK_SIZE-1 之间
    vqvae.eval()
    token_model.eval()

    total_acc = 0
    total_samples = 0
    # criterion = nn.CrossEntropyLoss()
    # # 建立一个反向映射，用于打印结果 (ID -> 文本)
    # id_to_text = {v: aligner.tokenizer.decode([v]) for v in aligner.gt_token_map.values()}
    all_preds = []
    all_gts = []
    with torch.no_grad():            
        for x, labels in test_loader:
            x, labels = x.to(device), labels.to(device)
            # 1. 提取 VQ 索引
            x_recon, z_e, z_q, indices = vqvae(x)  # print("VQVAE Indices shape: ", indices.shape)
            # 2. 获取 GT Tokens
            if 'vocab6' in save_path:
                gt_tokens = aligner.get_local_labels(labels) # [B]
            else:
                gt_tokens = aligner.get_gt_batch(labels) # [B]
            # print("GT Tokens shape: ", gt_tokens.shape) torch.Size([1])
            
            # 3. 前向传播 # 4. 计算损失
            logits = token_model(indices)
            # print(f"输入形状: {indices.shape}") # [1, 250]
            # print(f"输出形状 (Logits): {logits.shape}, GT {gt_tokens.shape}") # [1, 151643]
            # 输出形状 (Logits): torch.Size([1, 6])
                
            # loss = criterion(logits, gt_tokens)
            acc, preds = calculate_accuracy(logits, gt_tokens)
            
            # 收集数据用于混淆矩阵
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(gt_tokens.cpu().numpy())
            
            total_acc += acc * x.size(0)
            total_samples += x.size(0) 
            # print(f"Loss: {loss.item():.4f} | Batch Acc: {acc*100:.2f}%")
            # # 显示预测结果
            # for i in range(min(20000000, x.size(0))):
            #     pred_text = aligner.tokenizer.decode([preds[i]])
            #     gt_text = aligner.tokenizer.decode([gt_tokens[i]])
            #     print(f"  Sample {i}: Predicted: '{pred_text}'({preds[i]}), GT: '{gt_text}'({gt_tokens[i]})")

    final_acc = total_acc / total_samples # 1822
    print(f"\n✅ 验证完成! 总测试样本: {total_samples}, 平均准确率: {final_acc*100:.2f}%")
    # --- 核心逻辑：计算并保存混淆矩阵 ---
    save_confusion_matrix(all_gts, all_preds, epoch, final_acc, LABEL_MAP, save_path)
    
    print(f"\n✅ 验证完成! 平均准确率: {final_acc*100:.2f}%")
    return final_acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 42  # 
    EPOCHS = 1000
    LEARNING_RATE = 1e-4
    codebook = 512  # 128  # 256  #
    VQVAEembed_dim = 64  # 1  # 
    VQToTokenembed_dim = 256  # 128  # 64  # 
    subset = 'fingerflex'  # 'motor_basic'  # 'gestures' # 'BCIComp4'  #  'joystick_track2' #
    winsize = 1000  # 1024  # 
    WARMUP_EPOCHS = 10  # 前 10 个 epoch 慢慢增加 LR
    
    if subset =='BCIComp4': 
        save_path = './checkpoints/BCI Competition IV/TokenAlign_c%d_emb%d/' % (codebook, VQToTokenembed_dim)  # 
        VQVAEmodelpath = "./checkpoints/BCI Competition IV/VQTokenizer/best_vqvae_model.pth"
        
        datadir = '/mnt/home/user1/MCX/EEGLM/data/ECoG/BCICIV_4_mat/'
        full_dataset = BCIComp4Dataset4_single(datadir=datadir, window_size=winsize, 
                stride=256, single_channel=True, channel_idx=None) # 单通道stride=512,
        # full_dataset = BCIComp4Dataset4(datadir=datadir, window_size=winsize, stride=256)
    else:
        save_path = "./checkpoints/Stanford/TokenAlign/%s_vocab6-2/" % (subset)
        # save_path = "./checkpoints/Stanford/TokenAlign/%s_c%d/" % (subset, codebook)
        
        VQVAEmodelpath = "./checkpoints/Stanford/VQTokenizer/%s_c%d_2/vqvae_model.pth" % (subset, codebook) # best_
        # random.seed(0)
        datadir = '/disk2/user1/dataset/BCI-Standford/%s/%s/data/' % (subset, subset)
        full_dataset = MillerFingersDataset(datadir, window_size=winsize, stride=256,
                        single_channel=True, channel_idx=None, subset=subset)  # 单通道 channel_idx=0,

    num_classes = full_dataset.num_classes
    
    aligner = TokenAlignmentTrainer(subset=subset)    
    vqvae = NeuralVQVAE(in_channels=1, codebook_size=codebook, 
                        embed_dim=VQVAEembed_dim, num_classes=num_classes).to(device)  # 
    
    vqvae.load_state_dict(torch.load(VQVAEmodelpath)['model_state_dict'])
    
    # 初始化对齐模型
    os.makedirs(save_path, exist_ok=True)
    tokenmodel_path = f"{save_path}token_aligner.pth"
    # token_model = TransformerVQToTokenModel(codebook_size=codebook, embed_dim=VQToTokenembed_dim,
    #         nhead=8, num_layers=4, num_classes=len(LABEL_MAP), vocab_size=len(aligner.tokenizer), 
    #         max_seq_len=250).to(device)
    
    with torch.no_grad():
        vqvae.eval()
        test_input = torch.zeros(1, 1, winsize).to(device)
        _, _, _, dummy_indices = vqvae(test_input)
        actual_seq_len = dummy_indices.shape[1]
    print(f"Detected VQ Sequence Length: {actual_seq_len}")
    token_model = TransformerVQToTokenModel6_Robust(
        codebook_size=codebook, 
        embed_dim=VQToTokenembed_dim,
        num_classes=len(LABEL_MAP), 
        max_seq_len=actual_seq_len 
    ).to(device)
    
    print("...加载 ECoG 数据集...")
    # torch.manual_seed(0)
    train_loader, test_loader, test_data = prepare_dataloaders(full_dataset, BATCH_SIZE, num_classes=len(LABEL_MAP), ratio=0.9)
    # train_data, test_data = random_split(full_dataset, [int(len(full_dataset)*0.9),
    #     len(full_dataset)-int(len(full_dataset)*0.9)])    
    # print(f"数据集加载成功: 训练样本数={len(train_data)}, 测试样本数={len(test_data)}")
    # sampler = ClassBalancedBatchSampler(train_data, batch_size=BATCH_SIZE, num_classes=6, 
    #                         samples_per_class=BATCH_SIZE//6)
    # train_loader = DataLoader(train_data, batch_sampler=sampler)
    # # train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # analyze_token_distribution(vqvae, train_loader, test_loader, codebook, device)
    # exit()
    
    # diagnose_vq_collapse(test_data, vqvae, device, len(full_dataset)-int(len(full_dataset)*0.9))
    # # diagnose_vq_collapse(full_dataset, vqvae, device)

    train_alignment()
    
    # 执行测试
    print("加载已有的 Token 对齐模型权重...")
    token_model.load_state_dict(torch.load(tokenmodel_path))
    test(EPOCHS)
    token_model.load_state_dict(torch.load(tokenmodel_path.replace('.pth', '_best.pth')))
    test(0)
    
    