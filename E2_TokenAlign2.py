# 实验二  不同压缩条件下的Ours
# TokenAlign2.py

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
from E2_VQTokenizer2 import NeuralVQVAE
from data import DataLoader, MillerFingersDataset_TT, np, torch
import math
from torch.optim.lr_scheduler import LambdaLR
from collections import Counter

  

def calculate_compression_metrics_final(vqvae_model, sampling_rate, sample_idx=0):
    """
    计算 BCI Competition IV Dataset 4 的原始比特率与 Token 比特率
    """
    test_data = MillerFingersDataset_TT(datadir, window_size=winsize, stride=256, single_wind=False, 
                single_channel=True, channel_idx=None, subset=subset, split_ratio=split_ratio)
    test_data.set_mode('test')
    
    vqvae_model.eval() # 必须进入 eval 模式以关闭死索引重置逻辑
    device = next(vqvae_model.parameters()).device
    
    # 1. 原始比特率参数 (Raw Bitrate)
    # 解析数据  返回 (ecog_tensor, label)
    sample_data = test_data[sample_idx]
    ecog_tensor = sample_data[0]  # 提取特征 Tensor
    # 自动计算比特深度 (Bit Depth)
    # torch.float32 -> 32, torch.float64 -> 64, torch.int16 -> 16, torch.uint8 -> 8
    bit_depth = ecog_tensor.element_size() * 8  # bits_per_val = 16     # 采样精度 (bits)
    n_channels = ecog_tensor.shape[0]  # 动态获取通道数：如果是单通道 [1, 1000] 则 n_channels=1
    window_size = ecog_tensor.shape[1]
    print(f"DEBUG: Feature shape = {ecog_tensor.shape}") # [1, 1000]
    print(f"DEBUG: Detected Bit Depth = {bit_depth} bits")
    
    # 2. 原始比特率参数 (Raw Bitrate)
    raw_bps = sampling_rate * n_channels * bit_depth
        
    # 提取神经 Token
    ecog_tensor, _ = test_data[sample_idx]
    ecog_tensor = ecog_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 编码并获取索引
        z = vqvae_model.encoder(ecog_tensor)
        x_recon, z_e, z_q, indices= vqvae_model.vq_layer(z)
        print(f"DEBUG: VQ Indices shape = {indices.shape}")  # [1, T']
    
    # 3. Token 比特率参数 (Token Bitrate)
    # 每个窗口的时间长度 (秒)
    window_sec = window_size / sampling_rate 
    # 窗口内产生的 Token 数量
    num_tokens = indices.numel()
    print(f"DEBUG: Number of Tokens = {num_tokens}")
    # 每个 Token 的信息量 (Codebook 为 512，则为 9 bits)
    codebook_size = vqvae_model.vq_layer._embedding_dim # n_e
    bits_per_token = np.log2(codebook_size)
    
    token_bps = (num_tokens * bits_per_token) / window_sec
    
    # 4. 计算压缩比
    compression_ratio = raw_bps / token_bps
    
    print("\n" + "="*40, " 神经词汇压缩报告 (BCICIV 4)", "="*40)
    print(f"原始信号带宽 (Raw):    {raw_bps/1000:10.2f} kbps")
    print(f"神经 Token 带宽 (LLM): {token_bps/1000:10.4f} kbps")
    print(f"带宽缩减倍数:         {compression_ratio:10.2f} X")
    print(f"下采样倍率 (Stride):   {window_size/num_tokens:10.1f} : 1")
    print("-" * 40)
    print(f"Big Claim: 仅需原始带宽的 {1/compression_ratio:.4%}, 即可传输完整运动意图。")
    print("="*40)
    
    return {"raw_bps": raw_bps, "token_bps": token_bps, "ratio": compression_ratio}


# --- 1. 标签映射配置 ---
# 定义 fingerflex 子集的 label 到 文本的映射
LABEL_MAP = {
    0: "Inter-stimulus interval", # -> GT Token ID: 3306
    1: "thumb", # -> GT Token ID: 25036
    2: "index finger", #  -> GT Token ID: 1252
    3: "middle finger", #  -> GT Token ID: 19656
    4: "ring finger", #  -> GT Token ID: 12640
    5: "little finger" # -> GT Token ID: 55392
}


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


class TransformerVQToTokenModel6_Robust64(nn.Module):
    def __init__(self, codebook_size, embed_dim=256, nhead=8, num_layers=3, num_classes=6, 
                 max_seq_len=64, num_channels=64):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        self.embed_dim = embed_dim
        # 增加一个线性层，用于将通道维度压缩或整合
        # 方案：将 64 个通道的 embedding 融合
        self.channel_fusion = nn.Linear(num_channels * embed_dim, embed_dim)
        
        self.ms_extractor = MultiScaleFeatureExtractor(embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)
        
        # ... Transformer 定义保持不变 ...
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=0.4, batch_first=True, activation='gelu'
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
        # vq_indices: [B, 64, T_vq]
        B, C, T = vq_indices.shape
        E = self.embed_dim
        # 1. Embedding: [B, 64, T] -> [B, 64, T, E]
        x = self.embedding(vq_indices)
        # print('1. x.shape = ', x.shape)
        
        # 2. 空间融合 (Spatial-Temporal Interaction)
        # 变换形状为 [B, T, 64 * E]
        x = x.transpose(1, 2).reshape(B, T, C * self.embed_dim)
        # print('2. x.shape = ', x.shape)
        # # 第一步：调换维度顺序 [B, 64, T, E] -> [B, T, 64, E]
        # x = x.permute(0, 2, 1, 3).contiguous() 
        # # 第二步：合并最后两个维度 [B, T, 64 * E] 此时 x 的最后一维大小为 16384 (64 * 256)
        # x = x.view(B, T, C * E)
        
        # 投影回 [B, T, E] 输出形状将变为 [B, T, 256]
        # x 是 [B, T, 16384]，self.channel_fusion 是 Linear(16384, 256)
        x = self.channel_fusion(x)
        
        # 3. 进入鲁棒提取流程
        x = self.ms_extractor(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.ln_final(x)
        logits = self.fc_head(x)
        return logits


# --- 3. 训练对齐器类 ---
class TokenAlignmentTrainer:
    def __init__(self, model_path="/mnt/home/user1/MCX/EEGLM/Qwen2.5-7B-Instruct", subset='fingerflex'):
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


# --- 2. 训练 ---
def train():
    if os.path.exists(tokenmodel_path):
        print("检测到已有的 Token 对齐模型权重，直接加载...")
        token_model.load_state_dict(torch.load(tokenmodel_path))
        
    # 🌟 优化 A: 动态权重策略 给类0(静息态)极低的权重，给其他手指更高的权重，强制模型跳出舒适区
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
    best_epoch = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        token_model.train()
        vqvae.eval()
        
        all_preds, all_gts = [], []
        total_loss = 0

        for x, labels in train_loader:
            # x 现在的形状是 [Batch, 64, 1024]
            x, labels = x.to(device), labels.to(device)
            B, C, T = x.shape  # B=Batch, C=64, T=1024
            # print('x.shape = ', x.shape)

            optimizer.zero_grad()
            with torch.no_grad():
                # --- 🌟 关键修改：多通道处理 ---
                # 1. 重塑为 [B*C, 1, T] 以符合单通道 VQ-VAE 的输入格式
                x_reshaped = x.view(B * C, 1, T)
                
                # 2. 逐通道提取 VQ 索引
                # indices 形状为 [B*C, T_vq] (T_vq 是下采样后的序列长度)
                _, _, _, indices = vqvae(x_reshaped)
                
                # 3. 恢复通道维度 [B, C, T_vq]
                T_vq = indices.shape[-1]
                indices = indices.view(B, C, T_vq)
                
                # 🌟 优化 C: 随机平移 (对所有通道应用相同的位移，保持空间同步)
                if torch.rand(1) > 0.5:
                    shift = torch.randint(-4, 5, (1,)).item()
                    indices = torch.roll(indices, shifts=shift, dims=2)

            # 4. 输入 token_model # 注意：token_model 的输入现在需要处理 [B, 64, T_vq] 的形状
            # print('indices.shape = ', indices.shape)
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
        
        # 保存最优模型
        torch.save(token_model.state_dict(), tokenmodel_path)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(token_model.state_dict(), tokenmodel_path.replace(".pth", "_best.pth"))
            print(f"⭐ New Best Model Saved (Acc: {best_acc*100:.2f}%)")
        else:
            patience_counter += 1
        
        # 打印详细报告
        train_acc = np.mean(np.array(all_preds) == np.array(all_gts)) * 100
        pred_dist = Counter(all_preds)
        print(f"\n--- Epoch {epoch} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc*100:.2f}% | BestVal Acc: {best_acc*100:.2f}% at Epoch {best_epoch} | LR: {optimizer.param_groups[0]['lr']:.8f}")
        print(f"预测分布: {[f'C{k}: {v/len(all_preds)*100:.1f}%' for k, v in sorted(pred_dist.items())]}")

        ema.restore() # 恢复原始参数继续下一轮训练

        # 🌟 优化 E: 早停机制 (Early Stopping)
        if patience_counter > 500: # 如果 60 轮没提升则停止
            print("触发早停，训练结束。")
            break


# --- 3. 测试 ---
from evaluate import save_confusion_matrix
def test(epoch):
    token_model.eval()
    vqvae.eval()
    all_preds, all_gts = [], []
    
    with torch.no_grad():
        for x, labels in test_loader:
            # x 形状: [Batch, 64, 1000]
            x, labels = x.to(device), labels.to(device)
            B, C, T = x.shape 
            
            # --- 🌟 1. 维度折叠 (Channel Folding) ---
            # 将通道合并到 Batch 维度，以适应单通道 VQ-VAE
            x_reshaped = x.view(B * C, 1, T) # [B*64, 1, 1000]
            
            # --- 🌟 2. 逐通道提取 VQ 索引 ---
            # 注意：此处只获取 indices
            _, _, _, indices = vqvae(x_reshaped) 
            
            # --- 🌟 3. 维度还原 (Channel Unfolding) ---
            # 还原为 [Batch, Channels, Seq_Len]
            T_vq = indices.shape[-1]
            indices = indices.view(B, C, T_vq) # [B, 64, T_vq]
            
            # --- 🌟 4. 多通道特征融合与分类 ---
            # 此时 token_model 内部应包含处理 [B, 64, T_vq] 的融合层
            logits = token_model(indices)
            
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())

    # 计算准确率
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    final_acc = np.mean(all_preds == all_gts)
    
    # 保存混淆矩阵
    save_confusion_matrix(all_gts, all_preds, epoch, final_acc, LABEL_MAP, save_path)
    
    print(f"\n✅ Epoch {epoch} 验证完成! 总样本: {len(all_gts)}, 平均准确率: {final_acc*100:.2f}%")
    return final_acc


for modeltype in range(1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    EPOCHS = 1000
    LEARNING_RATE = 1e-4
    VQToTokenembed_dim = 256
    subset = 'fingerflex'
    winsize = 1000
    WARMUP_EPOCHS = 10
    split_ratio = 0.6
    num_classes = 6
    datadir = '/disk2/user1/dataset/BCI-Standford/%s/%s/data/' % (subset, subset)
    
    if modeltype == 0:
        codebook = 512; VQVAEembed_dim = 64; Tokenstride = 4
    elif modeltype == 1:
        codebook = 256; VQVAEembed_dim = 64; Tokenstride = 8
    elif modeltype == 2:
        codebook = 256; VQVAEembed_dim = 32; Tokenstride = 8
    elif modeltype == 3:
        codebook = 256; VQVAEembed_dim = 32; Tokenstride = 16
    elif modeltype == 4:
        codebook = 256; VQVAEembed_dim = 64; Tokenstride = 32
    # elif modeltype == 5:
    #     codebook = 256; VQVAEembed_dim = 64; Tokenstride = 16
    # elif modeltype == 6:
    #     codebook = 256; VQVAEembed_dim = 32; Tokenstride = 32
        
    vqvae = NeuralVQVAE(in_channels=1, codebook_size=codebook,
                embed_dim=VQVAEembed_dim, num_classes=num_classes, tokenstride=Tokenstride).to(device)
    
    if VQVAEembed_dim == 64: EM = ''
    else: EM = '_E%d' % VQVAEembed_dim
    if Tokenstride == 4: TStride = ''
    elif Tokenstride == 8: TStride = 'token31'
    elif Tokenstride == 16: TStride = 'token16'
    elif Tokenstride == 32: TStride = 'token8'    
    
    save_path = "/mnt/home/user1/MCX/EEGLM/ECogLM/checkpoints/Stanford/TokenAlign/%s_vocab6-4-c%d%s%s_ratio6/" % (subset, codebook, EM, TStride) # 64-2
    VQVAEmodelpath = "/mnt/home/user1/MCX/EEGLM/ECogLM/checkpoints/Stanford/VQTokenizer/%s_c%d_4%s%s_ratio6/best_vqvae_model.pth" % \
            (subset, codebook, EM, TStride)
    print(save_path, '\n', VQVAEmodelpath)
    
    full_dataset = MillerFingersDataset_TT(datadir, window_size=winsize, stride=256, single_wind=False, 
                single_channel=False, channel_idx=None, subset=subset, split_ratio=split_ratio)
    num_channels = full_dataset.samples[0].shape[-2]
    print('full_dataset.samples[0].shape, num_classes = ', full_dataset.samples[0].shape, num_classes)
    print("...加载 ECoG 数据集...")
    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    full_dataset.set_mode('test')
    test_data = full_dataset
    test_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)
    
    print("... Build Model ...")
    aligner = TokenAlignmentTrainer(subset=subset)

    _ = calculate_compression_metrics_final(vqvae, sampling_rate=1000, sample_idx=10)
        
    vqvae.load_state_dict(torch.load(VQVAEmodelpath)['model_state_dict'])
    # 初始化对齐模型
    os.makedirs(save_path, exist_ok=True)
    tokenmodel_path = f"{save_path}token_aligner.pth"
    
    with torch.no_grad():
        vqvae.eval()
        test_input = torch.zeros(1, 1, winsize).to(device)
        _, _, _, dummy_indices = vqvae(test_input)
        actual_seq_len = dummy_indices.shape[1]
    print(f"Detected VQ Sequence Length: {actual_seq_len}", "dummy_indices.shape = ", dummy_indices.shape)
    token_model = TransformerVQToTokenModel6_Robust64(
        codebook_size=codebook, embed_dim=VQToTokenembed_dim,
        num_classes=len(LABEL_MAP), max_seq_len=actual_seq_len, num_channels=num_channels).to(device)
    
    # train()
    
    print("加载已有的 Token 对齐模型权重...")
    # token_model.load_state_dict(torch.load(tokenmodel_path))
    # test(EPOCHS)
    token_model.load_state_dict(torch.load(tokenmodel_path.replace('.pth', '_best.pth')))
    test(0)
    

'''
DEBUG: Feature shape = torch.Size([1, 1000])
DEBUG: Detected Bit Depth = 32 bits
DEBUG: VQ Indices shape = torch.Size([1, 62])
DEBUG: Number of Tokens = 62
========================================  神经词汇压缩报告 (BCICIV 4) ========================================
原始信号带宽 (Raw):         32.00 kbps
神经 Token 带宽 (LLM):     0.3720 kbps
带宽缩减倍数:              86.02 X
下采样倍率 (Stride):         16.1 : 1
----------------------------------------
Big Claim: 仅需原始带宽的 1.1625%, 即可传输完整运动意图。
========================================
/checkpoints/Stanford/TokenAlign/fingerflex_vocab6-4_ratio6/cm_epoch_0.png
✅ Epoch 0 验证完成! 总样本: 7267, 平均准确率: 33.26%


DEBUG: Feature shape = torch.Size([1, 1000])
DEBUG: Detected Bit Depth = 32 bits
DEBUG: VQ Indices shape = torch.Size([1, 31])
DEBUG: Number of Tokens = 31
========================================  神经词汇压缩报告 (BCICIV 4) ========================================
原始信号带宽 (Raw):         32.00 kbps
神经 Token 带宽 (LLM):     0.1860 kbps
带宽缩减倍数:             172.04 X
下采样倍率 (Stride):         32.3 : 1
----------------------------------------
Big Claim: 仅需原始带宽的 0.5813%, 即可传输完整运动意图。
========================================
/checkpoints/Stanford/TokenAlign/fingerflex_vocab6-4-c256token31_ratio6/cm_epoch_0.png
✅ Epoch 0 验证完成! 总样本: 7267, 平均准确率: 100.00%


DEBUG: Feature shape = torch.Size([1, 1000])
DEBUG: Detected Bit Depth = 32 bits
DEBUG: VQ Indices shape = torch.Size([1, 31])
DEBUG: Number of Tokens = 31
========================================  神经词汇压缩报告 (BCICIV 4) ========================================
原始信号带宽 (Raw):         32.00 kbps
神经 Token 带宽 (LLM):     0.1550 kbps
带宽缩减倍数:             206.45 X
下采样倍率 (Stride):         32.3 : 1
----------------------------------------
Big Claim: 仅需原始带宽的 0.4844%, 即可传输完整运动意图。
========================================
/checkpoints/Stanford/TokenAlign/fingerflex_vocab6-4-c256_E32token31_ratio6/cm_epoch_0.png
✅ Epoch 0 验证完成! 总样本: 7267, 平均准确率: 99.99%


DEBUG: Feature shape = torch.Size([1, 1000])
DEBUG: Detected Bit Depth = 32 bits
DEBUG: VQ Indices shape = torch.Size([1, 16])
DEBUG: Number of Tokens = 16
========================================  神经词汇压缩报告 (BCICIV 4) ========================================
原始信号带宽 (Raw):         32.00 kbps
神经 Token 带宽 (LLM):     0.0800 kbps
带宽缩减倍数:             400.00 X
下采样倍率 (Stride):         62.5 : 1
----------------------------------------
Big Claim: 仅需原始带宽的 0.2500%, 即可传输完整运动意图。
========================================
/checkpoints/Stanford/TokenAlign/fingerflex_vocab6-4-c256_E32token16_ratio6/cm_epoch_0.png
✅ Epoch 0 验证完成! 总样本: 7267, 平均准确率: 100.00%

DEBUG: Feature shape = torch.Size([1, 1000])
DEBUG: Detected Bit Depth = 32 bits
DEBUG: VQ Indices shape = torch.Size([1, 8])
DEBUG: Number of Tokens = 8
========================================  神经词汇压缩报告 (BCICIV 4) ========================================
原始信号带宽 (Raw):         32.00 kbps
神经 Token 带宽 (LLM):     0.0480 kbps
带宽缩减倍数:             666.67 X
下采样倍率 (Stride):        125.0 : 1
----------------------------------------
Big Claim: 仅需原始带宽的 0.1500%, 即可传输完整运动意图。
========================================
eckpoints/Stanford/TokenAlign/fingerflex_vocab6-4-c256token8_ratio6/cm_epoch_0.png
✅ Epoch 0 验证完成! 总样本: 7267, 平均准确率: 100.00%
'''

