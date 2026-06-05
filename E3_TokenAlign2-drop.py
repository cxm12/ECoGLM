# 实验三  不同丢包率条件下的Ours   TokenAlign2-drop.py

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from E2_VQTokenizer2 import NeuralVQVAE
from data import MillerFingersDataset_Drop, DataLoader, np, os, torch
import math
from torch.optim.lr_scheduler import LambdaLR
from collections import Counter


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
def encode_to_indices(x, vqvae):
    B, C, T = x.shape
    x_reshaped = x.view(B * C, 1, T)  # [B*C, 1, T]
    
    # Encoder 是可训练的
    z_e = vqvae.encoder(x_reshaped)  # [B*C, D, T_vq]
    
    # vq_layer 处于 eval 模式 → 固定 codebook，无 EMA 更新
    _, quantized, _, indices = vqvae.vq_layer(z_e)
    # indices shape: [B*C, T_vq]
    
    T_vq = indices.shape[-1]
    indices = indices.view(B, C, T_vq)  # [B, C, T_vq]

    if torch.rand(1) > 0.5:
        shift = torch.randint(-4, 5, (1,)).item()
        indices = torch.roll(indices, shifts=shift, dims=2)
    return indices


def train():
    dataset = MillerFingersDataset_Drop(
        datadir=datadir, window_size=winsize, stride=stride,
        packet_loss_rate=plr,
        loss_type=LOSS_TYPE, burst_max_len=50, single_wind=False,
        single_channel=False, subset=subset, split_ratio=split_ratio)
    dataset.set_mode('train')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    dataset.set_mode('test')
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    num_channels = dataset[0][0].shape[0]
    num_classes = len(set(dataset.labels))
    
    save_path = "/mnt/home/user1/MCX/EEGLM/ECogLM/checkpoints/Stanford/Baseline/Drop/TokenAlign%s%s%s%s/" % (subset, LOSS_TYPE, str(int(plr*10)), ratio)
    os.makedirs(save_path, exist_ok=True)
    tokenmodel_path = os.path.join(save_path, "token_aligner.pth")
    vqvae_finetuned_path = os.path.join(save_path, "vqvae_finetuned.pth")
    best_token_path = os.path.join(save_path, "token_aligner_best.pth")
    best_vqvae_path = os.path.join(save_path, "vqvae_finetuned_best.pth")

    # ====== 1. Load & Freeze VQ-VAE ======
    print("... Loading and freezing VQ-VAE ...")
    vqvae = NeuralVQVAE(in_channels=1, codebook_size=codebook, 
                        embed_dim=VQVAEembed_dim, num_classes=num_classes, tokenstride=Tokenstride).to(device)
    vqvae.load_state_dict(torch.load(VQVAEmodelpath)['model_state_dict'])

    # ====== 冻结 VQ-VAE 的 decoder 和 vq_layer（固定 codebook）======
    for param in vqvae.decoder.parameters():
        param.requires_grad = False
    # 冻结 vq_layer 的 embedding（虽无梯度，但显式禁止）
    vqvae.vq_layer._embedding.weight.requires_grad = False
    # ⭐ 关键：设为 eval 模式，禁用 EMA 更新，彻底固定 codebook
    vqvae.vq_layer.eval()  # 确保 forward 中不执行 EMA 更新

    # 确保 encoder 可训练
    for param in vqvae.encoder.parameters():
        param.requires_grad = True

    # ====== 2. Build Token Model ======
    with torch.no_grad():
        vqvae.eval()
        test_input = torch.zeros(1, 1, winsize).to(device)
        _, _, _, dummy_indices = vqvae(test_input)
        actual_seq_len = dummy_indices.shape[1]
    print(f"Detected VQ Sequence Length: {actual_seq_len}")

    token_model = TransformerVQToTokenModel6_Robust64(
        codebook_size=codebook, 
        embed_dim=VQToTokenembed_dim,
        num_classes=len(LABEL_MAP),
        max_seq_len=actual_seq_len, 
        num_channels=num_channels
    ).to(device)

    # ====== 3. Loss & Optimizer ======
    if False:  # LOSS_TYPE == 'channel' and plr >= 0.5:
        class_weights = torch.tensor([0.5, 1.2, 1.2, 1.2, 1.2, 1.2]).to(device)  # channel>=0.5
    else:
        class_weights = torch.tensor([0.05, 1.2, 1.2, 1.2, 1.2, 1.2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # 联合优化：encoder (low LR) + token_model (high LR)
    optimizer = torch.optim.AdamW([
        {'params': vqvae.encoder.parameters(), 'lr': 1e-5},
        {'params': token_model.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=1e-2)

    warmup_scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, verbose=False)
    ema = EMA(token_model, 0.999)
    ema.register()

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    # ====== 5. Training Loop ======
    for epoch in range(EPOCHS):
        vqvae.encoder.train()
        token_model.train()
        all_preds, all_gts = [], []
        total_loss = 0

        for x, labels, x_HR in train_loader:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()

            # 获取 VQ indices（现在 encoder 是可训练的！）
            indices = encode_to_indices(x, vqvae)
            logits = token_model(indices)
            loss = criterion(logits, labels)
            loss.backward()

            # 分别裁剪梯度
            torch.nn.utils.clip_grad_norm_(token_model.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(vqvae.encoder.parameters(), max_norm=0.5)
            optimizer.step()
            ema.update()

            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())
            total_loss += loss.item()

        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()

        # --- Validation with EMA ---
        ema.apply_shadow()
        vqvae.eval()
        token_model.eval()
        val_preds, val_gts = [], []
        with torch.no_grad():
            for x, labels, x_HR in val_loader:
                x, labels = x.to(device), labels.to(device)
                # 验证时也走完整 encoder（但无梯度）
                indices = encode_to_indices(x, vqvae)
                logits = token_model(indices)
                _, preds = torch.max(logits, 1)
                val_preds.extend(preds.cpu().numpy())
                val_gts.extend(labels.cpu().numpy())
        val_acc = np.mean(np.array(val_preds) == np.array(val_gts))
        ema.restore()

        if epoch >= WARMUP_EPOCHS:
            plateau_scheduler.step(val_acc)

        # Save latest models
        torch.save(token_model.state_dict(), tokenmodel_path)
        torch.save(vqvae.state_dict(), vqvae_finetuned_path)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(token_model.state_dict(), best_token_path)
            torch.save(vqvae.state_dict(), best_vqvae_path)
            print(f"⭐ New Best Models Saved! Val Acc: {best_acc*100:.2f}%")
        else:
            patience_counter += 1

        # Logging
        train_acc = np.mean(np.array(all_preds) == np.array(all_gts)) * 100
        pred_dist = Counter(all_preds)
        lr1 = optimizer.param_groups[0]['lr']  # encoder
        lr2 = optimizer.param_groups[1]['lr']  # token_model
        print(f"\n--- Epoch {epoch} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc*100:.2f}% | Best: {best_acc*100:.2f}% @ Ep{best_epoch}")
        print(f"LRs: Encoder={lr1:.2e}, TokenModel={lr2:.2e}")
        print(f"Pred Dist: {[f'C{k}: {v/len(all_preds)*100:.1f}%' for k, v in sorted(pred_dist.items())]}")

        if patience_counter > 300:
            print("Early stopping triggered.")
            break
    print(f"=== Final Best Val Acc for PLR={plr}: {best_acc*100:.4f}% at Epoch {best_epoch} ===")
    return best_acc, best_epoch


# --- 4. 测试脚本 ---
def testdrop(VQVAEmodelpath, tokenmodel_path):
    full_dataset = MillerFingersDataset_Drop(
        datadir=datadir, window_size=winsize, stride=stride,
        packet_loss_rate=plr, # 基线：packet_loss_rate=0.0 低鲁棒性：packet_loss_rate=0.5
        loss_type=LOSS_TYPE, burst_max_len=50, single_wind=False,
        single_channel=False, subset=subset, split_ratio=split_ratio)
    num_classes = len(set(full_dataset.labels))
    num_channels = full_dataset.samples[0].shape[-2]
    print('full_dataset.samples[0].shape, num_classes = ', full_dataset.samples[0].shape, num_classes)
    print("...加载 ECoG 数据集...")
    
    full_dataset.set_mode('test')
    test_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)

    print("... Build Model ...")
    vqvae = NeuralVQVAE(in_channels=1, codebook_size=codebook, 
                embed_dim=VQVAEembed_dim, num_classes=num_classes, tokenstride=Tokenstride).to(device)
    try:
        vqvae.load_state_dict(torch.load(VQVAEmodelpath)['model_state_dict'])
    except:
        vqvae.load_state_dict(torch.load(VQVAEmodelpath))
        
    with torch.no_grad():
        vqvae.eval()
        test_input = torch.zeros(1, 1, winsize).to(device)
        _, _, _, dummy_indices = vqvae(test_input)
        actual_seq_len = dummy_indices.shape[1]
    print(f"Detected VQ Sequence Length: {actual_seq_len}", "dummy_indices.shape = ", dummy_indices.shape)
    token_model = TransformerVQToTokenModel6_Robust64(  # _Robust(  # 
        codebook_size=codebook, embed_dim=VQToTokenembed_dim,
        num_classes=len(LABEL_MAP), # vocab_size=len(aligner.tokenizer), 
        max_seq_len=actual_seq_len, num_channels=num_channels).to(device)
    print("检测到已有的 Token 对齐模型权重，直接加载...")
    token_model.load_state_dict(torch.load(tokenmodel_path))
    
    token_model.eval()
    vqvae.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for x, labels, x_HR in test_loader:
            # x 形状: [Batch, 64, 1000]
            x, labels = x.to(device), labels.to(device)
            B, C, T = x.shape
            x_reshaped = x.view(B * C, 1, T) # [B*64, 1, 1000]
            _, _, _, indices = vqvae(x_reshaped)
            T_vq = indices.shape[-1]
            indices = indices.view(B, C, T_vq) # [B, 64, T_vq]
            logits = token_model(indices)
            
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    final_acc = np.mean(all_preds == all_gts)
    print(f"\n✅ 验证完成! 平均准确率: {final_acc*100:.2f}%")
    return final_acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 500
    LEARNING_RATE = 1e-4
    codebook = 512
    VQVAEembed_dim = 64
    Tokenstride = 4
    stride = 256
    VQToTokenembed_dim = 256
    subset = 'fingerflex'
    winsize = 1000
    WARMUP_EPOCHS = 10
    split_ratio = 0.9  #  0.6  #
    if split_ratio == 0.6: ratio = '_ratio6'
    else: ratio = ''
    
    LOSS_TYPE = 'random'  # 'channel'  # 'burst'  # 
    BURST_MAX_LEN = 50  # 仅当 LOSS_TYPE='burst' 时有效
    resultlst = []
    result_epoch = 0
    VQVAEmodelpath = "/mnt/home/user1/MCX/EEGLM/ECogLM/checkpoints/Stanford/VQTokenizer/%s_c%d_4/vqvae_model.pth" % (subset, codebook)
    datadir = '/disk2/user1/dataset/BCI-Standford/%s/%s/data/' % (subset, subset)

    for plr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"\n=== Training with Packet Loss Rate = {plr} (Joint Fine-tuning VQ-VAE Encoder + Token Model) ===")
        # result0, result_epoch = train()
        
        save_path = "/mnt/home/user1/MCX/EEGLM/ECogLM/checkpoints/Stanford/Baseline/Drop/TokenAlign%s%s%s/" % (subset, LOSS_TYPE, str(int(plr*10)))
        best_token_path = os.path.join(save_path, "token_aligner_best.pth")
        best_vqvae_path = os.path.join(save_path, "vqvae_finetuned_best.pth")
        result = testdrop(best_vqvae_path, best_token_path)
        
        resultlst.append([result, plr, result_epoch])
        print(LOSS_TYPE, "=== Final Best Val Acc =:", resultlst)
