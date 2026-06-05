# 实验一  'joystick_track' 数据集上Ours  VQTTAlign_regress.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from vector_quantize_pytorch import VectorQuantize
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import MillerFingersDataset_TT_joystick_track
from evaluate import plot_reconstruction_comparison
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


##  === 可视化 3：动态轨迹对比动画 ===
def drawGIF(labels_orig, preds_orig, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(min(np.min(labels_orig[:, 0]), np.min(preds_orig[:, 0])) - 1,
                max(np.max(labels_orig[:, 0]), np.max(preds_orig[:, 0])) + 1)
    ax.set_ylim(min(np.min(labels_orig[:, 1]), np.min(preds_orig[:, 1])) - 1,
                max(np.max(labels_orig[:, 1]), np.max(preds_orig[:, 1])) + 1)
    line_gt, = ax.plot([], [], color='red', label='Ground Truth')
    line_pred, = ax.plot([], [], color='blue', linestyle='--', label='Prediction')
    ax.legend()
    ax.grid(True)

    def update(frame):
        line_gt.set_data(labels_orig[:frame+1, 0], labels_orig[:frame+1, 1])
        line_pred.set_data(preds_orig[:frame+1, 0], preds_orig[:frame+1, 1])
        return line_gt, line_pred,

    ani = FuncAnimation(fig, update, frames=len(labels_orig), interval=200, blit=True)
    # 保存动画为GIF文件
    ani.save(save_path+'trajectory_comparison.gif', writer='pillow')
    plt.show()


# === 绘制两条不同颜色的轨迹曲线 ===
def drawLine(labels_orig, preds_orig, save_path):
    plt.figure(figsize=(10, 6))
    # 绘制GT轨迹
    plt.plot(labels_orig[:, 0], labels_orig[:, 1], color='red', label='Ground Truth', linewidth=2)
    # 绘制预测轨迹
    plt.plot(preds_orig[:, 0], preds_orig[:, 1], color='blue', label='Prediction', linestyle='--', linewidth=2)
    plt.title('Trajectory Comparison')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.axis('equal') # 确保X轴和Y轴的比例相同，以正确显示轨迹
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path+'trajectory_comparison.png', dpi=300)
    plt.savefig(save_path+'trajectory_comparison.pdf', dpi=300)
    plt.show()


# === 绘制 X/Y 坐标随时间变化的折线图（PNG 格式）===
def drawLineXY(labels_orig, preds_orig, save_path):
    t = np.arange(len(labels_orig))  # 时间步（样本序号）

    # ===== 图 1：X 坐标对比 =====
    plt.figure(figsize=(12, 5))
    plt.plot(t, labels_orig[:, 0], color='red', label='Ground Truth (X)', linewidth=1.5)
    plt.plot(t, preds_orig[:, 0], color='blue', linestyle='--', label='Prediction (X)', linewidth=1.5)
    plt.title('X Coordinate Over Time')
    plt.xlabel('Time Step (Sample Index)')
    plt.ylabel('X Position')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path + 'x_coordinate_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path + 'x_coordinate_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # ===== 图 2：Y 坐标对比 =====
    plt.figure(figsize=(12, 5))
    plt.plot(t, labels_orig[:, 1], color='red', label='Ground Truth (Y)', linewidth=1.5)
    plt.plot(t, preds_orig[:, 1], color='blue', linestyle='--', label='Prediction (Y)', linewidth=1.5)
    plt.title('Y Coordinate Over Time')
    plt.xlabel('Time Step (Sample Index)')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path + 'y_coordinate_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path + 'y_coordinate_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ 已保存 X/Y 坐标对比图：x_coordinate_comparison.png, y_coordinate_comparison.png")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiChannelVQVAE(nn.Module):
    def __init__(self, input_channels=64, hidden_dim=256, codebook_size=512, 
                 codebook_dim=256, downsample_factor=8):
        super().__init__()
        self.downsample_factor = downsample_factor
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            self._make_downsample_block(hidden_dim, hidden_dim, stride=2),
            self._make_downsample_block(hidden_dim, hidden_dim, stride=2),
            self._make_downsample_block(hidden_dim, hidden_dim, stride=2),
            nn.Conv1d(hidden_dim, codebook_dim, kernel_size=3, padding=1)
        )
        
        self.vq = VectorQuantize(
            dim=codebook_dim,
            codebook_size=codebook_size,
            decay=0.8,
            commitment_weight=1.0
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(codebook_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            self._make_upsample_block(hidden_dim, hidden_dim, scale_factor=2),
            self._make_upsample_block(hidden_dim, hidden_dim, scale_factor=2),
            self._make_upsample_block(hidden_dim, hidden_dim, scale_factor=2),
            nn.Conv1d(hidden_dim, input_channels, kernel_size=7, padding=3)
        )

    def _make_downsample_block(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(out_ch)
        )
    
    def _make_upsample_block(self, in_ch, out_ch, scale_factor):
        return nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=False),
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(out_ch)
        )

    def forward(self, x):
        z_e = self.encoder(x)  # [B, D, T_vq]
        z_q, indices, vq_loss = self.vq(z_e.permute(0, 2, 1))  # [B, T_vq, D]
        z_q = z_q.permute(0, 2, 1)  # [B, D, T_vq]
        recon = self.decoder(z_q)
        return recon, vq_loss, indices  # indices: [B, T_vq]
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerVQToTokenModel6_Robust64(nn.Module):
    def __init__(self, codebook_size, embed_dim=256, nhead=8, num_layers=3, 
                 max_seq_len=64, output_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=0.3, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # self.fc_head = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(max_seq_len * embed_dim, 512),
        #     nn.GELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(512, output_dim))
        # 替换 Flatten + Fixed Linear 为 Adaptive 或 Mean Pooling
        # self.pool = nn.AdaptiveAvgPool1d(1)  # 或直接 mean
        self.fc_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, output_dim)
        )

    def forward(self, vq_indices):
        # vq_indices: [B, T_vq]
        x = self.embedding(vq_indices)  # [B, T_vq, E]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.ln_final(x)
        
        # Global average pooling over time
        x = x.mean(dim=1)  # [B, E]  ← 简单有效！
        # 或者: x = self.pool(x.permute(0,2,1)).squeeze(-1)
        
        logits = self.fc_head(x)  # [B, 2]
        return logits
    
    
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

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
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


from collections import Counter
def analyze_token_distribution1(vqvae, train_loader, test_loader, codebook_size, device):
    vqvae.eval()
    train_tokens = []
    test_tokens = []

    print("正在提取训练集 Token...")
    with torch.no_grad():
        for x, _ in train_loader:
            _, _, indices = vqvae(x.to(device))
            train_tokens.extend(indices.view(-1).cpu().numpy())

    print("正在提取验证集 Token...")
    with torch.no_grad():
        for x, _ in test_loader:
            _, _, indices = vqvae(x.to(device))
            test_tokens.extend(indices.view(-1).cpu().numpy())

    # 统计频率
    train_counts = Counter(train_tokens)
    test_counts = Counter(test_tokens)

    # 转换为密度分布（归一化）
    train_dist = np.array([train_counts.get(i, 0) for i in range(codebook_size)])
    test_dist = np.array([test_counts.get(i, 0) for i in range(codebook_size)])
    train_dist = train_dist / (train_dist.sum() + 1e-9)
    test_dist = test_dist / (test_dist.sum() + 1e-9)

    # 可视化
    plt.figure(figsize=(15, 6))
    plt.bar(range(codebook_size), train_dist, alpha=0.5, label='Train Tokens', color='blue')
    plt.bar(range(codebook_size), test_dist, alpha=0.5, label='Test Tokens', color='red')
    plt.title("Token Activation Distribution: Train vs Test")
    plt.xlabel("Token ID")
    plt.ylabel("Activation Density")
    plt.legend()
    plt.savefig("token_dist_comparison.png")
    plt.show()

    # 计算重合度 (Intersection over Union)
    active_train = set(np.where(train_dist > 0)[0])
    active_test = set(np.where(test_dist > 0)[0])
    overlap = active_train.intersection(active_test)
    
    print(f"\n--- 诊断报告 ---")
    print(f"训练集激活 Token 数: {len(active_train)}")
    print(f"验证集激活 Token 数: {len(active_test)}")
    print(f"两者重合 Token 数: {len(overlap)}")
    if len(active_test) > 0:
        print(f"验证集 Token 覆盖率: {len(overlap)/len(active_test)*100:.2f}%")


# ==================== Step 1: 训练 VQ-VAE（仅预训练）====================
def train_vqvae(save_path, codebook_size=512):
    modelpath = os.path.join(save_path, "vqvae_model.pth")
    bestmodelpath = os.path.join(save_path, "best_vqvae_model.pth")
    
    vqvae = MultiChannelVQVAE(codebook_size=codebook_size, codebook_dim=256).to(device)
    optimizer = optim.AdamW(vqvae.parameters(), lr=2e-4)
    recon_criterion = nn.MSELoss()
    best_val_loss = float('inf')

    for epoch in range(300):  # 减少 epoch，避免过拟合重构
        vqvae.train()
        total_loss = 0
        for signals, _ in train_loader:
            signals = signals.to(device)
            recon, vq_loss, _ = vqvae(signals)
            recon_loss = recon_criterion(recon, signals)
            loss = recon_loss + vq_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证
        vqvae.eval()
        val_loss = 0
        with torch.no_grad():
            for vx, _ in val_loader:
                vx = vx.to(device)
                recon, vq_loss, _ = vqvae(vx)
                val_loss += (F.mse_loss(recon, vx) + vq_loss).item()
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(vqvae.state_dict(), bestmodelpath)
            print(f"⭐ VQ-VAE Epoch {epoch} | Val Loss: {avg_val_loss:.4f}")
    torch.save(vqvae.state_dict(), modelpath)
    # vqvae.load_state_dict(torch.load(os.path.join(modelpath), map_location=device))
    
    print("\n--- 执行完整验证 Pipeline ---")
    print("--- 1. Waveform Fidelity Check ---")
    test_batch, test_labels = next(iter(val_loader))
    test_batch = test_batch.to(device)
    all_idx = []
    vqvae.eval()
    with torch.no_grad():
        x_recon, vq_loss, all_indices = vqvae(test_batch)
        all_idx.append(all_indices.view(-1).cpu())
    # 画出第 0 个样本的第 5 号通道重构图
    plot_reconstruction_comparison(test_batch, x_recon, channel_idx=0, savepath=save_path)
    
    print("\n--- 3. Codebook Utilization Check ---")  # 统计 Codebook 的激活比例
    unique_tokens = torch.unique(all_indices).cpu().numpy()
    print(f"Active Tokens in this batch: {len(unique_tokens)} / {codebook_size}")
    if len(unique_tokens) < 10:
        print("⚠️ Warning: Codebook Collapse detected! Use EMA or Dead Code Reset.")
    else:
        print("✅ Codebook utilization is healthy.")
    analyze_token_distribution1(vqvae, train_loader, val_loader, codebook, device)
    
    return vqvae


# ==================== Step 2: 联合微调 ====================
def train_token_model_joint(save_path, codebook_size=512):
    # 加载预训练 VQ-VAE
    vqvae = MultiChannelVQVAE(codebook_size=codebook_size).to(device)
    vqvae.load_state_dict(torch.load(os.path.join(save_path, "best_vqvae_model.pth"), map_location=device))
    
    # 🔥 冻结 decoder 和 codebook，只训练 encoder
    for param in vqvae.decoder.parameters():
        param.requires_grad = False
    vqvae.vq.requires_grad_(False)  # 冻结 codebook
    
    token_model = TransformerVQToTokenModel6_Robust64(
        codebook_size=codebook_size, embed_dim=256, output_dim=2, max_seq_len=winsize // 8
        ).to(device)
    
    criterion = nn.SmoothL1Loss()
    # 联合优化 encoder + token_model
    optimizer = optim.AdamW([
        {'params': vqvae.encoder.parameters(), 'lr': 1e-5},
        {'params': token_model.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
    ema = EMA(token_model, decay=0.999)
    ema.register()
    
    best_val_mae = float('inf')
    
    for epoch in range(500):
        vqvae.encoder.train()
        token_model.train()
        train_mae = 0
        for signals, targets in train_loader:
            signals = signals.to(device)
            targets = targets.to(device)
            
            # 前向（不 detach encoder）
            z_e = vqvae.encoder(signals)  # [B, D, T_vq]
            z_q, indices, _ = vqvae.vq(z_e.permute(0, 2, 1))
            preds = token_model(indices)
            
            loss = criterion(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(token_model.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(vqvae.encoder.parameters(), 0.5)
            optimizer.step()
            ema.update()
            
            train_mae += torch.mean(torch.abs(preds - targets)).item() * signals.size(0)
        
        train_mae /= len(train_loader.dataset)
        
        # --- Validate (with EMA) ---
        ema.apply_shadow()
        vqvae.eval()
        token_model.eval()
        val_mae = 0
        with torch.no_grad():
            for signals, targets in val_loader:
                signals = signals.to(device)
                targets = torch.tensor(targets, dtype=torch.float32).to(device)
                
                _, _, indices = vqvae(signals)
                preds = token_model(indices)
                val_mae += torch.mean(torch.abs(preds - targets)).item() * signals.size(0)
        val_mae /= len(val_loader.dataset)
        ema.restore()
        
        scheduler.step(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(token_model.state_dict(), os.path.join(save_path, "token_model_best.pth"))
            torch.save(vqvae.state_dict(), os.path.join(save_path, "best_vqvae_modelFT.pth")) 
    
            print(f"⭐ New Best Token Model! Val MAE: {val_mae:.4f}")
        
        print(f"Epoch {epoch} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")


# ==================== Step 3: 测试 ====================
def test_model(save_path, codebook_size):
    vqvae = MultiChannelVQVAE(codebook_size=codebook_size).to(device)
    vqvae.load_state_dict(torch.load(os.path.join(save_path, "best_vqvae_modelFT.pth"), map_location=device))
    vqvae.eval()
    
    token_model = TransformerVQToTokenModel6_Robust64(
        codebook_size=codebook_size, embed_dim=256, output_dim=2, max_seq_len=winsize // 8
        ).to(device)
    token_model.load_state_dict(torch.load(os.path.join(save_path, "token_model_best.pth")))
    token_model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for signals, targets in val_loader:
            signals = signals.to(device)
            _, _, indices = vqvae(signals)
            preds = token_model(indices)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mae = np.mean(np.abs(all_preds - all_targets))
    print(f"Test MAE (std scale): {mae:.4f}")
    
    # 从数据集获取全局统计量
    pos_mean = full_dataset.pos_mean  # shape (2,)
    pos_std = full_dataset.pos_std    # shape (2,)
    preds_original = all_preds * pos_std + pos_mean
    targets_original = all_targets * pos_std + pos_mean
    mae_original = np.mean(np.abs(preds_original - targets_original))
    print(f"Test MAE (original scale): {mae_original:.4f}")
    
    drawGIF(all_targets, all_preds, save_path)
    drawLine(all_targets[:10], all_preds[:10], save_path)
    drawLineXY(all_targets, all_preds, save_path)
    # plot_trajectory(all_targets.numpy(), all_preds.numpy(), save_path)


def plot_trajectory(targets, preds, save_dir, num_samples=100):
    """ 绘制前 num_samples 个样本的真实 vs 预测轨迹  """
    plt.figure(figsize=(8, 6))
    plt.plot(targets[:num_samples, 0], targets[:num_samples, 1], 'g-', alpha=0.7, label='Ground Truth')
    plt.plot(preds[:num_samples, 0], preds[:num_samples, 1], 'r--', alpha=0.7, label='Prediction')
    plt.title(f'Joystick Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'trajectory.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, f'trajectory.pdf'), dpi=300)
    plt.close()
    

# ==================== Main ====================
if __name__ == "__main__":
    winsize = 1000
    codebook = 512
    subset = 'joystick_track'
    datadir = '/disk2/user1/dataset/BCI-Standford/%s/%s/data/' % (subset, subset)
    split_ratio = 0.6
    save_path = "/mnt/home/user1/MCX/EEGLM/ECogLM/checkpoints/Stanford/VQTokenizer/%s_c%d_4/" % (subset, codebook) 
    os.makedirs(save_path, exist_ok=True)
    
    # 创建数据集
    full_dataset = MillerFingersDataset_TT_joystick_track(
        datadir, window_size=winsize, stride=256, single_wind=False,
        single_channel=False, channel_idx=None, subset=subset, split_ratio=split_ratio)
    '''📊 全局标签统计: mean=(15910.33, 16718.60), std=(10058.17, 9794.30)
    ✅ 数据集加载完成。总训练集: 2367, 总测试集: 404'''
    
    # === 正确收集训练标签（shape: [N, 2]）===
    full_dataset.set_mode('train')
    labels_list = []
    for i in range(len(full_dataset)):
        _, label = full_dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        labels_list.append(label)
    
    train_labels = np.stack(labels_list, axis=0)  # Shape: (N, 2)
    print(f"Collected train_labels shape: {train_labels.shape}") # (2367, 2)
    
    # 显式划分数据集
    train_dataset = full_dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    full_dataset.set_mode('test')
    test_dataset = full_dataset
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
    # # Step 1: 预训练 VQ-VAE
    # print("Training VQ-VAE...")
    # train_vqvae(save_path, codebook_size=codebook)
    
    # # Step 2: 联合微调（关键提升！）
    # print("Jointly Fine-tuning Encoder + Token Model...")
    # train_token_model_joint(save_path, codebook_size=codebook)
    
    # Step 3: 测试
    print("Testing...")
    test_model(save_path, codebook_size=codebook)
    # 0.1520
    # Test MAE (original scale): 1504.8945
