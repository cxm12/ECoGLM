# 实验四    个体刻画  VQTTAlign_emotion.py

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data import SEEDIVDataset_TT, os, np, torch
from evaluate import plot_reconstruction_comparison, diagnose_vq_collapse_multiClass, analyze_token_distribution
from collections import Counter
from transformers import AutoTokenizer
import math
from torch.optim.lr_scheduler import LambdaLR

LABEL_MAP = {
    0: "neutral",
    1: "sad",
    2: "fear",
    3: "happy"
}

# ==========================================
# 1. 快速更新的 VQ 层
# ==========================================
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.9, epsilon=1e-5):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.normal_()
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.Tensor(num_embeddings, embedding_dim))
        self._ema_w.data.copy_(self._embedding.weight.data)
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.permute(0, 2, 1).reshape(-1, self._embedding_dim)

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape[0], input_shape[2], self._embedding_dim).permute(0, 2, 1)
        
        if self.training:
            with torch.no_grad():
                self._ema_cluster_size.mul_(self._decay).add_(torch.sum(encodings, dim=0), alpha=1 - self._decay)
                n = torch.sum(self._ema_cluster_size)
                self._ema_cluster_size.copy_(
                    (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n)
                dw = torch.matmul(encodings.t(), flat_input)
                self._ema_w.mul_(self._decay).add_(dw, alpha=1 - self._decay)
                self._embedding.weight.data.copy_(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encoding_indices.view(input_shape[0], input_shape[2])


# ==========================================
# 2. 强化版 Encoder (加入原型平滑)
# ==========================================
class NeuralVQVAE(nn.Module):
    def __init__(self, in_channels=1, codebook_size=512, embed_dim=64, num_classes=6, tokenstride=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1), # [84, embed_dim, 250]
            nn.InstanceNorm1d(embed_dim),
            nn.ReLU(),
            # ✅ 新增 AvgPool：下采样抛弃高频噪声，强制提取缓慢变化的语义特征
            nn.AvgPool1d(kernel_size=4, stride=tokenstride), # (250 - 4) // 4 + 1 = (246 / 4) + 1 = 61.5
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
        )

        # ✅ 内置类原型 Buffer (不参与梯度下降，通过 EMA 更新)
        self.register_buffer('class_prototypes', torch.zeros(num_classes, embed_dim))
        self.proto_decay = 0.99 

        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.vq_layer = VectorQuantizerEMA(codebook_size, embed_dim, decay=0.9)

        self.decoder = nn.Sequential(
            # 这里的上采样需匹配 Encoder 的下采样倍数
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=16, stride=16), 
            nn.ReLU(),
            nn.ConvTranspose1d(embed_dim, in_channels, kernel_size=1, stride=1),
        )

    def forward(self, x):
        z_e = self.encoder(x)
        cls_logits = self.aux_classifier(z_e)        
        # print('z_e.shape = [1984, 64, 62]', z_e.shape)
        vq_loss, z_q, perplexity, indices = self.vq_layer(z_e)
        # print('indices.shape = [1984, 62]', indices.shape)
        x_recon = self.decoder(z_q)
        
        # 补全重构尺寸 (由于 Pooling 可能导致长度微差)
        if x_recon.shape[-1] != x.shape[-1]:
            x_recon = F.interpolate(x_recon, size=x.shape[-1], mode='linear', align_corners=False)

        if self.training:
            return vq_loss, x_recon, indices, cls_logits, perplexity, z_e
        else:
            return x_recon, z_e, z_q, indices

    @torch.no_grad()
    def update_prototypes(self, z_v, labels):
        """ 使用当前 Batch 更新全局类中心 """
        for l in range(self.class_prototypes.size(0)):
            mask = (labels == l)
            if mask.any():
                batch_mean = z_v[mask].mean(0)
                # EMA 更新原型
                self.class_prototypes[l] = self.proto_decay * self.class_prototypes[l] + \
                                          (1 - self.proto_decay) * batch_mean


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
                 max_seq_len=62, num_channels=62):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        self.embed_dim = embed_dim
        # 增加一个线性层，用于将通道维度压缩或整合
        # 方案：将 62 个通道的 embedding 融合
        self.channel_fusion = nn.Linear(num_channels * embed_dim, embed_dim)
        
        self.ms_extractor = MultiScaleFeatureExtractor(embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)
        
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
        # vq_indices: [32, 62, 62] [8, 62, 62]
        # print('vq_indices.shape = ', vq_indices.shape)
        B, C, T = vq_indices.shape
        E = self.embed_dim
        # 1. Embedding: [B, 62, T] -> [B, 62, T, E]
        x = self.embedding(vq_indices)
        # print('1. x.shape = ', x.shape)
        
        # 2. 空间融合 (Spatial-Temporal Interaction)
        # 变换形状为 [B, T, 62 * E]
        x = x.transpose(1, 2).reshape(B, T, C * self.embed_dim)
        # print('2. x.shape = ', x.shape)        
        # 投影回 [B, T, E] 输出形状将变为 [B, T, 256]
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


def train_token():
    full_dataset = SEEDIVDataset_TT(datadir=datadir, window_size=winsize, stride=stride,
        single_wind=False, single_channel=False, channel_idx=None,
        split_ratio=split_ratio, normalize=True)
    full_dataset.set_mode('train')
    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    full_dataset.set_mode('test')  # ('valid')
    val_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    # Train samples: 18312, Test samples: 3354
    
    # if os.path.exists(tokenmodel_path):
    #     token_model.load_state_dict(torch.load(tokenmodel_path, map_location=device))
    #     print("Load best model from" + tokenmodel_path)    

    # 🌟 动态权重策略
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(token_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    
    # 🌟 组合式学习率策略
    warmup_scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                      num_warmup_steps=WARMUP_EPOCHS, 
                                                      num_training_steps=EPOCHS)
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
            # x 现在的形状是 [Batch, 62, 1024]
            x, labels = x.to(device), labels.to(device)
            B, C, T = x.shape  # B=Batch, C=62, T=1024

            optimizer.zero_grad()
            with torch.no_grad():
                # 1. 重塑为 [B*C, 1, T] 以符合单通道 VQ-VAE 的输入格式
                x_reshaped = x.view(B * C, 1, T)
                
                # 2. 逐通道提取 VQ 索引
                # indices 形状为 [B*C, T_vq] (T_vq 是下采样后的序列长度)
                # print(x_reshaped.shape, 'x_reshaped = 1984, 1, 1000')
                _, _, _, indices = vqvae(x_reshaped)
                
                # 3. 恢复通道维度 [B, C, T_vq]
                T_vq = indices.shape[-1]
                indices = indices.view(B, C, T_vq)
                
                # 🌟 优化 C: 随机平移 (对所有通道应用相同的位移，保持空间同步)
                if torch.rand(1) > 0.5:
                    shift = torch.randint(-4, 5, (1,)).item()
                    indices = torch.roll(indices, shifts=shift, dims=2)

            # 4. 输入 token_model # 注意：token_model 的输入现在需要处理 [B, 62, T_vq] 的形状
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
        val_acc = testtoken(epoch,val_loader)
        # 更新 Plateau 调度器
        if epoch >= WARMUP_EPOCHS: plateau_scheduler.step(val_acc)
        
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
        
        train_acc = np.mean(np.array(all_preds) == np.array(all_gts)) * 100
        pred_dist = Counter(all_preds)
        print(f"\n--- Epoch {epoch} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc*100:.2f}% | BestVal Acc: {best_acc*100:.2f}% at Epoch {best_epoch} | LR: {optimizer.param_groups[0]['lr']:.8f}")
        print(f"预测分布: {[f'C{k}: {v/len(all_preds)*100:.1f}%' for k, v in sorted(pred_dist.items())]}")

        ema.restore() # 恢复原始参数继续下一轮训练

        # # 🌟 优化 E: 早停机制 (Early Stopping)
        # if patience_counter > 900: # 如果 60 轮没提升则停止
        #     print("触发早停，训练结束。")
        #     break


# ==================== 联合微调（关键！）====================
def train_token_model_joint():
    full_dataset = SEEDIVDataset_TT(
        datadir=datadir, window_size=winsize, stride=stride,
        single_wind=False, single_channel=False, channel_idx=None,
        split_ratio=split_ratio, normalize=True
    )
    full_dataset.set_mode('train')
    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    full_dataset.set_mode('test')
    val_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

    # ==================== 关键修改 1: 解冻 VQ-VAE Encoder，冻结其他部分 ====================
    vqvae.to(device)
    
    # 冻结 decoder 和 codebook
    for param in vqvae.decoder.parameters():
        param.requires_grad = False
    vqvae.vq_layer.requires_grad_(False)  # 冻结 codebook
    
    # 解冻 encoder（默认应为 True，但显式确保）
    for param in vqvae.encoder.parameters():
        param.requires_grad = True

    # ==================== 关键修改 2: 联合优化器 ====================
    optimizer = torch.optim.AdamW([
        {'params': vqvae.encoder.parameters(), 'lr': 1e-5},      # encoder 微调：小学习率
        {'params': token_model.parameters(), 'lr': LEARNING_RATE}  # token_model：原学习率
    ], weight_decay=1e-2)

    # 🌟 动态权重策略（可选，根据实际类别分布调整）
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # 🌟 组合式学习率策略
    warmup_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=WARMUP_EPOCHS, 
        num_training_steps=EPOCHS
    )
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, verbose=True)
    
    ema = EMA(token_model, 0.999)
    ema.register()

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        token_model.train()
        vqvae.train()
        all_preds, all_gts = [], []
        total_loss = 0

        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            B, C, T = x.shape
            # print('x.shape = ', x.shape)
            optimizer.zero_grad()
            # ==================== 关键修改 3: 移除 torch.no_grad()，允许梯度流经 encoder ====================
            # 1. 重塑为 [B*C, 1, T]
            x_reshaped = x.view(B * C, 1, T) # x_reshaped = x.view(-1, 1, x.size(-1))
            # print('x_reshaped.shape = ', x_reshaped.shape)
            
            z_e = vqvae.encoder(x_reshaped)  # [N, D, T_vq], N = B*C
            # print('z_e.shape = ', z_e.shape)  # [1984, 64, 62]            
            vq_loss, z_q, perplexity, indices = vqvae.vq_layer(z_e)                        
            # 4. Reshape indices
            indices = indices.view(B, C, -1)  # [B, C, T_vq]

            # 🌟 数据增强：随机时间平移
            if torch.rand(1) > 0.5:
                shift = torch.randint(-4, 5, (1,)).item()
                indices = torch.roll(indices, shifts=shift, dims=2)

            # 3. Token model 前向
            logits = token_model(indices)
            loss = criterion(logits, labels)
            loss.backward()
            
            # 🌟 梯度裁剪（对两个模块都做）
            torch.nn.utils.clip_grad_norm_(token_model.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(vqvae.encoder.parameters(), max_norm=0.5)
            
            optimizer.step()
            ema.update()

            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())
            total_loss += loss.item()

        # 调整学习率
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()

        # 🌟 验证阶段：使用 EMA 权重
        ema.apply_shadow()
        token_model.eval()
        vqvae.eval()  # 验证时全部 eval
        val_acc = testtoken(epoch, val_loader)
        ema.restore()

        # 更新 Plateau 调度器（注意：plateau 需要 metric，这里用 val_acc）
        if epoch >= WARMUP_EPOCHS:
            plateau_scheduler.step(val_acc)
        
        # 保存模型
        torch.save(token_model.state_dict(), tokenmodel_path)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(token_model.state_dict(), tokenmodel_path.replace(".pth", "_best.pth"))
            # 同时保存微调后的 VQ-VAE encoder（可选）
            torch.save(vqvae.state_dict(), os.path.join(os.path.dirname(tokenmodel_path), "vqvae_finetuned.pth"))
            print(f"⭐ New Best Model Saved (Acc: {best_acc*100:.2f}%)")
        else:
            patience_counter += 1
        
        # 打印报告
        train_acc = np.mean(np.array(all_preds) == np.array(all_gts)) * 100
        pred_dist = Counter(all_preds)
        print(f"\n--- Epoch {epoch} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc*100:.2f}% | BestVal Acc: {best_acc*100:.2f}% at Epoch {best_epoch} | LR_token: {optimizer.param_groups[1]['lr']:.8f}, LR_enc: {optimizer.param_groups[0]['lr']:.8f}")
        print(f"预测分布: {[f'C{k}: {v/len(all_preds)*100:.1f}%' for k, v in sorted(pred_dist.items())]}")

        # 早停（可取消注释）
        # if patience_counter > 60:
        #     print("触发早停，训练结束。")
        #     break


def testtoken(epoch, loader=None):
    if loader is None:
        full_dataset = SEEDIVDataset_TT(
            datadir=datadir, window_size=winsize, stride=stride,
            single_wind=False, single_channel=False, channel_idx=None,
            split_ratio=split_ratio, normalize=True)
        # num_classes = len(set(full_dataset.labels))
        # print('num_classes = ', num_classes)
        full_dataset.set_mode('test')
        loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

    token_model.eval()
    vqvae.eval()
    all_preds, all_gts = [], []
    
    with torch.no_grad():
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            B, C, T = x.shape
            x_reshaped = x.view(B * C, 1, T)
            
            _, _, _, indices = vqvae(x_reshaped) 
            T_vq = indices.shape[-1]
            indices = indices.view(B, C, T_vq)
            
            logits = token_model(indices)
            
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())

    # 计算准确率
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    final_acc = np.mean(all_preds == all_gts)
    
    print(f"\n✅ Epoch {epoch} 验证完成! 总样本: {len(all_gts)}, 平均准确率: {final_acc*100:.2f}%")
    return final_acc


# ==========================================
# 3. 功能完整的 Main
# ==========================================
def trainVQVAE():
    full_dataset = SEEDIVDataset_TT(datadir=datadir, window_size=winsize, stride=stride, single_wind=False, 
                                    single_channel=True, channel_idx=None, split_ratio=split_ratio, normalize=True)    
    full_dataset.set_mode('train')
    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)    
    full_dataset.set_mode('valid')
    val_data = full_dataset
    val_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
        
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=12, factor=0.5)
    
    start_epoch = 0
    modelpath = VQVAEmodelpath
    bestmodelpath = modelpath.replace("vqvae_model.pth", "best_vqvae_model.pth")
    best_val_loss = float('inf')
    if os.path.exists(modelpath):
        # model.load_state_dict(torch.load(modelpath, map_location=device))
        checkpoint = torch.load(bestmodelpath, map_location=device)
        vqvae.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print("Load model from", bestmodelpath)

    # 初始超参数
    aux_weight = 20.0     # 分类头权重
    sim_weight = 50.0     # 类间排斥权重
    sample_sim_weight = 30.0 # 样本级权重
    recon_weight = 0.2     # 极低权重，只给 Encoder 留一口气
    
    print("\n🚀 开始强化版全类聚合 VQ-Tokenizer 训练...")    
    for epoch in range(start_epoch, EPOCHS):
        vqvae.train()
        total_recon = 0
        total_aux = 0
        for x, labels in train_loader:
            # print('x.shape = ', x.shape)  # [32, 62, 1000]            
            x, labels = x.to(device), labels.to(device)
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
            optimizer.zero_grad()
            # 1. Forward
            vq_loss, x_recon, indices, cls_logits, _, z_e = vqvae(x)
            # 2. 基础损失
            recon_loss = F.mse_loss(x_recon, x)
            aux_loss = F.cross_entropy(cls_logits, labels)
            # 3. 核心：样本级对比损失 (Sample-level Contrastive Loss)
            z_v = z_e.mean(dim=-1)  # z_v: [Batch, Channel]       
            # 🔥 关键步骤：特征归一化到单位球，消除幅值干扰，强制关注方向差异
            z_v_norm = F.normalize(z_v, p=2, dim=1)            
            # 计算余弦相似度矩阵 [Batch, Batch]
            sim_matrix = torch.matmul(z_v_norm, z_v_norm.t())            
            # 构造掩码：只有标签不同的样本对才为 True
            diff_label_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
            # 🌟 样本级排斥：如果不同类样本相似度 > 0.3，就施加惩罚
            sample_inter_penalty = (torch.clamp(sim_matrix - 0.3, min=0) * diff_label_mask).sum() / (diff_label_mask.sum() + 1e-6)
            
            # 4. 原型级排斥 (保持原有的全局约束)
            vqvae.update_prototypes(z_v.detach(), labels)
            all_protos = F.normalize(vqvae.class_prototypes, p=2, dim=1)
            proto_sims = torch.matmul(all_protos, all_protos.t())
            mask = torch.eye(num_classes, device=device).bool()
            proto_inter_penalty = torch.clamp(proto_sims[~mask] - 0.4, min=0).pow(2).mean()

            # 5. 组合总损失
            loss = (recon_weight * recon_loss) + vq_loss + (aux_weight * aux_loss) + \
                   (sim_weight * proto_inter_penalty) + (sample_sim_weight * sample_inter_penalty)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vqvae.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_recon += recon_loss.item()
            total_aux += aux_loss.item()

        # 4. 验证与指标诊断
        vqvae.eval()
        val_recon_list = []
        # 1. 计算测试集的 Inter-Sim (类间中心相似度)
        current_inter_sim = get_inter_class_similarity(vqvae, val_data, device, num_samples=len(val_data))
        # 2. 计算测试集的平均 Intra-Dist (类内距离 1-Sim)
        total_val_intra_dist = 0
        samples_count = 0
        with torch.no_grad():
            for vx, vl in val_loader:
                vx, vl = vx.to(device), vl.to(device)
                v_recon, z_e, _, _ = vqvae(vx)
                
                # 计算重构 Loss
                val_recon_list.append(F.mse_loss(v_recon, vx).item())
                
                # 计算该样本到其类中心的距离
                z_v = z_e.mean(dim=-1)
                target_proto = vqvae.class_prototypes[vl]
                cos_sim = F.cosine_similarity(z_v, target_proto)
                total_val_intra_dist += (1.0 - cos_sim).mean().item()
                samples_count += 1

        avg_val_recon = sum(val_recon_list) / len(val_recon_list)
        avg_val_intra = total_val_intra_dist / samples_count
        # --- 3. 自动化状态判定逻辑 ---
        if current_inter_sim > 0.95 and avg_val_intra < 0.05:
            status = "🔴 特征坍塌 (Collapse) - 类间分不开"
        elif current_inter_sim > 0.85:
            status = "🟡 正在解耦 (Breakout) - 尝试推开边界"
        elif current_inter_sim <= 0.85 and avg_val_intra > 0.1:
            status = "🟢 理想状态 (Golden) - 类间疏离且类内有细节"
        else:
            status = "🔵 调优中 (Fine-tuning)"
        # --- 4. 格式化输出 ---
        print("-" * 70)
        print(status)
        if current_inter_sim > 0.8:
            sim_weight = min(sim_weight + 5.0, 100.0)
            aux_weight = min(aux_weight + 2.0, 50.0)
            sample_sim_weight = min(sample_sim_weight + 5.0, 100.0) # 🌟 也要动态加压样本级损失
            recon_weight = max(recon_weight - 0.05, 0.1) # 🌟 允许更低的重构权重，强制解耦
        else:
            recon_weight = min(recon_weight + 0.1, 1.0) # 🌟 解耦后慢慢恢复重构
            sample_sim_weight = max(sample_sim_weight - 2.0, 10.0) # 🌟 解耦后降低排斥压力
        
        avg_val_loss = avg_val_recon
        scheduler.step(avg_val_loss)
        
        # 5. 模型保存逻辑
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': vqvae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, bestmodelpath)
            print(f"⭐ New Best Model! Val Recon: {avg_val_loss:.4f}")

        # 6. 打印进度报告  Intra-Dist 越小代表类内越聚合；Inter-Sim 越小代表类间越区分
        print(f"Epoch {epoch:03d} | Recon: {total_recon/len(train_loader):.4f} | "
              f"In Valid: \nIntra-Dist: {avg_val_intra:.4f} | Inter-Sim: {current_inter_sim:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"⚙️ 当前权重执行: Recon_w: {recon_weight:.1f} | Sim_w: {sim_weight:.1f}")

        # 7. 定期诊断
        if (epoch) % 20 == 0:
            diagnose_vq_collapse_multiClass(val_data, vqvae, device, num_samples=20, num_classes=num_classes)  # ← 传入 4
            
        # 8. 热重启机制 (防止学习率过低导致死锁)
        if optimizer.param_groups[0]['lr'] < 1e-6:
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
            optimizer.state.clear() 
            print("🚀 Learning rate reset!")

        # 保存 Last Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': vqvae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, modelpath)

    print("\n训练流程结束。")
    

# 如果它开始下降到 0.8-0.9 左右，你的 VQVAE 就算真正训练成功了，此时再训练之前的 Transformer 就会有立竿见影的效果。
def get_inter_class_similarity(model, dataset, device, num_samples=100):
    model.eval()
    vectors = {i: [] for i in range(num_classes)}  # 动态类数
    with torch.no_grad():
        for x, label in dataset:
            l = label.item()
            if len(vectors[l]) < num_samples:
                x = x.unsqueeze(0).to(device)
                z_e = model.encoder(x)
                vectors[l].append(z_e.mean(dim=-1).view(-1).detach())
            if all(len(v) >= num_samples for v in vectors.values()):
                break
    
    # 计算所有类中心两两之间的最大相似度
    protos = []
    for i in range(num_classes):
        if len(vectors[i]) == 0:
            return 1.0
        protos.append(torch.stack(vectors[i]).mean(0))
    
    max_sim = 0.0
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            sim = F.cosine_similarity(protos[i].unsqueeze(0), protos[j].unsqueeze(0)).item()
            max_sim = max(max_sim, sim)
    return max_sim

 
def testVQVAE(save_path0, isbest=True, channel_idx=0):
    '''
    SEEDIVDataset_TT0
    ==================================================
    🔍 VQ 空间深度诊断报告 (每类样本数: 20, 总类数: 4)
    ==================================================
    ⚠️ 警告: 类别 0 未找到任何样本！
    ⚠️ 警告: 类别 1 未找到任何样本！
    1. Encoder 原始特征最大类间相似度 (z_e): -0.1246
    2. Codebook 量化特征最大类间相似度 (z_q): -0.2117
    ------------------------------
    ✅ 判定结果: [模型健康]
    特征在编码和量化阶段都保持了良好区分度。
    ==================================================
    --- 3. Codebook Utilization Check ---
    Active Tokens in this batch: 213 / 256
    ✅ Codebook utilization is healthy.
    正在提取训练集 Token...正在提取验证集 Token...
    --- 诊断报告 ---
    训练集激活 Token 数: 254
    验证集激活 Token 数: 253
    两者重合 Token 数: 252
    验证集 Token 覆盖率: 99.60%
    
    SEEDIVDataset_TT
    ==================================================
    🔍 VQ 空间深度诊断报告 (每类样本数: 20, 总类数: 4)
    ==================================================
    counts =  [20, 20, 20, 20]
    1. Encoder 原始特征最大类间相似度 (z_e): 0.2722
    2. Codebook 量化特征最大类间相似度 (z_q): 0.2992
    ------------------------------
    ✅ 判定结果: [模型健康]特征在编码和量化阶段都保持了良好区分度。
    ==================================================
    --- 执行完整验证 Pipeline ---
    --- 3. Codebook Utilization Check ---
    Active Tokens in this batch: 188 / 256
    ✅ Codebook utilization is healthy.
    训练集激活 Token 数: 254   验证集激活 Token 数: 255
    两者重合 Token 数: 254  验证集 Token 覆盖率: 99.61%
    '''
    
    full_dataset = SEEDIVDataset_TT(
        datadir=datadir, window_size=winsize, stride=stride,
        single_wind=False, single_channel=True, channel_idx=None,
        split_ratio=split_ratio, normalize=True)
    full_dataset.set_mode('test')
    test_data = full_dataset
    test_loader = DataLoader(full_dataset, batch_size=32, shuffle=False) 
    
    modelpath = VQVAEmodelpath
    bestmodelpath = modelpath.replace("vqvae_model.pth", "best_vqvae_model.pth")
    save_path1 = save_path0 + '/last/'   #
    if isbest:
        modelpath = bestmodelpath
        save_path1 = save_path0 + '/best/'  # 
    checkpoint = torch.load(modelpath)
    os.makedirs(save_path1, exist_ok=True)
    vqvae.load_state_dict(checkpoint['model_state_dict'])
    print('Finetune from', modelpath)
    vqvae.eval()
    
    diagnose_vq_collapse_multiClass(test_data, vqvae, device, num_samples=20,  # 每类采 20 个
        num_classes=num_classes)  # ← 传入 4

    # 6. 执行最终评估
    print("\n--- 执行完整验证 Pipeline ---")
    print("--- 1. Waveform Fidelity Check ---")
    test_batch, test_labels = next(iter(test_loader))
    test_batch = test_batch.to(device)
    all_idx = []
    
    vqvae.eval()
    with torch.no_grad():
        x_recon, z_e, z_q, all_indices= vqvae(test_batch)
        all_idx.append(all_indices.view(-1).cpu())
    # 画出第 0 个样本的第 5 号通道重构图
    plot_reconstruction_comparison(test_batch, x_recon, channel_idx=channel_idx, savepath=save_path1)
    
    print("\n--- 3. Codebook Utilization Check ---")
    # 统计 Codebook 的激活比例
    unique_tokens = torch.unique(all_indices).cpu().numpy()
    print(f"Active Tokens in this batch: {len(unique_tokens)} / {vqvae.vq_layer._num_embeddings}")

    if len(unique_tokens) < 10:
        print("⚠️ Warning: Codebook Collapse detected! Use EMA or Dead Code Reset.")
    else:
        print("✅ Codebook utilization is healthy.")
        
    analyze_token_distribution(vqvae, test_loader, test_loader, codebook, device)

    # 统计测试集里所有 indices 的分布  
    all_idx = torch.cat(all_idx)
    plt.hist(all_idx.numpy(), bins=512)
    plt.title("Codebook Usage Distribution")
    plt.savefig(save_path1+'Codebook_Usage_Distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 1000
    LEARNING_RATE = 1e-4
    codebook = 256
    winsize = 1000
    embed_dim = 64
    Tokenstride = 4
    datadir = '/mnt/home/user1/MCX/EEGLM/data/emotion recognition/SEED-IV/eeg_raw_data'
    split_ratio = 0.6
    VQToTokenembed_dim = 256
    stride = 256
    WARMUP_EPOCHS = 10
    num_classes = 4
    num_channels = 62

    save_path = "/mnt/home/user1/MCX/EEGLM/ECogLM/checkpoints/Stanford/VQTokenizer/SEED-IV_c%d_4_1/" % (codebook) 
    os.makedirs(save_path, exist_ok=True)
    VQVAEmodelpath = save_path + "vqvae_model.pth"

    # 4. 模型初始化
    vqvae = NeuralVQVAE(in_channels=1, codebook_size=codebook, 
                embed_dim=embed_dim, num_classes=num_classes, tokenstride=Tokenstride).to(device)
    
    # trainVQVAE()
    # print("\n--- 执行完整验证 Pipeline ---")
    # testVQVAE(save_path0=save_path, isbest=True)
    
    print("... Build Model ...") 
    vqvae.load_state_dict(torch.load(VQVAEmodelpath)['model_state_dict'])
    print("Load VQ-VAE model from", VQVAEmodelpath)
    
    # save_path = "/mnt/home/user1/MCX/EEGLM/ECogLM/checkpoints/Stanford/VQTokenizer/SEED-IV_c%d_4_FT/" % (codebook) 
    # os.makedirs(save_path, exist_ok=True)
    tokenmodel_path = f"{save_path}token_aligner.pth"    
    with torch.no_grad():
        vqvae.eval()
        test_input = torch.zeros(1, 1, winsize).to(device)
        _, _, _, dummy_indices = vqvae(test_input)
        actual_seq_len = dummy_indices.shape[1]
    print(f"Detected VQ Sequence Length: {actual_seq_len}", "dummy_indices.shape = ", dummy_indices.shape)
    token_model = TransformerVQToTokenModel6_Robust64(
        codebook_size=codebook, embed_dim=VQToTokenembed_dim,
        num_classes=num_classes,
        max_seq_len=actual_seq_len, num_channels=num_channels).to(device)
    
    # train_token()
    print("\n--- 执行完整验证 Pipeline ---")
    token_model.load_state_dict(torch.load(tokenmodel_path.replace(".pth", "_best.pth"), map_location=device))
    # token_model.load_state_dict(torch.load(tokenmodel_path, map_location=device))
    print('Finetune from', tokenmodel_path.replace(".pth", "_best.pth"))
    testtoken(epoch=999)
