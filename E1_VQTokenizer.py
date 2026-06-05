# 实验一 VQTokenizer3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data import BCIFingersDataset_TT, MillerFingersDataset_TT
from evaluate import plot_reconstruction_comparison, diagnose_vq_collapse, analyze_token_distribution
import os


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
        
        self.use_classification = (num_classes > 0)
        self.num_classes = num_classes
        
        if self.use_classification:
            self.register_buffer('class_prototypes', torch.zeros(num_classes, embed_dim))
            self.aux_classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        else:
            self.aux_classifier = None
            self.class_prototypes = None
        # ✅ 内置类原型 Buffer (不参与梯度下降，通过 EMA 更新)
        self.proto_decay = 0.99

        self.vq_layer = VectorQuantizerEMA(codebook_size, embed_dim, decay=0.9)

        self.decoder = nn.Sequential(
            # 这里的上采样需匹配 Encoder 的下采样倍数
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=16, stride=16), 
            nn.ReLU(),
            nn.ConvTranspose1d(embed_dim, in_channels, kernel_size=1, stride=1),
        )

    def forward(self, x):
        # print('x.shape = ', x.shape) # [84, 1, 1000]
        z_e = self.encoder(x)
        cls_logits = None
        if self.use_classification:
            cls_logits = self.aux_classifier(z_e)

        # print('z_e.shape = ', z_e.shape)  # [2688, 64, 62]
        vq_loss, z_q, perplexity, indices = self.vq_layer(z_e)
        # print('indices.shape = ', indices.shape)  # [2688, 62]
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


# ==========================================
# 3. 功能完整的 Main
# ==========================================
def main():
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=12, factor=0.5)
    
    start_epoch = 0
    modelpath = os.path.join(save_path, "vqvae_model.pth")
    bestmodelpath = modelpath.replace("vqvae_model.pth", "best_vqvae_model.pth")
    best_val_loss = float('inf')
    
    if os.path.exists(bestmodelpath):
        checkpoint = torch.load(bestmodelpath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
        print(f'恢复训练：Epoch {start_epoch}, Best Loss: {best_val_loss:.4f}')

    # 超参数（仅分类任务使用）
    aux_weight = 20.0
    sim_weight = 50.0
    sample_sim_weight = 30.0
    recon_weight = 0.2
    
    print("\n🚀 开始训练...")
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_recon = 0.0
        total_aux = 0.0
        
        for x, labels in train_loader:
            x = x.to(device)
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
            
            optimizer.zero_grad()
            
            # Forward
            if model.use_classification:
                labels = labels.to(device)
                vq_loss, x_recon, indices, cls_logits, _, z_e = model(x)
                
                recon_loss = F.mse_loss(x_recon, x)
                aux_loss = F.cross_entropy(cls_logits, labels)
                
                # 样本级对比损失
                z_v = z_e.mean(dim=-1)
                z_v_norm = F.normalize(z_v, p=2, dim=1)
                sim_matrix = torch.matmul(z_v_norm, z_v_norm.t())
                diff_label_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
                sample_inter_penalty = (
                    torch.clamp(sim_matrix - 0.3, min=0) * diff_label_mask
                ).sum() / (diff_label_mask.sum() + 1e-6)
                
                # 原型更新与原型级排斥
                model.update_prototypes(z_v.detach(), labels)
                all_protos = F.normalize(model.class_prototypes, p=2, dim=1)
                proto_sims = torch.matmul(all_protos, all_protos.t())
                mask = torch.eye(model.num_classes, device=device).bool()
                proto_inter_penalty = torch.clamp(proto_sims[~mask] - 0.4, min=0).pow(2).mean()
                
                loss = (recon_weight * recon_loss) + vq_loss + \
                       (aux_weight * aux_loss) + \
                       (sim_weight * proto_inter_penalty) + \
                       (sample_sim_weight * sample_inter_penalty)
            else:
                # 回归任务：只有重构 + VQ 损失
                vq_loss, x_recon, indices, cls_logits, _, z_e = model(x)
                recon_loss = F.mse_loss(x_recon, x)
                aux_loss = torch.tensor(0.0, device=device)
                loss = recon_weight * recon_loss + vq_loss
                # 其他 penalty 设为 0（不参与反向传播）
                sample_inter_penalty = 0.0
                proto_inter_penalty = 0.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_recon += recon_loss.item()
            total_aux += aux_loss.item()

        # ===== 验证阶段 =====
        model.eval()
        val_recon_list = []
        total_val_intra_dist = 0.0
        samples_count = 0
        current_inter_sim = 0.0

        with torch.no_grad():
            for vx, vl in test_loader:
                vx = vx.to(device)
                if model.use_classification:
                    vl = vl.to(device)
                
                outputs = model(vx)
                if model.use_classification:
                    v_recon, z_e, _, _ = outputs
                else:
                    v_recon, z_e, _, _ = outputs  # same unpacking
                
                val_recon_list.append(F.mse_loss(v_recon, vx).item())
                
                # 仅分类任务计算类内距离
                if model.use_classification:
                    z_v = z_e.mean(dim=-1)
                    target_proto = model.class_prototypes[vl]
                    cos_sim = F.cosine_similarity(z_v, target_proto)
                    total_val_intra_dist += (1.0 - cos_sim).mean().item()
                    samples_count += 1

        avg_val_recon = sum(val_recon_list) / len(val_recon_list)
        
        # 计算指标（仅分类任务）
        if model.use_classification:
            avg_val_intra = total_val_intra_dist / samples_count if samples_count > 0 else 0.0
            current_inter_sim = get_inter_class_similarity(
                model, test_data, device,
                num_samples=len(full_dataset) - int(len(full_dataset) * split_ratio)
            )
            
            # 状态判定
            if current_inter_sim > 0.95 and avg_val_intra < 0.05:
                status = "🔴 特征坍塌 (Collapse)"
            elif current_inter_sim > 0.85:
                status = "🟡 正在解耦 (Breakout)"
            elif current_inter_sim <= 0.85 and avg_val_intra > 0.1:
                status = "🟢 理想状态 (Golden)"
            else:
                status = "🔵 调优中"
            
            # 动态调整权重
            if current_inter_sim > 0.8:
                sim_weight = min(sim_weight + 5.0, 100.0)
                aux_weight = min(aux_weight + 2.0, 50.0)
                sample_sim_weight = min(sample_sim_weight + 5.0, 100.0)
                recon_weight = max(recon_weight - 0.05, 0.1)
            else:
                recon_weight = min(recon_weight + 0.1, 1.0)
                sample_sim_weight = max(sample_sim_weight - 2.0, 10.0)
        else:
            avg_val_intra = 0.0
            current_inter_sim = 0.0
            status = "📈 回归任务（仅重构）"

        # 学习率调度
        avg_val_loss = avg_val_recon
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, bestmodelpath)
            print(f"⭐ New Best Model! Val Recon: {avg_val_loss:.4f}")

        # 打印日志
        print("-" * 70)
        print(status)
        print(f"Epoch {epoch:03d} | Train Recon: {total_recon/len(train_loader):.4f} | "
              f"Val Recon: {avg_val_recon:.4f}")
        if model.use_classification:
            print(f"Intra-Dist: {avg_val_intra:.4f} | Inter-Sim: {current_inter_sim:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Recon_w: {recon_weight:.1f} | Sim_w: {sim_weight:.1f}")

        # 定期诊断（仅分类任务）
        if model.use_classification and (epoch + 1) % 20 == 0:
            diagnose_vq_collapse(
                test_data, model, device,
                num_samples=len(full_dataset) - int(len(full_dataset) * split_ratio)
            )

        # 热重启
        if optimizer.param_groups[0]['lr'] < 1e-6:
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
            optimizer.state.clear()
            print("🚀 Learning rate reset!")

        # 保存最新 checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, modelpath)

    print("\n✅ 训练完成！")
    
# 更大的分类权重 + 更强的对比压力 + 更低的重构权重，能让 VQVAE 更快地解耦出纯语义特征，避免过早陷入重构细节的局部最优。
def main1():
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=12, factor=0.5)
    
    start_epoch = 0
    modelpath = os.path.join(save_path, "vqvae_model.pth")
    bestmodelpath = modelpath.replace("vqvae_model.pth", "best_vqvae_model.pth")
    best_val_loss = float('inf')
    
    # if os.path.exists(bestmodelpath):
    #     checkpoint = torch.load(bestmodelpath, map_location=device)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     best_val_loss = checkpoint['loss']
    #     print(f'恢复训练：Epoch {start_epoch}, Best Loss: {best_val_loss:.4f}')

    # ===== 超参数（仅分类任务生效）=====
    # 🔥 强化解耦配置：高分类权重 + 高对比压力 + 低重构权重
    aux_weight = 50.0          # 分类头权重（↑ 提高）
    sim_weight = 100.0         # 原型间排斥权重（↑）
    sample_sim_weight = 100.0  # 样本级排斥权重（↑↑ 关键！）
    recon_weight = 0.05        # 重构权重（↓ 压低，优先语义解耦）

    print("\n🚀 开始训练...")
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_recon = 0.0
        total_aux = 0.0
        
        for x, labels in train_loader:
            x = x.to(device)
            # 标准化输入（沿时间维度）
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
            
            optimizer.zero_grad()
            
            if model.use_classification:
                labels = labels.to(device).long()  # 确保是整型
                vq_loss, x_recon, indices, cls_logits, _, z_e = model(x)
                
                recon_loss = F.mse_loss(x_recon, x)
                aux_loss = F.cross_entropy(cls_logits, labels)
                
                # === 样本级对比损失（更强 margin）===
                z_v = z_e.mean(dim=-1)  # [B, C]
                z_v_norm = F.normalize(z_v, p=2, dim=1)
                sim_matrix = torch.matmul(z_v_norm, z_v_norm.t())  # [B, B]
                diff_label_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
                
                # 🔥 更强惩罚：margin 从 0.3 → 0.1，只要相似度 >0.1 就罚
                sample_inter_penalty = (
                    torch.clamp(sim_matrix - 0.1, min=0) * diff_label_mask
                ).sum() / (diff_label_mask.sum() + 1e-6)
                
                # === 原型更新 + 原型级排斥 ===
                model.update_prototypes(z_v.detach(), labels)
                all_protos = F.normalize(model.class_prototypes, p=2, dim=1)
                proto_sims = torch.matmul(all_protos, all_protos.t())
                mask = torch.eye(model.num_classes, device=device).bool()
                # 🔥 原型 margin 从 0.4 → 0.2
                proto_inter_penalty = torch.clamp(proto_sims[~mask] - 0.2, min=0).pow(2).mean()
                
                loss = (recon_weight * recon_loss) + vq_loss + \
                       (aux_weight * aux_loss) + \
                       (sim_weight * proto_inter_penalty) + \
                       (sample_sim_weight * sample_inter_penalty)
            else:
                # 回归任务：仅重构 + VQ
                vq_loss, x_recon, indices, cls_logits, _, z_e = model(x)
                recon_loss = F.mse_loss(x_recon, x)
                aux_loss = torch.tensor(0.0, device=device)
                loss = recon_weight * recon_loss + vq_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_recon += recon_loss.item()
            total_aux += aux_loss.item()

        # ==================== 验证阶段 ====================
        model.eval()
        val_recon_list = []
        total_val_intra_dist = 0.0
        samples_count = 0
        current_inter_sim = 0.0

        with torch.no_grad():
            for vx, vl in test_loader:
                vx = vx.to(device)
                if model.use_classification:
                    vl = vl.to(device).long()
                
                outputs = model(vx)
                v_recon, z_e, _, _ = outputs[:4]  # 兼容返回值
                
                val_recon_list.append(F.mse_loss(v_recon, vx).item())
                
                if model.use_classification:
                    z_v = z_e.mean(dim=-1)
                    target_proto = model.class_prototypes[vl]
                    cos_sim = F.cosine_similarity(z_v, target_proto)
                    total_val_intra_dist += (1.0 - cos_sim).mean().item()
                    samples_count += 1

        avg_val_recon = sum(val_recon_list) / len(val_recon_list)
        
        # 分类任务指标计算
        if model.use_classification:
            avg_val_intra = total_val_intra_dist / samples_count if samples_count > 0 else 0.0
            current_inter_sim = get_inter_class_similarity(
                model, test_data, device,
                num_samples=len(full_dataset) - int(len(full_dataset) * split_ratio)
            )
            
            # 状态判定
            if current_inter_sim > 0.95 and avg_val_intra < 0.05:
                status = "🔴 特征坍塌 (Collapse)"
            elif current_inter_sim > 0.85:
                status = "🟡 正在解耦 (Breakout)"
            elif current_inter_sim <= 0.85 and avg_val_intra > 0.1:
                status = "🟢 理想状态 (Golden)"
            else:
                status = "🔵 调优中"
            
            # 动态调整超参数（强化解耦）
            if current_inter_sim > 0.8:
                sim_weight = min(sim_weight + 10.0, 300.0)
                aux_weight = min(aux_weight + 5.0, 200.0)
                sample_sim_weight = min(sample_sim_weight + 10.0, 500.0)
                recon_weight = max(recon_weight - 0.1, 0.01)  # 最低 0.01
            else:
                recon_weight = min(recon_weight + 0.1, 1.0)
                sample_sim_weight = max(sample_sim_weight - 5.0, 20.0)
        else:
            avg_val_intra = 0.0
            current_inter_sim = 0.0
            status = "📈 回归任务（仅重构）"

        avg_val_loss = avg_val_recon
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, bestmodelpath)
            print(f"⭐ New Best Model! Val Recon: {avg_val_loss:.4f}")

        # 打印日志
        print("-" * 70)
        print(status)
        print(f"Epoch {epoch:03d} | Train Recon: {total_recon/len(train_loader):.4f} | "
              f"Val Recon: {avg_val_recon:.4f}")
        if model.use_classification:
            print(f"Intra-Dist: {avg_val_intra:.4f} | Inter-Sim: {current_inter_sim:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Recon_w: {recon_weight:.2f} | Aux_w: {aux_weight:.1f} | "
              f"SampleSim_w: {sample_sim_weight:.1f}")

        # 定期诊断（仅分类）
        if model.use_classification and (epoch + 1) % 20 == 0:
            diagnose_vq_collapse(
                test_data, model, device,
                num_samples=len(full_dataset) - int(len(full_dataset) * split_ratio)
            )

        # 热重启
        if optimizer.param_groups[0]['lr'] < 1e-6:
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
            optimizer.state.clear()
            print("🚀 Learning rate reset!")

        # 保存最新 checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, modelpath)

    print("\n✅ 训练完成！")
    

# 如果它开始下降到 0.8-0.9 左右，你的 VQVAE 就算真正训练成功了，此时再训练之前的 Transformer 就会有立竿见影的效果。
def get_inter_class_similarity(model, dataset, device, num_samples=100):
    model.eval()
    vectors = {0: [], 1: []}
    with torch.no_grad(): # ✅ 强制不追踪梯度
        for x, label in dataset:
            l = label.item()
            if l in vectors and len(vectors[l]) < num_samples:
                x = x.unsqueeze(0).to(device)
                z_e = model.encoder(x)
                # 使用 .detach() 确保彻底断开计算图
                vectors[l].append(z_e.view(-1).detach()) 
            if len(vectors[0]) >= num_samples and len(vectors[1]) >= num_samples:
                break
    
    if len(vectors[0]) < 2 or len(vectors[1]) < 2: return 1.0
    
    mu0 = torch.stack(vectors[0]).mean(0)
    mu1 = torch.stack(vectors[1]).mean(0)
    # 返回纯 Python 浮点数
    sim = F.cosine_similarity(mu0.unsqueeze(0), mu1.unsqueeze(0)).item()
    return sim

 
from evaluate import run_full_validation, visualize_tsne, run_tsne_analysis
def test(save_path0, isbest=True, channel_idx=0):
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False) #BATCH_SIZE
    print(f"数据集加载成功: 测试样本数={len(test_data)}")
    modelpath = os.path.join(save_path, "vqvae_model.pth")
    bestmodelpath = modelpath.replace("vqvae_model.pth", "best_vqvae_model.pth")
    save_path1 = save_path0 + '/last/'   #
    if isbest:
        modelpath = bestmodelpath
        save_path1 = save_path0 + '/best/'  # 
    checkpoint = torch.load(modelpath)
    os.makedirs(save_path1, exist_ok=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Finetune from', modelpath)
    model.eval()
    
    if subset != 'joystick_track': diagnose_vq_collapse(test_data, model, device, len(full_dataset)-int(len(full_dataset)*split_ratio))
        
    # 6. 执行最终评估
    print("\n--- 执行完整验证 Pipeline ---")
    print("--- 1. Waveform Fidelity Check ---")
    test_batch, test_labels = next(iter(test_loader))
    test_batch = test_batch.to(device)
    all_idx = []
    
    model.eval()
    with torch.no_grad():
        x_recon, z_e, z_q, all_indices= model(test_batch)
        all_idx.append(all_indices.view(-1).cpu())
    # 画出第 0 个样本的第 5 号通道重构图
    plot_reconstruction_comparison(test_batch, x_recon, channel_idx=channel_idx, savepath=save_path1)
    
    # print("\n--- 2. Semantic Clustering Check ---")
    # silhouette = run_tsne_analysis(model, test_loader, device, save_path1)

    print("\n--- 3. Codebook Utilization Check ---")
    # 统计 Codebook 的激活比例
    unique_tokens = torch.unique(all_indices).cpu().numpy()
    print(f"Active Tokens in this batch: {len(unique_tokens)} / {model.vq_layer._num_embeddings}")

    if len(unique_tokens) < 10:
        print("⚠️ Warning: Codebook Collapse detected! Use EMA or Dead Code Reset.")
    else:
        print("✅ Codebook utilization is healthy.")
        
    analyze_token_distribution(model, train_loader, test_loader, codebook, device)

    # 统计测试集里所有 indices 的分布  
    all_idx = torch.cat(all_idx)
    plt.hist(all_idx.numpy(), bins=512)
    plt.title("Codebook Usage Distribution")
    plt.savefig(save_path1+'Codebook_Usage_Distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # print("\n--- 4. 生成最终 t-SNE 可视化 ---")# t-SNE 图：5个手指的聚类颜色区分明显
    # # 此处可以使用 test_loader 来评估模型在未见数据上的聚类效果
    # visualize_tsne(model, test_loader, device, save_path1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    EPOCHS = 1000
    LEARNING_RATE = 1e-4
    winsize = 1000
    embed_dim = 64
    codebook = 512
    Tokenstride = 4
    # subset = 'fingerflex'  
    # subset = 'gestures'  # 
    # subset = 'motor_basic'  # 
    subset = 'joystick_track'  #
    # subset = 'BCI_Competion4_dataset4_data_fingerflexions'  
    
    print("正在加载 ECoG 数据集...", subset)
    datadir = '/disk2/user1/dataset/BCI-Standford/%s/%s/data/' % (subset, subset)
    split_ratio = 0.9  # 0.6  # 
    
    if embed_dim == 64: EM = ''
    else: EM = '_E%d' % embed_dim
    if Tokenstride == 4: TStride = ''
    elif Tokenstride == 8: TStride = 'token31'
    elif Tokenstride == 16: TStride = 'token16'
    elif Tokenstride == 32: TStride = 'token8'

    # 3. 每个mat的前90%用于训练，后10%测试
    if subset == 'BCI_Competion4_dataset4_data_fingerflexions':
        save_path = "./checkpoints/BCI Competition IV/VQTokenizer_c%d_4/" % codebook #  
        full_dataset = BCIFingersDataset_TT(datadir, window_size=winsize, stride=256, single_wind=False, 
                 single_channel=True, channel_idx=None)
    else:
        save_path = "./checkpoints/Stanford/VQTokenizer/%s_c%d_4%s%s/" % (subset, codebook, EM, TStride) 
        # save_path = "./checkpoints/Stanford/VQTokenizer/%s_c%d_4%s%s_ratio6/" % (subset, codebook, EM, TStride) 
        full_dataset = MillerFingersDataset_TT(datadir, window_size=winsize, stride=256, single_wind=False, 
                single_channel=True, channel_idx=None, subset=subset, split_ratio=split_ratio)
    os.makedirs(save_path, exist_ok=True)

    full_dataset.set_mode('train')
    num_classes = len(set(full_dataset.labels))
    if subset == 'joystick_track': num_classes = 0
    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    full_dataset.set_mode('test')
    test_data = full_dataset
    test_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)
    
    # 4. 模型初始化 (in_channels 需匹配数据集通道数，Miller 数据通常是 64)
    model = NeuralVQVAE(in_channels=1, codebook_size=codebook, 
                embed_dim=embed_dim, num_classes=num_classes, tokenstride=Tokenstride).to(device)
    if subset == 'motor_basic': main1()
    else: main()
    
    print("\n--- 执行完整验证 Pipeline ---")
    test(save_path, isbest=True)
    test(save_path, isbest=False)

