import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from data import MillerFingersDataset, ClassBalancedBatchSampler, prepare_dataloaders
from test_VQTokenizer import plot_reconstruction_comparison, diagnose_vq_collapse, analyze_token_distribution
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
    def __init__(self, in_channels=1, codebook_size=512, embed_dim=64, num_classes=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(embed_dim),
            nn.ReLU(),
            # ✅ 新增 AvgPool：下采样抛弃高频噪声，强制提取缓慢变化的语义特征
            nn.AvgPool1d(kernel_size=4, stride=4), 
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
        vq_loss, z_q, perplexity, indices = self.vq_layer(z_e)
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
    # 使用 ReduceLROnPlateau，当重构损失进入平台期时降低学习率
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

    # 初始超参数
    aux_weight = 20.0     # 分类头权重
    sim_weight = 50.0     # 类间排斥权重
    # intra_weight = 0.0  # 🚀 强化：类内凝聚权重 (解决相似度低的核心)
    # 1. Intra-Dist 会回升（比如到 0.2），这是好事！说明特征空间不再是死的。
    # 2. Inter-Sim 会开始下降（目标是破掉 0.90），这代表类间区分度出来了。
    sample_sim_weight = 30.0 # 样本级权重
    recon_weight = 0.2     # 极低权重，只给 Encoder 留一口气
    
    print("\n🚀 开始强化版全类聚合 VQ-Tokenizer 训练...")    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_recon = 0
        total_aux = 0
        # total_intra = 0
        # total_inter = 0        
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

            optimizer.zero_grad()
            # 1. Forward
            vq_loss, x_recon, indices, cls_logits, _, z_e = model(x)            
            # 2. 基础损失
            recon_loss = F.mse_loss(x_recon, x)
            aux_loss = F.cross_entropy(cls_logits, labels)
            # 3. 核心：样本级对比损失 (Sample-level Contrastive Loss)
            # z_v: [Batch, Channel]
            z_v = z_e.mean(dim=-1)             
            # 🔥 关键步骤：特征归一化到单位球，消除幅值干扰，强制关注方向差异
            z_v_norm = F.normalize(z_v, p=2, dim=1)            
            # 计算余弦相似度矩阵 [Batch, Batch]
            sim_matrix = torch.matmul(z_v_norm, z_v_norm.t())            
            # 构造掩码：只有标签不同的样本对才为 True
            diff_label_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
            # 🌟 样本级排斥：如果不同类样本相似度 > 0.3，就施加惩罚
            # 使用较高的 Margin (0.3) 强制拉开距离
            sample_inter_penalty = (torch.clamp(sim_matrix - 0.3, min=0) * diff_label_mask).sum() / (diff_label_mask.sum() + 1e-6)
            
            # 4. 原型级排斥 (保持原有的全局约束)
            model.update_prototypes(z_v.detach(), labels)
            all_protos = F.normalize(model.class_prototypes, p=2, dim=1)
            proto_sims = torch.matmul(all_protos, all_protos.t())
            mask = torch.eye(num_classes, device=device).bool()
            proto_inter_penalty = torch.clamp(proto_sims[~mask] - 0.4, min=0).pow(2).mean()

            # 5. 组合总损失
            # ⚠️ 注意：初期我们将 recon_weight 设得很低，sample_sim_weight 设得很高
            loss = (recon_weight * recon_loss) + vq_loss + (aux_weight * aux_loss) + \
                   (sim_weight * proto_inter_penalty) + (sample_sim_weight * sample_inter_penalty)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_recon += recon_loss.item()
            total_aux += aux_loss.item()

        # 4. 验证与指标诊断
        model.eval()
        val_recon_list = []
        # 1. 计算测试集的 Inter-Sim (类间中心相似度)
        current_inter_sim = get_inter_class_similarity(model, test_data, device, num_samples=len(full_dataset)-int(len(full_dataset)*0.9))
        # 2. 计算测试集的平均 Intra-Dist (类内距离 1-Sim)
        total_val_intra_dist = 0
        samples_count = 0
        with torch.no_grad():
            for vx, vl in test_loader:
                vx, vl = vx.to(device), vl.to(device)
                v_recon, z_e, _, _ = model(vx)
                
                # 计算重构 Loss
                val_recon_list.append(F.mse_loss(v_recon, vx).item())
                
                # 计算该样本到其类中心的距离
                z_v = z_e.mean(dim=-1)
                target_proto = model.class_prototypes[vl]
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
        # if avg_val_intra < 0.05:
        #     intra_weight = 0.0
        # else:
        #     intra_weight = 1.0
        avg_val_loss = avg_val_recon
        scheduler.step(avg_val_loss)
        
        # 5. 模型保存逻辑
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
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
        if (epoch + 1) % 20 == 0:
            diagnose_vq_collapse(test_data, model, device, num_samples=len(full_dataset)-int(len(full_dataset)*0.9))
            
        # 8. 热重启机制 (防止学习率过低导致死锁)
        if optimizer.param_groups[0]['lr'] < 1e-6:
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
            optimizer.state.clear() 
            print("🚀 Learning rate reset!")

        # 保存 Last Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, modelpath)

    print("\n训练流程结束。")
    

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
    
    diagnose_vq_collapse(test_data, model, device, len(full_dataset)-int(len(full_dataset)*0.9))
        
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 42  # 64  #
    EPOCHS = 1000
    LEARNING_RATE = 1e-4
    codebook = 512  # 256  # 128  #
    winsize = 1024
    
    subset = 'fingerflex'  # 'joystick_track'  # 'gestures'  # 'motor_basic'  #
    save_path = "./checkpoints/Stanford/VQTokenizer/%s_c%d_3/" % (subset, codebook) #  
    # save_path = "./checkpoints/Stanford/VQTokenizer/%s_c%d_2/" % (subset, codebook) #  
    os.makedirs(save_path, exist_ok=True)
    
    print("正在加载 ECoG 数据集...")
    datadir = '/disk2/user1/dataset/BCI-Standford/%s/%s/data/' % (subset, subset)
    full_dataset = MillerFingersDataset(datadir, window_size=winsize, stride=256,
                        single_channel=True, channel_idx=None, subset=subset)
    num_classes = full_dataset.num_classes
    
    # 时序切分
    train_loader, test_loader, test_data = prepare_dataloaders(full_dataset, BATCH_SIZE, num_classes=num_classes, ratio=0.9)
    # torch.manual_seed(0)  
    # train_data, test_data = random_split(full_dataset, [int(len(full_dataset)*0.9),
    #     len(full_dataset)-int(len(full_dataset)*0.9)])
    # sampler = ClassBalancedBatchSampler(train_data, batch_size=BATCH_SIZE, num_classes=6, 
    #                         samples_per_class=BATCH_SIZE//6)
    # train_loader = DataLoader(train_data, batch_sampler=sampler)
    # # train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # 4. 模型初始化 (in_channels 需匹配数据集通道数，Miller 数据通常是 64)
    model = NeuralVQVAE(in_channels=1, codebook_size=codebook, 
                        embed_dim=64, num_classes=num_classes).to(device)
    
    main()
    
    print("\n--- 执行完整验证 Pipeline ---")
    test(save_path, isbest=True)
    test(save_path, isbest=False)
    
    
