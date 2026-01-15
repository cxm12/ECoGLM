import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, confusion_matrix
import seaborn as sns
import os
from torch.utils.data import DataLoader, random_split
from data import MillerFingersDataset, BCIComp4Dataset4, \
    BCIComp4TestDataset, BCIComp4Dataset4_single

# ==========================================
# 1. 物理维度：波形重构对比图 (Reconstruction Visualization)
# ==========================================
def plot_reconstruction_comparison(original, reconstructed, channel_idx=0, 
                                   num_samples=1, savepath='./'):
    """
    对比原始 ECoG 波形与重构后的波形
    original/reconstructed: [B, C, T]
    """
    plt.figure(figsize=(12, 4))
    orig = original[0, channel_idx, :].cpu().numpy()
    recon = reconstructed[0, channel_idx, :].cpu().numpy()
    
    time = np.arange(len(orig))
    plt.plot(time, orig, label='Original ECoG', color='blue', alpha=0.6)
    plt.plot(time, recon, label='Reconstructed (from Tokens)', color='red', linestyle='--')
    
    # 计算相关系数 (Pearson Correlation)
    correlation = np.corrcoef(orig, recon)[0, 1]
    
    plt.title(f"Channel {channel_idx} Reconstruction (Corr: {correlation:.3f})")
    plt.legend()
    plt.xlabel("Time Samples")
    plt.ylabel("Normalized Amplitude")
    plt.grid(True, alpha=0.3)
    plt.savefig(savepath+"rec_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


# ==========================================
# 2. 语义维度：t-SNE 聚类分析 (Semantic Clustering)
# ==========================================
def run_tsne_analysis(model, test_loader, device, savepath):
    """
    验证 Token 是否在隐空间内形成了具有语义的神经原语簇
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            # 提取量化后的特征索引，将其展平作为特征向量
            _, _, indices = model(x) # indices: [B, T_compressed, 1]
            feat = indices.view(x.size(0), -1).cpu().numpy()
            all_features.append(feat)
            all_labels.append(y.view(-1).numpy())
            
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 执行 t-SNE
    print("Running t-SNE... (this may take a minute)")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
    embeds = tsne.fit_transform(all_features)
    
    # 计算轮廓系数 (Silhouette Score) - 证明聚类质量的量化指标
    ss = silhouette_score(embeds, all_labels)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeds[:, 0], embeds[:, 1], c=all_labels, cmap='Set1', s=20, alpha=0.8)
    plt.colorbar(scatter, label='Action Label (Finger Index)')
    plt.title(f"t-SNE of Neural Tokens (Silhouette Score: {ss:.3f})")
    plt.savefig(savepath+"t-SNE_Neural_Tokens.png", dpi=300, bbox_inches='tight')
    plt.show()
    return ss


def run_tsne_analysis_raw(test_loader, savepath):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            feat = x.reshape(x.shape[0], -1) # x.view(x.size(0), -1)
            all_features.append(feat)
            all_labels.append(y.view(-1).numpy())
            
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 执行 t-SNE
    print("Running t-SNE... (this may take a minute)")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
    embeds = tsne.fit_transform(all_features)
    
    # 计算轮廓系数 (Silhouette Score) - 证明聚类质量的量化指标
    ss = silhouette_score(embeds, all_labels)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeds[:, 0], embeds[:, 1], c=all_labels, cmap='Set1', s=20, alpha=0.8)
    plt.colorbar(scatter, label='Action Label (Finger Index)')
    plt.title(f"t-SNE of Neural Tokens (Silhouette Score: {ss:.3f})")
    plt.savefig(savepath+"raw_t-SNE_Neural_Tokens.png", dpi=300, bbox_inches='tight')
    plt.show()
    return ss


# ==========================================
# 3. 性能维度：分类准确率与混淆矩阵 (Linear Probing)
# ==========================================
def evaluate_classification_performance(model, test_loader, device, savepath):
    """
    使用简单的线性层或简单的 MLP 验证 Token 的信息丰富度
    如果 Token 选得好，即使不看原始波形，分类准确率也应该很高
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            # 获取 Token 序列
            _, _, indices = model(x)
            # 简易解码器逻辑：这里我们直接用 Token 的分布特征做统计
            # 在实际论文中，这里会接一个微小的 Linear Layer
            # 此时为了快速验证，我们计算每个 Batch 的预测趋势
            feat = indices.view(x.size(0), -1) 
            # 模拟：简单统计每个样本最常出现的 Token（仅作展示）
            # 建议：实际科研中，训练一个 Linear Probe SVM 更好
            
    # 输出混淆矩阵 (Confusion Matrix)
    # 此处假设你已经训练了一个简易的 Linear Probe 分类器
    cm = confusion_matrix(targets, preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d')# , cmap='Blues'
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(savepath+'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# 4. 执行完整验证 Pipeline
# ==========================================
def run_full_validation(model, test_loader, device, savepath, channel_idx=5):
    print("--- 1. Waveform Fidelity Check ---")
    # 取一个 Batch 进行可视化
    test_batch, test_labels = next(iter(test_loader))
    test_batch = test_batch.to(device)
    
    model.eval()
    with torch.no_grad():
        _, x_recon, _ = model(test_batch)
    
    # 画出第 0 个样本的第 5 号通道重构图
    plot_reconstruction_comparison(test_batch, x_recon, channel_idx=channel_idx, savepath=savepath)
    
    print("\n--- 2. Semantic Clustering Check ---")
    silhouette = run_tsne_analysis(model, test_loader, device, savepath)
    
    print("\n--- 3. Codebook Utilization Check ---")
    # 统计 Codebook 的激活比例
    _, _, all_indices = model(test_batch)
    unique_tokens = torch.unique(all_indices).cpu().numpy()
    print(f"Active Tokens in this batch: {len(unique_tokens)} / {model.vq_layer._num_embeddings}")
    
    if len(unique_tokens) < 10:
        print("⚠️ Warning: Codebook Collapse detected! Use EMA or Dead Code Reset.")
    else:
        print("✅ Codebook utilization is healthy.")


def visualize_tsne(model, dataloader, device, savepath):
    """提取 t-SNE 的辅助函数"""
    model.eval()
    all_indices = []
    all_labels = []    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            _, _, indices = model(x)
            feat = indices.view(x.size(0), -1).cpu().numpy()
            all_indices.append(feat)
            all_labels.append(y.numpy())
            
    all_indices = np.concatenate(all_indices, axis=0)
    all_labels = np.concatenate(all_labels, axis=0).flatten()
    
    tsne = TSNE(n_components=2, perplexity=30)
    X_embedded = tsne.fit_transform(all_indices)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=all_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Finger Index')
    plt.title("Neural Primitive Clusters on Test Set")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(savepath+"tsne_clusters.png", dpi=300, bbox_inches='tight')
    plt.show()
    

def visualize_tsne_raw(dataloader, savepath):
    """提取 t-SNE 的辅助函数"""
    all_indices = []
    all_labels = []    
    with torch.no_grad():
        for x, y in dataloader:
            feat = x.view(x.size(0), -1)
            all_indices.append(feat)
            all_labels.append(y.numpy())
            
    all_indices = np.concatenate(all_indices, axis=0)
    all_labels = np.concatenate(all_labels, axis=0).flatten()
    
    tsne = TSNE(n_components=2, perplexity=30)
    X_embedded = tsne.fit_transform(all_indices)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=all_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Finger Index')
    plt.title("Neural Primitive Clusters on Test Set")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(savepath+"rawdata_tsne_clusters.png", dpi=300, bbox_inches='tight')
    plt.show()
 

def runRawdata():
    savepath = './checkpoints/rawdata_analysis/'
    os.makedirs(savepath, exist_ok=True)
    
    full_dataset = BCIComp4Dataset4_single(datadir = '/mnt/home/user1/MCX/EEGLM/data/ECoG/BCICIV_4_mat/',
                    window_size=1000, stride=512, single_channel=True # 单通道
                               , channel_idx=None)

    train_data, test_data = random_split(full_dataset, [int(len(full_dataset)*0.9),
        len(full_dataset)-int(len(full_dataset)*0.9)])
    
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    print("\n--- 生成最终 t-SNE 可视化 ---")# t-SNE 图：5个手指的聚类颜色区分明显
    # 此处可以使用 test_loader 来评估模型在未见数据上的聚类效果
    visualize_tsne_raw(test_loader, savepath)
    run_tsne_analysis_raw(test_loader, savepath)
    

# ======================================================
# VQ 空间深度诊断报告
# ======================================================
import torch.nn.functional as F
def diagnose_vq_collapse(dataset, vqvae, device, num_samples=20):
    vqvae.eval()
    # 存储原始特征 (Encoder Output)
    ze_0, ze_1 = [], []
    # 存储量化特征 (Quantized Output)
    zq_0, zq_1 = [], []
    class0_vectors = []
    class1_vectors = []
    
    print(f"\n" + "="*50)
    print(f"🔍 VQ 空间深度诊断报告 (样本数: {num_samples})")
    print("="*50)
    with torch.no_grad():
        found0, found1 = 0, 0
        for x, label in dataset:
            x = x.unsqueeze(0).to(device)
            # 获取所有中间变量
            x_recon, z_e, z_q, indices = vqvae(x)
            # 展平为 [T * embed_dim] 方便计算余弦相似度
            vec_q = z_q.view(-1) 
            vec_e = z_e.view(-1)
            
            if label == 0 and found0 < num_samples:
                class0_vectors.append(vec_q)
                ze_0.append(vec_e)
                zq_0.append(vec_q)
                found0 += 1
            elif label == 1 and found1 < num_samples:
                class1_vectors.append(vec_q)
                ze_1.append(vec_e)
                zq_1.append(vec_q)
                found1 += 1
            if found0 >= num_samples and found1 >= num_samples: break
                                                
    # 计算均值向量
    m_ze0, m_ze1 = torch.stack(ze_0).mean(0), torch.stack(ze_1).mean(0)
    m_zq0, m_zq1 = torch.stack(zq_0).mean(0), torch.stack(zq_1).mean(0)
    # --- 计算相似度 ---
    # 1. Encoder 阶段相似度 (判断 Encoder 是否有区分能力)
    sim_ze = F.cosine_similarity(m_ze0.unsqueeze(0), m_ze1.unsqueeze(0)).item()
    # 2. VQ 阶段相似度 (判断 Codebook 是否坍塌)
    sim_zq = F.cosine_similarity(m_zq0.unsqueeze(0), m_zq1.unsqueeze(0)).item()
    print(f"1. Encoder 原始特征相似度 (z_e): {sim_ze:.4f}")
    print(f"2. Codebook 量化特征相似度 (z_q): {sim_zq:.4f}")
    print("-" * 30)
    # --- 逻辑判定 ---
    if sim_ze > 0.98:
        print("❌ 判定结果: [Encoder 死亡]")
        print("原因: Encoder 根本没有尝试去区分不同的动作。")
        print("对策: 增加 Aux Loss 权重，或引入 Triplet Loss，检查数据是否归一化。")
    elif sim_zq > 0.98 and sim_ze < 0.90:
        print("❌ 判定结果: [VQ/Codebook 坍塌]")
        print("原因: Encoder 分开了特征，但所有特征都被映射到了同一个或极其相似的 Code 上。")
        print("对策: 降低 EMA Decay (至 0.8)，增加 Codebook 学习率，或开启死码重启。")
    elif sim_zq < 0.90:
        print("✅ 判定结果: [模型健康]")
        print("特征在编码和量化阶段都保持了区分度。")
    else:
        print("⚠️ 判定结果: [边缘状态]")
        print("有一定的区分度，但建议继续加大辅助分类压力。")
    print("="*50 + "\n")

    # 2. 计算相似度
    c0_mean = torch.stack(class0_vectors).mean(0)
    c1_mean = torch.stack(class1_vectors).mean(0)
    # 类内相似度 (Intra-class)
    # sim_00 = torch.nn.functional.cosine_similarity(class0_vectors[0], class0_vectors[1], dim=0)
    sim_00 = compute_intra_sim(ze_0)
    # 类间相似度 (Inter-class)
    sim_01 = torch.nn.functional.cosine_similarity(c0_mean, c1_mean, dim=0)
    print(f"类0 内部相似度: {sim_00:.4f} (越接近 1 说明编码越稳定)")
    print(f"类0 与 类1 的类间相似度: {sim_01.item():.4f}")
    if sim_01 > 0.95:
        print("⚠️ 警告: 类间相似度极高！VQVAE 无法区分静息和动作，Transformer 很难学。")
    else:
        print("✅ 编码具有区分度，请重点检查分类头的 Loss 权重。")

def compute_intra_sim(vectors):
    v_stack = torch.stack(vectors) # [N, D]
    v_norm = F.normalize(v_stack, p=2, dim=1)
    sim_matrix = torch.matmul(v_norm, v_norm.t()) # [N, N]
    # 取上三角（不含对角线）的平均值
    mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
    return sim_matrix[mask].mean().item()


def calculate_accuracy(logits, targets):
    """
    计算 Top-1 准确率
    logits: [Batch, Vocab_Size]
    targets: [Batch] (Qwen Token IDs)
    """
    assert logits.device == targets.device, "Tensor on different devices"

    # 获取预测的 Token ID (概率最大的索引)
    _, predicted = torch.max(logits, dim=1)

    # 对比预测值与真值
    correct = (predicted == targets).sum().item()
    total = targets.size(0)

    accuracy = correct / total
    return accuracy, predicted


def save_confusion_matrix(y_true, y_pred, epoch, acc, LABEL_MAP, save_path):
    """绘制并保存混淆矩阵图片"""
    # 获取类别名称
    classes = [LABEL_MAP[i] for i in range(len(LABEL_MAP))]
    
    # 计算矩阵 (归一化到 0-1 范围，看比例更直观)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cm_perc = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) # 防止除0
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.title(f'Confusion Matrix (Epoch {epoch} | Acc: {acc*100:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 保存图片
    img_path = os.path.join(save_path, f"cm_epoch_{epoch}.png")
    plt.savefig(img_path)
    plt.close() # 释放内存
    print(f"📊 混淆矩阵已保存至: {img_path}")

from collections import Counter
def analyze_token_distribution(vqvae, train_loader, test_loader, codebook_size, device):
    vqvae.eval()
    train_tokens = []
    test_tokens = []

    print("正在提取训练集 Token...")
    with torch.no_grad():
        for x, _ in train_loader:
            _, _, _, indices = vqvae(x.to(device))
            train_tokens.extend(indices.view(-1).cpu().numpy())

    print("正在提取验证集 Token...")
    with torch.no_grad():
        for x, _ in test_loader:
            _, _, _, indices = vqvae(x.to(device))
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
        
 
 
# runRawdata()

# 在训练脚本末尾调用：
# run_full_validation(model, test_loader, device)

'''
相关系数 (Correlation)：
对于 ECoG 信号，如果相关系数 $r > 0.7$，说明 VQ 编码器成功保留
大部分物理特征。在 1000:1 的压缩比下做到这一点。
t-SNE 聚类：
失败的表现： 所有颜色的点混在一起。
成功的表现： 同一种颜色（同一个动作）的点形成了一个明确的孤岛。
证明你定义的“神经原语”在数学上是可分的。
Codebook 利用率：
如果 512 个 Token 中有 50-200 个被频繁激活，
说明模型学到脑电信号中丰富的亚稳态。
'''