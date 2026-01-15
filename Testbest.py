import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import os
from VQTokenizer import NeuralVQVAE
from torch.utils.data import DataLoader, random_split
from data import MillerFingersDataset, BCIComp4Dataset4_single
# import math
from model.tokenalign_model import TokenAlignmentTrainer, TransformerVQToTokenModel

    
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


# --- 3. 训练脚本 ---
def train_alignment():
    vqvae.eval() # VQVAE 冻结或仅作为特征提取    
    if os.path.exists(f"{save_path}token_aligner_best.pth"):
        print("加载已有的 Token 对齐模型权重...")
        token_model.load_state_dict(torch.load(f"{save_path}token_aligner_best.pth"))

    # optimizer = torch.optim.AdamW(token_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(token_model.parameters(), lr=LEARNING_RATE)
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # 加入标签平滑提高泛化性    
    criterion = nn.CrossEntropyLoss() # 加入标签平滑提高泛化性    
    
    prototypes = []
    prototype_matrix=None
    if 'CtrLoss' in save_path:
        qwen_model = AutoModelForCausalLM.from_pretrained(
            "../Qwen2.5-7B-Instruct", torch_dtype=torch.float16, 
            device_map=device, trust_remote_code=True)
        with torch.no_grad():
            print(">>> 正在生成 6 个动作的文本原型向量...")
            embeddings_layer = qwen_model.get_input_embeddings()
            for i in range(len(LABEL_MAP)):
                text = LABEL_MAP[i]
                # 获取该动作对应的 Token ID
                token_ids = aligner.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
                # 获取 Embedding 并取平均（如果是多 token 词组）  # [1, seq_len, 4096] -> [4096]
                text_embeds = embeddings_layer(token_ids).mean(dim=1) 
                # L2 归一化，方便后续直接做点积计算余弦相似度
                text_embeds = F.normalize(text_embeds, p=2, dim=1)
                prototypes.append(text_embeds) # label -> list of embeddings
        # 将 6 个原型拼接成矩阵 [6, 4096]
        prototype_matrix = torch.cat(prototypes, dim=0)
    
    print("\n开始训练 Token 对齐模型...")
    # print(any(p.requires_grad for p in token_model.parameters()))  # True
    bestacc = 0.0
    bestepo = 0
    for epoch in range(EPOCHS):
        token_model.train() # transformer_tokenizer.train()
        total_loss = 0
        correct = 0  # total = 0
    
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            # print('x = ', x, '\nlabels = ', labels)  # shape: [B]
            # 1. 提取 VQVAE Indices
            with torch.no_grad():
                _, _, indices = vqvae(x) # indices shape: [B, 250]
                # print("VQVAE Indices shape: ", indices.shape, indices)
            
            # 2. 获取目标 GT Tokens (Qwen ids)
            if 'vocab6' in save_path:
                gt_tokens = aligner.get_local_labels(labels) # [B]
            else:
                gt_tokens = aligner.get_gt_batch(labels) # [B]
            # print("GT Tokens shape: ", gt_tokens.shape) torch.Size([32])
            
            # 3. 前向传播 # 4. 计算损失
            if 'CtrLoss' in save_path:
                embeddings = token_model(indices)  # [B, embed_dim], e.g., [32, 128]
                # loss = supervised_contrastive_loss(embeddings, labels, temperature=0.1)
                loss = contrastive_loss(embeddings, labels, 0.5)
            else:
                logits = token_model(indices) # [B, vocab_size]
                # print("Logits shape: ", logits.shape)  # [32, 151665]
                loss = criterion(logits, gt_tokens)
            
            # 5. 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪防止 Transformer 训练不稳定            
            torch.nn.utils.clip_grad_norm_(token_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 6. 计算本 Batch 准确率
            # 计算样本间的余弦相似度矩阵 # 通过看每个样本的“最近邻”是否是同类来计算 Acc
            if 'CtrLoss' in save_path:
                # sim_matrix = torch.matmul(embeddings, embeddings.T)
                # # print("Sim_matrix shape:", sim_matrix.shape, sim_matrix) #[B, B]
                # sim_matrix.fill_diagonal_(-1)
                # _, closest_idx = torch.max(sim_matrix, dim=1)
                # # print("Closest_idx:", closest_idx) # [ 7,  8,  4, 11, 19, 12, 19, 13,  4, 20, 13, 19, 15,  7, 19, 12,  7, 21, 6,  6,  6,  6, 17, 19]
                # predicted = labels[closest_idx]
                gt_tokens = labels
                # # print("Predicted labels:", predicted, "GT labels:", gt_tokens) 
                # # [1, 2, 4, 5, 1, 0, 1, 1, 4, 2, 1, 1, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 5, 1],[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
                
                # 计算信号与 6 个原型的相似度矩阵 [B, 6] # 这里的矩阵乘法等同于计算余弦相似度
                similarities = torch.matmul(embeddings.float(), prototype_matrix.T.float()) 
                predicted = torch.argmax(similarities, dim=1)
            else:
                _, predicted = torch.max(logits, 1)
                
            # total += gt_tokens.size(0)
            correct += (predicted == gt_tokens).sum().item()
            total_loss += loss.item()
            # print('loss.item()=', loss.item())
        print(f"Epoch {epoch} | Train Loss: {total_loss/len(train_loader):.4f} | Accuracy: {correct / len(train_loader):.2f}%")
    
        acc = test_model_forward(prototype_matrix)
        if acc > bestacc:
            bestacc = acc
            bestepo = epoch
            print("保存最佳模型权重...")
            torch.save(token_model.state_dict(), tokenmodel_path.replace(".pth", "_best.pth"))
        print(f"验证集准确率: {acc*100:.2f}%, Best Acc: {bestacc*100:.2f}% at Epoch {bestepo}")
    torch.save(token_model.state_dict(), tokenmodel_path)

    
# --- 4. 测试脚本 ---
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


def test_model_forward(prototype_matrix=None):
    print(">>> 开始测试 TransformerVQToTokenModel 前向传播...")
    # 形状为 [Batch, Seq_Len]，值在 0 到 CODEBOOK_SIZE-1 之间
    vqvae.eval()
    token_model.eval()
    
    total_acc = 0
    total_samples = 0
    # criterion = nn.CrossEntropyLoss()
    # # 建立一个反向映射，用于打印结果 (ID -> 文本)
    # id_to_text = {v: aligner.tokenizer.decode([v]) for v in aligner.gt_token_map.values()}
        
    # --- 如果是对比学习模式，我们需要准备“类别特征中心” ---
    if 'CtrLoss' in save_path:
        # 简单方案：从验证集中每类抽取特征求平均作为“标准向量”
        # 或者直接使用文本的 Embedding (如果你有 Projection 层对齐到 Qwen)
        all_embeds = []
        all_labels = []
                
    with torch.no_grad():
        for x, labels in test_loader:
            x, labels = x.to(device), labels.to(device)
            # 1. 提取 VQ 索引
            _, _, indices = vqvae(x)  # print("VQVAE Indices shape: ", indices.shape)
            # 2. 获取 GT Tokens
            if 'vocab6' in save_path:
                gt_tokens = aligner.get_local_labels(labels) # [B]
            else:
                gt_tokens = aligner.get_gt_batch(labels) # [B]
            # print("GT Tokens shape: ", gt_tokens.shape) torch.Size([1])
            
            # 3. 前向传播 # 4. 计算损失
            if 'CtrLoss' in save_path:
                output = token_model(indices)  # [B, embed_dim], e.g., [32, 128]
                all_embeds.append(output)
                all_labels.append(labels)  # shape: [B]
                
                # # 计算样本间的余弦相似度矩阵 # 通过看每个样本的“最近邻”是否是同类来计算 Acc
                # sim_matrix = torch.matmul(output, output.T)
                # # 排除掉自己和自己的对比
                # sim_matrix.fill_diagonal_(-1)
                # # 找到最相似的那个样本的索引
                # _, closest_idx = torch.max(sim_matrix, dim=1)
                # pred_labels = labels[closest_idx] 
                                
                # 计算信号与 6 个原型的相似度矩阵 [B, 6] # 这里的矩阵乘法等同于计算余弦相似度
                similarities = torch.matmul(output.float(), prototype_matrix.T.float()) 
                pred_labels = torch.argmax(similarities, dim=1)            
                acc = (pred_labels == labels).sum().item()
            else:
                logits = token_model(indices)
                # print(f"输入形状: {indices.shape}") # [1, 250]
                # print(f"输出形状 (Logits): {logits.shape}, GT {gt_tokens.shape}") # [1, 151643]
                # # 输出形状 (Logits): torch.Size([1, 6])
                
                # loss = criterion(logits, gt_tokens)
                acc, preds = calculate_accuracy(logits, gt_tokens)
                
            total_acc += acc * x.size(0)
            total_samples += x.size(0) 
            # print(f"Loss: {loss.item():.4f} | Batch Acc: {acc*100:.2f}%")
            # # 显示预测结果
            # for i in range(min(20000000, x.size(0))):
            #     pred_text = aligner.tokenizer.decode([preds[i]])
            #     gt_text = aligner.tokenizer.decode([gt_tokens[i]])
            #     print(f"  Sample {i}: Predicted: '{pred_text}'({preds[i]}), GT: '{gt_text}'({gt_tokens[i]})")
        
        if 'CtrLoss' in save_path:
            all_embeds = torch.cat(all_embeds, dim=0) # [N, D]
            all_labels = torch.cat(all_labels, dim=0) # [N]
            
            # # 计算样本间的余弦相似度矩阵 # 通过看每个样本的“最近邻”是否是同类来计算 Acc
            # sim_matrix = torch.matmul(all_embeds, all_embeds.T)
            # # 排除掉自己和自己的对比
            # sim_matrix.fill_diagonal_(-1)
            # # 找到最相似的那个样本的索引
            # _, closest_idx = torch.max(sim_matrix, dim=1)
            # pred_labels = all_labels[closest_idx]
            # correct = (pred_labels == all_labels).sum().item()

            # 计算信号与 6 个原型的相似度矩阵 [B, 6] # 这里的矩阵乘法等同于计算余弦相似度
            similarities = torch.matmul(all_embeds.float(), prototype_matrix.T.float()) 
            pred_labels = torch.argmax(similarities, dim=1)            
            correct = (pred_labels == labels).sum().item()
            
            final_acc = correct / all_labels.size(0)
        
    final_acc = total_acc / total_samples # 1822
    print(f"\n✅ 验证完成! 总测试样本: {total_samples}, 平均准确率: {final_acc*100:.2f}%")
    return final_acc


## ----- index 2 token 2 LLM 2 text ----- ##
def test_index2text():
    print(">>> 开始执行完整推理测试 (脑电 -> VQ -> Qwen Embedding -> 文本生成)...")
    qwen_model = AutoModelForCausalLM.from_pretrained(
            "../Qwen2.5-7B-Instruct", torch_dtype=torch.float16, 
            device_map=device, trust_remote_code=True)
    vqvae.eval()
    token_model.eval()
    
    # 1. 建立全局映射表 (用于在全词表中找回我们的 6 个手指标签)
    local_to_global_id = {} # {0: 3306, 1: 25036, ...}
    for idx, text in LABEL_MAP.items():
        token_id = aligner.tokenizer.encode(text, add_special_tokens=False)[0]
        local_to_global_id[idx] = token_id
    
    # 反向映射: {3306: 0, 25036: 1, ...} 用于计算准确率和获取标签文本
    global_to_local_idx = {v: k for k, v in local_to_global_id.items()}
    valid_global_ids = list(local_to_global_id.values())

    total_acc = 0
    total_samples = 0
    
    # 获取 Qwen 的 Embedding 层
    embeddings_layer = qwen_model.get_input_embeddings()

    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(test_loader):
            x, labels = x.to(device), labels.to(device)
            
            # --- Step A: 信号特征提取 ---
            _, _, vq_indices = vqvae(x) # [B, 250]
            
            # --- Step B: 预测全局 Token ID ---
            logits = token_model(vq_indices) # [B, 151643]
            
            # 策略：限制搜索范围在 6 个目标 Token 内，提高鲁棒性
            filtered_logits = logits[:, valid_global_ids] # [B, 6]
            best_local_sub_idx = torch.argmax(filtered_logits, dim=1) # 在 0-5 范围
                                    
            # --- Step D: 准确率计算 ---
            # 获取 Ground Truth 的全局 Token IDs
            gt_tokens = aligner.get_gt_batch(labels) # [B]
            # 转回全局 Token ID (例如 3306)
            gt_tokens_full = [GT_token2_FullToken.get(tid.item(), [tid.item()]) for tid in gt_tokens]
            print(f"\n[Batch {batch_idx}] GT全局 Token IDs: {gt_tokens_full}")
            
            pred_global_ids = [valid_global_ids[i.item()] for i in best_local_sub_idx]
            # 计算准确率 (这里使用限制范围后的预测值)
            preds_tensor = torch.tensor(pred_global_ids, device=device)
            acc = (preds_tensor == gt_tokens).sum().item() / x.size(0)
            total_acc += acc * x.size(0)
            total_samples += x.size(0)
                        
            # --- Step C: LLM 向量拼接推理 (仅演示 Batch 中的第一个样本) ---
            if batch_idx % 1 == 0: # 减少打印频率
                # 转回全局 Token ID (例如 3306)
                print(f"\n[Batch {batch_idx}] 预测的全局 Token IDs: {pred_global_ids}")
                target_global_id = pred_global_ids[0]
                # 获取完整序列 (如 [1252, 14317])
                full_token_seq = GT_token2_FullToken.get(target_global_id, [target_global_id])
                print(f"\n[Batch {batch_idx}] 预测的full_全局 Token IDs: {full_token_seq}")
            #     # 转换所有 Token 为 Embedding 向量
            #     token_vecs = []
            #     for tid in full_token_seq:
            #         t_tensor = torch.tensor([tid], device=device)
            #         vec = embeddings_layer(t_tensor).unsqueeze(0) # [1, 1, 4096]
            #         token_vecs.append(vec)
            #     action_embeds = torch.cat(token_vecs, dim=1) # [1, Seq, 4096]
                
                # 构造 Prompt Embedding
                prompt = "The patient is moving their "
                prompt_ids = aligner.tokenizer.encode(prompt, add_special_tokens=False) # , return_tensors="pt").to(device)
            #     prompt_embeds = embeddings_layer(prompt_ids)
            #     # 最终拼接并生成  Prompt + Action -> LLM -> New Tokens
            #     full_embeds = torch.cat([prompt_embeds, action_embeds], dim=1).to(torch.float16)
            #     output = qwen_model.generate(inputs_embeds=full_embeds, max_new_tokens=20,
            #         eos_token_id=aligner.tokenizer.eos_token_id, pad_token_id=aligner.tokenizer.pad_token_id)
            #     response = aligner.tokenizer.decode(output[0], skip_special_tokens=True)
            #     print(f"\n[Batch {batch_idx}] 预测动作: {LABEL_MAP[global_to_local_idx[target_global_id]]}")
            #     print(f"Qwen 回复: {prompt} {response}")
                
                # 3. 直接在 Token 级别进行拼接 (Concatenate)：[Prompt Tokens] + [Predicted Action Tokens]
                if full_token_seq==[3306, 5477, 318, 19425, 9873]:
                    full_token_sequence = full_token_seq
                else:
                    full_token_sequence = prompt_ids + full_token_seq
                # 4. 直接交给 Decoder 转回文本 注意：这里不再输入给 qwen_model 运行，而是直接通过 tokenizer 还原文本
                response = aligner.tokenizer.decode(full_token_sequence, skip_special_tokens=True)
                print(f"\n[Batch {batch_idx}] 拼接后的完整 Token ID 序列: {full_token_sequence}")
                print(f"Decoder 直接解码结果: {response}")
                
                p_text = aligner.tokenizer.decode([target_global_id])
                g_text = aligner.tokenizer.decode([gt_tokens[0].item()])
                print(f"  Sample {0}: Pred: '{p_text}', GT: '{g_text}'")
                        
    final_acc = total_acc / total_samples
    print(f"\n✅ 测试完成! 平均准确率 (Top-1 in 6 labels): {final_acc*100:.2f}%")
    return final_acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32  # 24 #
    EPOCHS = 1500
    LEARNING_RATE = 1e-3 # 4
    codebook = 512  # 128  # 256  #
    VQVAEembed_dim = 64  # 1  # 
    VQToTokenembed_dim = 256  # 64  # 
    subset = 'fingerflex'  # 'motor_basic'  # 'gestures' # 'BCIComp4'  #  'joystick_track2' #
    # 数据集加载
    if subset =='BCIComp4': 
        save_path = './checkpoints/BCI Competition IV/TokenAlign_c%d_emb%d/' % (codebook, VQToTokenembed_dim)  # 
        VQVAEmodelpath = "./checkpoints/BCI Competition IV/VQTokenizer/best_vqvae_model.pth"
        
        datadir = '/mnt/home/user1/MCX/EEGLM/data/ECoG/BCICIV_4_mat/'
        full_dataset = BCIComp4Dataset4_single(datadir=datadir, window_size=1000, 
                stride=256, single_channel=True, channel_idx=None) # 单通道stride=512,
        # full_dataset = BCIComp4Dataset4(datadir=datadir, window_size=1000, stride=256)
    else:
        save_path = "./checkpoints/Stanford/fingerflex_best/" #% (subset, codebook, VQToTokenembed_dim)
        VQVAEmodelpath = "./checkpoints/Stanford/fingerflex_best/best_vqvae_model.pth" # % subset
        datadir = '/disk2/user1/dataset/BCI-Standford/%s/%s/data/' % (subset, subset)
        full_dataset = MillerFingersDataset(datadir, window_size=1000, stride=256,
                        single_channel=True, channel_idx=0, subset=subset)  # 单通道 channel_idx=0,

    os.makedirs(save_path, exist_ok=True)
    tokenmodel_path = f"{save_path}token_aligner.pth"
    
    aligner = TokenAlignmentTrainer(subset=subset)
    
    vqvae = NeuralVQVAE(in_channels=1, codebook_size=codebook, embed_dim=VQVAEembed_dim,
                window_size=1000).to(device)  # 
    vqvae.load_state_dict(torch.load(VQVAEmodelpath)['model_state_dict'])
    
    # 初始化对齐模型
    token_model = TransformerVQToTokenModel(codebook_size=codebook, embed_dim=VQToTokenembed_dim,
            nhead=8, num_layers=4, num_classes=len(LABEL_MAP), vocab_size=len(aligner.tokenizer), 
            max_seq_len=250, save_path=save_path).to(device)
        
    print("...加载 ECoG 数据集...")
    # torch.manual_seed(0)  # 任意整数，如 0, 42, 1234 等
    train_data, test_data = random_split(full_dataset, [int(len(full_dataset)*0.9),
        len(full_dataset)-int(len(full_dataset)*0.9)])
    
    print(f"数据集加载成功: 训练样本数={len(train_data)}, 测试样本数={len(test_data)}")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # train_alignment()
    
    # 执行测试
    print("加载已有的 Token 对齐模型权重...")
    token_model.load_state_dict(torch.load(tokenmodel_path))
    test_index2text()
    test_model_forward()
    
    
    

