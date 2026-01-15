
import numpy as np
from scipy.io import loadmat
import torch
from data import BCIComp4Dataset4_single
from VQTokenizer import NeuralVQVAE
from torch.utils.data import random_split


""" 计算 BCI 数据的压缩性能指标 """
def calculate_compression_metrics(num_channels, sampling_rate, bit_depth, 
                                 window_size, stride, codebook_size):
    # 1. 原始数据码率计算 (bps)
    # Dataset 4: 64 channels, 1000Hz, 16-bit
    original_bps = num_channels * sampling_rate * bit_depth
    
    # 2. Token 码率计算 (bps)
    # 每个窗口(window_size)产生一个 Token
    # 每秒产生的 Token 数量由步长 (stride) 决定
    tokens_per_second = sampling_rate / stride
    
    # 每个 Token 索引需要的比特数 (例如 512 = 2^9, 需要 9 bits)
    bits_per_token = np.ceil(np.log2(codebook_size))
    
    token_bps = tokens_per_second * bits_per_token
    
    # 3. 压缩比
    compression_ratio = original_bps / token_bps
    
    # 4. 节省带宽百分比
    space_saving = (1 - (token_bps / original_bps)) * 100

    print("--- BCI 压缩性能分析报告 ---")
    print(f"原始信号比特率: {original_bps/1000:.2f} kbps")
    print(f"Token 序列比特率: {token_bps/1000:.4f} kbps")
    print(f"理论压缩比: {compression_ratio:.2f} : 1")
    print(f"带宽节省率: {space_saving:.4f}%")
    print("-" * 30)
    
    return compression_ratio

  

def calculate_compression_metrics_final(vqvae_model, dataset, sampling_rate, bit_depth,\
    window_size=256, sample_idx=0):
    """
    计算 BCI Competition IV Dataset 4 的原始比特率与 Token 比特率
    """
    vqvae_model.eval() # 必须进入 eval 模式以关闭死索引重置逻辑
    device = next(vqvae_model.parameters()).device
    
    # 1. 原始比特率参数 (Raw Bitrate)
    # Dataset 4 原始为 64 通道，1000Hz 采样，通常以 float32 存储，但传感器精度多为 16bit
    # fs = 1000             # 采样率
    # 1. 解析数据
    # dataset[i] 返回 (ecog_tensor, label)
    sample_data = dataset[sample_idx]
    ecog_tensor = sample_data[0]  # 提取特征 Tensor
    # 自动计算比特深度 (Bit Depth)
    # torch.float32 -> 32, torch.float64 -> 64, torch.int16 -> 16, torch.uint8 -> 8
    bit_depth = ecog_tensor.element_size() * 8  # bits_per_val = 16     # 采样精度 (bits)
    n_channels = ecog_tensor.shape[0]  # 动态获取通道数：如果是单通道 [1, 1000] 则 n_channels=1
    print(f"DEBUG: Feature shape = {ecog_tensor.shape}") 
    print(f"DEBUG: Detected Bit Depth = {bit_depth} bits")
    # label = sample_data[1]
    
    # 2. 原始比特率参数 (Raw Bitrate)
    raw_bps = sampling_rate * n_channels * bit_depth
    
    # 2. 提取神经 Token
    ecog_tensor, _ = dataset[sample_idx]
    ecog_tensor = ecog_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 编码并获取索引
        z = vqvae_model.encoder(ecog_tensor)
        # 注意：这里直接调用 vq_layer，因为它在 eval 模式下不会报错
        _, _, indices = vqvae_model.vq_layer(z)
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
    
    print("\n" + "="*40)
    print("      神经词汇压缩报告 (BCICIV 4)")
    print("="*40)
    print(f"原始信号带宽 (Raw):    {raw_bps/1000:10.2f} kbps")
    print(f"神经 Token 带宽 (LLM): {token_bps/1000:10.4f} kbps")
    print(f"带宽缩减倍数:         {compression_ratio:10.2f} X")
    print(f"下采样倍率 (Stride):   {window_size/num_tokens:10.1f} : 1")
    print("-" * 40)
    print(f"Big Claim: 仅需原始带宽的 {1/compression_ratio:.4%}, 即可传输完整运动意图。")
    print("="*40)
    
    return {
        "raw_bps": raw_bps,
        "token_bps": token_bps,
        "ratio": compression_ratio
    }


# from QwenNeuralCorrector2 import indices_to_string_tokens
def indices_to_string_tokens(indices, loss_rate=0.5):
    """
    将 VQ 索引转换为字符串级 Token，并模拟丢包
    indices: [Seq_Len]
    """
    token_str_list = []
    for i, idx in enumerate(indices):
        # 模拟丢包
        if torch.rand(1).item() < loss_rate:
            token_str_list.append("[SIGNAL_LOSS]")
        else:
            # 转换为字符串级 Token: [NEURAL_ID_042]
            token_str_list.append(f"[NEURAL_ID_{int(idx):03d}]")
    
    return " ".join(token_str_list)


def calculate_string_token_compression(vqvae_model, dataset, sampling_rate=1000, 
                                       loss_rate=0.5, sample_idx=0):
    """
    计算基于文本字符串传输的压缩比
    衡量从原始浮点信号到最终喂给 LLM 的字符串 token_str 的带宽变化
    """
    vqvae_model.eval()
    device = next(vqvae_model.parameters()).device
    
    # 1. 原始数据分析
    sample_data = dataset[sample_idx]
    ecog_tensor = sample_data[0] # [Channels, Time]
    
    # 计算原始比特率 (基于 Tensor 数据类型的物理位宽)
    bit_depth_raw = ecog_tensor.element_size() * 8 
    n_channels = ecog_tensor.shape[0]
    raw_bps = sampling_rate * n_channels * bit_depth_raw

    # 2. 生成字符串 Token
    ecog_input = ecog_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        z = vqvae_model.encoder(ecog_input)
        _, _, indices = vqvae_model.vq_layer(z)
        indices = indices.flatten()
    
    # 使用你定义的函数生成最终传输的字符串
    # 例如: "[NEURAL_ID_042] [SIGNAL_LOSS] [NEURAL_ID_105]"
    token_str = indices_to_string_tokens(indices, loss_rate=loss_rate)
    
    # 3. 计算字符串的传输成本 (String Transmission Bitrate)
    # 计算原则：在网络传输中，字符串通常以 utf-8 编码传输
    token_str_bytes = len(token_str.encode('utf-8'))
    token_str_bits = token_str_bytes * 8
    
    # 计算这段字符串对应的持续时间
    window_samples = ecog_tensor.shape[-1]
    window_sec = window_samples / sampling_rate
    
    # 得到字符串级带宽 (bps)
    string_bps = token_str_bits / window_sec
    
    # 4. 计算压缩比
    compression_ratio = raw_bps / string_bps
    
    print("\n" + "="*50)
    print("      字符串级传输报告 (String-based Transmission)")
    print("="*50)
    print(f"原始信号带宽 (Raw):      {raw_bps/1000:10.2f} kbps")
    print(f"字符串 Token 带宽 (Str):  {string_bps/1000:10.2f} kbps")
    print(f"字符串级压缩比 (Ratio):   {compression_ratio:10.2f} X")
    print("-" * 50)
    print(f"传输内容示例: {token_str[:60]}...")
    print(f"单次发送字节数: {token_str_bytes} Bytes")
    print("="*50)
    
    return compression_ratio


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datadir = '/mnt/home/user1/MCX/EEGLM/data/ECoG/BCICIV_4_mat/'
    
    # 1. 加载数据集
    full_dataset = BCIComp4Dataset4_single(datadir=datadir, window_size=1000, stride=512, single_channel=True)
    train_data, test_data = random_split(full_dataset, [int(len(full_dataset)*0.9), len(full_dataset)-int(len(full_dataset)*0.9)])
    
    # 2. 加载第一周 VQ-VAE
    vq_model = NeuralVQVAE(in_channels=1, codebook_size=512, embed_dim=64, window_size=1000).to(device)
    ratio = calculate_compression_metrics_final(vq_model, test_data, 
        sampling_rate=1000, bit_depth=16, window_size=1000, sample_idx=10)
    # ratio = calculate_string_token_compression(vq_model, test_data, sampling_rate=1000, 
    #                                    loss_rate=0.5, sample_idx=0)
'''
DEBUG: Feature shape = torch.Size([1, 1000])
DEBUG: Detected Bit Depth = 32 bits
DEBUG: VQ Indices shape = torch.Size([1, 250])
DEBUG: Number of Tokens = 250

========================================
      神经词汇压缩报告 (BCICIV 4)
========================================
原始信号带宽 (Raw):         32.00 kbps
神经 Token 带宽 (LLM):     1.5000 kbps
带宽缩减倍数:              21.33 X
下采样倍率 (Stride):          4.0 : 1
----------------------------------------
Big Claim: 仅需原始带宽的 4.6875%, 即可传输完整运动意图。
========================================

==================================================
      字符串级传输报告 (String-based Transmission)
==================================================
原始信号带宽 (Raw):           32.00 kbps
字符串 Token 带宽 (Str):       29.94 kbps
字符串级压缩比 (Ratio):         1.07 X
--------------------------------------------------
传输内容示例: [SIGNAL_LOSS] [NEURAL_ID_482] [NEURAL_ID_007] [NEURAL_ID_007...
单次发送字节数: 3743 Bytes
==================================================
'''


# mat = loadmat('/mnt/home/user1/MCX/EEGLM/data/ECoG/BCICIV_4_mat/sub1_comp.mat')
# X = data = mat['train_data']  # [Samples, Channels]
# cue = mat['train_dg']  # [Samples]
# # 将数据 reshape 为连续时间序列（可选，但 nbytes 不受影响）
# # 如果确实是 int32，可保留（或转为 int16 更合理）
# if X.dtype == np.int32:
#     # 可选：检查数值范围是否在 int16 范围内 [-32768, 32767]
#     if X.min() >= -32768 and X.max() <= 32767:
#         original_data = X.astype(np.int16)  # 更贴近真实采样
#     else:
#         original_data = X.astype(np.int32)  # 确实需要 int32
# else:
#     original_data = X
# # original_data = X  # shape: (T, C)

# # 数值类型'bit_depth'  float64 占 8 字节/元素，float32 占 4 字节, int32 占 4 字节/元素 
# original_size_bytes = original_data.nbytes

# print(f"Data shape: {original_data.shape}")
# print(f"Data dtype: {original_data.dtype}")  # int32
# print(f"Original size: {original_size_bytes} bytes ({original_size_bytes / (1024**2):.2f} MB)")

  
# # 针对你的项目参数：
# params = {
#     'num_channels': 1,   # original_data.shape[1] # 对齐后的通道数64
#     'sampling_rate': 1000,    # 1000Hz
#     'bit_depth': 32,          # 假设 16位 采样
#     'window_size': 256,       # 你的窗口大小
#     'stride': 256,            # 假设非重叠传输，每个窗口传一次
#     'codebook_size': 1024      # 词典大小
# }

# cr = calculate_compression_metrics(**params)

'''
Data shape: (400000, 62)
Data dtype: int32
Original size: 99200000 bytes (94.60 MB)
--- BCI 压缩性能分析报告 ---
原始信号比特率: 1984.00 kbps
Token 序列比特率: 0.0391 kbps
理论压缩比: 50790.40 : 1
带宽节省率: 99.9980%
'''