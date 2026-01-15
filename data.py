from pathlib import Path
import os
import glob
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, Sampler, DataLoader #, random_split, 
import random
# ==========================================
# 3. 增强版数据集处理 (支持多文件加载)
# ==========================================
class MillerFingersDataset(Dataset):
    def __init__(self, datadir, window_size=256, stride=128, single_wind=False, \
        single_channel=False, channel_idx=None, subset='fingerflex'):
        """
        Args:
            datadir (str): 包含多个 .mat 文件的目录路径
            window_size (int): 滑动窗口大小
            stride (int): 滑动步长
        """
        self.single_channel = single_channel
        self.channel_idx = channel_idx  # 可选：固定通道索引
        self.samples = []
        self.labels = []
        self.window_size = window_size
        self.stride = stride
        self.subset = subset

        # 1. 获取目录下所有 .mat 文件路径
        if subset=='fingerflex': 
            mat_files = list(Path(datadir).rglob("*_fingerflex.mat"))
        if subset== 'gestures':
            mat_files = list(Path(datadir).rglob("*_fingerflex.mat"))
            # mat_files = list(Path(datadir).rglob("*.mat"))
        if subset == 'motor_basic':
            mat_files = glob.glob(os.path.join(datadir, "*.mat"))
        if subset == 'joystick_track':
            mat_files = glob.glob(os.path.join(datadir, "*.mat"))
        # mat_files = mat_files[:1]
        print(f"正在从 {datadir} 加载 {len(mat_files)} 个数据文件...")#  18

        # 2. 遍历并处理每个文件
        for file_path in mat_files:
            mat = loadmat(file_path)
            # print(mat.keys()) # 注意：某些版本的 Miller 数据集 key 可能不同，若报错请检查 key 名
            if 'data' not in mat or \
                    ('cue' not in mat and 'stim' not in mat and 'TargetPosX' not in mat):
                    print(f"跳过文件 {file_path}: 缺少 'data' 或 'cue' 键")
                    continue
            data = mat['data']  # [Samples, Channels] 
            if subset=='fingerflex':
                cue = mat['cue'].flatten() # [Samples] # fingerflex  # (610040/444840, 46/64/38) (610040/444840,)
            if subset== 'gestures':
                cue = mat['stim'].flatten() # [Samples] # gestures  # (130040/730080, 84/64) (130040/730080,)
            if subset == 'motor_basic':
                cue = mat['stim'].flatten() # motor_basic # (376240/571720, 64/48) (376240/571720,) 总样本数: 29773
            if subset == 'joystick_track':
                cue = mat['TargetPosX'].flatten() # [Samples] # joystick_track # (372760/134360, 60/64) (372760/134360,)总样本数: 3954
            print('data.shape, cue.shape = ', data.shape, cue.shape) 
            # for m in range(len(cue)):
            #     print(cue[m])
            
            # 3. 预处理 (按文件独立进行标准化，避免不同 session 间的幅值干扰)
            # CAR (共模去噪)
            data = data - np.mean(data, axis=1, keepdims=True)
            # Z-Score 标准化
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6)
            
            # 4. 滑动窗口切片  训练集 16393, 测试集 1822
            count = 0
            if not single_wind:
                for i in range(0, len(data) - self.window_size, self.stride):
                    # 提取窗口数据 [C, T]
                    window = data[i:i+self.window_size, :].T #.astype(np.float32) # 节省内存
                    # 标签提取逻辑：取窗口内出现频次最高的 cue
                    # np.bincount 不接受负数，确保 cue >= 0
                    window_cues = cue[i:i+self.window_size].astype(int)
                    if len(window_cues) > 0:
                        label = np.bincount(window_cues).argmax() # 统计非负整数数组的频次,np.argmax()返回数组中最大值的索引
                        self.samples.append(window)
                        self.labels.append(label)
                        count += 1
            else:
                # --- 修改后的4. 允许重叠的纯净区域提取逻辑 ---训练集 8803, 测试集 979
                n_samples = len(cue)
                i = 0
                while i < n_samples:
                    start_idx = i
                    current_label = cue[i]
                    # 1. 探测标签连续相同的区间边界
                    while i < n_samples and cue[i] == current_label:
                        i += 1
                    end_idx = i
                    segment_len = end_idx - start_idx
                    # 2. 如果连续区间长度足以容纳至少一个窗口
                    if segment_len >= self.window_size :
                        # 3. 在该纯净区域内进行“带重叠”的滑动窗口切片
                        # 使用 self.stride 控制重叠程度
                        for j in range(start_idx, end_idx - self.window_size + 1, self.stride):
                            # 提取窗口数据 [C, T]
                            window = data[j:j+self.window_size, :].T
                            # 确保标签纯净（虽然逻辑上已经是了，这里做强制转换确保类型正确）
                            self.samples.append(window.astype(np.float32))
                            self.labels.append(int(current_label))
                            count += 1
            print(f"  - 已处理: {os.path.basename(file_path)} | 提取样本数: {count}")

        
        self.num_classes = len(set(self.labels))
        # print(f"共有 {self.num_classes} 种不同的标签。")
        # unique_labels = sorted(set(self.labels))
        # print("所有出现的标签种类：", unique_labels)
        # exit()
        # # 'joystick_track' : 1166种不同的标签
        if self.subset == 'joystick_track':
            unique_labels = sorted(set(self.labels))
            label_map = {old_label: new_id for new_id, old_label in enumerate(unique_labels)}
            self.labels = [label_map[label] for label in self.labels]
        
        # # 将样本和标签配对后打乱
        # combined = list(zip(self.samples, self.labels))
        # random.shuffle(combined)
        # self.samples, self.labels = zip(*combined)
        # self.samples = list(self.samples)
        # self.labels = list(self.labels)

        print(f"数据集构建完成。总样本数: {len(self.labels)}")  # 36472

    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]      # [64, window_size]
        label = self.labels[idx]
        if self.subset == 'motor_basic':
            if label == 11:
                label = 1
            elif label == 12:
                label = 2
            elif label == 13:
                label = 3
            elif label == 15:
                label = 4
            
        if self.single_channel:
            if self.channel_idx is not None:
                # 固定通道
                ch = self.channel_idx
            else:
                # 随机选一个通道（每次调用可能不同）
                ch = np.random.randint(0, sample.shape[0])
            sample = sample[ch:ch+1, :]  # 保持维度 [1, window_size]
        
        # 保持返回格式一致，y 为 LongTensor 以匹配交叉熵损失函数
        # print('sample.shape, label = ', sample.shape, label)  # (ch, windowsize) 0
        
        return torch.FloatTensor(sample), torch.tensor(label, dtype=torch.long)


class BCIComp4Dataset4(Dataset):
    def __init__(self, datadir, window_size=256, stride=128):
        '''Dataset4采样率1000Hz。window_size=256：每个Token代表256ms神经活动，
        对于捕捉手指运动的动态过程非常理想。'''
        self.samples = []
        self.labels = []
        self.window_size = window_size
        self.stride = stride
        # 匹配该数据集常见的训练文件名模式
        mat_files = glob.glob(os.path.join(datadir, "*.mat"))
        print(f"正在加载 BCI Competition IV Dataset 4，共 {len(mat_files)} 个受试者数据...")
        for file_path in mat_files:
            mat = loadmat(file_path)            
            # Dataset 4 的典型 Key: 'train_data' (ECoG), 'train_dg' (Finger Flexion)
            if 'train_data' not in mat:
                print(f"跳过 {file_path}: 格式不匹配")
                continue
                
            data = mat['train_data']  # [Samples, 62 Channels]
            # train_dg 是 5 个手指的连续弯曲值 [Samples, 5]
            dg = mat['train_dg'] 
            # print('data.shape, dg.shape = ', data.shape, dg.shape) (400000, 48) (400000, 5)
            
            if data.shape[1] < 64:
                pad = np.zeros((data.shape[0], 64 - data.shape[1]))
                data = np.hstack((data, pad))
                
            # --- 1. 数据预处理 ---
            # CAR (共模去噪)
            data = data - np.mean(data, axis=1, keepdims=True)
            # 标准化
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6)
            # print('data.shape = ', data.shape)  # (400000, 64)
            
            # --- 2. 连续弯曲值转离散标签 ---
            # 策略：找到当前时刻弯曲程度最大的手指作为 label
            # 如果 5 个手指都没有明显弯曲（低于阈值），设为 0 (Resting)
            # 1. 计算每个样本点 5 个手指中的最大弯曲度
            dg_max = np.max(dg, axis=1)
            '''BCI Comp IV Dataset 4 的原始任务是预测连续的手指弯曲度。
            由于要构建 VQ-Tokenizer（语义字母表），必须将其转换为离散状态。提取当前最显著的动作原语。'''
            # print('np.max(dg_max) = ', np.max(dg_max)) # 4.8177630
            # print('dg_max ', dg_max) # [-0.41384039 -0.41384039 -0.41384039 ...  0.22336148  0.22336148  0.22336148]
            # 2. 动态计算阈值：70% 的时间是静息或微动，30% 是显著动作，值越高判定动作越严格
            threshold = np.quantile(dg_max, 0.7)
            dg_labels = np.argmax(dg, axis=1) + 1 # 1-5 代表手指
            dg_labels[dg_max < threshold] = 0    # 0 代表静息
            # print('dg_labels ', dg_labels) # dg_labels  [0 0 0 ... 0 0 0]
            # nonzero_count = np.count_nonzero(dg_labels)
            # print(f"非零标签个数: {nonzero_count}")
            # print(f"动作占比: {nonzero_count / len(dg_labels) * 100:.2f}%") # 30%
            
            # --- 3. 滑动窗口切片 ---
            count = 0
            for i in range(0, len(data) - self.window_size, self.stride):
                # ECoG 窗口 [Channels, Window]
                window = data[i:i+self.window_size, :].T
                
                # 取窗口内最频繁出现的动作作为标签
                window_labels = dg_labels[i:i+self.window_size]
                label = np.bincount(window_labels.astype(int)).argmax()
                
                self.samples.append(window)
                self.labels.append(label)
                # print('label = ', label)
                count += 1
                
            print(f"  - 受试者文件 {os.path.basename(file_path)}: 提取 {count} 个样本")
            '''- 受试者文件 sub3_comp.mat: 提取 3123 个样本 '''
        print(f"✅ 全量数据集构建完成。总样本数: {len(self.labels)}")  # 9369

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        # 样本形状: torch.Size([64, 1000]), 标签: 0
        return torch.FloatTensor(self.samples[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

# 单通道
class BCIComp4Dataset4_single(Dataset):
    def __init__(self, datadir, window_size=256, stride=128, \
        single_channel=True, channel_idx=None):
        '''
        新增参数:
        - single_channel: 是否启用单通道模式（默认 True）
        - channel_idx: 若指定（如 0），则固定使用该通道；若 None，则每次随机选
        '''
        self.samples = []
        self.labels = []
        self.window_size = window_size
        self.stride = stride
        self.single_channel = single_channel
        self.channel_idx = channel_idx  # 可选：固定通道索引
        
        mat_files = glob.glob(os.path.join(datadir, "*.mat"))
        print(f"正在加载 BCI Competition IV Dataset 4，共 {len(mat_files)} 个受试者数据...")
        for file_path in mat_files:
            mat = loadmat(file_path)            
            if 'train_data' not in mat:
                print(f"跳过 {file_path}: 格式不匹配")
                continue
                
            data = mat['train_data']  # [Samples, Channels]
            dg = mat['train_dg']      # [Samples, 5]
            # # 补零到 64 通道
            # if data.shape[1] < 64:
            #     pad = np.zeros((data.shape[0], 64 - data.shape[1]))
            #     data = np.hstack((data, pad))
                
            # 预处理
            data = data - np.mean(data, axis=1, keepdims=True)  # CAR
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6)
            
            # 转离散标签
            dg_max = np.max(dg, axis=1)
            threshold = np.quantile(dg_max, 0.7)
            dg_labels = np.argmax(dg, axis=1) + 1
            dg_labels[dg_max < threshold] = 0
            
            # 滑动窗口
            count = 0
            for i in range(0, len(data) - self.window_size, self.stride):
                window = data[i:i+self.window_size, :].T  # [64, window_size]
                window_labels = dg_labels[i:i+self.window_size]
                label = np.bincount(window_labels.astype(int)).argmax()
                # print('label = ', label)
                self.samples.append(window)   # 仍存多通道
                self.labels.append(label)
                count += 1
                
            print(f"  - 受试者文件 {os.path.basename(file_path)}: 提取 {count} 个样本")
        print(f"✅ 全量数据集构建完成。总样本数: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]      # [64, window_size]
        label = self.labels[idx]
        
        if self.single_channel:
            if self.channel_idx is not None:
                # 固定通道
                ch = self.channel_idx
            else:
                # 随机选一个通道（每次调用可能不同）
                ch = np.random.randint(0, sample.shape[0])
            sample = sample[ch:ch+1, :]  # 保持维度 [1, window_size]
        # print('sample.shape, label.shape = ', sample.shape, label.shape)
        # print('label = ', label)
        return torch.FloatTensor(sample), torch.tensor(label, dtype=torch.long)


# 无标签返回，返回一个 dummy label 0
class BCIComp4TestDataset(Dataset):
    """ 专门用于加载没有标签的竞赛原始测试集 (*_comp.mat) """
    def __init__(self, datadir, window_size=256, stride=128):
        self.samples = []
        self.window_size = window_size
        self.stride = stride
        
        # 匹配竞赛测试集文件名
        mat_files = glob.glob(os.path.join(datadir, "*_comp.mat"))
        
        for file_path in mat_files:
            mat = loadmat(file_path)
            # 测试集的 key 通常是 'test_data'
            if 'test_data' not in mat:
                continue
                
            data = mat['test_data']
            print('test_data.shape = ', data.shape)  # 
            
            # --- 预处理 (必须与训练集一致) ---
            data = data - np.mean(data, axis=1, keepdims=True)
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6)
            # --- 切片 ---
            for i in range(0, len(data) - self.window_size, self.stride):
                window = data[i:i+self.window_size, :].T
                self.samples.append(window)
        
        print(f"✅ 纯测试集加载完成。总窗口数: {len(self.samples)}") # 4683

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        # 样本形状: torch.Size([64, 256]), 标签: 0 
        return torch.FloatTensor(self.samples[idx]), torch.tensor(0) 


######
from collections import defaultdict
        
class ClassBalancedBatchSampler(Sampler):
    def __init__(self, dataset, labels, batch_size, num_classes=6, samples_per_class=2):
        """
        Args:
            dataset: your MillerFingersDataset instance
            labels: 对应 dataset 顺序的标签列表 (重要)
            batch_size: total batch size (should equal num_classes * samples_per_class)
            num_classes: number of classes (e.g., 5 for fingers)
            samples_per_class: number of samples per class in each batch
        """
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        
        assert batch_size == num_classes * samples_per_class, \
            f"batch_size ({batch_size}) must equal num_classes * samples_per_class ({num_classes} * {samples_per_class})"
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.class_indices[label.item()].append(idx)
        
        # Ensure all classes exist
        assert len(self.class_indices) == num_classes, \
            f"Expected {num_classes} classes, got {len(self.class_indices)}"
        
        
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        
        assert batch_size == num_classes * samples_per_class
        
        # --- 修改点：直接使用传入的 labels 列表 ---
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            # 确保 label 是整数
            l = label.item() if torch.is_tensor(label) else int(label)
            self.class_indices[l].append(idx)
            
        # Shuffle each class list
        for cls in self.class_indices:
            np.random.shuffle(self.class_indices[cls])
        
        self.class_iters = {cls: 0 for cls in self.class_indices}
        self.epoch = 0

    def __iter__(self):
        # Reset iterators every epoch
        for cls in self.class_indices:
            if self.class_iters[cls] >= len(self.class_indices[cls]):
                # Reshuffle and reset if exhausted
                np.random.shuffle(self.class_indices[cls])
                self.class_iters[cls] = 0
        
        batch = []
        while len(batch) < len(self):  # total number of batches * batch_size
            for cls in range(self.num_classes):
                idx_list = self.class_indices[cls]
                pos = self.class_iters[cls]
                batch.append(idx_list[pos % len(idx_list)])
                self.class_iters[cls] += 1
            if len(batch) % self.batch_size == 0:
                yield batch[-self.batch_size:]  # yield one balanced batch

    def __len__(self):
        # Total number of batches per epoch
        min_samples = min(len(indices) for indices in self.class_indices.values())
        total_batches = (min_samples // self.samples_per_class)
        return total_batches * self.batch_size
    

def prepare_dataloaders(full_dataset, batch_size, num_classes=6, ratio=0.9):
    # 1. 严格按时间顺序切分 (前 90% 训练，后 10% 测试)
    total_len = len(full_dataset)
    train_size = int(total_len * ratio)
    test_size = total_len - train_size
    
    # 获取索引
    indices = list(range(total_len))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # 构建 Subset
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    test_subset = torch.utils.data.Subset(full_dataset, test_indices)
    
    print(f"时序切分完成: 训练集 {len(train_subset)}, 测试集 {len(test_subset)}")

    # 2. 训练集使用类别平衡采样 (解决手指运动频率不均问题) # 注意：需要提取子集的标签
    # 确保 sampler 拿到的标签是对应训练集索引的
    train_labels = [full_dataset.labels[i] for i in train_indices]
    # 如果你的 Sampler 构造函数需要 labels 列表，请显式传入
    sampler = ClassBalancedBatchSampler(
        train_subset, 
        labels=train_labels, # 确保传入的是子集的标签，而不是全集的
        batch_size=batch_size, 
        num_classes=num_classes, 
        samples_per_class=batch_size // num_classes)

    train_loader = DataLoader(train_subset, batch_sampler=sampler)
    
    # 3. 测试集保持时序顺序，不打乱，方便观察混淆矩阵
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
    
    return train_loader, test_loader, test_subset


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# set_seed(42)

# train_size = int(len(full_dataset) * 0.9)
# test_size = len(full_dataset) - train_size
# train_data, test_data = random_split(full_dataset, [train_size, test_size])
# if __name__ == "__main__":
    # test_dataset = BCIComp4TestDataset('/mnt/home/user1/MCX/EEGLM/data/ECoG/BCICIV_4_mat/')
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # # 4683
        
    # print(f"test_数据集样本数: {len(test_dataset)}") # 4683
    # sample, label = test_dataset[0]
    # print(f"样本形状: {sample.shape}, 标签: {label}")


    # dataset = BCIComp4Dataset4(datadir='/mnt/home/user1/MCX/EEGLM/data/ECoG/BCICIV_4_mat/', 
    #                             window_size=256, stride=128)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}") 
    # # 7495, 1874

    # dataset = MillerFingersDataset(datadir = '/disk2/user1/dataset/BCI-Standford/fingerflex/fingerflex/data/', 
    #             window_size=256, stride=128, single_channel=True, channel_idx=0)
    

    # print(f"数据集样本数: {len(dataset)}")
    # sample, label = dataset[0]
    # print(f"样本形状: {sample.shape}, 标签: {label}")



