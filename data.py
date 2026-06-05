from pathlib import Path
import os
import glob
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, Sampler, DataLoader #, random_split, 
import random
from collections import Counter
import h5py
import scipy.io
import re
import mne
from mne.io import read_raw_edf
from mne import pick_types

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
            
            if (not self.single_channel) and data.shape[1] < 64:
                pad = np.zeros((data.shape[0], 64 - data.shape[1]))
                data = np.hstack((data, pad))
                
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

        
        # self.num_classes = len(set(self.labels))
        # print(f"共有 {self.num_classes} 种不同的标签。")
        # unique_labels = sorted(set(self.labels))
        # print("所有出现的标签种类：", unique_labels)
        # exit()

        if self.subset == 'motor_basic':
            self.labels_continue = []
            for n in self.labels:
                if n == 11:
                    self.labels_continue.append(1)
                elif n == 12:
                    self.labels_continue.append(2)
                elif n == 13:
                    self.labels_continue.append(3)
                elif n == 15:
                    self.labels_continue.append(4)
                else:
                    self.labels_continue.append(n)
            self.labels = self.labels_continue

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


# 每个mat的前90%用于训练，后10%测试
class MillerFingersDataset_TT(Dataset):
    def __init__(self, datadir, window_size=256, stride=128, single_wind=False, 
                 single_channel=False, channel_idx=None, subset='fingerflex', split_ratio=0.9):
        
        self.single_channel = single_channel
        self.channel_idx = channel_idx
        self.window_size = window_size
        self.stride = stride
        self.subset = subset
        
        # 定义四个核心存储列表
        self.samples_train, self.labels_train = [], []
        self.samples_test, self.labels_test = [], []
        
        # 默认指向训练集，方便 __getitem__ 调用
        self.samples, self.labels = self.samples_train, self.labels_train

        # 1. 获取文件列表
        if subset in ['fingerflex']:  # , 'gestures'
            mat_files = list(Path(datadir).rglob("*_fingerflex.mat"))
            maxchannel = 64 # 
        elif subset == 'gestures':
            mat_files = list(Path(datadir).rglob("*_fingerflex.mat"))
            maxchannel = 84 
        elif subset == 'motor_basic':
            mat_files = glob.glob(os.path.join(datadir, "*.mat"))
            maxchannel = 64
        else:
            mat_files = glob.glob(os.path.join(datadir, "*.mat"))
            maxchannel = 64
        # print('mat_files = ', mat_files)

        for file_path in mat_files:
            mat = loadmat(file_path)
            if 'data' not in mat: 
                print('data not in ', file_path)
                continue
            
            # 读取原始数据
            data_full = mat['data']
            if subset == 'fingerflex':
                cue_full = mat['cue'].flatten()
            else:
                cue_full = mat['stim'].flatten() if 'stim' in mat else mat['TargetPosX'].flatten()

            print('data.shape, cue.shape = ', data_full.shape, cue_full.shape) 
            # 2. 预处理 (CAR & Z-Score) - 在划分前对整个文件进行标准化
            if (not self.single_channel) and data_full.shape[1] < maxchannel:
                pad = np.zeros((data_full.shape[0], maxchannel - data_full.shape[1]))
                data_full = np.hstack((data_full, pad))
            
            data_full = data_full - np.mean(data_full, axis=1, keepdims=True)
            data_full = (data_full - np.mean(data_full, axis=0)) / (np.std(data_full, axis=0) + 1e-6)

            # 3. 核心：按文件时长划分训练与测试索引
            split_idx = int(len(data_full) * split_ratio)

            # --- 处理函数：内部切片逻辑 ---
            def extract_segments(data_part, cue_part, is_train=True):
                samples_out, labels_out = [], []
                count = 0
                
                if not single_wind:
                    # 普通滑动窗口
                    for i in range(0, len(data_part) - self.window_size, self.stride):
                        window = data_part[i:i+self.window_size, :].T
                        label = np.bincount(cue_part[i:i+self.window_size].astype(int)).argmax()
                        samples_out.append(window.astype(np.float32))
                        labels_out.append(label)
                        count += 1
                else:
                    # 纯净区域提取逻辑
                    n_samples = len(cue_part)
                    k = 0
                    while k < n_samples:
                        start_k = k
                        curr_label = cue_part[k]
                        while k < n_samples and cue_part[k] == curr_label:
                            k += 1
                        seg_len = k - start_k
                        if seg_len >= self.window_size:
                            # 训练集用 stride，测试集用 window_size (不重叠) 保证公平
                            curr_stride = self.stride if is_train else self.window_size
                            for j in range(start_k, k - self.window_size + 1, curr_stride):
                                window = data_part[j:j+self.window_size, :].T
                                samples_out.append(window.astype(np.float32))
                                labels_out.append(int(curr_label))
                                count += 1
                return samples_out, labels_out, count

            # 4. 执行划分
            # 训练部分 (前 90%)
            s_tr, l_tr, c_tr = extract_segments(data_full[:split_idx], cue_full[:split_idx], is_train=True)
            self.samples_train.extend(s_tr)
            self.labels_train.extend(l_tr)

            # 测试部分 (后 10%)
            s_te, l_te, c_te = extract_segments(data_full[split_idx:], cue_full[split_idx:], is_train=False)
            self.samples_test.extend(s_te)
            self.labels_test.extend(l_te)

            print(f"文件 {os.path.basename(file_path)}: 训练样本 {c_tr}, 测试样本 {c_te}")

        # 5. 统一标签映射 (Motor_basic 等逻辑)
        self.labels_train = self._map_labels(self.labels_train)
        self.labels_test = self._map_labels(self.labels_test)
        self.set_mode(mode='train')
        print(f"数据集加载完成。总训练集: {len(self.labels_train)}, 总测试集: {len(self.labels_test)}")

        # # 6. 类别分布统计
        # def get_dist(labels):
        #     count_dict = Counter(labels)
        #     total = len(labels)
        #     return {k: (count_dict[k], count_dict[k]/total*100) for k in sorted(count_dict.keys())}

        # train_dist = get_dist(self.labels_train)
        # test_dist = get_dist(self.labels_test)

        # print(f"\n✅ 数据集加载完成统计:")
        # print(f"{'':<10} | {'训练集 (Train)':<25} | {'测试集 (Test)':<25}")
        # print(f"{'类别':<10} | {'数量 (占比)':<25} | {'数量 (占比)':<25}")
        # print("-" * 70)
        
        # all_classes = sorted(list(set(self.labels_train) | set(self.labels_test)))
        # for c in all_classes:
        #     tr_num, tr_per = train_dist.get(c, (0, 0))
        #     te_num, te_per = test_dist.get(c, (0, 0))
        #     print(f"{c:<12} | {tr_num:<6} ({tr_per:>5.2f}%) {'':<10} | {te_num:<6} ({te_per:>5.2f}%)")
            
        # print(f"{'总计':<10} | {len(self.labels_train):<25} | {len(self.labels_test):<25}")
        # exit()

    def _map_labels(self, labels):
        if self.subset == 'motor_basic':
            mapping = {11: 1, 12: 2, 13: 3, 15: 4}
            return [mapping.get(n, n) for n in labels]
        return labels

    def set_mode(self, mode='train'):
        """手动切换 Dataset 当前指向的数据"""
        if mode == 'train':
            self.samples, self.labels = self.samples_train, self.labels_train
        else:
            self.samples, self.labels = self.samples_test, self.labels_test

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        if self.single_channel:
            ch = self.channel_idx if self.channel_idx is not None else np.random.randint(0, sample.shape[0])
            sample = sample[ch:ch+1, :]
        return torch.FloatTensor(sample), torch.tensor(label, dtype=torch.long)


class MillerFingersDataset_TT_generalization(Dataset):
    def __init__(self, datadir, window_size=1000, stride=256, single_wind=False,
                 single_channel=False, channel_idx=None, subset='fingerflex', 
                 leave_one_out_subject_idx=9, train_ratio_in_subject=0.0):
        """
        Args:
            datadir: 数据目录路径
            window_size: 滑动窗口大小
            stride: 步长
            single_wind: 是否按事件边界切分
            single_channel: 是否只取单通道
            channel_idx: 指定通道索引
            subset: 数据集子集名称
            leave_one_out_subject_idx: 留出的被试索引 (默认第9个，即索引8)
            train_ratio_in_subject: 第9个被试用于训练的比例 (0.0 ~ 1.0)
        """
        self.single_wind = single_wind
        self.single_channel = single_channel
        self.channel_idx = channel_idx
        self.window_size = window_size
        self.stride = stride
        self.subset = subset
        self.leave_one_out_subject_idx = leave_one_out_subject_idx
        self.train_ratio_in_subject = train_ratio_in_subject
        
        self.samples_train, self.labels_train = [], []
        self.samples_test, self.labels_test = [], []
        
        # 获取所有文件并排序 (确保每次运行顺序一致，第9个文件是固定的)
        if subset in ['fingerflex']: 
            mat_files = sorted(list(Path(datadir).rglob("*_fingerflex.mat")))
            maxchannel = 64
        
        print(f"共找到 {len(mat_files)} 个文件。设定留出第 {leave_one_out_subject_idx} 个文件作为测试主体。")

        for file_idx, file_path in enumerate(mat_files):
            # 文件索引从1开始计数，方便理解
            current_subject_idx = file_idx + 1
            mat = loadmat(file_path)
            if 'data' not in mat: 
                print(f'数据缺失: {file_path}')
                continue
            
            data_full = mat['data']            
            # 标签处理
            if subset == 'fingerflex':
                cue_full = mat['cue'].flatten()
            else:
                cue_full = mat['stim'].flatten() if 'stim' in mat else mat['TargetPosX'].flatten()
            # 通道填充
            if (not self.single_channel) and data_full.shape[1] < maxchannel:
                pad = np.zeros((data_full.shape[0], maxchannel - data_full.shape[1]))
                data_full = np.hstack((data_full, pad))
            # 归一化 (保持原逻辑)
            data_full = data_full - np.mean(data_full, axis=1, keepdims=True)
            data_full = (data_full - np.mean(data_full, axis=0)) / (np.std(data_full, axis=0) + 1e-6)

            # --- 核心修改逻辑：决定当前文件的数据去向 ---            
            # 情况1: 前8个人 (完全加入训练集)
            if current_subject_idx < self.leave_one_out_subject_idx:
                s_tr, l_tr, c_tr = self._extract_segments(data_full, cue_full, is_train=True)
                self.samples_train.extend(s_tr)
                self.labels_train.extend(l_tr)
                print(f"主体 {current_subject_idx}: 全部加入训练集 (样本数: {c_tr})")
            # 情况2: 第9个人 (按比例切分)
            elif current_subject_idx == self.leave_one_out_subject_idx:
                split_point = int(len(data_full) * self.train_ratio_in_subject)
                # 训练部分 (前 X%)
                if split_point > 0:
                    s_tr, l_tr, c_tr = self._extract_segments(data_full[:split_point], cue_full[:split_point], is_train=True)
                    self.samples_train.extend(s_tr)
                    self.labels_train.extend(l_tr)
                
                # 测试部分 (后 1-X%)
                # 注意：如果 train_ratio=1.0，测试集将为空
                if split_point < len(data_full):
                    s_te, l_te, c_te = self._extract_segments(data_full[split_point:], cue_full[split_point:], is_train=False)
                    self.samples_test.extend(s_te)
                    self.labels_test.extend(l_te)
                print(f"主体 {current_subject_idx}: 训练比例 {self.train_ratio_in_subject}, 训练样本 {c_tr if split_point > 0 else 0}, 测试样本 {c_te if split_point < len(data_full) else 0}")
        
        self.set_mode(mode='train')
        print(f"数据集加载完成。总训练集: {len(self.labels_train)}, 总测试集: {len(self.labels_test)}")

    def _extract_segments(self, data_part, cue_part, is_train=True):
        """
        提取片段的辅助函数 (原 extract_segments 逻辑)
        """
        samples_out, labels_out = [], []
        count = 0
        
        if not self.single_wind:
            # 标准滑动窗口
            for i in range(0, len(data_part) - self.window_size, self.stride):
                window = data_part[i:i+self.window_size, :].T
                # 使用窗口内出现次数最多的标签
                label = np.bincount(cue_part[i:i+self.window_size].astype(int)).argmax()
                samples_out.append(window.astype(np.float32))
                labels_out.append(label)
                count += 1
        else:
            # 按事件边界切分
            n_samples = len(cue_part)
            k = 0
            while k < n_samples:
                start_k = k
                curr_label = cue_part[k]
                while k < n_samples and cue_part[k] == curr_label:
                    k += 1
                seg_len = k - start_k
                if seg_len >= self.window_size:
                    # 训练集用 stride，测试集用 window_size (不重叠)
                    curr_stride = self.stride if is_train else self.window_size
                    for j in range(start_k, k - self.window_size + 1, curr_stride):
                        window = data_part[j:j+self.window_size, :].T
                        samples_out.append(window.astype(np.float32))
                        labels_out.append(int(curr_label))
                        count += 1
        return samples_out, labels_out, count

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.samples, self.labels = self.samples_train, self.labels_train
        else:
            self.samples, self.labels = self.samples_test, self.labels_test

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        if self.single_channel:
            ch = self.channel_idx if self.channel_idx is not None else np.random.randint(0, sample.shape[0])
            sample = sample[ch:ch+1, :]
        return torch.FloatTensor(sample), torch.tensor(label, dtype=torch.long)


# regression task
class MillerFingersDataset_TT_joystick_track(Dataset):
    def __init__(self, datadir, window_size=256, stride=128, single_wind=False, 
                 single_channel=False, channel_idx=None, subset='joystick_track', split_ratio=0.9):
        assert subset == 'joystick_track', "This class is only for joystick_track regression task."
        self.single_channel = single_channel
        self.channel_idx = channel_idx
        self.window_size = window_size
        self.stride = stride
        self.subset = subset
        self.task_type = 'regression'
        self.label_dim = 2
        self.maxchannel = 64
        
        # === 第一步：收集所有位置数据，计算全局统计量 ===
        mat_files = list(Path(datadir).rglob("*.mat"))
        all_pos_x, all_pos_y = [], []
        
        for file_path in mat_files:
            mat = loadmat(file_path)
            if 'data' not in mat:
                continue                
            # 收集位置数据（不处理 ECoG）
            if 'TargetPosX' in mat and 'TargetPosY' in mat:
                all_pos_x.append(mat['TargetPosX'].flatten())
                all_pos_y.append(mat['TargetPosY'].flatten())
            elif 'CursorPosX' in mat and 'CursorPosY' in mat:
                all_pos_x.append(mat['CursorPosX'].flatten())
                all_pos_y.append(mat['CursorPosY'].flatten())
            else:
                raise KeyError(f"File {file_path} missing position fields.")
                
        # 合并并计算全局均值/标准差
        all_pos_x = np.concatenate(all_pos_x)
        all_pos_y = np.concatenate(all_pos_y)
        self.pos_mean = np.array([all_pos_x.mean(), all_pos_y.mean()], dtype=np.float32)
        self.pos_std = np.array([all_pos_x.std(), all_pos_y.std()], dtype=np.float32) + 1e-6
        
        print(f"📊 全局标签统计: mean=({self.pos_mean[0]:.2f}, {self.pos_mean[1]:.2f}), "
              f"std=({self.pos_std[0]:.2f}, {self.pos_std[1]:.2f})")

        # === 第二步：正式加载数据并标准化标签 ===
        self.samples_train, self.labels_train = [], []
        self.samples_test, self.labels_test = [], []
        self.samples, self.labels = self.samples_train, self.labels_train

        for file_path in mat_files:
            mat = loadmat(file_path)
            if 'data' not in mat:
                print('data not in ', file_path)
                continue
            data_full = mat['data']  # [T, C]

            # 加载标签
            # if 'TargetPosX' in mat and 'TargetPosY' in mat:
            #     pos_x = mat['TargetPosX'].flatten()
            #     pos_y = mat['TargetPosY'].flatten()
            if 'CursorPosX' in mat and 'CursorPosY' in mat:
                pos_x = mat['CursorPosX'].flatten()
                pos_y = mat['CursorPosY'].flatten()
            cue_full = np.stack([pos_x, pos_y], axis=1)  # [T, 2]

            # Z-score 标准化，均值0 标准差1 取值范围[-3, 3]或更广（如果原始数据有长尾）
            cue_full = (cue_full - self.pos_mean) / self.pos_std
            print(f'data.shape = {data_full.shape}, cue.shape = {cue_full.shape}')

            # 预处理 ECoG 信号
            if not self.single_channel and data_full.shape[1] < self.maxchannel:
                pad = np.zeros((data_full.shape[0], self.maxchannel - data_full.shape[1]))
                data_full = np.hstack((data_full, pad))
        
            # CAR + Z-score per channel
            data_full = data_full - np.mean(data_full, axis=1, keepdims=True)
            data_full = (data_full - np.mean(data_full, axis=0)) / (np.std(data_full, axis=0) + 1e-6)

            # 划分训练/测试
            split_idx = int(len(data_full) * split_ratio)

            def extract_segments(data_part, cue_part, is_train=True):
                samples_out, labels_out = [], []
                T = len(data_part)
                step = self.stride if is_train else self.window_size
                
                for i in range(0, T - self.window_size + 1, step):
                    window = data_part[i:i+self.window_size, :].T  # [C, T]
                    label = cue_part[i:i+self.window_size].mean(axis=0)  # [2] —— 已标准化！
                    
                    samples_out.append(window.astype(np.float32))
                    labels_out.append(label.astype(np.float32))
                    
                return samples_out, labels_out, len(samples_out)

            # 执行划分
            s_tr, l_tr, c_tr = extract_segments(data_full[:split_idx], cue_full[:split_idx], is_train=True)
            s_te, l_te, c_te = extract_segments(data_full[split_idx:], cue_full[split_idx:], is_train=False)

            self.samples_train.extend(s_tr)
            self.labels_train.extend(l_tr)
            self.samples_test.extend(s_te)
            self.labels_test.extend(l_te)

            print(f"文件 {os.path.basename(file_path)}: 训练样本 {c_tr}, 测试样本 {c_te}")

        self.set_mode(mode='train')
        print(f"✅ 数据集加载完成。总训练集: {len(self.labels_train)}, 总测试集: {len(self.labels_test)}")

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.samples, self.labels = self.samples_train, self.labels_train
        else:
            self.samples, self.labels = self.samples_test, self.labels_test

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]  # [C, T]
        label = self.labels[idx]    # [2], 已标准化
        # print(label)
        if self.single_channel:
            ch = self.channel_idx if self.channel_idx is not None else np.random.randint(0, sample.shape[0])
            sample = sample[ch:ch+1, :]  # [1, T]

        return torch.FloatTensor(sample), torch.FloatTensor(label)


# 丢包率
class MillerFingersDataset_Drop(Dataset):
    def __init__(self, datadir, window_size=256, stride=128,
                 packet_loss_rate=0.0, loss_type='random', burst_max_len=50,
                 single_wind=False, single_channel=False, channel_idx=None, 
                 subset='fingerflex', split_ratio=0.6):
        self.packet_loss_rate = packet_loss_rate
        self.loss_type = loss_type
        self.burst_max_len = burst_max_len

        self.single_channel = single_channel
        self.channel_idx = channel_idx
        self.window_size = window_size
        self.stride = stride
        self.subset = subset
        
        self.samples_train, self.labels_train = [], []
        self.samples_test, self.labels_test = [], []
        
        # 默认指向训练集
        self.samples, self.labels = self.samples_train, self.labels_train

        # 1. 获取文件列表
        if subset == 'fingerflex':
            mat_files = list(Path(datadir).rglob("*_fingerflex.mat"))
            self.maxchannel = 64
        elif subset == 'gestures':
            mat_files = list(Path(datadir).rglob("*_fingerflex.mat"))
            self.maxchannel = 84 
        elif subset == 'motor_basic':
            mat_files = glob.glob(os.path.join(datadir, "*.mat"))
            self.maxchannel = 64 
        else:
            mat_files = glob.glob(os.path.join(datadir, "*.mat"))
            self.maxchannel = 64

        for file_path in mat_files:
            mat = loadmat(file_path)
            if 'data' not in mat: continue
            
            data_full = mat['data']
            if subset == 'fingerflex':
                cue_full = mat['cue'].flatten()
            else:
                cue_full = mat['stim'].flatten() if 'stim' in mat else mat['TargetPosX'].flatten()

            # 2. 预处理 (补齐通道 & 标准化)
            if (not self.single_channel) and data_full.shape[1] < self.maxchannel:
                pad = np.zeros((data_full.shape[0], self.maxchannel - data_full.shape[1]))
                data_full = np.hstack((data_full, pad))
            
            # CAR (共模去噪) & Z-Score
            data_full = data_full - np.mean(data_full, axis=1, keepdims=True)
            data_full = (data_full - np.mean(data_full, axis=0)) / (np.std(data_full, axis=0) + 1e-6)

            # 3. 划分索引
            split_idx = int(len(data_full) * split_ratio)

            # 4. 执行切片 (使用内部函数确保逻辑一致)
            s_tr, l_tr, c_tr = self._extract_segments(data_full[:split_idx], cue_full[:split_idx], True, single_wind)
            self.samples_train.extend(s_tr)
            self.labels_train.extend(l_tr)

            s_te, l_te, c_te = self._extract_segments(data_full[split_idx:], cue_full[split_idx:], False, single_wind)
            self.samples_test.extend(s_te)
            self.labels_test.extend(l_te)
            # print(f"处理文件: {os.path.basename(file_path)} | 训练: {c_tr} | 测试: {c_te}")

        # 5. 标签映射
        self.set_mode('train')
        self._print_stats()

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.samples[idx]).float()  # [C, T]
        label = self.labels[idx]
        if self.single_channel:
            ch = self.channel_idx if self.channel_idx is not None else np.random.randint(0, sample.shape[0])
            sample = sample[ch:ch+1, :]  # [1, T]

        # === 注入丢包 ===
        if self.packet_loss_rate > 0:
            sampledrop = self._apply_packet_loss(sample)
        else:
            sampledrop = sample.clone()
        return sampledrop, torch.tensor(label, dtype=torch.long), sample

    def _apply_packet_loss(self, x):
        """
        x: [C, T]
        Returns: [C, T] with packet loss applied
        """
        C, T = x.shape
        x = x.clone()  # 避免修改原始数据

        if self.loss_type == 'random':
            # 每个时间点独立以概率 p 被丢弃（置0）
            mask = torch.rand(T) > self.packet_loss_rate  # True=保留
            x[:, ~mask] = 0.0
            # x[:, ~mask] = x.mean(dim=1, keepdim=True)  # 通道均值填充
        elif self.loss_type == 'burst':
            # 模拟突发丢包：随机选择起始点，丢弃连续 L 个点
            num_bursts = int(self.packet_loss_rate * T / (self.burst_max_len / 2 + 1e-6))
            for _ in range(num_bursts):
                start = np.random.randint(0, T)
                length = np.random.randint(1, min(self.burst_max_len, T - start) + 1)
                x[:, start:start+length] = 0.0
        elif self.loss_type == 'channel':
            # 随机丢弃整个通道
            mask_ch = torch.rand(C) > self.packet_loss_rate
            x[~mask_ch, :] = 0.0
        return x

    def _extract_segments(self, data_part, cue_part, is_train, single_wind):
        samples_out, labels_out = [], []
        count = 0
        n_samples = len(cue_part)
        
        if not single_wind:
            for i in range(0, n_samples - self.window_size, self.stride):
                window = data_part[i:i+self.window_size, :].T
                label = np.bincount(cue_part[i:i+self.window_size].astype(int)).argmax()
                samples_out.append(window.astype(np.float32))
                labels_out.append(label)
                count += 1
        else:
            k = 0
            while k < n_samples:
                start_k = k
                curr_label = cue_part[k]
                while k < n_samples and cue_part[k] == curr_label: k += 1
                seg_len = k - start_k
                if seg_len >= self.window_size:
                    curr_stride = self.stride if is_train else self.window_size
                    for j in range(start_k, k - self.window_size + 1, curr_stride):
                        window = data_part[j:j+self.window_size, :].T
                        samples_out.append(window.astype(np.float32))
                        labels_out.append(int(curr_label))
                        count += 1
        return samples_out, labels_out, count

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.samples, self.labels = self.samples_train, self.labels_train
        else:
            self.samples, self.labels = self.samples_test, self.labels_test

    def _print_stats(self):
        def get_info(labels):
            c = Counter(labels)
            total = len(labels)
            return {k: (c[k], c[k]/total*100) for k in sorted(c.keys())}
        tr_i, te_i = get_info(self.labels_train), get_info(self.labels_test)
        print(f"{'类别':<6} | {'训练 (占比)':<20} | {'测试 (占比)':<20}")
        for c in sorted(set(self.labels_train) | set(self.labels_test)):
            tr = tr_i.get(c, (0,0)); te = te_i.get(c, (0,0))
            print(f"{c:<6} | {tr[0]:>6} ({tr[1]:>5.1f}%) | {te[0]:>6} ({te[1]:>5.1f}%)")

    def __len__(self):
        return len(self.labels)



'''
 fingerflex  | 训练集 (Train)               | 测试集 (Test)               
类别         | 数量 (占比)                   | 数量 (占比)                  
----------------------------------------------------------------------
0            | 7058   (48.46%)            | 1761   (48.69%)
1            | 1840   (12.63%)            | 539    (14.90%)
2            | 1367   ( 9.39%)            | 362    (10.01%)
3            | 1472   (10.11%)            | 304    ( 8.40%)
4            | 1430   ( 9.82%)            | 365    (10.09%)
5            | 1397   ( 9.59%)            | 286    ( 7.91%)
总计         | 14564                     | 3617
'''
# ------------------------------ BCI 比赛数据集 ------------------------------
class BCIFingersDataset_TT(Dataset):
    def __init__(self, datadir, window_size=256, stride=128, single_wind=False, 
                 single_channel=False, channel_idx=None):
        self.single_channel = single_channel
        self.channel_idx = channel_idx
        self.window_size = window_size
        self.stride = stride
        self.maxchannel = 64
        
        self.samples_train, self.labels_train = [], []
        self.samples_test, self.labels_test = [], []
        
        # 默认指向训练集
        self.samples, self.labels = self.samples_train, self.labels_train

        # 1. 匹配数据文件和标签文件
        mat_files = list(Path(datadir).rglob("*_comp.mat"))# 
        # print('mat_files = ', datadir, mat_files)
        # # /disk2/user1/dataset/BCI-Standford/BCI_Competion4_dataset4_data_fingerflexions/BCI_Competion4_dataset4_data_fingerflexions/data/
        # exit()
        for file_path in mat_files:
            # 构建对应的标签文件名，例如 sub1_comp.mat -> sub1_testlabels.mat
            base_name = file_path.name.replace('_comp.mat', '')
            label_path = file_path.parent / f"{base_name}_testlabels.mat"
            
            if not label_path.exists():
                print(f"警告: 找不到标签文件 {label_path}，跳过该主体。")
                continue

            # 加载数据
            mat_data = loadmat(file_path)
            mat_label = loadmat(label_path)

            # 获取训练和测试数据
            # train_data/test_data: [Samples, Channels]
            # train_dg/test_dg: [Samples, 5] (5个手指的运动轨迹)
            tr_data = mat_data['train_data']
            te_data = mat_data['test_data']
            tr_dg = mat_data['train_dg']
            te_dg = mat_label['test_dg'] # 从 label 文件读取测试标签

            # 2. 预处理 (标准化) - 训练和测试应分别或统一标准化
            def preprocess(data):
                # 补齐 64 通道
                if (not self.single_channel) and data.shape[1] < self.maxchannel:
                    pad = np.zeros((data.shape[0], self.maxchannel - data.shape[1]))
                    data = np.hstack((data, pad))
                # CAR & Z-Score
                data = data - np.mean(data, axis=1, keepdims=True)
                data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6)
                return data

            tr_data = preprocess(tr_data)
            te_data = preprocess(te_data)

            # 3. 标签转换 (从连续手指轨迹 dg 转换为分类标签)
            # 简化逻辑：取 5 个手指中位移最大的那个作为当前类索引 (0-4)，若都小则为 5 (静息)
            def dg_to_label(dg):
                # dg 形状 [Samples, 5]
                main_finger = np.argmax(dg, axis=1) # 拿到位移最大的手指索引
                # 如果位移都很小（阈值化），可以设为背景类，这里暂时直接取最大
                return main_finger 

            tr_cue = dg_to_label(tr_dg)
            te_cue = dg_to_label(te_dg)

            # 4. 执行切片 (利用你之前的 extract_segments 逻辑)
            # 训练集
            s_tr, l_tr, c_tr = self.extract_segments(tr_data, tr_cue, is_train=True, single_wind=single_wind)
            self.samples_train.extend(s_tr)
            self.labels_train.extend(l_tr)

            # 测试集
            s_te, l_te, c_te = self.extract_segments(te_data, te_cue, is_train=False, single_wind=single_wind)
            self.samples_test.extend(s_te)
            self.labels_test.extend(l_te)

            print(f"主体 {base_name}: 提取训练样本 {c_tr}, 测试样本 {c_te}")

        self.set_mode('train')
        print(f"✅ 加载完成。总训练样本: {len(self.labels_train)}, 总测试样本: {len(self.labels_test)}")

    def extract_segments(self, data_part, cue_part, is_train=True, single_wind=False):
        samples_out, labels_out = [], []
        count = 0
        n_samples = len(cue_part)
        
        if not single_wind:
            for i in range(0, n_samples - self.window_size, self.stride):
                window = data_part[i:i+self.window_size, :].T
                label = np.bincount(cue_part[i:i+self.window_size].astype(int)).argmax()
                samples_out.append(window.astype(np.float32))
                labels_out.append(label)
                count += 1
        else:
            k = 0
            while k < n_samples:
                start_k = k
                curr_label = cue_part[k]
                while k < n_samples and cue_part[k] == curr_label:
                    k += 1
                seg_len = k - start_k
                if seg_len >= self.window_size:
                    curr_stride = self.stride if is_train else self.window_size
                    for j in range(start_k, k - self.window_size + 1, curr_stride):
                        window = data_part[j:j+self.window_size, :].T
                        samples_out.append(window.astype(np.float32))
                        labels_out.append(int(curr_label))
                        count += 1
        return samples_out, labels_out, count

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.samples, self.labels = self.samples_train, self.labels_train
        else:
            self.samples, self.labels = self.samples_test, self.labels_test

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        if self.single_channel:
            ch = self.channel_idx if self.channel_idx is not None else np.random.randint(0, sample.shape[0])
            sample = sample[ch:ch+1, :]
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


# ==========================================
# dataloader处理 数据平衡、时序切分
# ==========================================
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
        print('self.class_iters, self.num_classes = ', self.class_iters, self.num_classes)

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


## SEED-IV Emotion Recognition Dataset with Train/Test split per session (90%/10%).
class SEEDIVDataset_TT0(Dataset):
    """
    把EEG直接按照90% 10%划分，可能把一个trail分开，归一化方式不太准确
    """
    def __init__(self, datadir="", window_size=0, stride=0, single_wind=False, single_channel=False, channel_idx=None, split_ratio=0.6, normalize=True):
        self.window_size = window_size
        self.stride = stride
        self.single_wind = single_wind
        self.single_channel = single_channel
        self.channel_idx = channel_idx
        self.split_ratio = split_ratio
        self.normalize = normalize

        # Storage
        self.samples_train, self.labels_train = [], []
        self.samples_test, self.labels_test = [], []
        self.samples_v, self.labels_v = [], []
        self.samples, self.labels = self.samples_train, self.labels_train  # default: train

        # Get all subject-session paths: /subject_id/session.mat
        subject_dirs = sorted([d for d in Path(datadir).iterdir() if d.is_dir() and d.name.isdigit()])
        # print('subject_dirs = ', subject_dirs)

        for subj_dir in subject_dirs:  # [:1]:  
            # 🔧 FIX: Correctly collect session files like "1_xxx.mat", "2_yyy.mat", etc.
            session_files = []
            for f in subj_dir.glob("*.mat"):
                stem = f.stem
                sess_id = stem.split('_')[0]
                if sess_id in ('1', '2', '3'):
                    session_files.append(f)
            # Sort by session ID (1, 2, 3)
            session_files.sort(key=lambda x: int(x.stem.split('_')[0]))
            print(f"Found session files in {subj_dir}: {[f.name for f in session_files]}")
            for sess_file in session_files:  # [:1]:  #
                print(f"Loading {sess_file}...")
                mat_contents = scipy.io.loadmat(str(sess_file))
                # 🔑 SEED-IV stores trials
                # trial_keys = [f'cz_eeg{i}' for i in range(1, 25)]
                for candidate_prefix in ['cz', 'ha', 'hql']:
                    trial_keys = [f"{candidate_prefix}_eeg{i}" for i in range(1, 25)]
                    if all(k in mat_contents for k in trial_keys):
                        # prefix = candidate_prefix
                        break
    
                # Load each trial: shape (62, T)
                trials = [mat_contents[key] for key in trial_keys]  # list of (62, T)
                # Ensure all trials have same number of time points (optional: pad/trim if needed)
                T_trial = trials[0].shape[1]
                for i, tr in enumerate(trials):
                    if tr.shape[1] != T_trial:
                        print(f"⚠️ Trial {i+1} has different length ({tr.shape[1]} vs {T_trial}), trimming/padding...")
                        if tr.shape[1] < T_trial:
                            # Pad with last value
                            tr = np.pad(tr, ((0,0), (0, T_trial - tr.shape[1])), mode='edge')
                        else:
                            tr = tr[:, :T_trial]
                        trials[i] = tr        
                # Concatenate along time axis → (62, T_total = 24 * T_trial)
                data_full = np.concatenate(trials, axis=1)  # (62, T_total)        
                # Build label vector: [0,1,2,3] * 6 → 24 labels
                label_seq = [0, 1, 2, 3] * 6  # 24 emotions in order
                cue_full = np.concatenate([
                    np.full(T_trial, label_seq[i], dtype=np.int64) for i in range(24)
                ])  # (T_total,)
                # print('cue_full: ', cue_full.shape, cue_full)
                
                # Normalize (z-score per channel)
                if self.normalize:
                    data_full = data_full - np.mean(data_full, axis=1, keepdims=True)
                    data_full = (data_full - np.mean(data_full, axis=0)) / (np.std(data_full, axis=0) + 1e-8)
                # Transpose to (T, C) for easier slicing
                data_full = data_full.T  # (T, 62)
                cue_full = cue_full      # (T,)
                # print(f"  → Continuous data shape: {data_full.shape}, labels: {cue_full.shape}")
                split_idx = int(len(data_full) * self.split_ratio)
                # --- Core segment extraction ---
                def extract_segments(data_part, cue_part, is_train=True):
                    samples_out, labels_out = [], []
                    count = 0
                    n_samples = len(data_part)

                    if not self.single_wind:
                        for i in range(0, n_samples - self.window_size + 1, self.stride):
                            window = data_part[i:i+self.window_size, :].T  # (C, T)
                            label = np.bincount(cue_part[i:i+self.window_size]).argmax()
                            samples_out.append(window.astype(np.float32))
                            labels_out.append(int(label))
                            count += 1
                    else:
                        # Pure-label segments
                        k = 0
                        while k < n_samples:
                            start_k = k
                            curr_label = cue_part[k]
                            while k < n_samples and cue_part[k] == curr_label:
                                k += 1
                            seg_len = k - start_k
                            if seg_len >= self.window_size:
                                curr_stride = self.stride if is_train else self.window_size
                                for j in range(start_k, k - self.window_size + 1, curr_stride):
                                    window = data_part[j:j+self.window_size, :].T
                                    samples_out.append(window.astype(np.float32))
                                    labels_out.append(int(curr_label))
                                    count += 1
                    return samples_out, labels_out, count

                # Extract train/test
                s_tr, l_tr, c_tr = extract_segments(data_full[:split_idx], cue_full[:split_idx], is_train=True)
                s_te, l_te, c_te = extract_segments(data_full[split_idx:], cue_full[split_idx:], is_train=False)
                self.samples_train.extend(s_tr)
                self.labels_train.extend(l_tr)
                self.samples_test.extend(s_te)
                self.labels_test.extend(l_te)
                self.samples_v.extend(s_te[:200])
                self.labels_v.extend(l_te[:200])
                # print('l_tr = ', l_tr)
                # print(f"    → Train samples: {c_tr}, Test samples: {c_te}")
        print(f"\n✅ SEED-IV loading complete.")
        print(f"   Train samples: {len(self.labels_train)}, Test samples: {len(self.labels_test)}")
        self.set_mode('train')
        
    def set_mode(self, mode='train'):
        if mode == 'train':
            self.samples, self.labels = self.samples_train, self.labels_train
        elif mode == 'valid':
            self.samples, self.labels = self.samples_v, self.labels_v
        else:
            self.samples, self.labels = self.samples_test, self.labels_test

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]  # (C, T)
        label = self.labels[idx]

        if self.single_channel:
            ch = self.channel_idx if self.channel_idx is not None else np.random.randint(0, sample.shape[0])
            sample = sample[ch:ch+1, :]  # (1, T)

        return torch.FloatTensor(sample), torch.tensor(label, dtype=torch.long)       

## SEED-IV Emotion Recognition Dataset with Train/Test split per session (trial-level).
class SEEDIVDataset_TT(Dataset):
    """
    Each .mat file contains 24 trials. We split by trial boundaries (not time points) 
    to avoid data leakage. Label order is fixed: [0: neutral, 1: sad, 2: fear, 3: happy] repeated 6 times.
    """
    def __init__(self, datadir="", window_size=1000, stride=500, single_wind=False, 
                 single_channel=False, channel_idx=None, split_ratio=0.9, normalize=True):
        self.window_size = window_size
        self.stride = stride
        self.single_wind = single_wind
        self.single_channel = single_channel
        self.channel_idx = channel_idx
        self.split_ratio = split_ratio  # Fraction of trials for training
        self.normalize = normalize

        # Storage
        self.samples_train, self.labels_train = [], []
        self.samples_test, self.labels_test = [], []
        self.samples_v, self.labels_v = [], []
        self.samples, self.labels = self.samples_train, self.labels_train  # default: train

        # Get all subject-session paths: /subject_id/session.mat
        subject_dirs = sorted([d for d in Path(datadir).iterdir() if d.is_dir() and d.name.isdigit()])
        
        for subj_dir in subject_dirs:  # [:1]:  #   # Process first subject only
            session_files = []
            for f in subj_dir.glob("*.mat"):
                stem = f.stem
                sess_id = stem.split('_')[0]
                if sess_id in ('1', '2', '3'):
                    session_files.append(f)
            session_files.sort(key=lambda x: int(x.stem.split('_')[0]))
            
            for sess_file in session_files:  # [:1]:  # 
                print(f"Loading {sess_file}...")
                mat_contents = scipy.io.loadmat(str(sess_file))
                
                # Find correct prefix (cz, ha, or hql)
                for candidate_prefix in ['cz', 'ha', 'hql']:
                    trial_keys = [f"{candidate_prefix}_eeg{i}" for i in range(1, 25)]
                    if all(k in mat_contents for k in trial_keys):
                        break
                
                # Load trials and ensure consistent length
                trials = [mat_contents[key] for key in trial_keys]  # list of (62, T)
                T_trial = trials[0].shape[1]
                for i, tr in enumerate(trials):
                    # print(tr.shape) = (62, 33601)(62, 19001)(62, 39801)(62, 41801)(62, 35801) (62, 9601) ...
                    if tr.shape[1] != T_trial:  # pad 0
                        if tr.shape[1] < T_trial:
                            tr = np.pad(tr, ((0,0), (0, T_trial - tr.shape[1])), mode='edge')
                        else:
                            tr = tr[:, :T_trial]
                        trials[i] = tr  # len(trials[i]) = 62
                # Fixed label sequence per SEED-IV protocol
                label_seq = [0, 1, 2, 3] * 6  # 24 emotions in fixed order                
                # Normalize per channel (EEG standard practice)
                if self.normalize:
                    for i in range(len(trials)):
                        trials[i] = (trials[i] - np.mean(trials[i], axis=1, keepdims=True)) / \
                                   (np.std(trials[i], axis=1, keepdims=True) + 1e-8)
                
                # Split by trial count (NOT time points) to prevent data leakage
                n_trials = len(trials)  # 24
                split_n = int(n_trials * self.split_ratio)
                
                # Process training trials
                data_train = np.concatenate(trials[:split_n], axis=1) if split_n > 0 else np.zeros((62, 0))
                cue_train = np.concatenate([
                    np.full(trials[i].shape[1], label_seq[i], dtype=np.int64) 
                    for i in range(split_n)
                ]) if split_n > 0 else np.array([])
                
                # Process test trials
                data_test = np.concatenate(trials[split_n:], axis=1) if split_n < n_trials else np.zeros((62, 0))
                cue_test = np.concatenate([
                    np.full(trials[i].shape[1], label_seq[i], dtype=np.int64) 
                    for i in range(split_n, n_trials)
                ]) if split_n < n_trials else np.array([])
                
                # Transpose to (T, C) for easier slicing
                data_train = data_train.T  # (T_train, 62)
                data_test = data_test.T    # (T_test, 62)
                
                # Extract segments
                s_tr, l_tr = self._extract_segments(data_train, cue_train, is_train=True)
                s_te, l_te = self._extract_segments(data_test, cue_test, is_train=False)
                
                self.samples_train.extend(s_tr)
                self.labels_train.extend(l_tr)
                self.samples_test.extend(s_te)
                self.labels_test.extend(l_te)
                self.samples_v.extend(s_te[:200])
                self.labels_v.extend(l_te[:200])
                # print('l_te = ', l_te)
        
        print(f"\n✅ SEED-IV loading complete.")
        print(f"   Train samples: {len(self.labels_train)}, Test samples: {len(self.labels_test)}")
        self.set_mode('train')
    
    def _extract_segments(self, data_part, cue_part, is_train=True):
        """Extract segments from continuous data with proper labeling"""
        if len(data_part) == 0:
            return [], []
            
        samples_out, labels_out = [], []
        n_samples = len(data_part)
        
        if not self.single_wind:
            # Sliding window approach
            step = self.stride if is_train else self.window_size
            for i in range(0, n_samples - self.window_size + 1, step):
                window = data_part[i:i+self.window_size, :].T  # (C, T)
                # Use majority vote for label in window
                label = np.bincount(cue_part[i:i+self.window_size]).argmax()
                samples_out.append(window.astype(np.float32))
                labels_out.append(int(label))
        else:
            # Pure-label segments (non-overlapping within same label)
            k = 0
            while k < n_samples:
                start_k = k
                curr_label = cue_part[k]
                while k < n_samples and cue_part[k] == curr_label:
                    k += 1
                seg_len = k - start_k
                if seg_len >= self.window_size:
                    step = self.stride if is_train else self.window_size
                    for j in range(start_k, k - self.window_size + 1, step):
                        window = data_part[j:j+self.window_size, :].T
                        samples_out.append(window.astype(np.float32))
                        labels_out.append(int(curr_label))
        return samples_out, labels_out
        
    def set_mode(self, mode='train'):
        if mode == 'train':
            self.samples, self.labels = self.samples_train, self.labels_train
        elif mode == 'valid':
            self.samples, self.labels = self.samples_v, self.labels_v
        else:
            self.samples, self.labels = self.samples_test, self.labels_test

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]  # (C, T)
        label = self.labels[idx]

        if self.single_channel:
            ch = self.channel_idx if self.channel_idx is not None else np.random.randint(0, sample.shape[0])
            sample = sample[ch:ch+1, :]  # (1, T)

        return torch.FloatTensor(sample), torch.tensor(label, dtype=torch.long)
    

## ==================================================
import warnings
# 仅抑制 MNE 中关于重复通道名的警告（推荐）
warnings.filterwarnings(
    "ignore",
    message="Channel names are not unique, found duplicates for:",
    category=RuntimeWarning
)
# 简单粗暴：忽略所有的 RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
class CHBMITDataset_TT(Dataset):
    def __init__(self, root_dir: str, window_size_sec: float = 4.0, stride_sec: float = 2.0,
                 sampling_rate: int = 256, patient_ids=None, split_ratio=0.6,
                 single_channel=False, target_channels=23):
        """
        CHB-MIT Scalp EEG Dataset (内存友好版：延迟加载 + 进程内缓存)
        """
        self.root_dir = root_dir
        self.window_size = int(window_size_sec * sampling_rate)
        self.stride = int(stride_sec * sampling_rate)
        self.sampling_rate = sampling_rate
        self.split_ratio = split_ratio
        self.single_channel = single_channel
        self.target_channels = target_channels

        # 核心优化：进程内文件缓存（每个 DataLoader worker 会拥有独立的缓存）
        self.file_cache = {}
        
        self.samples_train = []
        self.samples_test = []
        self.samples = []

        # 获取患者列表
        if patient_ids is None:
            patient_folders = [d for d in os.listdir(root_dir) 
                              if d.startswith('chb') and os.path.isdir(os.path.join(root_dir, d))]
        else:
            patient_folders = [pid for pid in patient_ids 
                              if os.path.isdir(os.path.join(root_dir, pid))]
        patient_folders.sort()

        for patient in patient_folders:
            print(f"Processing {patient}...")
            patient_dir = os.path.join(root_dir, patient)
            summary_file = os.path.join(patient_dir, f"{patient}-summary.txt")
            if not os.path.exists(summary_file):
                print(f"⚠️ Warning: Summary file not found for {patient}")
                continue

            seizure_intervals = self._parse_seizure_times(summary_file)
            edf_files = sorted([f for f in os.listdir(patient_dir) if f.endswith('.edf')])
            if not edf_files:
                continue

            n_files = len(edf_files)
            n_train_files = max(1, int(n_files * self.split_ratio))
            train_files = edf_files[:n_train_files]
            test_files = edf_files[n_train_files:]

            self._process_files(patient, patient_dir, train_files, seizure_intervals, is_train=True)
            self._process_files(patient, patient_dir, test_files, seizure_intervals, is_train=False)
        
        self.samples_valid = self.samples_test[:1000]

        print(f"\n✅ CHB-MIT loading complete.")
        print(f"   Train samples: {len(self.samples_train)}, Test samples: {len(self.samples_test)}")
        self.set_mode('train')

    def _process_files(self, patient, patient_dir, edf_files, seizure_intervals, is_train):
        current_samples = self.samples_train if is_train else self.samples_test
        cumulative_time_sec = 0

        for edf_file in edf_files:
            file_path = os.path.join(patient_dir, edf_file)
            try:
                # ⚡️ 只打开文件读取元数据，不加载实际波形数据（preload=False）
                raw = read_raw_edf(file_path, preload=False, verbose=False)
            except Exception as e:
                print(f"❌ Failed to load {edf_file}: {e}")
                continue

            # 统一重采样
            if raw.info['sfreq'] != self.sampling_rate:
                raw.resample(self.sampling_rate, verbose=False)
            
            # 只提取 EEG 通道
            eeg_picks = pick_types(raw.info, eeg=True, exclude='bads')
            if len(eeg_picks) == 0:
                print(f"⚠️ No EEG channels in {edf_file}")
                continue

            total_samples = len(raw.times)
            total_seconds = total_samples / self.sampling_rate

            # 计算该文件对应的发作区间
            file_seizure_intervals = []
            for start_sec, end_sec in seizure_intervals:
                if self._is_seizure_in_file(edf_file, patient, start_sec, end_sec):
                    rel_start = max(0, start_sec - cumulative_time_sec)
                    rel_end = min(total_seconds, end_sec - cumulative_time_sec)
                    if rel_end > rel_start:
                        file_seizure_intervals.append((rel_start, rel_end))

            # 滑动窗口生成样本索引
            start_sample = 0
            while start_sample + self.window_size <= total_samples:
                win_start_sec = start_sample / self.sampling_rate
                win_end_sec = win_start_sec + (self.window_size / self.sampling_rate)

                label = 0
                for s_sec, e_sec in file_seizure_intervals:
                    if win_end_sec > s_sec and win_start_sec < e_sec:
                        label = 1
                        break

                # 样本只保存文件路径和切片位置
                sample_info = {
                    'label': label,
                    'file_path': file_path,
                    'start_sample': start_sample
                }
                current_samples.append(sample_info)
                start_sample += self.stride

            cumulative_time_sec += total_seconds

    def _parse_seizure_times(self, summary_file):
        intervals = []
        with open(summary_file, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            if 'Seizure' in lines[i] and 'Start' in lines[i]:
                start_sec = self._extract_time_in_seconds(lines[i])
                end_sec = self._extract_time_in_seconds(lines[i+1]) if i+1 < len(lines) else None
                if start_sec is not None and end_sec is not None:
                    intervals.append((start_sec, end_sec))
                i += 2
            else:
                i += 1
        return intervals

    def _extract_time_in_seconds(self, line):
        match = re.search(r'(\d+)\s*seconds?', line, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _is_seizure_in_file(self, edf_filename, patient_id, seizure_start_sec, seizure_end_sec):
        base_name = os.path.splitext(edf_filename)[0]
        parts = base_name.split('_')
        if len(parts) < 2:
            return False
        try:
            file_index = int(parts[1])
        except ValueError:
            return False
        file_start_sec = (file_index - 1) * 3600
        file_end_sec = file_index * 3600
        return not (seizure_end_sec <= file_start_sec or seizure_start_sec >= file_end_sec)

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.samples = self.samples_train  
        elif mode == 'test':
            self.samples = self.samples_test
        elif mode == 'valid':
            self.samples = self.samples_valid

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]    
        file_path = info['file_path']
        
        # ⚡️ 核心优化：检查当前进程的缓存中是否已有该文件的数据
        if file_path in self.file_cache:
            full_data = self.file_cache[file_path]
        else:
            # 缓存未命中，去硬盘读取并预处理
            raw = read_raw_edf(file_path, preload=True, verbose=False)
            if raw.info['sfreq'] != self.sampling_rate:
                raw.resample(self.sampling_rate, verbose=False)
            
            eeg_picks = pick_types(raw.info, eeg=True, exclude='bads')
            data = raw.get_data(picks=eeg_picks)
            C_actual, T = data.shape
            
            # 通道对齐
            if C_actual >= self.target_channels:
                aligned_data = data[:self.target_channels, :]
            else:
                aligned_data = np.zeros((self.target_channels, T), dtype=np.float32)
                aligned_data[:C_actual, :] = data
            
            # 放入当前进程的缓存中
            self.file_cache[file_path] = aligned_data
            full_data = aligned_data

        # 从内存中切片提取当前样本
        start = info['start_sample']
        data = full_data[:, start : start + self.window_size]
        
        label = int(info['label'])
        
        if self.single_channel:
            ch = np.random.randint(0, self.target_channels)
            data = data[ch:ch+1, :]
        
        return torch.from_numpy(data).float(), torch.tensor(label, dtype=torch.long)


def preprocess_and_save(root_dir, save_dir, window_size_sec=4.0, stride_sec=2.0, 
                        sampling_rate=256, split_ratio=0.6, target_channels=23):
    """
    离线预处理 CHB-MIT 数据集，保存为 .npy 格式并生成样本索引
    """
    os.makedirs(save_dir, exist_ok=True)
    window_size = int(window_size_sec * sampling_rate)
    stride = int(stride_sec * sampling_rate)
    
    all_samples = [] # 保存所有样本的元数据

    patient_folders = sorted([d for d in os.listdir(root_dir) 
                              if d.startswith('chb') and os.path.isdir(os.path.join(root_dir, d))])

    for patient in patient_folders[20:]:
        print(f"正在预处理患者: {patient}...")
        patient_dir = os.path.join(root_dir, patient)
        summary_file = os.path.join(patient_dir, f"{patient}-summary.txt")
        if not os.path.exists(summary_file):
            continue

        # 1. 解析发作时间 (复用之前的逻辑)
        seizure_intervals = parse_seizure_times(summary_file)
        edf_files = sorted([f for f in os.listdir(patient_dir) if f.endswith('.edf')])

        n_files = len(edf_files)
        # 确保至少留 2 个文件给测试集，防止某个发作跨越倒数第二个和最后一个文件
        n_train_files = max(1, n_files - 2)
        
        for idx, edf_file in enumerate(edf_files):
            file_path = os.path.join(patient_dir, edf_file)
            mode = 'train' if idx < n_train_files else 'test'
            patient_save_dir = os.path.join(save_dir, mode, patient)
            os.makedirs(patient_save_dir, exist_ok=True)
            
            try:
                # 读取并预处理
                raw = read_raw_edf(file_path, preload=True, verbose=False)
                if raw.info['sfreq'] != sampling_rate:
                    raw.resample(sampling_rate, verbose=False)
                
                eeg_picks = pick_types(raw.info, eeg=True, exclude='bads')
                data = raw.get_data(picks=eeg_picks)
                
                # 通道对齐
                C_actual, T = data.shape
                if C_actual >= target_channels:
                    aligned_data = data[:target_channels, :]
                else:
                    aligned_data = np.zeros((target_channels, T), dtype=np.float32)
                    aligned_data[:C_actual, :] = data
                
                # 保存处理好的完整文件数据
                save_npy_path = os.path.join(patient_save_dir, edf_file.replace('.edf', '.npy'))
                np.save(save_npy_path, aligned_data)
                
                # 2. 生成样本索引
                # (这里复用你原有的发作区间计算和滑动窗口逻辑)
                # 为了简化展示，这里假设已经获取了 file_seizure_intervals
                file_seizure_intervals = [] # 请填入你原有的计算逻辑
                cumulative_time_sec = 0 # 请填入你原有的计算逻辑
                
                start_sample = 0
                while start_sample + window_size <= T:
                    win_start_sec = start_sample / sampling_rate
                    win_end_sec = win_start_sec + (window_size / sampling_rate)
                    
                    label = 0
                    for s_sec, e_sec in file_seizure_intervals:
                        if win_end_sec > s_sec and win_start_sec < e_sec:
                            label = 1
                            break
                    
                    # 记录样本元数据：保存后的 .npy 路径、起始位置、标签
                    all_samples.append({
                        'npy_path': save_npy_path,
                        'start_sample': start_sample,
                        'label': label,
                        'mode': mode
                    })
                    start_sample += stride
                    
            except Exception as e:
                print(f"处理 {edf_file} 失败: {e}")

    # 3. 保存样本索引到文件
    np.save(os.path.join(save_dir, 'samples_metadata.npy'), all_samples)
    print(f"✅ 预处理完成！共生成 {len(all_samples)} 个样本，已保存至 {save_dir}")

def parse_seizure_times(summary_file):
    # 复用你原有的解析逻辑
    intervals = []
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if 'Seizure' in lines[i] and 'Start' in lines[i]:
            start_sec = extract_time_in_seconds(lines[i])
            end_sec = extract_time_in_seconds(lines[i+1]) if i+1 < len(lines) else None
            if start_sec is not None and end_sec is not None:
                intervals.append((start_sec, end_sec))
            i += 2
        else:
            i += 1
    return intervals

def extract_time_in_seconds(line):
    match = re.search(r'(\d+)\s*seconds?', line, re.IGNORECASE)
    return int(match.group(1)) if match else None


class CHBMITDataset_Fast(Dataset):
    def __init__(self, preprocessed_dir, mode='train', valid=False, single_channel=False):
        """
        极速版 Dataset，直接读取预处理好的 .npy 文件
        
        Args:
            single_channel (bool): 如果为 True，每次随机抽取 1 个通道返回。
        """
        self.mode = mode
        self.single_channel = single_channel  # 保存参数
        
        # 加载预处理阶段生成的样本元数据
        all_metadata = np.load(os.path.join(preprocessed_dir, 'samples_metadata.npy'), allow_pickle=True)
        
        # 筛选对应模式的样本
        self.samples = []
        for s in all_metadata:
            info = s if isinstance(s, dict) else s.item()
            if info['mode'] == mode:
                self.samples.append(info)
                
        # # 如果是在测试模式下且开启了 valid，进行降采样（按你原有的逻辑）
        # if mode == 'test' and valid:
        #     self.samples = self.samples[::1000]
            
        print(f"✅ 极速加载完成。{mode} 模式样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        
        # 核心：直接从 .npy 文件中读取切片
        full_data = np.load(info['npy_path'], mmap_mode='r') 
        start = info['start_sample']
        # 假设 window_size 固定为 4*256=1024
        data = full_data[:, start : start + 1024] 
        
        # ⚡️ 新增：单通道处理逻辑
        if self.single_channel:
            # 随机抽取一个通道的索引 (0 到 总通道数-1)
            random_ch = np.random.randint(0, data.shape[0])
            # 抽取该通道，并使用 [None, :] 或 keepdims=True 保持二维形状 [1, 1024]
            data = data[random_ch, :][None, :] 
        
        label = int(info['label'])
        # print(label, ' = label')
        
        return torch.from_numpy(data).float(), torch.tensor(label, dtype=torch.long)
    

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

