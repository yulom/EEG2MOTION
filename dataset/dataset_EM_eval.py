import os
import re
import numpy as np
import scipy.io as sio
import torch
from torch.utils import data
import random
from os.path import join as pjoin
from collections import defaultdict
from utils.word_vectorizer import WordVectorizer
from dataset import dataset_TM_eval
import torch.nn.functional as F

def average_sample_by_key(sample_list):
    """
    对样本列表按 motion_key 分组，对 eeg 求平均
    返回：去重 + 平均后的新样本列表
    EEG2MOTION中未启用
    """
    # key: motion_key, value: [eeg1, eeg2, ...]
    key_eeg_dict = defaultdict(list)

    # 第一步：把相同 key 的 eeg 收集起来
    for item in sample_list:
        key = item['motion_key']
        eeg = item['eeg']
        key_eeg_dict[key].append(eeg)

    # 第二步：对每个 key 求平均
    averaged_sample = []
    for key, eeg_list in key_eeg_dict.items():
        # 转成 numpy 数组求平均（支持任意形状 eeg：一维/二维都可以）
        eeg_avg = np.mean(np.array(eeg_list), axis=0)
        averaged_sample.append({
            'motion_key': key,
            'eeg': eeg_avg,
            'subj_idx': item.get('subj_idx', 0)  # 保留被试索引
        })

    return averaged_sample


class EEG2MotionTestDataset(data.Dataset):
    def __init__(self,
                 dataset_name,
                 eeg_roots,  # 改为列表形式
                 eeg_name,
                 test_avg=False,
                 dataset_type='test',  # test, val, test_and_val
                 device='cuda',
                 eeg_ch='all',  # 新增参数：'all', 'only_motor', 'only_visual', 'wo_motor', 'wo_visual'
                 channels_txt='EEG_data/channels_labels.txt',
                 t2m_test_data_path='EEG_data/HumanML3D_test_processed.pt',
                 t2m_val_data_path='EEG_data/HumanML3D_val_processed.pt'):

        print("Loading EEG2MotionTestDataset...")
        self.eeg_ch = eeg_ch
        self.channels_txt = channels_txt
        
       # ===== 定义脑区电极名称 =====
        self.frontal_areas = ['Fpz','Fp1','Fp2','AF3','AF4','AF7','AF8','Fz','F1','F2','F3','F4','F5','F6','F7','F8','FC1','FC2','FC5','FC6']
        self.motor_areas = ['FCz', 'FC3', 'FC4','FT7','FT8','C1', 'C2', 'C3', 'C4', 'Cz', 'C5', 'C6', 'T7', 'T8','CP1','CP2','CP3','CP4','CP5','CP6']
        self.visual_areas = ['O1', 'Oz', 'O2', 'PO3', 'PO4', 'POz', 'PO5','PO6','PO7','PO8','TP7','TP8','Pz','P3','P4','P5','P6','P7','P8']

        # ===== 加载所有通道名称 =====
        self.all_channel_names = self._load_channel_names(channels_txt)
        print(f"Total channels loaded: {len(self.all_channel_names)}")
        
        # ===== 根据 eeg_ch 参数选择通道索引 =====
        self.selected_channels, self.selected_channel_names = self._select_channels()
        self.num_selected_channels = len(self.selected_channels)
        print(f"Selected {self.num_selected_channels} channels for mode '{eeg_ch}'")
        print(f"Selected channel names: {self.selected_channel_names}")

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        self.dataset_type = dataset_type
        self.device = device
        
        # 确保 eeg_roots 是列表
        if isinstance(eeg_roots, str):
            eeg_roots = [eeg_roots]
        self.eeg_roots = eeg_roots
        
        # 加载 motion dataset（所有被试共享）
        if dataset_type == "test":
            is_test = True
            if t2m_test_data_path is not None:
                self.t2m_dataset_test = dataset_TM_eval.Text2MotionDataset.load(t2m_test_data_path)
            else:
                self.t2m_dataset_test = dataset_TM_eval.Text2MotionDataset(dataset_name, is_test, w_vectorizer, unit_length=4, shuffle=False)
        elif dataset_type == "val":
            is_test = False
            if t2m_val_data_path is not None:
                self.t2m_dataset_val = dataset_TM_eval.Text2MotionDataset.load(t2m_val_data_path)
            else:
                self.t2m_dataset_val = dataset_TM_eval.Text2MotionDataset(dataset_name, is_test, w_vectorizer, unit_length=4, shuffle=False)
        elif dataset_type == "test_and_val":
            is_test = True
            if t2m_test_data_path is not None:
                self.t2m_dataset_test = dataset_TM_eval.Text2MotionDataset.load(t2m_test_data_path)
            else:
                self.t2m_dataset_test = dataset_TM_eval.Text2MotionDataset(dataset_name, is_test, w_vectorizer, unit_length=4, shuffle=False)
            is_test = False
            if t2m_val_data_path is not None:
                self.t2m_dataset_val = dataset_TM_eval.Text2MotionDataset.load(t2m_val_data_path)
            else:
                self.t2m_dataset_val = dataset_TM_eval.Text2MotionDataset(dataset_name, is_test, w_vectorizer, unit_length=4, shuffle=False)
                    
        self.test_avg = test_avg
        
        self.test_ids_range = [786, 1852]
        self.val_ids_range = [252, 584]
        
        # 存储所有被试的样本
        self.samples = []  # 用于 test_avg=True 时的平均样本
        self.motion_to_eegs = defaultdict(list)  # 用于 test_avg=False 时的映射
        self.motion_key_list = []  # 唯一的 motion_key 列表
        
        # ===================== 遍历所有文件夹 =====================
        for subj_idx, eeg_root in enumerate(self.eeg_roots):
            print(f"Processing subject {subj_idx}: {eeg_root}")
            
            # ===== 加载该被试的归一化参数 =====
            stats_path = pjoin(eeg_root, 'eeg_stats.npz')
            if not os.path.exists(stats_path):
                raise FileNotFoundError(f"Statistics file not found: {stats_path}. Please run training dataset first to generate stats.")
            
            stats = np.load(stats_path)
            channel_mean = stats['mean']  # (59, 1)
            channel_std = stats['std']    # (59, 1)
            print(f"Loaded stats from {stats_path}")
            print(f"  Mean shape: {channel_mean.shape}, range: [{channel_mean.min():.3f}, {channel_mean.max():.3f}]")
            print(f"  Std shape: {channel_std.shape}, range: [{channel_std.min():.3f}, {channel_std.max():.3f}]")
            
            # ===== 匹配 block_test_and_val / block_test_and_val_数字 =====
            all_items = os.listdir(eeg_root)
            blocks = [
                d for d in all_items
                if (
                    re.fullmatch(r'block_test_and_val', d) or
                    re.fullmatch(r'block_test_and_val_\d+', d)
                )
                and os.path.isdir(os.path.join(eeg_root, d))
            ]
            
            blocks = sorted(blocks, key=lambda x: (len(x), x))
            print("Detected test blocks:", blocks)
            
            # ===== 读取 EEG =====
            all_ids = []
            all_eeg_data = []
            
            for block in blocks:
                print(f'Loading {block}...')
                
                # ---- 读取 index ----
                txt_path = pjoin(eeg_root, f'{block}.txt')
                with open(txt_path, 'r') as f:
                    ids = [line.strip() for line in f.readlines()]
                
                # ---- 读取 EEG ----
                mat_path = pjoin(eeg_root, block, f'{block}' + eeg_name)
                mat_data = sio.loadmat(mat_path)
                
                eeg_data = None
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        eeg_data = mat_data[key]
                        print(f'Using mat key: {key}')
                        break
                
                # (N, 59, 500)
                assert len(ids) == eeg_data.shape[0], \
                    f"Mismatch: ids={len(ids)}, eeg={eeg_data.shape[0]}"
                
                all_ids.extend(ids)
                if len(all_eeg_data) == 0:
                    all_eeg_data = eeg_data
                else:
                    all_eeg_data = np.concatenate([all_eeg_data, eeg_data], axis=0)
            
            # ===== 根据 dataset_type 和 ids_range 筛选样本 =====
            test_and_val_sample = []
            test_sample = []
            val_sample = []
            
            for i, line in enumerate(all_ids):
                idx = int(line)
                
                # 映射到真实 motion key
                motion_key = idx
                
                # 归一化
                # 取出当前eeg，选择特定通道，并做 Z-score 归一化
                eeg_full = all_eeg_data[i]  # (59, 500)
                eeg_selected = eeg_full[self.selected_channels, :]  # (num_channels, 500)
                eeg_norm = (eeg_selected - channel_mean) / channel_std
                # eeg_norm = (all_eeg_data[i] - channel_mean) / channel_std
                
                # 根据 ID 范围分类
                if idx >= self.test_ids_range[0] and idx <= self.test_ids_range[1]:
                    test_sample.append({
                        'motion_key': motion_key,
                        'eeg': eeg_norm,
                        'subj_idx': subj_idx
                    })
                elif idx >= self.val_ids_range[0] and idx <= self.val_ids_range[1]:
                    val_sample.append({
                        'motion_key': motion_key,
                        'eeg': eeg_norm,
                        'subj_idx': subj_idx
                    })
                else:
                    print(f"Warning: index {idx} not in test or val range")
                
                test_and_val_sample.append({
                    'motion_key': motion_key,
                    'eeg': eeg_norm,
                    'subj_idx': subj_idx
                })
            
            print(f"Subject {subj_idx} statistics:")
            print(f"  Test samples: {len(test_sample)}")
            print(f"  Val samples: {len(val_sample)}")
            print(f"  Total samples: {len(test_and_val_sample)}")
            
            # ===== 根据 test_avg 和 dataset_type 处理当前被试的数据 =====
            if test_avg:
                if dataset_type == "test":
                    current_processed = average_sample_by_key(test_sample)
                elif dataset_type == "val":
                    current_processed = average_sample_by_key(val_sample)
                elif dataset_type == "test_and_val":
                    current_processed = average_sample_by_key(test_and_val_sample)
                else:
                    raise ValueError(f"Unknown dataset_type: {dataset_type}")
                
                # 添加到总样本列表
                self.samples.extend(current_processed)
                
            else:
                # 选择当前数据集对应的原始样本
                if dataset_type == "test":
                    current_samples = test_sample
                elif dataset_type == "val":
                    current_samples = val_sample
                elif dataset_type == "test_and_val":
                    current_samples = test_and_val_sample
                else:
                    raise ValueError(f"Unknown dataset_type: {dataset_type}")
                
                # 填充 motion -> [eeg1, eeg2, ...] 的映射（包含被试信息）
                for item in current_samples:
                    key = item['motion_key']
                    eeg = item['eeg']
                    subj_idx = item['subj_idx']
                    self.motion_to_eegs[key].append({
                        'eeg': eeg,
                        'subj_idx': subj_idx
                    })
        
        # ===== 处理非平均模式 =====
        if not test_avg:
            # 保存唯一的 motion_key 列表
            self.motion_key_list = list(self.motion_to_eegs.keys())
            print(f"\nTotal unique motion keys: {len(self.motion_key_list)}")
        
        # ===== 打印最终统计信息 =====
        print(f"Final dataset summary:")
        if test_avg:
            print(f"  Total averaged samples: {len(self.samples)}")
        else:
            print(f"  Total unique motion keys: {len(self.motion_key_list)}")
            # 统计每个 motion_key 对应的 EEG 数量
            total_eeg_trials = sum(len(eeg_list) for eeg_list in self.motion_to_eegs.values())
            print(f"  Total EEG trials: {total_eeg_trials}")
        print(f"  Total subjects: {len(self.eeg_roots)}")

    def _load_channel_names(self, channels_txt):
        """加载通道名称文件"""
        if not os.path.exists(channels_txt):
            raise FileNotFoundError(f"Channel names file not found: {channels_txt}")
        
        with open(channels_txt, 'r') as f:
            channel_names = [line.strip() for line in f.readlines() if line.strip()]
        
        return channel_names
    
    def _select_channels(self):
        """根据 eeg_ch 参数选择通道索引"""
        # 创建通道名称到索引的映射
        channel_to_idx = {name: idx for idx, name in enumerate(self.all_channel_names)}
        
        # 获取运动区和视觉区的索引
        frontal_indices = []
        for ch in self.frontal_areas:
            if ch in channel_to_idx:
                frontal_indices.append(channel_to_idx[ch])
            else:
                print(f"Warning: Frontal channel '{ch}' not found in channel list")

        motor_indices = []
        for ch in self.motor_areas:
            if ch in channel_to_idx:
                motor_indices.append(channel_to_idx[ch])
            else:
                print(f"Warning: Motor channel '{ch}' not found in channel list")
        
        visual_indices = []
        for ch in self.visual_areas:
            if ch in channel_to_idx:
                visual_indices.append(channel_to_idx[ch])
            else:
                print(f"Warning: Visual channel '{ch}' not found in channel list")
        
        all_indices = set(range(len(self.all_channel_names)))
        frontal_set = set(frontal_indices)
        motor_set = set(motor_indices)
        visual_set = set(visual_indices)
        
        if self.eeg_ch == 'all':
            selected_indices = sorted(all_indices)
            selected_names = [self.all_channel_names[i] for i in selected_indices]

        elif self.eeg_ch == 'only_frontal':
            selected_indices = sorted(frontal_set)
            selected_names = [self.all_channel_names[i] for i in selected_indices]
            if len(selected_indices) == 0:
                raise ValueError("No frontal channels found!")
        
        elif self.eeg_ch == 'only_motor':
            selected_indices = sorted(motor_set)
            selected_names = [self.all_channel_names[i] for i in selected_indices]
            if len(selected_indices) == 0:
                raise ValueError("No motor channels found!")
        
        elif self.eeg_ch == 'only_visual':
            selected_indices = sorted(visual_set)
            selected_names = [self.all_channel_names[i] for i in selected_indices]
            if len(selected_indices) == 0:
                raise ValueError("No visual channels found!")
        
        elif self.eeg_ch == 'wo_frontal':
            selected_indices = sorted(all_indices - frontal_set)
            selected_names = [self.all_channel_names[i] for i in selected_indices]
            print(f"Removed {len(frontal_set)} motor channels")

        elif self.eeg_ch == 'wo_motor':
            selected_indices = sorted(all_indices - motor_set)
            selected_names = [self.all_channel_names[i] for i in selected_indices]
            print(f"Removed {len(motor_set)} motor channels")
        
        elif self.eeg_ch == 'wo_visual':
            selected_indices = sorted(all_indices - visual_set)
            selected_names = [self.all_channel_names[i] for i in selected_indices]
            print(f"Removed {len(visual_set)} visual channels")
        
        else:
            raise ValueError(f"Invalid eeg_ch value: {self.eeg_ch}. "
                           f"Must be one of: 'all','only_frontal', 'only_motor', 'only_visual', "
                           f"'wo_frontal','wo_motor', 'wo_visual'")
        
        print(f"Mode '{self.eeg_ch}': selected {len(selected_indices)} channels")
        return selected_indices, selected_names
    
    def __len__(self):
        if self.test_avg:
            return len(self.samples)
        else:
            return len(self.motion_key_list)

    def __getitem__(self, idx):
        if self.test_avg:
            # 原逻辑：直接取平均后的样本
            sample = self.samples[idx]
            motion_key = sample['motion_key']
            eeg = sample['eeg']
            subj_idx = sample['subj_idx']
        else:
            # 新逻辑：先取 motion_key，再随机选一个对应的 EEG 试次
            motion_key = self.motion_key_list[idx]
            eeg_items = self.motion_to_eegs[motion_key]
            # 随机选择一个 EEG 试次
            rand_idx = random.randint(0, len(eeg_items) - 1)
            eeg = eeg_items[rand_idx]['eeg']
            subj_idx = eeg_items[rand_idx]['subj_idx']
        
        # 获取对应的 motion 数据
        if self.dataset_type == "test" or (self.dataset_type == "test_and_val" and motion_key >= self.test_ids_range[0] and motion_key <= self.test_ids_range[1]):
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = self.t2m_dataset_test[motion_key]
        elif self.dataset_type == "val" or (self.dataset_type == "test_and_val" and motion_key >= self.val_ids_range[0] and motion_key <= self.val_ids_range[1]):
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = self.t2m_dataset_val[motion_key]
        else:
            raise ValueError(f"Motion key {motion_key} not in valid range")
        
        # ===== 转换为 tensor =====
        eeg = torch.tensor(eeg).float()  # (59, 500)
        subj_idx = torch.tensor(subj_idx).long()
        
        return word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, eeg, subj_idx,motion_key

# ===== Loader =====

class EEG2MotionTestDatasetWithFeat(torch.utils.data.Dataset):
    def __init__(self, base_dataset, target_feats):
        super().__init__()

        self.base_dataset = base_dataset
        self.target_feats = target_feats 

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, eeg,subid,motion_key = self.base_dataset[idx]
        
        # 需要填充的 0 的行数
        pad_len = 196 - m_length
        
        # F.pad 填充格式：(左边填0, 右边填0, 上边填0, 下边填0)
        # 只对 第0维（行数）向下填充0
        pose = torch.tensor(pose)
        pose = F.pad(pose, (0, 0, 0, pad_len))

        target_feats = self.target_feats[idx]

        return clip_text, pose, m_length, eeg, target_feats,subid
    
def EEG2MotionLoaderWithFeat(base_dataset,target_feats):

    dataset = EEG2MotionTestDatasetWithFeat( base_dataset,target_feats)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        drop_last=True)

    return loader