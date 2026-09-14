import os
import re
import numpy as np
import scipy.io as sio
import torch
from torch.utils import data
import random
from os.path import join as pjoin
from tqdm import tqdm
from dataset.dataset_TM_train import Text2MotionDataset


import os
import re
import torch
import torch.utils.data as data
import scipy.io as sio
from os.path import join as pjoin

class EEG2MotionDataset(data.Dataset):
    def __init__(self,
                 dataset_name,
                 eeg_roots,  # 改为列表形式，可以接收多个路径
                 eeg_name,
                 tokenizer_name,
                 codebook_size=1024,
                 unit_length=4,
                 up_low_sep=False,
                 eeg_ch='all',  # 新增参数：'all', 'only_motor', 'only_visual', 'wo_motor', 'wo_visual'
                 channels_txt='EEG_data/channels_labels.txt',
                 t2m_train_data_path='EEG_data/HumanML3D_train_processed.pt'):

        print("Loading EEG2MotionDataset...")
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

        if t2m_train_data_path is not None:
            self.motion_dataset = Text2MotionDataset.load(t2m_train_data_path)
        else:
            self.motion_dataset = Text2MotionDataset(
                dataset_name,
                codebook_size=codebook_size,
                tokenizer_name=tokenizer_name,
                unit_length=unit_length,
                up_low_sep=up_low_sep
            )

        # 确保 eeg_roots 是列表
        if isinstance(eeg_roots, str):
            eeg_roots = [eeg_roots]
        self.eeg_roots = eeg_roots
        
        self.samples = []
        
        # ===== 用于存储每个被试的自定义信息 =====
        self.subject_ids = []  # 存储被试ID，用于后续映射
        
        # ===================== 遍历所有文件夹 =====================
        for subj_idx, eeg_root in enumerate(self.eeg_roots):
            print(f"Processing subject {subj_idx}: {eeg_root}")
            
            # ===== 自动检测 block =====
            all_items = os.listdir(eeg_root)
            blocks = [
                d for d in all_items
                if re.fullmatch(r'block_\d+', d)
                and os.path.isdir(os.path.join(eeg_root, d))
            ]
            blocks = sorted(blocks, key=lambda x: int(x.split('_')[1]))
            
            print("Detected blocks:", blocks)
            
            # ===================== 收集当前被试的所有 EEG 数据 =====================
            all_eeg_list = []
            
            # ===== 读取 EEG =====
            for block in blocks:
                print(f'Loading {block}...')
                
                # ---- 读取 index ----
                txt_path = pjoin(eeg_root, f'{block}.txt')
                with open(txt_path, 'r') as f:
                    ids = [line.strip() for line in f.readlines()]
                
                # ---- 读取 EEG ----
                mat_path = pjoin(eeg_root, block, f'{block}'+eeg_name)
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
                
                all_eeg_list.append(eeg_data)
            
            # ===================== 【核心】计算当前被试的全局均值和标准差 =====================
            print(f"\n=== 计算 Subject {subj_idx} 全部 EEG 的全局均值 & 标准差 ===")
            all_eeg = np.concatenate(all_eeg_list, axis=0)
            
            # 形状：(59, 1) → 每个通道一个均值、一个标准差
            # channel_mean = np.mean(all_eeg, axis=(0, 2))[..., None]
            # channel_std = np.std(all_eeg, axis=(0, 2))[..., None]
            channel_mean = np.mean(all_eeg[:, self.selected_channels, :], axis=(0, 2))[..., None]
            channel_std = np.std(all_eeg[:, self.selected_channels, :], axis=(0, 2))[..., None]
            channel_std[channel_std < 1e-6] = 1.0
            
            print(f"被试 {subj_idx} 原始数据形状: {all_eeg.shape}")
            print(f"通道均值 shape: {channel_mean.shape}")
            print(f"第一个通道均值: {channel_mean[0,0]:.4f}")
            print(f"第一个通道标准差: {channel_std[0,0]:.4f}")
            
            # ===================== 保存均值和标准差到文件 =====================
            save_path = pjoin(eeg_root, 'eeg_stats.npz')
            np.savez(save_path, 
                     mean=channel_mean, 
                     std=channel_std,
                     num_samples=all_eeg.shape[0],
                     shape=all_eeg.shape[1:])  # 保存形状信息 (59, 500)
            print(f"统计信息已保存到: {save_path}")
            
            # ===================== 重新构建样本（已归一化） =====================
            print(f"\n=== 构建被试 {subj_idx} 的归一化数据集 ===")
            ptr = 0
            
            for block in blocks:
                txt_path = pjoin(eeg_root, f'{block}.txt')
                with open(txt_path, 'r') as f:
                    ids = [line.strip() for line in f.readlines()]
                
                valid_cnt = 0
                for line in ids:
                    idx = int(line)
                    motion_key = idx
                    
                    # 取出当前eeg，选择特定通道，并做 Z-score 归一化
                    eeg_full = all_eeg[ptr]  # (59, 500)
                    eeg_selected = eeg_full[self.selected_channels, :]  # (num_channels, 500)
                    eeg_norm = (eeg_selected - channel_mean) / channel_std
                    
                    self.samples.append({
                        'motion_key': motion_key,
                        'eeg': eeg_norm,
                        'subj_idx': subj_idx,  # 保存被试索引
                        'subj_root': eeg_root,  # 可选：保存原始路径
                        'channel_indices': self.selected_channels,  # 保存使用的通道索引
                        'channel_names': self.selected_channel_names  # 保存通道名称
                    })
                    valid_cnt += 1
                    ptr += 1
                
                print(f'{block}: valid samples = {valid_cnt}')
            
            print(f"被试 {subj_idx} 总样本数: {ptr}")
        
        print(f"Final dataset size: {len(self.samples)}")
        print(f"Total subjects: {len(self.eeg_roots)}")

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
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        motion_key = sample['motion_key']
        eeg = sample['eeg']           # (59, 500) - 归一化后的数据
        subj_id = sample['subj_idx']  # 被试ID (0, 1, 2, ...)
        
        caption, m_tokens, m_tokens_len = self.motion_dataset[motion_key]
        
        # ===== 转换为 tensor =====
        eeg = torch.tensor(eeg).float()        # (59, 500)
        m_tokens = torch.tensor(m_tokens).long()
        subj_id = torch.tensor(subj_id).long()  # 转换为tensor
        
        return caption, m_tokens, m_tokens_len, eeg, subj_id,motion_key

# ===== DataLoader =====
def EEG2MotionLoader(args, codebook_dir):

    dataset = EEG2MotionDataset(
        args.dataname,
        eeg_root=args.eeg_data_root,
        eeg_name=args.eeg_data_name,
        tokenizer_name=codebook_dir,
        codebook_size=args.nb_code,
        unit_length=2**args.down_t
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )

    return loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class EEG2MotionDatasetWithFeat(torch.utils.data.Dataset):
    def __init__(self,base_dataset, target_feats):
        super().__init__()

        self.base_dataset = base_dataset

        self.target_feats = target_feats

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        clip_text_train, train_motion, train_motion_len, eeg_train,subid,motion_key = self.base_dataset[idx]

        target_feats = self.target_feats[idx]
        
        return clip_text_train, train_motion, train_motion_len, eeg_train,target_feats, subid
    
def EEG2MotionLoaderWithFeat(args, base_dataset,target_feats,batch_size=None):

    dataset = EEG2MotionDatasetWithFeat( base_dataset, target_feats)

    if batch_size is None:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True
        )

    return loader
