"""
PyTorch数据集类
"""

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import random


class PulmonaryArteryDataset(Dataset):
    """
    肺动脉数据集 - 支持38类流向
    
    数据结构：
    data_dir/
        images/
            case_001.nii.gz
        labels_seg/
            case_001.nii.gz
        labels_dist/
            case_001.nii.gz
        labels_flow/
            case_001.nii.gz  (38类：0-35方向, 36端点, 37分支点)
        labels_skeleton/
            case_001.nii.gz
    """
    
    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        num_samples_per_volume: int = 4,
        augment: bool = None  # None表示根据mode自动决定
    ):
        """
        Args:
            data_dir: 数据目录
            mode: 'train' 或 'val'
            patch_size: 裁剪patch大小
            num_samples_per_volume: 每个volume采样次数
        """
        self.data_dir = data_dir
        self.mode = mode
        self.patch_size = patch_size
        self.num_samples = num_samples_per_volume
        # 数据增强开关：None表示根据mode决定，True/False强制开启/关闭
        if augment is None:
            self.augment = (mode == 'train')
        else:
            self.augment = augment
        
        # 加载方向翻转映射表（用于数据增强）
        self.flip_mapping_z = None
        self.flip_mapping_y = None
        self.flip_mapping_x = None
        self.n_direction_classes = 36  # 方向类别数（不包括端点36和分支点37）
        
        flip_z_path = os.path.join(data_dir, 'flip_mapping_z.npy')
        flip_y_path = os.path.join(data_dir, 'flip_mapping_y.npy')
        flip_x_path = os.path.join(data_dir, 'flip_mapping_x.npy')
        
        if os.path.exists(flip_z_path):
            self.flip_mapping_z = np.load(flip_z_path)
            self.flip_mapping_y = np.load(flip_y_path)
            self.flip_mapping_x = np.load(flip_x_path)
            print(f"Loaded flip mappings for data augmentation")
        
        # 获取case列表
        self.cases = self._get_case_list()
        
        print(f"Dataset initialized: {len(self.cases)} cases, mode={mode}")
    
    def _get_case_list(self) -> List[str]:
        """获取所有case ID"""
        image_dir = os.path.join(self.data_dir, 'images')
        if not os.path.exists(image_dir):
            print(f"Warning: {image_dir} does not exist")
            return []
        
        cases = [f.replace('.nii.gz', '') for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
        return sorted(cases)
    
    def __len__(self):
        return len(self.cases) * self.num_samples
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict:
                'image': (1, D, H, W), 归一化后的CT图像
                'gt': dict of GT tensors
        """
        case_idx = idx // self.num_samples
        case_id = self.cases[case_idx]
        
        # 加载数据（38类版本不需要mask）
        image = self._load_nifti(f'images/{case_id}.nii.gz')
        seg = self._load_nifti(f'labels_seg/{case_id}.nii.gz')
        dist = self._load_nifti(f'labels_dist/{case_id}.nii.gz')
        flow = self._load_nifti(f'labels_flow/{case_id}.nii.gz')
        skeleton = self._load_nifti(f'labels_skeleton/{case_id}.nii.gz')
        
        # 随机裁剪patch（以血管为中心）
        if self.mode == 'train':
            image, seg, dist, flow, skeleton = self._random_crop(
                image, seg, dist, flow, skeleton
            )
            # 数据增强（可通过augment参数控制）
            if self.augment:
                image, seg, dist, flow, skeleton = self._augment(
                    image, seg, dist, flow, skeleton
                )
        else:
            # 验证时使用中心裁剪或全图
            image, seg, dist, flow, skeleton = self._center_crop(
                image, seg, dist, flow, skeleton
            )
        
        # 图像归一化
        image = self._normalize(image)
        
        # 转为tensor（38类版本：flow_labels包含0-37）
        return {
            'image': torch.from_numpy(image[None]).float(),
            'gt': {
                'seg': torch.from_numpy(seg).long(),
                'dist': torch.from_numpy(dist).float(),
                'flow_labels': torch.from_numpy(flow).long(),  # 38类：0-35方向, 36端点, 37分支点
                'skeleton': torch.from_numpy(skeleton).float()
            },
            'case_id': case_id
        }
    
    def _load_nifti(self, rel_path: str) -> np.ndarray:
        """加载NIfTI文件"""
        path = os.path.join(self.data_dir, rel_path)
        return nib.load(path).get_fdata().astype(np.float32)
    
    def _random_crop(self, *arrays) -> Tuple[np.ndarray, ...]:
        """随机裁剪（以血管为中心）"""
        seg = arrays[1]  # 分割标注
        
        # 找到血管区域
        vessel_coords = np.argwhere(seg > 0.5)
        
        if len(vessel_coords) == 0:
            # 如果没有血管，使用中心裁剪
            return self._center_crop(*arrays)
        
        # 在血管区域内随机选择中心点
        center_idx = random.randint(0, len(vessel_coords) - 1)
        center = vessel_coords[center_idx]
        
        # 计算裁剪范围
        crops = []
        for i, (c, size) in enumerate(zip(center, self.patch_size)):
            shape = arrays[0].shape[i]
            start = max(0, c - size // 2)
            end = min(shape, start + size)
            start = max(0, end - size)  # 确保patch大小
            crops.append((start, end))
        
        # 裁剪所有数组
        result = []
        for arr in arrays:
            cropped = arr[
                crops[0][0]:crops[0][1],
                crops[1][0]:crops[1][1],
                crops[2][0]:crops[2][1]
            ]
            result.append(cropped)
        
        return tuple(result)
    
    def _center_crop(self, *arrays) -> Tuple[np.ndarray, ...]:
        """中心裁剪"""
        shape = arrays[0].shape
        
        crops = []
        for i, size in enumerate(self.patch_size):
            if shape[i] <= size:
                crops.append((0, shape[i]))
            else:
                start = (shape[i] - size) // 2
                crops.append((start, start + size))
        
        result = []
        for arr in arrays:
            cropped = arr[
                crops[0][0]:crops[0][1],
                crops[1][0]:crops[1][1],
                crops[2][0]:crops[2][1]
            ]
            result.append(cropped)
        
        return tuple(result)
    
    def _augment(self, *arrays) -> Tuple[np.ndarray, ...]:
        """数据增强 - 正确处理方向标签的翻转"""
        image, seg, dist, flow_labels, skeleton = arrays
        
        # 记录翻转操作
        flip_axes = []
        if random.random() > 0.5:
            flip_axes.append(0)  # z轴
        if random.random() > 0.5:
            flip_axes.append(1)  # y轴
        if random.random() > 0.5:
            flip_axes.append(2)  # x轴
        
        if len(flip_axes) == 0:
            return arrays
        
        # 翻转所有数组
        for axis in flip_axes:
            image = np.flip(image, axis=axis).copy()
            seg = np.flip(seg, axis=axis).copy()
            dist = np.flip(dist, axis=axis).copy()
            skeleton = np.flip(skeleton, axis=axis).copy()
            flow_labels = np.flip(flow_labels, axis=axis).copy()
        
        # 关键：更新方向标签以匹配翻转后的方向
        if self.flip_mapping_z is not None:
            flow_labels = self._update_flow_labels_after_flip(flow_labels, seg, flip_axes)
        
        return image, seg, dist, flow_labels, skeleton
    
    def _update_flow_labels_after_flip(
        self, 
        flow_labels: np.ndarray, 
        seg: np.ndarray,
        flip_axes: List[int]
    ) -> np.ndarray:
        """
        翻转后更新方向标签（使用预计算的映射表）
        
        Args:
            flow_labels: 原始方向标签 [D, H, W]
            seg: 分割标签 [D, H, W]
            flip_axes: 翻转的轴 (0=z, 1=y, 2=x)
        
        Returns:
            更新后的方向标签
        """
        # 只处理有方向的点（0-35），端点（36）和分支点（37）不需要更新
        vessel_mask = seg > 0
        direction_mask = (flow_labels < self.n_direction_classes) & vessel_mask
        
        if direction_mask.sum() == 0:
            return flow_labels
        
        # 获取当前方向标签
        current_labels = flow_labels[direction_mask].astype(np.int64)
        
        # 根据翻转轴依次应用映射
        new_labels = current_labels.copy()
        for axis in flip_axes:
            if axis == 0:  # z轴
                new_labels = self.flip_mapping_z[new_labels]
            elif axis == 1:  # y轴
                new_labels = self.flip_mapping_y[new_labels]
            elif axis == 2:  # x轴
                new_labels = self.flip_mapping_x[new_labels]
        
        # 更新flow_labels
        new_flow_labels = flow_labels.copy()
        new_flow_labels[direction_mask] = new_labels
        
        return new_flow_labels
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """CT图像归一化"""
        # 截断到[-1000, 400] HU
        image = np.clip(image, -1000, 600)
        # 归一化到[0, 1]
        image = (image + 1000) / 1600
        return image


if __name__ == '__main__':
    # 测试代码
    print("Testing PulmonaryArteryDataset...")
    
    data_dir = "/home/agr/DUOFENZHI_gujia/data/processed"
    
    if os.path.exists(data_dir):
        dataset = PulmonaryArteryDataset(
            data_dir,
            mode='train',
            patch_size=(64, 64, 64),
            num_samples_per_volume=2
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample shapes:")
            print(f"  Image: {sample['image'].shape}")
            print(f"  Seg: {sample['gt']['seg'].shape}")
            print(f"  Dist: {sample['gt']['dist'].shape}")
            print(f"  Flow labels: {sample['gt']['flow_labels'].shape}")
            print(f"  Flow mask: {sample['gt']['flow_mask'].shape}")
            print(f"  Skeleton: {sample['gt']['skeleton'].shape}")
    else:
        print(f"Data directory {data_dir} does not exist yet")
