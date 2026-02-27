"""
基础训练器 - 封装共用的训练逻辑
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Optional, Tuple

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)  # 优先使用项目根目录

from data.dataset import PulmonaryArteryDataset
from data.direction_codec import DirectionCodec
from models.losses import MultiTaskLoss
from utils.train_metrics import compute_all_metrics, format_metrics


class BaseTrainer:
    """
    基础训练器 - 封装数据加载、日志、验证等共用逻辑
    """
    
    def __init__(
        self,
        config: Dict,
        model: nn.Module,
        device: str = 'cuda:0',
        exp_name: Optional[str] = None
    ):
        """
        Args:
            config: 配置字典
            model: 网络模型
            device: 设备
            exp_name: 实验名称（用于日志目录）
        """
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # 实验名称
        if exp_name is None:
            exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_name = exp_name
        
        # 为每个实验创建独立的根目录
        exp_root = os.path.join(config['paths']['experiments_dir'], exp_name)
        
        # 创建子目录结构
        self.exp_root = exp_root
        self.checkpoint_dir = os.path.join(exp_root, 'checkpoints')
        self.log_dir = os.path.join(exp_root, 'logs')
        self.tensorboard_dir = os.path.join(exp_root, 'tensorboard')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"  Root: {exp_root}")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Logs: {self.log_dir}")
        print(f"  TensorBoard: {self.tensorboard_dir}")
        print(f"{'='*60}\n")
        
        # TensorBoard
        self.writer = SummaryWriter(self.tensorboard_dir)
        
        # 保存配置到实验根目录和logs目录
        import yaml
        config_path = os.path.join(self.exp_root, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # 同时保存到logs目录（向后兼容）
        with open(os.path.join(self.log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
        
        # 创建README记录实验信息
        readme_path = os.path.join(self.exp_root, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(f"# Experiment: {exp_name}\n\n")
            f.write(f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Directory Structure\n\n")
            f.write(f"```\n")
            f.write(f"{exp_name}/\n")
            f.write(f"├── checkpoints/       # Model checkpoints\n")
            f.write(f"├── logs/             # Training logs and metrics\n")
            f.write(f"├── tensorboard/      # TensorBoard logs\n")
            f.write(f"├── config.yaml       # Training configuration\n")
            f.write(f"└── README.md         # This file\n")
            f.write(f"```\n\n")
            f.write(f"## Configuration\n\n")
            f.write(f"See `config.yaml` for full configuration.\n\n")
            f.write(f"## Notes\n\n")
            f.write(f"Add your experiment notes here...\n")
        
        # 方向编码器
        codec_path = os.path.join(
            config['paths']['processed_dir'],
            'direction_codec.npy'
        )
        self.direction_codec = DirectionCodec.load(codec_path)
        self.direction_vectors = torch.from_numpy(
            self.direction_codec.directions
        ).float().to(device)
        
        # 损失函数
        self.criterion = MultiTaskLoss(
            weights=config['loss_weights'],
            n_flow_classes=config['model']['num_flow_classes']
        ).to(device)
        
        # 混合精度
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # 最佳验证loss
        self.best_val_loss = float('inf')
        
        print(f"Trainer initialized: {exp_name}")
        print(f"  - Checkpoint dir: {self.checkpoint_dir}")
        print(f"  - Log dir: {self.log_dir}")
        print(f"  - TensorBoard dir: {self.tensorboard_dir}")
    
    def setup_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        设置数据加载器
        
        Returns:
            train_loader, val_loader
        """
        # 创建数据集
        full_dataset = PulmonaryArteryDataset(
            data_dir=self.config['paths']['processed_dir'],
            patch_size=self.config['training']['patch_size'],
            num_samples_per_volume=self.config['training']['num_samples_per_volume']
        )
        
        # 划分训练/验证集
        train_cases = self.config['data'].get('train_cases', None)
        val_cases = self.config['data'].get('val_cases', None)
        random_seed = self.config['data'].get('random_seed', 42)
        
        train_dataset, val_dataset = self._split_dataset(
            full_dataset,
            train_cases=train_cases,
            val_cases=val_cases,
            random_seed=random_seed
        )
        
        print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _split_dataset(
        self,
        dataset,
        train_cases=None,
        val_cases=None,
        random_seed=42
    ):
        """划分数据集"""
        n_total = len(dataset)
        
        if train_cases is not None and val_cases is not None:
            # 指定划分
            train_indices = list(range(train_cases))
            val_indices = list(range(n_total - val_cases, n_total))
        else:
            # 随机划分
            np.random.seed(random_seed)
            indices = np.arange(n_total)
            np.random.shuffle(indices)
            
            train_ratio = 0.8
            n_train = int(n_total * train_ratio)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        return train_dataset, val_dataset
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, Dict]:
        """
        验证
        
        Args:
            val_loader: 验证数据加载器
            epoch: 当前epoch
        
        Returns:
            avg_loss, avg_metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}", leave=False):
                # 数据移到设备
                image = batch['image'].to(self.device)
                gt = {k: v.to(self.device) for k, v in batch['gt'].items()}
                
                # 前向传播
                with autocast(enabled=self.use_amp):
                    pred = self.model(image)
                    loss, loss_dict = self.criterion(pred, gt)
                
                total_loss += loss.item()
                
                # 计算指标
                metrics = compute_all_metrics(
                    pred, gt, self.direction_vectors
                )
                all_metrics.append(metrics)
        
        # 平均loss和metrics
        avg_loss = total_loss / len(val_loader)
        avg_metrics = self._average_metrics(all_metrics)
        
        return avg_loss, avg_metrics
    
    def _average_metrics(self, metrics_list):
        """平均多个batch的指标"""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        is_best: bool = False
    ):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # 保存常规checkpoint
        if epoch % self.config['training']['save_interval'] == 0:
            path = os.path.join(
                self.checkpoint_dir,
                f'model_epoch_{epoch}.pth'
            )
            torch.save(checkpoint, path)
            print(f"Saved checkpoint: {path}")
        
        # 保存最佳模型
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, path)
            print(f"Saved best model: {path}")
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.log_dir, 'train_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
    
    def log_to_tensorboard(
        self,
        metrics: Dict,
        epoch: int,
        phase: str = 'train'
    ):
        """记录到TensorBoard"""
        for key, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{key}', value, epoch)
    
    def close(self):
        """关闭训练器"""
        self.writer.close()
        self.save_training_history()
        print("Trainer closed.")
