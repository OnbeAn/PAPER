"""
预训练+微调训练器 - 带跨分支注意力和一致性损失
继承PretrainFinetuneTrainer，只添加一致性损失，保留所有原有功能
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from typing import Dict, Optional

from .pretrain_finetune_trainer import PretrainFinetuneTrainer
from models.losses_with_consistency import create_loss_with_consistency


class PretrainFinetuneTrainerWithAttention(PretrainFinetuneTrainer):
    """
    预训练+微调训练器 - 带跨分支注意力和一致性损失
    
    继承PretrainFinetuneTrainer的所有功能：
        - 两阶段训练（冻结/解冻编码器）
        - nnU-Net预训练权重
        - 分阶段学习率
    
    新增功能：
        - 跨分支注意力机制（在模型中）
        - 一致性损失（边界、梯度、平滑）
    """
    
    def __init__(
        self,
        config: Dict,
        model: nn.Module,
        device: str = 'cuda:0',
        exp_name: Optional[str] = None,
        freeze_encoder: bool = True
    ):
        """
        Args:
            config: 配置
            model: nnUNetMultiTaskWithAttention模型
            device: 设备
            exp_name: 实验名称
            freeze_encoder: 是否冻结编码器（阶段1）
        """
        # 先不调用父类__init__，因为我们要替换损失函数
        # 手动初始化BaseTrainer的部分
        self.config = config
        self.model = model
        self.device = device
        self.freeze_encoder = freeze_encoder
        self.current_stage = 1
        
        # 创建实验目录
        from datetime import datetime
        if exp_name is None:
            exp_name = f"exp_attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.exp_name = exp_name
        self.exp_dir = os.path.join(config['paths']['experiments_dir'], exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, 'logs'), exist_ok=True)
        
        # TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(os.path.join(self.exp_dir, 'tensorboard'))
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        self.best_val_loss = float('inf')
        
        # 混合精度
        self.use_amp = config['training'].get('use_amp', True)
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        
        # 加载DirectionCodec
        from data.direction_codec import DirectionCodec
        codec_path = os.path.join(config['data']['processed_dir'], 'direction_codec.npy')
        codec = DirectionCodec.load(codec_path)
        self.direction_vectors = codec.directions
        
        # ===== 关键：使用带一致性的损失函数 =====
        print("\n" + "="*60)
        print("Creating Loss with Consistency Constraints")
        print("="*60)
        
        # 计算类别权重（从配置读取，默认更强的背景权重抑制过分割）
        #seg_class_weights_cfg = self.config.get('training', {}).get('seg_class_weights', [1.0, 300.0])
        #seg_class_weights = torch.tensor(seg_class_weights_cfg, device=device)
        
        # 任务损失权重
        task_weights = {
            'seg_dice': config['loss_weights']['seg_dice'],
            'seg_ce': config['loss_weights']['seg_ce'],
            'dist': config['loss_weights']['dist'],
            'flow': config['loss_weights']['flow'],
            'cldice': config['loss_weights']['cldice']
        }
        
        # 一致性损失权重
        consistency_weights = config.get('consistency_weights', {
            'seg_dist_boundary': 0.1,
            'dist_gradient': 0.05,
            'seg_dist_smooth': 0.05
        })
        # 额外损失选项
        dice_ignore_bg = bool(config.get('training', {}).get('dice_ignore_bg', False))
        seg_ce_topk = config.get('training', {}).get('seg_ce_topk', None)
        
        self.criterion = create_loss_with_consistency(
            task_weights=task_weights,
            #seg_class_weights=seg_class_weights,
            consistency_weights=consistency_weights
        )
        
        print(f"✓ Task weights: {task_weights}")
        print(f"✓ Consistency weights: {consistency_weights}")
        print(f"✓ Dice ignore background: {dice_ignore_bg}")
        print(f"✓ Seg CE TopK: {seg_ce_topk}")
        print("="*60)
        
        # 打印训练模式信息
        print("\n" + "=" * 60)
        print("Pretrain + Finetune Training with Attention & Consistency")
        print(f"  - Stage 1: Freeze encoder = {freeze_encoder}")
        print(f"  - Stage 2: Finetune all (optional)")
        print(f"  - Cross-branch attention: Enabled")
        print(f"  - Consistency losses: Enabled")
        print("=" * 60)
    
    def setup_data(self):
        """设置数据加载器（训练与验证使用不同模式与按case划分）"""
        from data.dataset import PulmonaryArteryDataset
        from torch.utils.data import DataLoader, Subset
        import numpy as np
        
        print("\nLoading datasets...")
        # 从配置读取困难负样本采样参数（当前数据集类不支持，忽略）

        # 训练集：随机血管中心裁剪 + 增强 + （可选）困难负样本采样
        train_full = PulmonaryArteryDataset(
            data_dir=self.config['data']['processed_dir'],
            patch_size=self.config['training']['patch_size'],
            num_samples_per_volume=self.config['training']['num_samples_per_volume'],
            mode='train'
        )
        # 验证集：中心裁剪/无增强；每个case仅取1个样本，避免样本级偏置
        val_full = PulmonaryArteryDataset(
            data_dir=self.config['data']['processed_dir'],
            patch_size=self.config['training']['patch_size'],
            num_samples_per_volume=1,
            mode='val'
        )
        
        case_ids = train_full.cases  # 两者应一致
        n_cases = len(case_ids)
        print(f"Total cases: {n_cases}")
        
        # 基于case划分
        cfg_train_cases = self.config['data'].get('train_cases', None)
        cfg_val_cases = self.config['data'].get('val_cases', None)
        if isinstance(cfg_val_cases, int) and cfg_val_cases > 0 and cfg_val_cases < n_cases:
            val_case_indices = list(range(n_cases - cfg_val_cases, n_cases))
            train_case_indices = list(range(0, n_cases - cfg_val_cases))
        else:
            n_train = int(n_cases * 0.8)
            train_case_indices = list(range(n_train))
            val_case_indices = list(range(n_train, n_cases))
        
        # 将case划分映射为样本级索引
        def build_indices_for_dataset(num_samples_per_volume, selected_case_indices):
            idxs = []
            for c in selected_case_indices:
                start = c * num_samples_per_volume
                end = start + num_samples_per_volume
                idxs.extend(list(range(start, end)))
            return idxs
        
        train_indices = build_indices_for_dataset(train_full.num_samples, train_case_indices)
        val_indices = build_indices_for_dataset(val_full.num_samples, val_case_indices)
        
        train_dataset = Subset(train_full, train_indices)
        val_dataset = Subset(val_full, val_indices) if len(val_indices) > 0 else None
        
        print(f"Train cases: {len(train_case_indices)} -> samples: {len(train_dataset)}")
        print(f"Val cases:   {len(val_case_indices)} -> samples: {len(val_dataset) if val_dataset is not None else 0}")
        
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
        ) if val_dataset is not None else None
        
        return train_loader, val_loader
    
    def validate(self, val_loader, epoch):
        """验证函数（继承自BaseTrainer的逻辑）"""
        from torch.cuda.amp import autocast
        from tqdm import tqdm
        from utils.train_metrics import compute_all_metrics
        import numpy as np
        
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False):
                image = batch['image'].to(self.device)
                gt = {k: v.to(self.device) for k, v in batch['gt'].items()}
                
                with autocast(enabled=self.use_amp):
                    pred = self.model(image)
                    loss, loss_dict = self.criterion(pred, gt)
                
                total_loss += loss.item()
                
                metrics = compute_all_metrics(pred, gt, self.direction_vectors)
                all_metrics.append(metrics)
        
        avg_loss = total_loss / len(val_loader)
        avg_metrics = self._average_metrics(all_metrics)
        
        return avg_loss, avg_metrics
    
    def _average_metrics(self, all_metrics):
        """平均指标"""
        import numpy as np
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        return avg_metrics
    
    def log_to_tensorboard(self, metrics, epoch, phase='train'):
        """记录到TensorBoard"""
        for key, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{key}', value, epoch)
    
    def save_checkpoint(self, epoch, optimizer, scheduler, is_best):
        """保存checkpoint"""
        import yaml
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'config': self.config
        }
        
        # 保存最新checkpoint
        latest_path = os.path.join(self.exp_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.exp_dir, 'checkpoints', 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved (val_loss: {self.best_val_loss:.4f})")
        
        # 保存训练历史
        import json
        history_path = os.path.join(self.exp_dir, 'logs', 'train_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        # 保存配置
        config_path = os.path.join(self.exp_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
    
    def close(self):
        """关闭资源"""
        if hasattr(self, 'writer'):
            self.writer.close()
        
        print(f"\n✓ Training logs saved to: {self.exp_dir}")
        print(f"✓ Best model: {os.path.join(self.exp_dir, 'checkpoints', 'best_model.pth')}")
        print(f"✓ TensorBoard: tensorboard --logdir {os.path.join(self.exp_dir, 'tensorboard')}")
