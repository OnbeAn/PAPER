"""
预训练+微调训练器 - 两阶段训练策略
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from typing import Dict, Optional

from .base_trainer import BaseTrainer
from utils.train_metrics import compute_all_metrics


class PretrainFinetuneTrainer(BaseTrainer):
    """
    预训练+微调训练器
    
    两阶段训练:
        阶段1: 使用预训练的nnU-Net编码器，冻结编码器，只训练解码器和任务头
        阶段2 (可选): 解冻编码器，微调整个网络
    
    特点:
        - 利用nnU-Net的预训练权重
        - 编码器可冻结/解冻
        - 支持分阶段学习率
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
            model: nnUNetMultiTask模型
            device: 设备
            exp_name: 实验名称
            freeze_encoder: 是否冻结编码器（阶段1）
        """
        super().__init__(config, model, device, exp_name)
        
        self.freeze_encoder = freeze_encoder
        self.current_stage = 1  # 当前训练阶段
        
        print("=" * 60)
        print("Pretrain + Finetune Training Mode")
        print(f"  - Stage 1: Freeze encoder = {freeze_encoder}")
        print("  - Stage 2: Finetune all (optional)")
        print("=" * 60)
    
    def train(self, stage1_epochs: Optional[int] = None, stage2_epochs: Optional[int] = None):
        """
        两阶段训练
        
        Args:
            stage1_epochs: 阶段1的epoch数（冻结编码器）
            stage2_epochs: 阶段2的epoch数（解冻编码器），如果为None则不进行阶段2
        """
        # 设置数据
        train_loader, val_loader = self.setup_data()
        
        # 阶段1: 冻结编码器
        if stage1_epochs is None:
            stage1_epochs = self.config['training']['max_epochs']
        
        print(f"\n{'='*60}")
        print(f"Stage 1: Training with frozen encoder ({stage1_epochs} epochs)")
        print(f"{'='*60}")
        
        self._train_stage(
            train_loader,
            val_loader,
            num_epochs=stage1_epochs,
            freeze_encoder=True,
            stage=1
        )
        
        # 阶段2: 解冻编码器（可选）
        if stage2_epochs is not None and stage2_epochs > 0:
            print(f"\n{'='*60}")
            print(f"Stage 2: Finetuning all parameters ({stage2_epochs} epochs)")
            print(f"{'='*60}")
            
            # 解冻编码器
            if hasattr(self.model, 'unfreeze_encoder'):
                self.model.unfreeze_encoder()
            
            self.current_stage = 2
            
            self._train_stage(
                train_loader,
                val_loader,
                num_epochs=stage2_epochs,
                freeze_encoder=False,
                stage=2
            )
        
        # 训练结束
        self.close()
        print("\nTraining completed!")
    
    def _train_stage(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        freeze_encoder: bool,
        stage: int
    ):
        """训练一个阶段"""
        # 设置优化器
        if freeze_encoder:
            # 只优化解码器和任务头
            if hasattr(self.model, 'get_decoder_parameters'):
                params = (
                    self.model.get_decoder_parameters() +
                    self.model.get_task_head_parameters()
                )
            else:
                params = [p for p in self.model.parameters() if p.requires_grad]
            
            lr = self.config['training']['learning_rate']
        else:
            # 优化所有参数，编码器用较小学习率
            if hasattr(self.model, 'get_encoder_parameters'):
                encoder_params = self.model.get_encoder_parameters()
                other_params = (
                    self.model.get_decoder_parameters() +
                    self.model.get_task_head_parameters()
                )
                
                params = [
                    {'params': encoder_params, 'lr': self.config['training']['learning_rate'] * 0.1},
                    {'params': other_params, 'lr': self.config['training']['learning_rate']}
                ]
            else:
                params = self.model.parameters()
            
            lr = self.config['training']['learning_rate']
        
        optimizer = torch.optim.Adam(params, lr=lr)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs
        )
        
        # 训练循环
        val_interval = self.config['training']['val_interval']
        start_epoch = len(self.train_history['train_loss']) + 1
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            print(f"\n{'='*60}")
            print(f"Stage {stage} - Epoch {epoch - start_epoch + 1}/{num_epochs} (Global: {epoch})")
            print(f"{'='*60}")
            
            # 训练一个epoch
            train_loss, train_metrics = self._train_epoch(
                train_loader, optimizer, epoch
            )
            
            # 学习率调度
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_metrics'].append(train_metrics)
            self.train_history['learning_rates'].append(current_lr)
            
            self.log_to_tensorboard(
                {'loss': train_loss, **train_metrics},
                epoch,
                phase=f'train_stage{stage}'
            )
            self.writer.add_scalar(f'learning_rate_stage{stage}', current_lr, epoch)
            
            # 验证
            if epoch % val_interval == 0:
                val_loss, val_metrics = self.validate(val_loader, epoch)
                
                self.train_history['val_loss'].append(val_loss)
                self.train_history['val_metrics'].append(val_metrics)
                
                self.log_to_tensorboard(
                    {'loss': val_loss, **val_metrics},
                    epoch,
                    phase=f'val_stage{stage}'
                )
                
                # 保存最佳模型
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                self.save_checkpoint(epoch, optimizer, scheduler, is_best)
                
                print(f"\nValidation - Loss: {val_loss:.4f}")
                print(f"  Seg Dice: {val_metrics.get('seg_dice', 0):.4f}")
                print(f"  Best Val Loss: {self.best_val_loss:.4f}")
            
            # 定期保存
            if epoch % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch, optimizer, scheduler, False)
    
    def _train_epoch(self, train_loader, optimizer, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_metrics = []
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            image = batch['image'].to(self.device)
            gt = {k: v.to(self.device) for k, v in batch['gt'].items()}
            
            # 前向传播
            optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                pred = self.model(image)
                loss, loss_dict = self.criterion(pred, gt)
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # 记录
            total_loss += loss.item()
            
            # 计算指标
            with torch.no_grad():
                metrics = compute_all_metrics(
                    pred, gt, self.direction_vectors
                )
                all_metrics.append(metrics)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{metrics.get('seg_dice', 0):.4f}"
            })
        
        # 平均
        avg_loss = total_loss / len(train_loader)
        avg_metrics = self._average_metrics(all_metrics)
        
        return avg_loss, avg_metrics
