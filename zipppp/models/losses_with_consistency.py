"""
带一致性约束的多任务损失函数
专注于分割和距离场的一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .losses import MultiTaskLoss, DiceLoss


class SegDistConsistencyLoss(nn.Module):
    """
    分割-距离场一致性损失
    约束：分割边界应该对应距离场的零点
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_seg: torch.Tensor, pred_dist: torch.Tensor, gt_seg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_seg: 预测的分割logits (B, 2, D, H, W)
            pred_dist: 预测的距离场 (B, 1, D, H, W)
            gt_seg: GT分割 (B, D, H, W)
        
        Returns:
            loss: 一致性损失
        """
        # 获取预测的分割mask
        pred_seg_binary = (pred_seg.argmax(1) == 1).float()  # (B, D, H, W)
        
        # 计算边界：使用形态学操作
        # 膨胀 - 腐蚀 = 边界
        kernel_size = 3
        padding = kernel_size // 2
        
        # 膨胀
        dilated = F.max_pool3d(
            pred_seg_binary.unsqueeze(1), 
            kernel_size, 
            stride=1, 
            padding=padding
        ).squeeze(1)
        
        # 腐蚀
        eroded = -F.max_pool3d(
            -pred_seg_binary.unsqueeze(1), 
            kernel_size, 
            stride=1, 
            padding=padding
        ).squeeze(1)
        
        # 边界 = 膨胀 - 腐蚀
        boundary = (dilated - eroded).clamp(0, 1)  # (B, D, H, W)
        
        # 距离场在边界处应该接近0
        dist_at_boundary = pred_dist[:, 0] * boundary  # (B, D, H, W)
        
        # L1损失
        loss = dist_at_boundary.abs().mean()
        
        return loss


class DistGradientConsistencyLoss(nn.Module):
    """
    距离场梯度一致性损失
    约束：距离场梯度应该指向最近的边界
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_dist: torch.Tensor, gt_seg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_dist: 预测的距离场 (B, 1, D, H, W)
            gt_seg: GT分割 (B, D, H, W)
        
        Returns:
            loss: 梯度一致性损失
        """
        # 计算距离场梯度
        dist = pred_dist[:, 0]  # (B, D, H, W)
        
        # 使用Sobel算子计算梯度
        grad_d = torch.gradient(dist, dim=(1, 2, 3))  # 3个方向的梯度
        
        # 梯度模长
        grad_norm = torch.sqrt(sum([g**2 for g in grad_d]) + 1e-8)  # (B, D, H, W)
        
        # 在血管内部，梯度模长应该接近1（单位梯度）
        vessel_mask = (gt_seg == 1).float()
        
        # L2损失：鼓励梯度模长为1
        loss = ((grad_norm - 1.0) ** 2 * vessel_mask).sum() / (vessel_mask.sum() + 1e-8)
        
        return loss


class SegDistSmoothConsistencyLoss(nn.Module):
    """
    分割-距离场平滑一致性损失
    约束：距离场应该在血管内部平滑变化
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_dist: torch.Tensor, gt_seg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_dist: 预测的距离场 (B, 1, D, H, W)
            gt_seg: GT分割 (B, D, H, W)
        
        Returns:
            loss: 平滑一致性损失
        """
        dist = pred_dist[:, 0]  # (B, D, H, W)
        
        # 计算二阶导数（拉普拉斯算子）
        # ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
        
        # 使用卷积近似拉普拉斯算子
        laplacian_kernel = torch.tensor([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=dist.dtype, device=dist.device).view(1, 1, 3, 3, 3)
        
        laplacian = F.conv3d(
            dist.unsqueeze(1), 
            laplacian_kernel, 
            padding=1
        ).squeeze(1)  # (B, D, H, W)
        
        # 在血管内部，拉普拉斯应该接近0（平滑）
        vessel_mask = (gt_seg == 1).float()
        
        loss = (laplacian ** 2 * vessel_mask).sum() / (vessel_mask.sum() + 1e-8)
        
        return loss


class MultiTaskLossWithConsistency(MultiTaskLoss):
    """
    带一致性约束的多任务损失
    继承原有的MultiTaskLoss，添加一致性损失
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        seg_class_weights: Optional[torch.Tensor] = None,
        consistency_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            weights: 任务损失权重
            seg_class_weights: 分割类别权重
            consistency_weights: 一致性损失权重
        """
        super().__init__(weights=weights, seg_class_weights=seg_class_weights)
        
        # 一致性损失模块
        self.seg_dist_consistency = SegDistConsistencyLoss()
        self.dist_gradient_consistency = DistGradientConsistencyLoss()
        self.seg_dist_smooth_consistency = SegDistSmoothConsistencyLoss()
        
        # 一致性损失权重
        self.consistency_weights = consistency_weights or {
            'seg_dist_boundary': 0.1,      # 边界一致性
            'dist_gradient': 0.05,          # 梯度一致性
            'seg_dist_smooth': 0.05         # 平滑一致性
        }
        
        print(f"✓ Consistency losses initialized with weights: {self.consistency_weights}")
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor]
    ) -> tuple:
        """
        计算总损失（任务损失 + 一致性损失）
        
        Args:
            pred: 预测结果
            gt: Ground truth
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 基础任务损失
        base_loss, loss_dict = super().forward(pred, gt)
        
        # 一致性损失
        consistency_losses = {}
        
        # 1. 分割-距离场边界一致性
        if self.consistency_weights['seg_dist_boundary'] > 0:
            consistency_losses['seg_dist_boundary'] = self.seg_dist_consistency(
                pred['seg'], 
                pred['dist'], 
                gt['seg']
            )
        
        # 2. 距离场梯度一致性
        if self.consistency_weights['dist_gradient'] > 0:
            consistency_losses['dist_gradient'] = self.dist_gradient_consistency(
                pred['dist'], 
                gt['seg']
            )
        
        # 3. 分割-距离场平滑一致性
        if self.consistency_weights['seg_dist_smooth'] > 0:
            consistency_losses['seg_dist_smooth'] = self.seg_dist_smooth_consistency(
                pred['dist'], 
                gt['seg']
            )
        
        # 加权一致性损失
        if len(consistency_losses) > 0:
            weighted_consistency_loss = sum([
                self.consistency_weights[key] * loss 
                for key, loss in consistency_losses.items()
            ])
        else:
            weighted_consistency_loss = base_loss.new_tensor(0.0)
        
        # 总损失
        total_loss = base_loss + weighted_consistency_loss
        
        # 更新损失字典
        loss_dict.update({
            f'consistency_{key}': loss.item() 
            for key, loss in consistency_losses.items()
        })
        loss_dict['consistency_total'] = weighted_consistency_loss.item()
        
        return total_loss, loss_dict


def create_loss_with_consistency(
    task_weights: Optional[Dict[str, float]] = None,
    seg_class_weights: Optional[torch.Tensor] = None,
    consistency_weights: Optional[Dict[str, float]] = None
) -> MultiTaskLossWithConsistency:
    """
    便捷函数：创建带一致性的损失函数
    
    Args:
        task_weights: 任务损失权重
        seg_class_weights: 分割类别权重
        consistency_weights: 一致性损失权重
    
    Returns:
        loss_fn: 损失函数
    """
    return MultiTaskLossWithConsistency(
        weights=task_weights,
        seg_class_weights=seg_class_weights,
        consistency_weights=consistency_weights
    )
