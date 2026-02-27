"""
多任务损失函数

包含：
1. 分割损失：Dice + CE
2. 距离场损失：L1
3. 流向损失：CE（只在有效区域）
4. clDice损失：中心线Dice，减少断裂
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DiceLoss(nn.Module):
    """Dice损失"""
    
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, C, D, H, W), logits
            target: (B, D, H, W), 类别标签
        """
        # 1. softmax得到概率
        pred_prob = F.softmax(pred, dim=1)
        
        # 2. one-hot编码target
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        
        # 3. 计算每个类别的Dice
        dice_scores = []
        for c in range(num_classes):
            pred_c = pred_prob[:, c]
            target_c = target_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # 4. 返回1 - mean(Dice)
        mean_dice = torch.stack(dice_scores).mean()
        return 1.0 - mean_dice


class SoftSkeletonize(nn.Module):
    """
    可微分的软骨架化
    用于clDice损失
    """
    
    def __init__(self, num_iterations: int = 10):
        super().__init__()
        self.num_iterations = num_iterations
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W), 概率图
            
        Returns:
            skeleton: (B, 1, D, H, W), 软骨架
        """
        # 迭代形态学操作
        for _ in range(self.num_iterations):
            # min_pool = -max_pool(-x)
            min_pool = -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)
            
            # max_pool(min_pool)
            max_min = F.max_pool3d(min_pool, kernel_size=3, stride=1, padding=1)
            
            # x = relu(x - max_pool) + min_pool
            x = F.relu(x - max_min) + min_pool
        
        return x


class CLDiceLoss(nn.Module):
    """
    中心线Dice损失
    强制预测结果保持拓扑连通性
    """
    
    def __init__(self, num_iterations: int = 10, smooth: float = 1e-5):
        super().__init__()
        self.soft_skel = SoftSkeletonize(num_iterations)
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target_skeleton: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, D, H, W), 分割概率
            target_skeleton: (B, 1, D, H, W), GT骨架
        """
        # 1. 对pred进行软骨架化
        pred_skeleton = self.soft_skel(pred)
        
        # 2. 计算预测骨架和GT骨架的Dice
        intersection = (pred_skeleton * target_skeleton).sum()
        union = pred_skeleton.sum() + target_skeleton.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice


class MultiTaskLoss(nn.Module):
    """
    多任务总损失
    """
    
    def __init__(
        self,
        weights: Dict[str, float] = None,
        n_flow_classes: int = 36,
        seg_class_weights: list = None
    ):
        super().__init__()
        
        self.weights = weights or {
            'seg_dice': 1.0,
            'seg_ce': 1.0,
            'dist': 0.5,
            'flow': 0.3,
            'cldice': 0.5
        }
        
        self.dice_loss = DiceLoss()
        self.cldice_loss = CLDiceLoss()
        self.n_flow_classes = n_flow_classes
        
        # 类别权重（背景:血管 = 1:300，因为血管只占0.3%）
        if seg_class_weights is None:
            seg_class_weights = [1.0, 300.0]
        self.seg_class_weights = torch.tensor(seg_class_weights, dtype=torch.float32)
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor]
    ) -> tuple:
        """
        Args:
            pred: 网络输出dict
            gt: GT dict，包含:
                'seg': (B, D, H, W) 分割标签
                'dist': (B, D, H, W) 距离场
                'flow_labels': (B, D, H, W) 流向类别（38类：0-35方向, 36端点, 37分支点）
                'skeleton': (B, D, H, W) 骨架
                
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失字典
        """
        losses = {}
        
        # 1. 分割Dice损失
        losses['seg_dice'] = self.dice_loss(pred['seg'], gt['seg'])
        
        # 2. 分割CE损失（使用类别权重处理类别不平衡）
        losses['seg_ce'] = F.cross_entropy(
            pred['seg'], 
            gt['seg'].long(),
            weight=self.seg_class_weights.to(pred['seg'].device)
        )
        
        # 3. 距离场L1损失（只在血管内）
        vessel_mask = gt['seg'] > 0.5
        if vessel_mask.sum() > 0:
            losses['dist'] = F.l1_loss(
                pred['dist'].squeeze(1)[vessel_mask],
                gt['dist'][vessel_mask]
            )
        else:
            losses['dist'] = torch.tensor(0.0, device=pred['dist'].device)
        
        # 4. 流向CE损失（38类版本：在血管区域计算）
        vessel_mask = gt['seg'] > 0.5
        if vessel_mask.sum() > 0:
            # pred['flow']: (B, N, D, H, W) -> 调整维度
            pred_flow = pred['flow'].permute(0, 2, 3, 4, 1)  # (B, D, H, W, N)
            pred_flow_masked = pred_flow[vessel_mask]  # (M, N)
            gt_flow_masked = gt['flow_labels'][vessel_mask]  # (M,)
            losses['flow'] = F.cross_entropy(pred_flow_masked, gt_flow_masked.long())
        else:
            losses['flow'] = torch.tensor(0.0, device=pred['flow'].device)
        
        # 5. clDice损失
        pred_seg_prob = F.softmax(pred['seg'], dim=1)[:, 1:2]  # 取血管类概率
        gt_skeleton = gt['skeleton'].unsqueeze(1).float()
        losses['cldice'] = self.cldice_loss(pred_seg_prob, gt_skeleton)
        
        # 加权求和
        total_loss = sum(
            self.weights.get(k, 0) * v 
            for k, v in losses.items()
        )
        
        return total_loss, losses


if __name__ == '__main__':
    # 测试代码
    print("Testing loss functions...")
    
    # 创建模拟数据
    B, D, H, W = 2, 32, 32, 32
    N = 36
    
    pred = {
        'seg': torch.randn(B, 2, D, H, W),
        'dist': torch.rand(B, 1, D, H, W) * 10,
        'flow': torch.randn(B, N, D, H, W)
    }
    
    gt = {
        'seg': torch.randint(0, 2, (B, D, H, W)),
        'dist': torch.rand(B, D, H, W) * 10,
        'flow_labels': torch.randint(0, N, (B, D, H, W)),
        'flow_mask': torch.rand(B, D, H, W) > 0.5,
        'skeleton': torch.rand(B, D, H, W) > 0.9
    }
    
    # 创建损失函数
    criterion = MultiTaskLoss(n_flow_classes=N)
    
    # 计算损失
    total_loss, losses = criterion(pred, gt)
    
    print(f"\nLoss values:")
    print(f"  Total loss: {total_loss.item():.4f}")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
