"""
训练评价指标计算
包含分割、距离场、流向和骨架的详细指标
"""

import torch
import numpy as np
from typing import Dict, Tuple
from scipy.ndimage import distance_transform_edt


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    计算Dice系数
    
    Args:
        pred: 预测概率 [B, C, D, H, W] 或 [B, D, H, W]
        target: 真值标签 [B, D, H, W]
        smooth: 平滑项
    
    Returns:
        dice: Dice系数
    """
    if pred.dim() == 5:  # [B, C, D, H, W]
        pred = torch.argmax(pred, dim=1)  # [B, D, H, W]
    
    pred = (pred > 0).float()
    target = (target > 0).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    计算IoU (Intersection over Union)
    
    Args:
        pred: 预测概率 [B, C, D, H, W] 或 [B, D, H, W]
        target: 真值标签 [B, D, H, W]
        smooth: 平滑项
    
    Returns:
        iou: IoU分数
    """
    if pred.dim() == 5:
        pred = torch.argmax(pred, dim=1)
    
    pred = (pred > 0).float()
    target = (target > 0).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算像素准确率
    
    Args:
        pred: 预测 [B, C, D, H, W]
        target: 真值 [B, D, H, W]
    
    Returns:
        accuracy: 准确率
    """
    pred_labels = torch.argmax(pred, dim=1)
    correct = (pred_labels == target).float()
    accuracy = correct.mean()
    return accuracy.item()


def compute_precision_recall(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """
    计算精确率和召回率
    
    Args:
        pred: 预测 [B, C, D, H, W]
        target: 真值 [B, D, H, W]
    
    Returns:
        precision, recall
    """
    pred_labels = torch.argmax(pred, dim=1)
    pred_pos = (pred_labels > 0).float()
    target_pos = (target > 0).float()
    
    tp = (pred_pos * target_pos).sum()
    fp = (pred_pos * (1 - target_pos)).sum()
    fn = ((1 - pred_pos) * target_pos).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return precision.item(), recall.item()


def compute_distance_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    """
    计算距离场的平均绝对误差
    
    Args:
        pred: 预测距离场 [B, 1, D, H, W]
        target: 真值距离场 [B, D, H, W]
        mask: 有效区域mask [B, D, H, W]
    
    Returns:
        mae: 平均绝对误差
    """
    if pred.dim() == 5:
        pred = pred.squeeze(1)  # [B, D, H, W]
    
    if mask is not None:
        mask = mask.float()
        error = torch.abs(pred - target) * mask
        mae = error.sum() / (mask.sum() + 1e-8)
    else:
        mae = torch.abs(pred - target).mean()
    
    return mae.item()


def compute_flow_accuracy(pred_logits: torch.Tensor, target_labels: torch.Tensor, mask: torch.Tensor) -> float:
    """
    计算流向分类准确率
    
    Args:
        pred_logits: 预测logits [B, N, D, H, W]
        target_labels: 真值标签 [B, D, H, W]
        mask: 有效区域mask [B, D, H, W]
    
    Returns:
        accuracy: 分类准确率
    """
    pred_labels = torch.argmax(pred_logits, dim=1)  # [B, D, H, W]
    mask = mask.bool()
    
    correct = (pred_labels == target_labels) & mask
    accuracy = correct.sum().float() / (mask.sum().float() + 1e-8)
    
    return accuracy.item()


def compute_flow_angular_error(
    pred_logits: torch.Tensor,
    target_labels: torch.Tensor,
    direction_vectors: np.ndarray,
    mask: torch.Tensor
) -> float:
    """
    计算流向角度误差
    
    Args:
        pred_logits: 预测logits [B, N, D, H, W]
        target_labels: 真值标签 [B, D, H, W]
        direction_vectors: 方向向量 [N, 3]
        mask: 有效区域mask [B, D, H, W]
    
    Returns:
        mean_angular_error: 平均角度误差（度）
    """
    pred_labels = torch.argmax(pred_logits, dim=1).cpu().numpy()  # [B, D, H, W]
    target_labels = target_labels.cpu().numpy()
    mask = mask.cpu().numpy()
    
    # direction_vectors转为numpy（如果是torch tensor）
    if isinstance(direction_vectors, torch.Tensor):
        direction_vectors = direction_vectors.cpu().numpy()
    
    # 获取有效位置（血管区域）
    valid_positions = mask > 0
    
    if valid_positions.sum() == 0:
        return 0.0
    
    # 获取有效位置的标签
    pred_labels_valid = pred_labels[valid_positions]
    target_labels_valid = target_labels[valid_positions]
    
    # 38类版本：只计算有方向的点（0-35），跳过端点（36）和分支点（37）
    n_directions = len(direction_vectors)  # 36
    direction_mask = (target_labels_valid < n_directions) & (pred_labels_valid < n_directions)
    
    if direction_mask.sum() == 0:
        return 0.0
    
    # 获取预测和真值的方向向量
    pred_dirs = direction_vectors[pred_labels_valid[direction_mask]]  # [N_valid, 3]
    target_dirs = direction_vectors[target_labels_valid[direction_mask]]  # [N_valid, 3]
    
    # 计算角度
    dot_products = np.sum(pred_dirs * target_dirs, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles = np.arccos(dot_products) * 180.0 / np.pi
    
    return float(np.mean(angles))


def compute_all_metrics(
    pred: Dict[str, torch.Tensor],
    gt: Dict[str, torch.Tensor],
    direction_vectors: np.ndarray = None
) -> Dict[str, float]:
    """
    计算所有评价指标
    
    Args:
        pred: 预测结果字典
            - 'seg': [B, 2, D, H, W]
            - 'dist': [B, 1, D, H, W]
            - 'flow': [B, N, D, H, W]
        gt: 真值字典
            - 'seg': [B, D, H, W]
            - 'dist': [B, D, H, W]
            - 'flow_labels': [B, D, H, W]
            - 'flow_mask': [B, D, H, W]
            - 'skeleton': [B, D, H, W]
        direction_vectors: 方向向量 [N, 3]
    
    Returns:
        metrics: 指标字典
    """
    metrics = {}
    
    # 分割指标
    metrics['seg_dice'] = compute_dice_score(pred['seg'], gt['seg'])
    metrics['seg_iou'] = compute_iou(pred['seg'], gt['seg'])
    metrics['seg_accuracy'] = compute_accuracy(pred['seg'], gt['seg'])
    precision, recall = compute_precision_recall(pred['seg'], gt['seg'])
    metrics['seg_precision'] = precision
    metrics['seg_recall'] = recall
    metrics['seg_f1'] = 2 * precision * recall / (precision + recall + 1e-8)
    
    # 距离场指标
    metrics['dist_mae'] = compute_distance_mae(pred['dist'], gt['dist'], gt['seg'] > 0)
    
    # 流向指标（38类版本：使用seg mask而不是flow_mask）
    vessel_mask = gt['seg'] > 0
    metrics['flow_accuracy'] = compute_flow_accuracy(pred['flow'], gt['flow_labels'], vessel_mask)
    
    if direction_vectors is not None:
        metrics['flow_angular_error'] = compute_flow_angular_error(
            pred['flow'], gt['flow_labels'], direction_vectors, vessel_mask
        )
    
    return metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    格式化指标输出
    
    Args:
        metrics: 指标字典
        prefix: 前缀（如"Train"或"Val"）
    
    Returns:
        formatted_string: 格式化的字符串
    """
    lines = []
    if prefix:
        lines.append(f"{prefix} Metrics:")
    
    # 分割指标
    lines.append("  Segmentation:")
    lines.append(f"    Dice: {metrics.get('seg_dice', 0):.4f}")
    lines.append(f"    IoU: {metrics.get('seg_iou', 0):.4f}")
    lines.append(f"    Accuracy: {metrics.get('seg_accuracy', 0):.4f}")
    lines.append(f"    Precision: {metrics.get('seg_precision', 0):.4f}")
    lines.append(f"    Recall: {metrics.get('seg_recall', 0):.4f}")
    lines.append(f"    F1: {metrics.get('seg_f1', 0):.4f}")
    
    # 距离场指标
    lines.append("  Distance Field:")
    lines.append(f"    MAE: {metrics.get('dist_mae', 0):.4f}")
    
    # 流向指标
    lines.append("  Flow Direction:")
    lines.append(f"    Accuracy: {metrics.get('flow_accuracy', 0):.4f}")
    if 'flow_angular_error' in metrics:
        lines.append(f"    Angular Error: {metrics.get('flow_angular_error', 0):.2f}°")
    
    return "\n".join(lines)


if __name__ == '__main__':
    # 测试
    print("Testing train_metrics...")
    
    B, C, D, H, W = 2, 2, 32, 32, 32
    N = 36
    
    pred = {
        'seg': torch.randn(B, C, D, H, W).softmax(dim=1),
        'dist': torch.rand(B, 1, D, H, W) * 5,
        'flow': torch.randn(B, N, D, H, W)
    }
    
    gt = {
        'seg': torch.randint(0, 2, (B, D, H, W)),
        'dist': torch.rand(B, D, H, W) * 5,
        'flow_labels': torch.randint(0, N, (B, D, H, W)),
        'flow_mask': torch.rand(B, D, H, W) > 0.5,
        'skeleton': torch.rand(B, D, H, W) > 0.9
    }
    
    # 模拟方向向量
    direction_vectors = np.random.randn(N, 3)
    direction_vectors = direction_vectors / np.linalg.norm(direction_vectors, axis=1, keepdims=True)
    
    metrics = compute_all_metrics(pred, gt, direction_vectors)
    print(format_metrics(metrics, "Test"))
    
    print("\n✓ All metrics computed successfully!")
