"""
带跨分支注意力的nnU-Net多任务网络
专注于分割和距离场的交互
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .nnunet_multitask import nnUNetMultiTask


class CrossBranchAttention(nn.Module):
    """
    跨分支注意力模块 - 轻量级实现
    使用通道注意力机制，计算开销小
    """
    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: 输入通道数
            reduction: 通道压缩比例
        """
        super().__init__()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x_source: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        从source分支获取注意力，应用到target分支
        
        Args:
            x_source: 源分支特征 (B, C, D, H, W)
            x_target: 目标分支特征 (B, C, D, H, W)
        
        Returns:
            增强后的target特征
        """
        # 从source计算注意力权重
        attention = self.channel_attention(x_source)
        
        # 应用注意力到target
        attended = x_target * attention
        
        # 残差连接
        out = self.alpha * attended + (1 - self.alpha) * x_target
        
        return out


class SpatialCrossBranchAttention(nn.Module):
    """
    空间跨分支注意力模块 - 更强大但计算开销更大
    """
    def __init__(self, channels: int):
        super().__init__()
        
        self.query_conv = nn.Conv3d(channels, channels // 8, 1)
        self.key_conv = nn.Conv3d(channels, channels // 8, 1)
        self.value_conv = nn.Conv3d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x_source: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_source: 源分支特征 (B, C, D, H, W)
            x_target: 目标分支特征 (B, C, D, H, W)
        """
        B, C, D, H, W = x_source.shape
        
        # Query from target, Key/Value from source
        query = self.query_conv(x_target).view(B, -1, D*H*W).permute(0, 2, 1)  # (B, N, C')
        key = self.key_conv(x_source).view(B, -1, D*H*W)  # (B, C', N)
        value = self.value_conv(x_source).view(B, -1, D*H*W)  # (B, C, N)
        
        # Attention map
        attention = F.softmax(torch.bmm(query, key), dim=-1)  # (B, N, N)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(B, C, D, H, W)
        
        # Residual
        out = self.gamma * out + x_target
        
        return out


class nnUNetMultiTaskWithAttention(nnUNetMultiTask):
    """
    带跨分支注意力的nnU-Net多任务网络
    在分割和距离场分支之间添加注意力机制
    """
    
    def __init__(
        self,
        num_flow_classes: int = 36,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
        use_spatial_attention: bool = False,
        attention_reduction: int = 16,
        random_init: bool = False,
        pretrain_encoder_only: bool = False,
        dual_decoders: bool = False
    ):
        """
        Args:
            num_flow_classes: 流向类别数
            pretrained_path: nnU-Net预训练权重路径
            freeze_encoder: 是否冻结编码器
            use_spatial_attention: 是否使用空间注意力（False则使用通道注意力）
            attention_reduction: 通道注意力的压缩比例
        """
        # 初始化基类
        super().__init__(
            num_flow_classes=num_flow_classes,
            pretrained_path=pretrained_path,
            freeze_encoder=freeze_encoder,
            random_init=random_init,
            pretrain_encoder_only=pretrain_encoder_only,
            dual_decoders=dual_decoders
        )
        
        self.use_spatial_attention = use_spatial_attention
        
        # 获取decoder输出通道数
        decoder_channels = self._get_decoder_out_channels()
        
        # 创建跨分支注意力模块
        if use_spatial_attention:
            # 空间注意力（更强大但更慢）
            self.seg_to_dist_attention = SpatialCrossBranchAttention(decoder_channels)
            self.dist_to_seg_attention = SpatialCrossBranchAttention(decoder_channels)
        else:
            # 通道注意力（轻量级）
            self.seg_to_dist_attention = CrossBranchAttention(decoder_channels, attention_reduction)
            self.dist_to_seg_attention = CrossBranchAttention(decoder_channels, attention_reduction)
        
        print(f"✓ Cross-branch attention initialized ({'spatial' if use_spatial_attention else 'channel'})")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播，包含跨分支注意力
        
        Args:
            x: 输入图像 (B, 1, D, H, W)
        
        Returns:
            pred: {'seg': (B,2,D,H,W), 'dist': (B,1,D,H,W), 'flow': (B,C,D,H,W)}
        """
        # 编码器提取共享特征
        features = self.nnunet.encoder(x)

        # 单/双解码器路径
        if hasattr(self, 'decoder_seg') and hasattr(self, 'decoder_dist'):
            seg_decoded = self.decoder_seg(features)
            dist_decoded = self.decoder_dist(features)
            if isinstance(seg_decoded, (list, tuple)):
                seg_decoded = seg_decoded[-1]
            if isinstance(dist_decoded, (list, tuple)):
                dist_decoded = dist_decoded[-1]

            # 跨分支注意力：分割 <-> 距离场（源->目标）
            seg_features_enhanced = self.dist_to_seg_attention(dist_decoded, seg_decoded)
            dist_features_enhanced = self.seg_to_dist_attention(seg_decoded, dist_decoded)

            # 通过task heads
            pred_seg = self.seg_head(seg_features_enhanced)
            pred_dist = self.dist_head(dist_features_enhanced)
            pred_flow = self.flow_head(seg_decoded)  # 流向复用分割解码器特征
        else:
            decoded = self.nnunet.decoder(features)
            if isinstance(decoded, (list, tuple)):
                decoded = decoded[-1]
            seg_features = decoded
            dist_features = decoded

            # 跨分支注意力：分割 <-> 距离场
            seg_features_enhanced = self.dist_to_seg_attention(dist_features, seg_features)
            dist_features_enhanced = self.seg_to_dist_attention(seg_features, dist_features)

            # heads
            pred_seg = self.seg_head(seg_features_enhanced)
            pred_dist = self.dist_head(dist_features_enhanced)
            pred_flow = self.flow_head(decoded)  # 流向不参与注意力
        
        return {
            'seg': pred_seg,
            'dist': pred_dist,
            'flow': pred_flow
        }


def load_nnunet_multitask_with_attention(
    checkpoint_path: str,
    num_flow_classes: int = 36,
    freeze_encoder: bool = False,
    use_spatial_attention: bool = False,
    attention_reduction: int = 16,
    device: str = 'cuda',
    random_init: bool = False,
    pretrain_encoder_only: bool = False,
    dual_decoders: bool = False
) -> nnUNetMultiTaskWithAttention:
    """
    便捷函数：加载带注意力的多任务网络
    
    Args:
        checkpoint_path: nnU-Net checkpoint路径
        num_flow_classes: 流向类别数
        freeze_encoder: 是否冻结编码器
        use_spatial_attention: 是否使用空间注意力
        attention_reduction: 通道注意力压缩比例
        device: 设备
    
    Returns:
        model: nnUNetMultiTaskWithAttention实例
    """
    model = nnUNetMultiTaskWithAttention(
        num_flow_classes=num_flow_classes,
        pretrained_path=checkpoint_path,
        freeze_encoder=freeze_encoder,
        use_spatial_attention=use_spatial_attention,
        attention_reduction=attention_reduction,
        random_init=random_init,
        pretrain_encoder_only=pretrain_encoder_only,
        dual_decoders=dual_decoders
    )
    
    model = model.to(device)
    
    return model
