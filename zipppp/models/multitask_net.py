"""
多任务网络：共享编码器 + 三个解码器

输出：
1. 分割：2通道（背景+血管）
2. 距离场：1通道
3. 流向分类：N通道（N=方向类别数）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class ConvBlock3D(nn.Module):
    """3D卷积块：Conv + Norm + ReLU"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    """
    共享编码器（类似nnUNet编码器）
    
    结构：逐层下采样，通道数递增
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_stages: int = 5
    ):
        """
        Args:
            in_channels: 输入通道数
            base_channels: 基础通道数
            num_stages: 编码器层数
        """
        super().__init__()
        
        self.num_stages = num_stages
        self.stages = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(num_stages):
            out_channels = base_channels * (2 ** i)
            
            # 每个stage：两个ConvBlock
            stage = nn.Sequential(
                ConvBlock3D(current_channels, out_channels),
                ConvBlock3D(out_channels, out_channels)
            )
            self.stages.append(stage)
            
            # 下采样（除了最后一层）
            if i < num_stages - 1:
                self.downsample.append(nn.MaxPool3d(2))
            
            current_channels = out_channels
    
    def forward(self, x) -> List[torch.Tensor]:
        """
        Returns:
            features: 各层特征列表，用于skip connection
        """
        features = []
        
        for i in range(self.num_stages):
            x = self.stages[i](x)
            features.append(x)
            
            if i < self.num_stages - 1:
                x = self.downsample[i](x)
        
        return features


class Decoder(nn.Module):
    """
    解码器
    
    结构：逐层上采样 + skip connection
    """
    
    def __init__(
        self,
        encoder_channels: List[int],
        out_channels: int,
        deep_supervision: bool = False
    ):
        """
        Args:
            encoder_channels: 编码器各层通道数 [32, 64, 128, 256, 512]
            out_channels: 最终输出通道数
            deep_supervision: 是否使用深监督
        """
        super().__init__()
        
        self.deep_supervision = deep_supervision
        self.num_stages = len(encoder_channels) - 1
        
        self.upsamples = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        
        # 从深层到浅层
        for i in range(self.num_stages):
            # 当前层和skip connection层的通道数
            deep_channels = encoder_channels[-(i+1)]  # 从最深层开始
            shallow_channels = encoder_channels[-(i+2)]
            
            # 上采样
            self.upsamples.append(
                nn.ConvTranspose3d(
                    deep_channels,
                    shallow_channels,
                    kernel_size=2,
                    stride=2
                )
            )
            
            # Concat后的卷积
            self.conv_blocks.append(
                nn.Sequential(
                    ConvBlock3D(shallow_channels * 2, shallow_channels),
                    ConvBlock3D(shallow_channels, shallow_channels)
                )
            )
        
        # 最终输出层
        self.output_conv = nn.Conv3d(
            encoder_channels[0],
            out_channels,
            kernel_size=1
        )
    
    def forward(
        self,
        encoder_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            encoder_features: 编码器各层特征 [浅->深]
            
        Returns:
            output: shape (B, out_channels, D, H, W)
        """
        # 从最深层开始
        x = encoder_features[-1]
        
        # 逐层上采样
        for i in range(self.num_stages):
            # 上采样
            x = self.upsamples[i](x)
            
            # Skip connection
            skip = encoder_features[-(i+2)]
            x = torch.cat([x, skip], dim=1)
            
            # 卷积
            x = self.conv_blocks[i](x)
        
        # 最终输出
        output = self.output_conv(x)
        
        return output


class MultiTaskNetwork(nn.Module):
    """
    多任务网络主类
    
    结构：
    - 1个共享编码器
    - 3个独立解码器（分割、距离场、流向）
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_stages: int = 5,
        n_flow_classes: int = 36
    ):
        super().__init__()
        
        self.encoder = Encoder(in_channels, base_channels, num_stages)
        
        # 计算编码器通道数
        encoder_channels = [base_channels * (2**i) for i in range(num_stages)]
        
        # 三个解码器
        self.decoder_seg = Decoder(encoder_channels, out_channels=2)
        self.decoder_dist = Decoder(encoder_channels, out_channels=1)
        self.decoder_flow = Decoder(encoder_channels, out_channels=n_flow_classes)
        
        self.n_flow_classes = n_flow_classes
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: shape (B, 1, D, H, W), CT图像
            
        Returns:
            dict:
                'seg': (B, 2, D, H, W), 分割logits
                'dist': (B, 1, D, H, W), 距离场
                'flow': (B, N, D, H, W), 流向logits
        """
        # 共享编码
        encoder_features = self.encoder(x)
        
        # 多任务解码
        seg = self.decoder_seg(encoder_features)
        dist = self.decoder_dist(encoder_features)
        flow = self.decoder_flow(encoder_features)
        
        # 距离场使用ReLU保证非负
        dist = F.relu(dist)
        
        return {
            'seg': seg,
            'dist': dist,
            'flow': flow
        }


if __name__ == '__main__':
    # 测试代码
    print("Testing MultiTaskNetwork...")
    
    # 创建模型
    model = MultiTaskNetwork(n_flow_classes=36)
    
    # 测试前向传播
    x = torch.randn(1, 1, 64, 64, 64)
    print(f"Input shape: {x.shape}")
    
    out = model(x)
    print(f"\nOutput shapes:")
    print(f"  Segmentation: {out['seg'].shape}")
    print(f"  Distance: {out['dist'].shape}")
    print(f"  Flow: {out['flow'].shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
