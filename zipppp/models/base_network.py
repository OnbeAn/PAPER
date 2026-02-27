"""
基础网络模块 - 供不同网络架构共用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TaskHead(nn.Module):
    """通用任务头"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        name: str = "task"
    ):
        super().__init__()
        self.name = name
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
