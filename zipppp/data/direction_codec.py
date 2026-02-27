"""
方向编解码器：将3D方向向量离散化为类别

关键点：
1. 使用Fibonacci球面采样生成均匀分布的方向
2. 血管流向是无向的，只使用半球（z>=0）
3. 编码：连续向量 -> 最近邻类别
4. 解码：类别 -> 方向向量
"""

import numpy as np
from typing import Tuple, Union


class DirectionCodec:
    """
    方向编解码器
    
    属性:
        directions: np.ndarray, shape (n_classes, 3), 方向模板
        n_classes: int, 类别数
    """
    
    def __init__(self, n_directions: int = 72):
        """
        初始化方向编解码器
        
        Args:
            n_directions: 全球面采样点数，实际使用半球n_directions//2个类别
        """
        # 1. 生成全球面方向
        full_sphere = self.fibonacci_sphere(n_directions)
        
        # 2. 只保留上半球（z >= 0）
        upper_hemisphere = full_sphere[full_sphere[:, 0] >= 0]
        
        # 3. 保存为方向模板
        self.directions = upper_hemisphere
        self.n_classes = len(self.directions)
        
        print(f"DirectionCodec initialized with {self.n_classes} direction classes")
    
    @staticmethod
    def fibonacci_sphere(n_points: int) -> np.ndarray:
        """
        Fibonacci球面均匀采样
        
        Args:
            n_points: 采样点数
            
        Returns:
            directions: shape (n_points, 3), 单位向量 (z, y, x)
        """
        # 黄金角
        phi = np.pi * (3.0 - np.sqrt(5.0))
        
        indices = np.arange(n_points)
        
        # z坐标：从1到-1均匀分布
        z = 1 - (2.0 * indices) / (n_points - 1)
        
        # 计算半径
        radius = np.sqrt(1 - z * z)
        
        # 角度
        theta = phi * indices
        
        # x, y坐标
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        
        # 返回 (z, y, x) 顺序
        directions = np.stack([z, y, x], axis=1)
        
        return directions
    
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        将连续方向向量编码为类别标签
        
        Args:
            vectors: shape (..., 3), 方向向量 (z, y, x)
            
        Returns:
            labels: shape (...), int64类别标签
        """
        original_shape = vectors.shape[:-1]
        vectors = vectors.reshape(-1, 3)
        
        # 1. 归一化输入向量
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)  # 避免除零
        vectors_normalized = vectors / norms
        
        # 2. 如果z<0，翻转向量（保证在上半球）
        flip_mask = vectors_normalized[:, 0] < 0
        vectors_normalized[flip_mask] = -vectors_normalized[flip_mask]
        
        # 3. 计算与所有模板的余弦相似度
        # vectors_normalized: (N, 3), self.directions: (M, 3)
        # 余弦相似度 = dot product (已归一化)
        similarities = np.dot(vectors_normalized, self.directions.T)  # (N, M)
        
        # 4. 返回最大相似度对应的索引
        labels = np.argmax(similarities, axis=1)
        
        return labels.reshape(original_shape).astype(np.int64)
    
    def decode(self, labels: np.ndarray) -> np.ndarray:
        """
        将类别标签解码为方向向量
        
        Args:
            labels: shape (...), 类别标签
            
        Returns:
            vectors: shape (..., 3), 单位方向向量 (z, y, x)
        """
        original_shape = labels.shape
        labels_flat = labels.flatten()
        
        # 直接从模板中查找
        vectors = self.directions[labels_flat]
        
        # 恢复原始形状
        return vectors.reshape(original_shape + (3,))
    
    def save(self, path: str):
        """保存方向模板到文件"""
        np.save(path, self.directions)
        print(f"Direction templates saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'DirectionCodec':
        """从文件加载方向模板"""
        codec = cls.__new__(cls)
        codec.directions = np.load(path)
        codec.n_classes = len(codec.directions)
        print(f"DirectionCodec loaded with {codec.n_classes} classes from {path}")
        return codec


if __name__ == '__main__':
    # 测试代码
    print("Testing DirectionCodec...")
    
    # 创建编解码器
    codec = DirectionCodec(72)
    print(f"Number of classes: {codec.n_classes}")
    
    # 测试编码-解码一致性
    test_vec = np.array([[0.5, 0.5, 0.707]])
    print(f"\nTest vector: {test_vec}")
    
    label = codec.encode(test_vec)
    print(f"Encoded label: {label}")
    
    recovered = codec.decode(label)
    print(f"Decoded vector: {recovered}")
    
    # 计算角度误差
    test_vec_norm = test_vec / np.linalg.norm(test_vec)
    angle_error = np.arccos(np.clip(np.dot(test_vec_norm[0], recovered[0]), -1, 1)) * 180 / np.pi
    print(f"Angle error: {angle_error:.2f}°")
    
    # 测试批量编码
    random_vecs = np.random.randn(10, 3)
    labels = codec.encode(random_vecs)
    recovered_vecs = codec.decode(labels)
    print(f"\nBatch test: {len(labels)} vectors encoded and decoded")
    
    # 测试保存和加载
    codec.save('/tmp/test_directions.npy')
    codec_loaded = DirectionCodec.load('/tmp/test_directions.npy')
    print(f"Loaded codec has {codec_loaded.n_classes} classes")
