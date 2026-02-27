"""
基于nnU-Net的多任务网络
支持加载预训练的nnU-Net编码器权重
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, Optional
from .base_network import TaskHead


class nnUNetMultiTask(nn.Module):
    """
    基于nnU-Net的多任务网络
    
    架构:
        - 使用nnU-Net的完整网络结构
        - 替换分割头为三个任务头（seg, dist, flow）
        - 支持加载预训练权重
        - 支持冻结编码器
    """
    
    def __init__(
        self,
        num_flow_classes: int = 36,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
        random_init: bool = False,
        pretrain_encoder_only: bool = False,
        dual_decoders: bool = False
    ):
        """
        Args:
            num_flow_classes: 流向类别数（36或38）
            pretrained_path: nnU-Net预训练权重路径
            freeze_encoder: 是否冻结编码器
        """
        super().__init__()
        
        self.num_flow_classes = num_flow_classes
        self.freeze_encoder = freeze_encoder
        self.dual_decoders = bool(dual_decoders)
        
        # 导入nnU-Net网络
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        
        # 如果提供了预训练路径，从中加载plans
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            plans = checkpoint.get('init_args', None)
            
            if plans is None:
                raise ValueError(f"Cannot find 'init_args' in checkpoint: {pretrained_path}")
            
            # 创建nnU-Net网络
            self.nnunet = self._build_nnunet_from_checkpoint(checkpoint)
            
            # 加载编码器和/或解码器权重（若不随机初始化）
            if random_init:
                print("[nnUNetMultiTask] Random initialization enabled: skipping pretrained weight loading.")
            else:
                self._load_pretrained_weights(checkpoint, encoder_only=pretrain_encoder_only)
            
            # 移除nnU-Net的分割头，替换为恒等映射
            # 这样decoder输出就是特征而不是分割结果
            self._remove_segmentation_head()

            # 可选：双解码器（共享同一编码器）
            if self.dual_decoders:
                # 深拷贝当前decoder（已移除seg head）以创建两个独立的解码器
                self.decoder_seg = copy.deepcopy(self.nnunet.decoder)
                self.decoder_dist = copy.deepcopy(self.nnunet.decoder)
                print("Initialized dual decoders: decoder_seg and decoder_dist")
            
        else:
            raise NotImplementedError(
                "Currently only support loading from pretrained nnU-Net checkpoint. "
                "Please provide pretrained_path."
            )
        
        # 获取解码器输出通道数
        # nnU-Net的decoder最后一层输出通道数
        decoder_out_channels = self._get_decoder_out_channels()
        
        # 创建三个任务头
        self.seg_head = TaskHead(decoder_out_channels, 2, name="seg")
        self.dist_head = TaskHead(decoder_out_channels, 1, name="dist")
        self.flow_head = TaskHead(decoder_out_channels, num_flow_classes, name="flow")
        
        # 冻结编码器
        if freeze_encoder:
            self._freeze_encoder()
    
    def _build_nnunet_from_checkpoint(self, checkpoint):
        """从checkpoint构建nnU-Net网络"""
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        
        # 获取初始化参数
        init_args = checkpoint['init_args']
        plans = init_args['plans']
        configuration = init_args['configuration']
        
        # 创建PlansManager
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration)
        
        # 获取输入通道数
        num_input_channels = plans.get('num_input_channels', plans.get('num_modalities', 1))
        
        # 获取网络架构参数
        arch_class_name = configuration_manager.network_arch_class_name
        arch_kwargs = configuration_manager.network_arch_init_kwargs
        arch_kwargs_req_import = configuration_manager.network_arch_init_kwargs_req_import
        
        # 构建网络
        network = get_network_from_plans(
            arch_class_name=arch_class_name,
            arch_kwargs=arch_kwargs,
            arch_kwargs_req_import=arch_kwargs_req_import,
            input_channels=num_input_channels,
            output_channels=2,  # 临时值，我们会替换分割头
            deep_supervision=False  # 不需要deep supervision
        )
        
        return network
    
    def _load_pretrained_weights(self, checkpoint, encoder_only: bool = False):
        """加载预训练权重。
        encoder_only=True 时仅加载 encoder；否则加载 encoder+decoder（跳过seg头）。"""
        pretrained_state_dict = checkpoint['network_weights']
        model_state_dict = self.nnunet.state_dict()
        
        loaded_keys = []
        skipped_keys = []
        
        for key, value in pretrained_state_dict.items():
            # 跳过分割头
            if 'seg_layers' in key or 'segmentation_output' in key:
                skipped_keys.append(key)
                continue
            # 仅加载encoder时跳过decoder相关权重
            if encoder_only and key.startswith('decoder.'):
                skipped_keys.append(key)
                continue
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                model_state_dict[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
        
        self.nnunet.load_state_dict(model_state_dict, strict=False)
        
        scope = 'encoder only' if encoder_only else 'encoder+decoder'
        print(f"Loaded {len(loaded_keys)} keys from pretrained nnU-Net ({scope})")
        print(f"Skipped {len(skipped_keys)} keys (head/decoder skipped or shape mismatch)")
    
    def _remove_segmentation_head(self):
        """移除nnU-Net的分割头，使decoder输出特征"""
        # 将seg_layers替换为恒等映射
        class Identity(nn.Module):
            def forward(self, x):
                return x
        
        # 替换所有seg_layers为恒等映射
        for i in range(len(self.nnunet.decoder.seg_layers)):
            self.nnunet.decoder.seg_layers[i] = Identity()
        
        print("Removed nnU-Net segmentation head")
    
    def _get_decoder_out_channels(self):
        """获取解码器输出通道数"""
        # nnU-Net的decoder最后一层stage的输出通道数
        try:
            # decoder.stages[-1]是最后一个stage
            # 获取最后一个ConvBlock的输出通道数
            last_stage = self.nnunet.decoder.stages[-1]
            if hasattr(last_stage[-1], 'conv'):
                return last_stage[-1].conv.out_channels
            else:
                return last_stage[-1].all_modules[0].out_channels
        except:
            # 如果失败，使用默认值
            return 32
    
    def _freeze_encoder(self):
        """冻结编码器参数"""
        for name, param in self.nnunet.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
        
        print("Encoder frozen!")
    
    def unfreeze_encoder(self):
        """解冻编码器"""
        for name, param in self.nnunet.named_parameters():
            if 'encoder' in name:
                param.requires_grad = True
        
        self.freeze_encoder = False
        print("Encoder unfrozen!")
    
    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [B, 1, D, H, W]
        
        Returns:
            {
                'seg': [B, 2, D, H, W],
                'dist': [B, 1, D, H, W],
                'flow': [B, num_flow_classes, D, H, W]
            }
        """
        # nnU-Net前向传播（encoder + decoder）
        # seg_layers已被替换为恒等映射，所以decoder输出是特征
        features = self.nnunet.encoder(x)

        if getattr(self, 'dual_decoders', False):
            decoded_seg = self.decoder_seg(features)
            decoded_dist = self.decoder_dist(features)
            if isinstance(decoded_seg, (list, tuple)):
                decoded_seg = decoded_seg[-1]
            if isinstance(decoded_dist, (list, tuple)):
                decoded_dist = decoded_dist[-1]
            seg = self.seg_head(decoded_seg)
            dist = self.dist_head(decoded_dist)
            # 流向分支默认复用分割解码器的特征
            flow = self.flow_head(decoded_seg)
        else:
            decoded = self.nnunet.decoder(features)
            # 如果decoder返回列表（deep supervision），取最后一个
            if isinstance(decoded, (list, tuple)):
                decoded = decoded[-1]
            # 三个任务头共用同一解码器特征
            seg = self.seg_head(decoded)
            dist = self.dist_head(decoded)
            flow = self.flow_head(decoded)
        
        return {
            'seg': seg,
            'dist': dist,
            'flow': flow
        }
    
    def get_encoder_parameters(self):
        """获取编码器参数"""
        return [p for name, p in self.nnunet.named_parameters() if 'encoder' in name]
    
    def get_decoder_parameters(self):
        """获取解码器参数"""
        if getattr(self, 'dual_decoders', False):
            return list(self.decoder_seg.parameters()) + list(self.decoder_dist.parameters())
        else:
            return [p for name, p in self.nnunet.named_parameters() if 'decoder' in name]
    
    def get_task_head_parameters(self):
        """获取任务头参数"""
        return list(self.seg_head.parameters()) + \
               list(self.dist_head.parameters()) + \
               list(self.flow_head.parameters())


def load_nnunet_multitask(
    checkpoint_path: str,
    num_flow_classes: int = 36,
    freeze_encoder: bool = False,
    device: str = 'cuda'
) -> nnUNetMultiTask:
    """
    便捷函数：加载基于nnU-Net的多任务网络
    
    Args:
        checkpoint_path: nnU-Net checkpoint路径
        num_flow_classes: 流向类别数
        freeze_encoder: 是否冻结编码器
        device: 设备
    
    Returns:
        model: nnUNetMultiTask实例
    
    Example:
        # 加载预训练nnU-Net，冻结编码器
        model = load_nnunet_multitask(
            checkpoint_path='/path/to/checkpoint_best.pth',
            num_flow_classes=38,
            freeze_encoder=True,
            device='cuda:0'
        )
    """
    model = nnUNetMultiTask(
        num_flow_classes=num_flow_classes,
        pretrained_path=checkpoint_path,
        freeze_encoder=freeze_encoder
    )
    
    model = model.to(device)
    
    return model
