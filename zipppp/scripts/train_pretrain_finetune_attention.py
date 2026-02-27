"""
预训练+微调训练脚本 - 使用nnU-Net预训练权重 + 跨分支注意力 + 一致性损失
两阶段训练策略
基于train_pretrain_finetune.py，添加注意力和一致性损失
"""

import os
import sys
import yaml
import argparse
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.nnunet_multitask_attention import load_nnunet_multitask_with_attention
from training.pretrain_finetune_trainer_attention import PretrainFinetuneTrainerWithAttention


def main():
    parser = argparse.ArgumentParser(description='Pretrain + Finetune Training with Attention')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--nnunet_checkpoint', type=str, required=True, 
                        help='nnU-Net预训练checkpoint路径')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称')
    parser.add_argument('--stage1_epochs', type=int, default=None, 
                        help='阶段1 epoch数（冻结编码器）')
    parser.add_argument('--stage2_epochs', type=int, default=None, 
                        help='阶段2 epoch数（解冻编码器），不指定则不进行阶段2')
    parser.add_argument('--freeze_encoder', action='store_true', 
                        help='是否冻结编码器（阶段1）')
    parser.add_argument('--spatial_attention', action='store_true',
                        help='使用空间注意力（默认使用通道注意力）')
    parser.add_argument('--attention_reduction', type=int, default=16,
                        help='通道注意力压缩比例')
    parser.add_argument('--random_init', action='store_true',
                        help='不加载预训练权重，使用随机初始化（仍使用checkpoint的plans构建架构）')
    parser.add_argument('--pretrain_encoder_only', action='store_true',
                        help='仅加载编码器预训练权重，解码器从随机初始化开始')
    parser.add_argument('--dual_decoders', action='store_true',
                        help='启用双解码器：分割与距离各自独立的decoder（共享同一encoder）')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置GPU
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    print(f"Using device: {device}")
    print(f"Loading nnU-Net checkpoint: {args.nnunet_checkpoint}")
    print(f"Attention type: {'Spatial' if args.spatial_attention else 'Channel'}")
    print(f"Attention reduction: {args.attention_reduction}")
    if args.random_init:
        print("[Ablation] Random init enabled: weights will NOT be loaded; only plans are used to build the net.")
    if args.pretrain_encoder_only:
        print("[Option] Pretrain encoder only: decoder will start from random init.")
    if args.dual_decoders:
        print("[Option] Dual decoders enabled: separate decoders for seg & dist.")

    # 将模型结构选项写入配置，便于测试阶段重建相同架构
    config.setdefault('model_options', {})
    config['model_options'].update({
        'random_init': bool(args.random_init),
        'pretrain_encoder_only': bool(args.pretrain_encoder_only),
        'dual_decoders': bool(args.dual_decoders)
    })
    
    # 创建模型（加载nnU-Net预训练权重 + 注意力机制）
    model = load_nnunet_multitask_with_attention(
        checkpoint_path=args.nnunet_checkpoint,
        num_flow_classes=config['model']['num_flow_classes'],
        freeze_encoder=args.freeze_encoder,
        use_spatial_attention=args.spatial_attention,
        attention_reduction=args.attention_reduction,
        device=device,
        random_init=args.random_init,
        pretrain_encoder_only=args.pretrain_encoder_only,
        dual_decoders=args.dual_decoders
    )
    
    if args.random_init:
        print(f"Model created (random init) with nnU-Net architecture + Cross-Branch Attention")
        if args.freeze_encoder:
            print("[Note] You set --freeze_encoder with random init. Consider not freezing the encoder for Stage-1 when training from scratch.")
    else:
        print(f"Model created with pretrained nnU-Net encoder + Cross-Branch Attention")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 创建训练器（带一致性损失）
    trainer = PretrainFinetuneTrainerWithAttention(
        config=config,
        model=model,
        device=device,
        exp_name=args.exp_name,
        freeze_encoder=args.freeze_encoder
    )
    
    # 开始训练
    trainer.train(
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs
    )


if __name__ == '__main__':
    main()
