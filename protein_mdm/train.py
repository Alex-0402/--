"""
训练脚本

使用方法:
    python train.py --pdb_path data/pdb_files --epochs 50 --batch_size 4
    或者:
    python -m train --pdb_path data/pdb_files --epochs 50 --batch_size 4
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径，以支持相对导入
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from models.encoder import BackboneEncoder
from models.decoder import FragmentDecoder
from data.dataset import ProteinStructureDataset, collate_fn
from data.vocabulary import get_vocab
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="训练蛋白质侧链设计模型"
    )
    
    # 数据参数
    parser.add_argument("--pdb_path", type=str, required=True,
                       help="PDB 文件路径或目录（或缓存目录）")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="缓存目录（如果提供，将使用缓存加速加载）")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="验证集比例（如果使用预处理的数据集划分，则忽略此参数）")
    
    # 模型参数
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="隐藏层维度")
    parser.add_argument("--num_encoder_layers", type=int, default=3,
                       help="Encoder 层数")
    parser.add_argument("--num_decoder_layers", type=int, default=3,
                       help="Decoder 层数")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="注意力头数")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=50,
                       help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="权重衰减")
    parser.add_argument("--mask_ratio", type=float, default=0.15,
                       help="掩码比例")
    parser.add_argument("--masking_strategy", type=str, default="random",
                       choices=["random", "block"],
                       help="掩码策略")
    
    # 其他参数
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                       help="保存目录")
    parser.add_argument("--device", type=str, default=None,
                       help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("="*70)
    print("蛋白质侧链设计模型 - 训练")
    print("="*70)
    print(f"设备: {device}")
    print(f"PDB 路径: {args.pdb_path}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"掩码比例: {args.mask_ratio}")
    print(f"掩码策略: {args.masking_strategy}")
    print("="*70)
    
    # 加载数据集
    print("\n1. 加载数据集...")
    dataset = ProteinStructureDataset(
        args.pdb_path,
        cache_dir=args.cache_dir
    )
    print(f"   数据集大小: {len(dataset)}")
    if args.cache_dir:
        print(f"   使用缓存目录: {args.cache_dir}")
    
    # 划分训练/验证集
    if args.val_split > 0:
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        print(f"   训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    else:
        train_dataset = dataset
        val_dataset = None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    # 初始化模型
    print("\n2. 初始化模型...")
    vocab = get_vocab()
    encoder = BackboneEncoder(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_encoder_layers,
        k_neighbors=30
    )
    decoder = FragmentDecoder(
        input_dim=args.hidden_dim,
        vocab_size=vocab.get_vocab_size(),
        num_torsion_bins=72,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_decoder_layers,
        num_heads=args.num_heads
    )
    
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"   Encoder 参数: {encoder_params:,}")
    print(f"   Decoder 参数: {decoder_params:,}")
    print(f"   总参数: {encoder_params + decoder_params:,}")
    
    # 初始化训练器
    print("\n3. 初始化训练器...")
    trainer = Trainer(
        encoder=encoder,
        decoder=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mask_ratio=args.mask_ratio,
        masking_strategy=args.masking_strategy
    )
    
    # 开始训练
    print("\n4. 开始训练...")
    trainer.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        save_every=10
    )
    
    print("\n" + "="*70)
    print("训练完成！")
    print(f"模型保存在: {args.save_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
