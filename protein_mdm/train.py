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

import matplotlib
# 强制使用无窗口后端
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

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


def load_train_val_splits(cache_dir: str):
    """
    从缓存目录加载预定义的训练集和验证集
    
    Args:
        cache_dir: 缓存目录路径
        
    Returns:
        (train_paths, val_paths) 元组，如果文件不存在则返回 (None, None)
    """
    train_file = os.path.join(cache_dir, 'train.txt')
    val_file = os.path.join(cache_dir, 'val.txt')
    
    train_paths = None
    val_paths = None
    
    # 加载训练集
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            train_files = [line.strip() for line in f if line.strip()]
        # 构建完整的缓存文件路径
        train_paths = [
            os.path.join(cache_dir, f"{name}.pt")
            for name in train_files
            if os.path.exists(os.path.join(cache_dir, f"{name}.pt"))
        ]
    
    # 加载验证集
    if os.path.exists(val_file):
        with open(val_file, 'r') as f:
            val_files = [line.strip() for line in f if line.strip()]
        # 构建完整的缓存文件路径
        val_paths = [
            os.path.join(cache_dir, f"{name}.pt")
            for name in val_files
            if os.path.exists(os.path.join(cache_dir, f"{name}.pt"))
        ]
    
    return train_paths, val_paths


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
                       help="验证集比例（如果使用预定义划分，则忽略此参数）")
    parser.add_argument("--use_predefined_split", action="store_true",
                       help="使用预定义的 train.txt 和 val.txt 划分（如果存在）")
    
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
    parser.add_argument("--epochs", type=int, default=300,
                       help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                       help="学习率（最大学习率）")
    parser.add_argument("--warmup_epochs", type=int, default=20,
                       help="Warmup 轮数")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="权重衰减")
    parser.add_argument("--masking_strategy", type=str, default="random",
                       choices=["random", "block"],
                       help="掩码策略")
    parser.add_argument("--num_diffusion_steps", type=int, default=1000,
                       help="扩散模型的时间步数（默认1000）")
    
    # 其他参数
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                       help="保存目录")
    parser.add_argument("--device", type=str, default=None,
                       help="设备 (cuda/cpu)")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="启用训练可视化（默认启用）")
    parser.add_argument("--no_visualize", dest="visualize", action="store_false",
                       help="禁用训练可视化")
    parser.add_argument("--plot_every", type=int, default=5,
                       help="每 N 个 epoch 绘制一次图表（默认 5）")
    parser.add_argument("--resume", type=str, default=None,
                       help="从检查点恢复训练（提供checkpoint路径，如 checkpoints/best_model.pt）")
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                       help="早停耐心值，验证损失连续N轮不下降则停止训练（None表示禁用）")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0,
                       help="早停最小改进阈值，只有改进超过此值才认为是有效改进")
    
    args = parser.parse_args()
    
    # 单GPU模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("蛋白质侧链设计模型 - 训练")
    print("="*70)
    print(f"设备: {device}")
    print(f"PDB 路径: {args.pdb_path}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"扩散模型: 启用 (时间步数: {args.num_diffusion_steps})")
    print(f"掩码比例: 动态 (Cosine Schedule, t=0时0%, t=1时100%)")
    print(f"掩码策略: {args.masking_strategy}")
    print("="*70)
    
    # 加载数据集
    print("\n1. 加载数据集...")
    
    # 尝试使用预定义的划分
    # 如果指定了 --use_predefined_split，或者 cache_dir 存在且包含 train.txt 和 val.txt，则使用预定义划分
    use_predefined = args.use_predefined_split
    train_paths = None
    val_paths = None
    
    if args.cache_dir:
        train_paths, val_paths = load_train_val_splits(args.cache_dir)
        if train_paths is not None and val_paths is not None and len(train_paths) > 0 and len(val_paths) > 0:
            # 如果找到了预定义划分，自动使用（除非明确指定不使用）
            if not args.use_predefined_split:
                # 自动检测：如果存在预定义文件，默认使用
                use_predefined = True
            print(f"   ✅ 使用预定义的数据集划分")
            print(f"   训练集文件: {len(train_paths)} 个")
            print(f"   验证集文件: {len(val_paths)} 个")
        else:
            if args.use_predefined_split:
                print(f"   ⚠️  预定义划分文件不存在，将使用随机划分")
                use_predefined = False
            else:
                use_predefined = False
    
    # 根据是否使用预定义划分来加载数据集
    if use_predefined and train_paths is not None and val_paths is not None:
        # 使用预定义的划分
        train_dataset = ProteinStructureDataset(
            train_paths,
            cache_dir=args.cache_dir
        )
        val_dataset = ProteinStructureDataset(
            val_paths,
            cache_dir=args.cache_dir
        )
        print(f"   训练集: {len(train_dataset)} 个样本")
        print(f"   验证集: {len(val_dataset)} 个样本")
    else:
        # 使用随机划分或全部数据
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
            print(f"   训练集: {len(train_dataset)}, 验证集: {len(val_dataset)} (随机划分)")
        else:
            train_dataset = dataset
            val_dataset = None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
            drop_last=True
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
    
    # 移动到设备
    encoder.to(device)
    decoder.to(device)
    
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
        masking_strategy=args.masking_strategy,
        num_diffusion_steps=args.num_diffusion_steps,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs
    )
    
    # 开始训练
    print("\n4. 开始训练...")
    if args.visualize:
        print(f"   可视化: 启用 (每 {args.plot_every} 个 epoch 绘制一次)")
    trainer.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        save_every=10,
        visualize=args.visualize,
        plot_every=args.plot_every,
        resume_from=args.resume,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta
    )
    
    print("\n" + "="*70)
    print("训练完成！")
    print(f"模型保存在: {args.save_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
