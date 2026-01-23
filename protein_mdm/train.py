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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    parser.add_argument("--mask_ratio", type=float, default=0.15,
                       help="掩码比例")
    parser.add_argument("--masking_strategy", type=str, default="random",
                       choices=["random", "block"],
                       help="掩码策略")
    parser.add_argument("--use_discrete_diffusion", action="store_true", default=True,
                       help="使用离散扩散模型训练（默认启用）")
    parser.add_argument("--no_discrete_diffusion", dest="use_discrete_diffusion", action="store_false",
                       help="禁用离散扩散模型训练")
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
    
    # DDP 参数
    parser.add_argument("--ddp", action="store_true",
                       help="使用分布式数据并行 (DDP)")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="本地 GPU rank（DDP 自动设置）")
    parser.add_argument("--world_size", type=int, default=-1,
                       help="总进程数（DDP 自动设置）")
    parser.add_argument("--master_addr", type=str, default="localhost",
                       help="主节点地址（DDP）")
    parser.add_argument("--master_port", type=str, default="12355",
                       help="主节点端口（DDP）")
    
    args = parser.parse_args()
    
    # 初始化 DDP（如果启用）
    # 检查是否在分布式环境中（torchrun 会自动设置这些环境变量）
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    rank = int(os.environ.get("RANK", -1))
    
    # 如果环境变量存在，或者明确指定了 --ddp，则启用 DDP
    ddp_enabled = (local_rank >= 0) or args.ddp
    
    if ddp_enabled and local_rank >= 0:
        # 使用环境变量初始化（torchrun 模式）
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = rank if rank >= 0 else local_rank
        world_size = world_size if world_size > 0 else torch.cuda.device_count()
    elif ddp_enabled and args.local_rank >= 0:
        # 手动指定模式（兼容旧方式）
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            rank=args.local_rank,
            world_size=args.world_size if args.world_size > 0 else torch.cuda.device_count()
        )
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
        rank = args.local_rank
        world_size = args.world_size if args.world_size > 0 else torch.cuda.device_count()
    else:
        # 单GPU模式
        ddp_enabled = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
    
    # 只在 rank 0 打印信息
    if rank == 0:
        print("="*70)
        print("蛋白质侧链设计模型 - 训练")
        print("="*70)
        if ddp_enabled:
            print(f"DDP 模式: 启用 (world_size={world_size})")
        print(f"设备: {device}")
        print(f"PDB 路径: {args.pdb_path}")
        print(f"批次大小: {args.batch_size} (每GPU)")
        if ddp_enabled:
            print(f"总批次大小: {args.batch_size * world_size} (所有GPU)")
        print(f"训练轮数: {args.epochs}")
        if args.use_discrete_diffusion:
            print(f"扩散模型: 启用 (时间步数: {args.num_diffusion_steps})")
            print(f"掩码比例: 动态 (Cosine Schedule, t=0时0%, t=1时100%)")
        else:
            print(f"掩码比例: {args.mask_ratio} (固定)")
        print(f"掩码策略: {args.masking_strategy}")
        print("="*70)
    
    # 加载数据集
    if rank == 0:
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
            if rank == 0:
                print(f"   ✅ 使用预定义的数据集划分")
                print(f"   训练集文件: {len(train_paths)} 个")
                print(f"   验证集文件: {len(val_paths)} 个")
        else:
            if args.use_predefined_split:
                if rank == 0:
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
        if rank == 0:
            print(f"   训练集: {len(train_dataset)} 个样本")
            print(f"   验证集: {len(val_dataset)} 个样本")
    else:
        # 使用随机划分或全部数据
        dataset = ProteinStructureDataset(
            args.pdb_path,
            cache_dir=args.cache_dir
        )
        if rank == 0:
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
            if rank == 0:
                print(f"   训练集: {len(train_dataset)}, 验证集: {len(val_dataset)} (随机划分)")
        else:
            train_dataset = dataset
            val_dataset = None
    
    # 创建数据加载器
    train_sampler = None
    if ddp_enabled:
        # DDP 模式：使用 DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True
            )
    else:
        # 单GPU模式：使用普通DataLoader
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
    if rank == 0:
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
    if rank == 0:
        print(f"   Encoder 参数: {encoder_params:,}")
        print(f"   Decoder 参数: {decoder_params:,}")
        print(f"   总参数: {encoder_params + decoder_params:,}")
    
    # 移动到设备并包装为DDP（如果启用）
    encoder.to(device)
    decoder.to(device)
    
    if ddp_enabled:
        # 使用 DDP 包装模型
        # 获取正确的设备ID（优先使用环境变量中的 local_rank）
        if local_rank >= 0:
            device_id = local_rank
        elif args.local_rank >= 0:
            device_id = args.local_rank
        else:
            device_id = 0  # 默认值
        
        # 使用 find_unused_parameters=True 因为 encoder 中的 physicochemical_proj 可能在某些情况下不被使用
        encoder = DDP(encoder, device_ids=[device_id], output_device=device_id, find_unused_parameters=True)
        decoder = DDP(decoder, device_ids=[device_id], output_device=device_id)
        if rank == 0:
            print(f"   模型已包装为 DDP (device_id={device_id})")
    
    # 初始化训练器
    if rank == 0:
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
        masking_strategy=args.masking_strategy,
        ddp_enabled=ddp_enabled,
        rank=rank,
        use_discrete_diffusion=args.use_discrete_diffusion,
        num_diffusion_steps=args.num_diffusion_steps,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs
    )
    
    # 开始训练
    if rank == 0:
        print("\n4. 开始训练...")
        if args.visualize:
            print(f"   可视化: 启用 (每 {args.plot_every} 个 epoch 绘制一次)")
    trainer.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        save_every=10,
        train_sampler=train_sampler if ddp_enabled else None,
        visualize=args.visualize,
        plot_every=args.plot_every
    )
    
    if rank == 0:
        print("\n" + "="*70)
        print("训练完成！")
        print(f"模型保存在: {args.save_dir}")
        print("="*70)
    
    # 清理 DDP
    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
