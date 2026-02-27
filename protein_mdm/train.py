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
from datetime import timedelta

import matplotlib
# 强制使用无窗口后端
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 添加项目根目录到 Python 路径，以支持相对导入
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 注意：torch 的导入将在 if __name__ == "__main__" 中，在设置 CUDA_VISIBLE_DEVICES 之后


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


def parse_args():
    """解析命令行参数"""
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
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout 率（默认 0.3，用于增强正则化）")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=300,
                       help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                       help="学习率（最大学习率）")
    parser.add_argument("--warmup_epochs", type=int, default=20,
                       help="Warmup 轮数")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                       help="权重衰减（默认 1e-3，用于增强正则化）")
    parser.add_argument("--masking_strategy", type=str, default="random",
                       choices=["random", "block"],
                       help="掩码策略")
    parser.add_argument("--num_diffusion_steps", type=int, default=1000,
                       help="扩散模型的时间步数（默认1000）")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                       help="交叉熵标签平滑系数（默认0.1，诊断可尝试0.0/0.05）")
    parser.add_argument("--max_train_samples", type=int, default=0,
                       help="仅使用前N个训练样本（0表示不限制，用于快速诊断）")
    parser.add_argument("--max_val_samples", type=int, default=0,
                       help="仅使用前N个验证样本（0表示不限制，用于快速诊断）")
    parser.add_argument("--overfit_train_subset", type=int, default=0,
                       help="容量诊断：从训练集取N个样本，并将验证集也设为同一子集（0表示关闭）")
    
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
    parser.add_argument("--early_stopping_patience", type=int, default=50,
                       help="早停耐心值，验证损失连续N轮不下降则停止训练（默认 50，给模型更多震荡收敛的时间）")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0,
                       help="早停最小改进阈值，只有改进超过此值才认为是有效改进")
    parser.add_argument("--debug_mode", action="store_true",
                       help="启用详细调试日志（包含traceback和张量统计信息）")
    
    # DDP 相关参数
    parser.add_argument("--gpu_ids", type=str, default=None,
                       help="指定使用的GPU ID（例如：'1,2,3,4,5,6,7'），在所有torch调用之前设置CUDA_VISIBLE_DEVICES")
    parser.add_argument("--master_port", type=str, default="29500",
                       help="DDP master port（默认：29500）")
    parser.add_argument("--ddp", action="store_true",
                       help="启用 DDP 模式（通常由 torchrun 自动检测，此参数用于手动模式）")
    
    return parser.parse_args()


def main():
    # 缓解显存碎片化，降低 OOM 概率（必须在任何 torch/cuda 调用之前设置）
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    # 关键：在 main 函数最开始解析参数并设置 CUDA_VISIBLE_DEVICES
    # 这必须在任何 torch cuda 调用之前完成
    args = globals().get('_args')
    if args is None:
        args = parse_args()
    
    # 在所有 torch cuda 调用之前设置 CUDA_VISIBLE_DEVICES
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        # 只在非 DDP 模式下打印，避免多进程重复打印
        if "RANK" not in os.environ and "LOCAL_RANK" not in os.environ:
            print(f"✅ 已设置 CUDA_VISIBLE_DEVICES={args.gpu_ids}")
    
    # 导入必要的模块（在设置 CUDA_VISIBLE_DEVICES 之后）
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    from models.encoder import BackboneEncoder
    from models.decoder import FragmentDecoder
    from data.dataset import ProteinStructureDataset, collate_fn
    from data.vocabulary import get_vocab
    from training.trainer import Trainer
    
    # 检查是否使用 DDP（通过环境变量判断，torchrun 会自动设置，或通过 --ddp 参数）
    ddp_enabled = args.ddp or (
        "RANK" in os.environ and "WORLD_SIZE" in os.environ
    ) or (
        "LOCAL_RANK" in os.environ
    )
    
    if ddp_enabled:
        # 初始化 DDP
        # 设置 NCCL 环境变量以调试和防止死锁
        os.environ.setdefault("NCCL_TIMEOUT", "1800")  # 30分钟
        os.environ["NCCL_P2P_DISABLE"] = "1"  # 禁用 P2P 防止 2080Ti 可能出现的 P2P 死锁
        os.environ["NCCL_BLOCKING_WAIT"] = "1"  # 阻塞等待，报错时提供更多信息
        # 可选：如果需要详细调试日志，取消下面的注释
        # os.environ.setdefault("NCCL_DEBUG", "INFO")
        
        # 获取环境变量
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            rank = int(os.environ.get("RANK", local_rank))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
        else:
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            local_rank = rank % torch.cuda.device_count()
        
        # 设置当前设备（必须在 init_process_group 之前）
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        # 初始化进程组，设置超时时间为 30 分钟
        # 如果使用 torchrun，环境变量会自动设置，不需要手动指定 init_method
        # 如果手动启动，可以使用 init_method
        if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
            # torchrun 模式：使用环境变量自动初始化
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(minutes=30)  # 30分钟超时
            )
        else:
            # 手动模式：使用指定的 master_port
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://127.0.0.1:{args.master_port}",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=30)
            )
        
        # 只在 rank 0 打印信息
        if rank == 0:
            print("="*70)
            print("蛋白质侧链设计模型 - 训练 (DDP 模式)")
            print("="*70)
            print(f"DDP 模式: 启用 (world_size={world_size})")
            print(f"设备: {device} (rank={rank}, local_rank={local_rank})")
            print(f"NCCL_TIMEOUT: {os.environ.get('NCCL_TIMEOUT', 'default')}")
    else:
        # 单GPU模式
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
        local_rank = 0
        
        print("="*70)
        print("蛋白质侧链设计模型 - 训练")
        print("="*70)
        print(f"设备: {device} (单GPU模式)")
    
    # 只在 rank 0 打印信息
    if rank == 0:
        print(f"PDB 路径: {args.pdb_path}")
        print(f"批次大小: {args.batch_size} (每个GPU)")
        if ddp_enabled:
            print(f"总批次大小: {args.batch_size * world_size} (所有GPU)")
        print(f"训练轮数: {args.epochs}")
        print(f"扩散模型: 启用 (时间步数: {args.num_diffusion_steps})")
        print(f"Label smoothing: {args.label_smoothing}")
        print(f"掩码比例: 动态 (Cosine Schedule, t=0时0%, t=1时100%)")
        print(f"掩码策略: {args.masking_strategy}")
        print("="*70)

    if not (0.0 <= args.label_smoothing <= 1.0):
        raise ValueError("--label_smoothing 必须在 [0, 1] 范围内")
    
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
        # 训练集启用数据增强，验证集禁用数据增强以确保稳定的验证指标
        train_dataset = ProteinStructureDataset(
            train_paths,
            cache_dir=args.cache_dir,
            augment=True  # 训练时启用数据增强
        )
        val_dataset = ProteinStructureDataset(
            val_paths,
            cache_dir=args.cache_dir,
            augment=False  # 验证时禁用数据增强，确保稳定的验证损失
        )
        if rank == 0:
            print(f"   训练集: {len(train_dataset)} 个样本")
            print(f"   验证集: {len(val_dataset)} 个样本")
    else:
        # 使用随机划分或全部数据
        # 先创建一个临时数据集以获取大小和文件列表
        temp_dataset = ProteinStructureDataset(
            args.pdb_path,
            cache_dir=args.cache_dir,
            augment=False  # 临时数据集，augment 参数不重要
        )
        if rank == 0:
            print(f"   数据集大小: {len(temp_dataset)}")
            if args.cache_dir:
                print(f"   使用缓存目录: {args.cache_dir}")
        
        # 划分训练/验证集
        if args.val_split > 0:
            val_size = int(len(temp_dataset) * args.val_split)
            train_size = len(temp_dataset) - val_size
            train_indices, val_indices = torch.utils.data.random_split(
                range(len(temp_dataset)), [train_size, val_size]
            )
            
            # 创建训练集（启用数据增强）
            train_dataset = torch.utils.data.Subset(
                ProteinStructureDataset(
                    args.pdb_path,
                    cache_dir=args.cache_dir,
                    augment=True  # 训练时启用数据增强
                ),
                train_indices.indices
            )
            
            # 创建验证集（禁用数据增强）
            val_dataset = torch.utils.data.Subset(
                ProteinStructureDataset(
                    args.pdb_path,
                    cache_dir=args.cache_dir,
                    augment=False  # 验证时禁用数据增强，确保稳定的验证损失
                ),
                val_indices.indices
            )
            
            if rank == 0:
                print(f"   训练集: {len(train_dataset)}, 验证集: {len(val_dataset)} (随机划分)")
        else:
            # 全部数据作为训练集，启用数据增强
            train_dataset = ProteinStructureDataset(
                args.pdb_path,
                cache_dir=args.cache_dir,
                augment=True  # 训练时启用数据增强
            )
            val_dataset = None

    # 诊断选项：限制样本数量（快速实验）
    if args.max_train_samples > 0 and len(train_dataset) > args.max_train_samples:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(args.max_train_samples)))
        if rank == 0:
            print(f"   🔬 训练集已截断为: {len(train_dataset)} 个样本")

    if val_dataset is not None and args.max_val_samples > 0 and len(val_dataset) > args.max_val_samples:
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(args.max_val_samples)))
        if rank == 0:
            print(f"   🔬 验证集已截断为: {len(val_dataset)} 个样本")

    # 容量诊断：让模型在小训练子集上过拟合，验证是否具备表达能力
    if args.overfit_train_subset > 0:
        overfit_n = min(args.overfit_train_subset, len(train_dataset))
        overfit_indices = list(range(overfit_n))
        train_dataset = torch.utils.data.Subset(train_dataset, overfit_indices)
        val_dataset = torch.utils.data.Subset(train_dataset, list(range(overfit_n)))
        if rank == 0:
            print(f"   🧪 过拟合诊断模式: train=val={overfit_n} 个样本")
    
    # 创建数据加载器
    # DDP 模式：使用 DistributedSampler
    if ddp_enabled:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True  # 防止最后一个 batch 大小不一致导致的 DDP 同步挂起
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,  # 使用 sampler 时不能设置 shuffle
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,  # 最安全的设置，防止 epoch 切换死锁
            drop_last=True  # 强制设置，防止最后一个 batch 大小不一致
        )
        
        val_sampler = None
        if val_dataset is not None:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,  # 验证集不需要 shuffle
                drop_last=True
            )
            val_batch_size = max(1, args.batch_size // 2)  # 验证集减半以降低显存压力
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True,
                persistent_workers=False,
                drop_last=True
            )
        else:
            val_loader = None
    else:
        # 单GPU模式：使用普通 DataLoader
        train_sampler = None
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
        val_sampler = None
        if val_dataset is not None:
            val_batch_size = max(1, args.batch_size // 2)  # 验证集减半以降低显存压力
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True,
                persistent_workers=False,
                drop_last=True
            )
    
    # 初始化模型
    if rank == 0:
        print("\n2. 初始化模型...")
    vocab = get_vocab()
    encoder = BackboneEncoder(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_encoder_layers,
        k_neighbors=30,
        dropout=args.dropout  # 传递 dropout 参数以增强正则化
    )
    decoder = FragmentDecoder(
        input_dim=args.hidden_dim,
        vocab_size=vocab.get_vocab_size(),
        num_torsion_bins=72,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout  # 传递 dropout 参数以增强正则化
    )
    
    if rank == 0:
        encoder_params = sum(p.numel() for p in encoder.parameters())
        decoder_params = sum(p.numel() for p in decoder.parameters())
        print(f"   Encoder 参数: {encoder_params:,}")
        print(f"   Decoder 参数: {decoder_params:,}")
        print(f"   总参数: {encoder_params + decoder_params:,}")
    
    # 移动到设备
    encoder.to(device)
    decoder.to(device)
    
    # DDP 模式：在模型初始化后、DDP 包装前添加 barrier
    # 确保 Rank 0 加载完配置/词表后，所有进程再一起包装 DDP
    if ddp_enabled:
        dist.barrier()
        if rank == 0:
            print("   ✅ 所有进程已完成模型初始化，开始 DDP 包装...")
    
    # DDP 模式：包装模型
    if ddp_enabled:
        encoder = torch.nn.parallel.DistributedDataParallel(
            encoder,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,  # 如果所有参数都被使用，设为 False 可以提升性能
            broadcast_buffers=False  # 模型未使用BatchNorm，关闭前向buffer同步可降低死锁风险
        )
        decoder = torch.nn.parallel.DistributedDataParallel(
            decoder,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False
        )
        if rank == 0:
            print(f"   ✅ 模型已包装为 DDP (device_id={local_rank})")
    
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
        masking_strategy=args.masking_strategy,
        num_diffusion_steps=args.num_diffusion_steps,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        ddp_enabled=ddp_enabled,
        rank=rank,
        world_size=world_size,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        debug_mode=args.debug_mode,
        label_smoothing=args.label_smoothing
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
        visualize=args.visualize,
        plot_every=args.plot_every,
        resume_from=args.resume,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta
    )
    
    # 清理 DDP
    if ddp_enabled:
        dist.destroy_process_group()
    
    if rank == 0:
        print("\n" + "="*70)
        print("训练完成！")
        print(f"模型保存在: {args.save_dir}")
        print("="*70)


if __name__ == "__main__":
    # 关键：在 if __name__ == "__main__": 的最开始解析参数
    # 这样可以在所有 torch 引用之前设置 CUDA_VISIBLE_DEVICES
    _args = parse_args()
    
    # 将 args 存储到全局命名空间，供 main() 使用
    # main() 函数内部会在最开始设置 CUDA_VISIBLE_DEVICES
    globals()['_args'] = _args
    
    # 调用主函数（main() 内部会设置 CUDA_VISIBLE_DEVICES 并导入 torch 等模块）
    main()
