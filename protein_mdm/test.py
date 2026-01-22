"""
测试脚本 - 在测试集上评估训练好的模型

使用方法:
    python test.py --model_path checkpoints/best_model.pt --pdb_path data/cache --cache_dir data/cache
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
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


def load_test_set(cache_dir: str):
    """
    从缓存目录加载测试集
    
    Args:
        cache_dir: 缓存目录路径
        
    Returns:
        测试集文件列表
    """
    test_file = os.path.join(cache_dir, 'test.txt')
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试集文件不存在: {test_file}")
    
    with open(test_file, 'r') as f:
        test_files = [line.strip() for line in f if line.strip()]
    
    # 构建完整的缓存文件路径
    test_paths = [
        os.path.join(cache_dir, f"{name}.pt")
        for name in test_files
        if os.path.exists(os.path.join(cache_dir, f"{name}.pt"))
    ]
    
    return test_paths


def main():
    parser = argparse.ArgumentParser(
        description="在测试集上评估训练好的模型"
    )
    
    # 必需参数
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型检查点路径")
    parser.add_argument("--pdb_path", type=str, required=True,
                       help="PDB 文件路径或缓存目录")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="缓存目录（如果提供，将使用缓存加速加载）")
    
    # 可选参数
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="隐藏层维度（需与训练时一致）")
    parser.add_argument("--num_encoder_layers", type=int, default=3,
                       help="Encoder 层数（需与训练时一致）")
    parser.add_argument("--num_decoder_layers", type=int, default=3,
                       help="Decoder 层数（需与训练时一致）")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="注意力头数（需与训练时一致）")
    parser.add_argument("--mask_ratio", type=float, default=0.15,
                       help="掩码比例（用于评估）")
    parser.add_argument("--masking_strategy", type=str, default="random",
                       choices=["random", "block"],
                       help="掩码策略")
    parser.add_argument("--device", type=str, default=None,
                       help="设备 (cuda/cpu)")
    parser.add_argument("--use_test_split", action="store_true",
                       help="使用 test.txt 文件中的测试集划分（如果存在）")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("="*70)
    print("蛋白质侧链设计模型 - 测试集评估")
    print("="*70)
    print(f"设备: {device}")
    print(f"模型: {args.model_path}")
    print(f"数据路径: {args.pdb_path}")
    if args.cache_dir:
        print(f"缓存目录: {args.cache_dir}")
    print(f"批次大小: {args.batch_size}")
    print("="*70)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"\n❌ 错误: 模型文件不存在: {args.model_path}")
        print("\n提示:")
        print("1. 请先训练模型:")
        print("   python train.py --pdb_path data/cache --cache_dir data/cache --epochs 50")
        print("\n2. 或者检查模型文件路径是否正确")
        
        # 检查 checkpoints 目录
        checkpoints_dir = "checkpoints"
        if os.path.exists(checkpoints_dir):
            pt_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
            if pt_files:
                print(f"\n在 {checkpoints_dir} 目录中找到以下模型文件:")
                for f in pt_files:
                    print(f"  - {os.path.join(checkpoints_dir, f)}")
                print(f"\n可以使用其中一个文件:")
                print(f"  python test.py --model_path {os.path.join(checkpoints_dir, pt_files[0])} ...")
        else:
            print(f"\n{checkpoints_dir} 目录不存在，请先训练模型")
        
        sys.exit(1)
    
    # 加载模型
    print("\n1. 加载模型...")
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
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
    except Exception as e:
        print(f"\n❌ 错误: 无法加载模型文件: {e}")
        sys.exit(1)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"   模型加载成功 (epoch {epoch})")
    if 'loss' in checkpoint:
        print(f"   训练损失: {checkpoint['loss']:.4f}")
    
    # 加载测试集
    print("\n2. 加载测试集...")
    
    if args.use_test_split and args.cache_dir:
        # 使用预定义的测试集划分
        test_paths = load_test_set(args.cache_dir)
        if len(test_paths) == 0:
            print("   ⚠️  测试集文件列表为空，使用全部数据")
            test_dataset = ProteinStructureDataset(
                args.pdb_path,
                cache_dir=args.cache_dir
            )
        else:
            print(f"   从 test.txt 加载了 {len(test_paths)} 个测试样本")
            # 创建只包含测试集文件的数据集
            test_dataset = ProteinStructureDataset(
                test_paths,
                cache_dir=args.cache_dir
            )
    else:
        # 使用全部数据作为测试集
        test_dataset = ProteinStructureDataset(
            args.pdb_path,
            cache_dir=args.cache_dir
        )
    
    print(f"   测试集大小: {len(test_dataset)}")
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 初始化训练器（用于评估）
    print("\n3. 初始化评估器...")
    trainer = Trainer(
        encoder=encoder,
        decoder=decoder,
        train_loader=None,  # 不需要训练数据
        val_loader=test_loader,  # 使用测试集作为验证集
        device=device,
        learning_rate=1e-4,  # 不会使用
        weight_decay=1e-5,   # 不会使用
        mask_ratio=args.mask_ratio,
        masking_strategy=args.masking_strategy
    )
    
    # 在测试集上评估
    print("\n4. 在测试集上评估...")
    print("-" * 70)
    test_metrics = trainer.validate()
    
    if test_metrics:
        print("\n" + "="*70)
        print("测试结果:")
        print("="*70)
        print(f"总损失: {test_metrics['loss']:.4f}")
        print(f"片段损失: {test_metrics['fragment_loss']:.4f}")
        print(f"扭转角损失: {test_metrics['torsion_loss']:.4f}")
        print("="*70)
        
        # 保存结果
        results_file = os.path.join(os.path.dirname(args.model_path), 'test_results.txt')
        with open(results_file, 'w') as f:
            f.write("测试集评估结果\n")
            f.write("="*70 + "\n")
            f.write(f"模型: {args.model_path}\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"测试集大小: {len(test_dataset)}\n")
            f.write(f"批次大小: {args.batch_size}\n")
            f.write(f"掩码比例: {args.mask_ratio}\n")
            f.write(f"掩码策略: {args.masking_strategy}\n")
            f.write("-"*70 + "\n")
            f.write(f"总损失: {test_metrics['loss']:.4f}\n")
            f.write(f"片段损失: {test_metrics['fragment_loss']:.4f}\n")
            f.write(f"扭转角损失: {test_metrics['torsion_loss']:.4f}\n")
            f.write("="*70 + "\n")
        
        print(f"\n结果已保存到: {results_file}")
    else:
        print("   ⚠️  测试集为空，无法评估")
    
    print("\n" + "="*70)
    print("评估完成！")
    print("="*70)


if __name__ == "__main__":
    main()
