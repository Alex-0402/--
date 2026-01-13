"""
数据集预处理脚本

将 PDB 文件批量转换为缓存的 .pt 文件，加快训练时的数据加载速度。

使用方法:
    python scripts/preprocess_dataset.py \
        --pdb_dir data/pdb_files \
        --output_dir data/cache \
        --num_workers 4 \
        --train_ratio 0.8 \
        --val_ratio 0.1 \
        --test_ratio 0.1
"""

import argparse
import os
import sys
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ProteinStructureDataset
from data.vocabulary import get_vocab


def process_pdb_wrapper(args):
    """
    包装函数，用于多进程处理单个 PDB 文件
    
    Args:
        args: 元组 (pdb_path, cache_dir, use_mmcif)
    
    Returns:
        元组 (pdb_path, success, error_message)
    """
    pdb_path, cache_dir, use_mmcif = args
    try:
        # 使用数据集类的方法处理
        # 注意：不使用缓存，我们手动保存
        dataset = ProteinStructureDataset(
            pdb_path,
            cache_dir=None,  # 不使用缓存，我们手动保存
            use_mmcif=use_mmcif
        )
        
        # 获取数据
        data = dataset[0]
        
        if data is None:
            return (pdb_path, False, "Failed to parse PDB file")
        
        # 保存到缓存
        pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
        cache_path = os.path.join(cache_dir, f"{pdb_name}.pt")
        torch.save(data, cache_path)
        
        return (pdb_path, True, None)
    
    except Exception as e:
        return (pdb_path, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="预处理 PDB 数据集，生成缓存文件"
    )
    
    parser.add_argument(
        "--pdb_dir",
        type=str,
        required=True,
        help="PDB 文件目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="缓存输出目录"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="并行处理进程数（默认：CPU 核心数）"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例（默认：0.8）"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="验证集比例（默认：0.1）"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="测试集比例（默认：0.1）"
    )
    parser.add_argument(
        "--use_mmcif",
        action="store_true",
        help="使用 mmCIF 格式"
    )
    
    args = parser.parse_args()
    
    # 验证比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例总和必须为 1.0，当前为 {total_ratio}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 收集 PDB 文件
    print(f"扫描 PDB 文件目录: {args.pdb_dir}")
    extensions = ['.pdb', '.cif'] if args.use_mmcif else ['.pdb']
    pdb_files = [
        os.path.join(args.pdb_dir, f)
        for f in os.listdir(args.pdb_dir)
        if any(f.endswith(ext) for ext in extensions)
    ]
    
    if len(pdb_files) == 0:
        raise ValueError(f"在 {args.pdb_dir} 中未找到 PDB 文件")
    
    print(f"找到 {len(pdb_files)} 个 PDB 文件")
    
    # 设置进程数
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    print(f"使用 {num_workers} 个进程并行处理")
    
    # 准备参数
    process_args = [(pdb_path, args.output_dir, args.use_mmcif) for pdb_path in pdb_files]
    
    # 多进程处理
    print("\n开始处理 PDB 文件...")
    success_count = 0
    failed_files = []
    
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_pdb_wrapper, process_args),
                total=len(pdb_files),
                desc="处理进度"
            ))
    else:
        # 单进程模式（用于调试）
        results = [process_pdb_wrapper(args) for args in tqdm(process_args, desc="处理进度")]
    
    # 统计结果
    successful_files = []
    for pdb_path, success, error_msg in results:
        if success:
            success_count += 1
            successful_files.append(os.path.splitext(os.path.basename(pdb_path))[0])
        else:
            failed_files.append((pdb_path, error_msg))
    
    print(f"\n处理完成:")
    print(f"  成功: {success_count}/{len(pdb_files)}")
    print(f"  失败: {len(failed_files)}/{len(pdb_files)}")
    
    if failed_files:
        print(f"\n失败的文件:")
        for pdb_path, error_msg in failed_files[:10]:  # 只显示前10个
            print(f"  {os.path.basename(pdb_path)}: {error_msg}")
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files) - 10} 个失败文件")
    
    # 数据划分
    print(f"\n划分数据集...")
    import random
    random.seed(42)  # 固定随机种子以确保可复现
    random.shuffle(successful_files)
    
    total = len(successful_files)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)
    
    train_files = successful_files[:train_end]
    val_files = successful_files[train_end:val_end]
    test_files = successful_files[val_end:]
    
    # 保存文件列表
    train_path = os.path.join(args.output_dir, 'train.txt')
    val_path = os.path.join(args.output_dir, 'val.txt')
    test_path = os.path.join(args.output_dir, 'test.txt')
    
    with open(train_path, 'w') as f:
        f.write('\n'.join(train_files))
    with open(val_path, 'w') as f:
        f.write('\n'.join(val_files))
    with open(test_path, 'w') as f:
        f.write('\n'.join(test_files))
    
    print(f"数据集划分:")
    print(f"  训练集: {len(train_files)} ({len(train_files)/total*100:.1f}%)")
    print(f"  验证集: {len(val_files)} ({len(val_files)/total*100:.1f}%)")
    print(f"  测试集: {len(test_files)} ({len(test_files)/total*100:.1f}%)")
    print(f"\n文件列表已保存:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    
    print("\n" + "="*70)
    print("预处理完成！")
    print("="*70)
    print(f"\n使用缓存数据集:")
    print(f"  python train.py \\")
    print(f"      --pdb_path {args.output_dir} \\")
    print(f"      --cache_dir {args.output_dir}")


if __name__ == "__main__":
    main()
