#!/usr/bin/env python3
"""
根据现有的 train.txt 和 val.txt 重新生成缓存文件
保留数据集划分，只重新生成 .pt 缓存文件（使用更新后的 vocabulary）
"""

import os
import sys
import torch
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import ProteinStructureDataset
from data.vocabulary import get_vocab

def main():
    cache_dir = "data/cache"
    raw_data_dir = "raw_data"
    
    # 检查目录
    if not os.path.exists(cache_dir):
        print(f"❌ 缓存目录不存在: {cache_dir}")
        return
    
    if not os.path.exists(raw_data_dir):
        print(f"❌ 原始数据目录不存在: {raw_data_dir}")
        return
    
    # 检查划分文件
    train_file = os.path.join(cache_dir, "train.txt")
    val_file = os.path.join(cache_dir, "val.txt")
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print("❌ 未找到 train.txt 或 val.txt")
        print("   请先运行: bash 重新生成缓存.sh")
        return
    
    # 读取文件列表
    with open(train_file, 'r') as f:
        train_files = [line.strip() for line in f if line.strip()]
    with open(val_file, 'r') as f:
        val_files = [line.strip() for line in f if line.strip()]
    
    all_files = train_files + val_files
    print(f"找到 {len(train_files)} 个训练文件，{len(val_files)} 个验证文件")
    print(f"总共需要重新生成 {len(all_files)} 个缓存文件")
    
    # 删除旧的 .pt 文件
    print("\n1. 删除旧的缓存文件...")
    pt_files = [f for f in os.listdir(cache_dir) if f.endswith('.pt')]
    deleted_count = 0
    for pt_file in pt_files:
        pt_path = os.path.join(cache_dir, pt_file)
        os.remove(pt_path)
        deleted_count += 1
    print(f"   ✅ 已删除 {deleted_count} 个旧缓存文件")
    
    # 重新生成缓存
    print("\n2. 重新生成缓存文件（使用更新后的 vocabulary）...")
    print("   注意: SER, CYS, ILE 的片段映射已修复")
    
    vocab = get_vocab()
    success_count = 0
    failed_count = 0
    
    for file_name in tqdm(all_files, desc="处理进度"):
        # 查找对应的 PDB 文件
        pdb_path = None
        for ext in ['.pdb', '.cif']:
            potential_path = os.path.join(raw_data_dir, f"{file_name}{ext}")
            if os.path.exists(potential_path):
                pdb_path = potential_path
                break
        
        if pdb_path is None:
            print(f"\n   ⚠️  警告: 找不到文件 {file_name}.pdb 或 {file_name}.cif")
            failed_count += 1
            continue
        
        try:
            # 使用数据集类处理
            dataset = ProteinStructureDataset(
                pdb_path,
                cache_dir=None,  # 不使用自动缓存
                use_mmcif=pdb_path.endswith('.cif')
            )
            
            # 获取数据
            data = dataset[0]
            
            if data is None:
                print(f"\n   ⚠️  警告: 无法解析 {file_name}")
                failed_count += 1
                continue
            
            # 保存到缓存
            cache_path = os.path.join(cache_dir, f"{file_name}.pt")
            torch.save(data, cache_path)
            success_count += 1
            
        except Exception as e:
            print(f"\n   ❌ 错误: 处理 {file_name} 时失败: {e}")
            failed_count += 1
    
    print(f"\n处理完成:")
    print(f"   ✅ 成功: {success_count}/{len(all_files)}")
    print(f"   ❌ 失败: {failed_count}/{len(all_files)}")
    
    print("\n" + "="*70)
    print("缓存重新生成完成！")
    print("="*70)
    print("\n下一步: 可以开始训练了")
    print("  bash 重新开始训练.sh")

if __name__ == "__main__":
    main()
