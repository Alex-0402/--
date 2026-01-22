"""
快速诊断脚本 - 检查关键问题
"""

import os
from pathlib import Path

print("="*70)
print("快速诊断")
print("="*70)

# 1. 检查数据集规模
print("\n1. 数据集规模:")
cache_dir = "data/cache"
train_file = os.path.join(cache_dir, "train.txt")
val_file = os.path.join(cache_dir, "val.txt")
test_file = os.path.join(cache_dir, "test.txt")

train_size = 0
val_size = 0
test_size = 0

if os.path.exists(train_file):
    with open(train_file, 'r') as f:
        train_size = len([l for l in f if l.strip()])
    print(f"   训练集: {train_size} 个样本")

if os.path.exists(val_file):
    with open(val_file, 'r') as f:
        val_size = len([l for l in f if l.strip()])
    print(f"   验证集: {val_size} 个样本")

if os.path.exists(test_file):
    with open(test_file, 'r') as f:
        test_size = len([l for l in f if l.strip()])
    print(f"   测试集: {test_size} 个样本")

total = train_size + val_size + test_size
print(f"   总计: {total} 个样本")

if total < 50:
    print("   ⚠️  警告: 数据集规模过小！建议至少 100+ 样本")
elif total < 100:
    print("   ⚠️  注意: 数据集规模较小，建议增加数据量")

# 2. 检查测试结果
print("\n2. 测试结果分析:")
test_results = "checkpoints/test_results.txt"
if os.path.exists(test_results):
    with open(test_results, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "总损失" in line or "片段损失" in line or "扭转角损失" in line:
                print(f"   {line.strip()}")
    
    # 理论基线
    import numpy as np
    random_frag = np.log(16)  # vocab_size = 16
    random_tors = np.log(72)  # num_bins = 72
    random_total = random_frag + random_tors
    
    print(f"\n   随机猜测基线:")
    print(f"   片段损失: {random_frag:.4f}")
    print(f"   扭转角损失: {random_tors:.4f}")
    print(f"   总损失: {random_total:.4f}")
    print(f"\n   ⚠️  当前结果接近随机猜测，需要改进！")

# 3. 检查检查点文件
print("\n3. 检查点文件:")
checkpoints_dir = "checkpoints"
if os.path.exists(checkpoints_dir):
    pt_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
    print(f"   找到 {len(pt_files)} 个检查点文件")
    for f in sorted(pt_files)[:5]:  # 只显示前5个
        filepath = os.path.join(checkpoints_dir, f)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"   - {f} ({size_mb:.1f} MB)")

# 4. 建议
print("\n" + "="*70)
print("建议的下一步操作:")
print("="*70)
print("\n1. 立即执行 - 扩大数据集:")
print("   python scripts/download_cath_subset.py --max_structures 200")
print("   python scripts/preprocess_dataset.py --pdb_dir raw_data --output_dir data/cache")
print("\n2. 重新训练 - 使用更多数据:")
print("   python train.py --pdb_path data/cache --cache_dir data/cache --epochs 100 --learning_rate 5e-5")
print("\n3. 评估改进效果:")
print("   python test.py --model_path checkpoints/best_model.pt --pdb_path data/cache --cache_dir data/cache")
print("\n详细改进计划请查看: IMPROVEMENT_PLAN.md")

print("\n" + "="*70)
