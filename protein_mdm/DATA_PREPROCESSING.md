# 数据预处理指南

## 概述

为了提高训练时的数据加载速度，我们提供了数据预处理功能，可以将 PDB 文件预先解析并缓存为 `.pt` 文件。

## 快速开始

### 1. 预处理数据集

```bash
# 基本用法
python scripts/preprocess_dataset.py \
    --pdb_dir data/pdb_files \
    --output_dir data/cache

# 使用自定义参数
python scripts/preprocess_dataset.py \
    --pdb_dir data/pdb_files \
    --output_dir data/cache \
    --num_workers 8 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

### 2. 使用缓存数据集训练

```bash
# 使用缓存目录训练（自动使用缓存）
python train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --epochs 50
```

## 预处理脚本参数

- `--pdb_dir`: PDB 文件目录（必需）
- `--output_dir`: 缓存输出目录（必需）
- `--num_workers`: 并行处理进程数（默认：CPU 核心数）
- `--train_ratio`: 训练集比例（默认：0.8）
- `--val_ratio`: 验证集比例（默认：0.1）
- `--test_ratio`: 测试集比例（默认：0.1）
- `--use_mmcif`: 如果 PDB 文件是 mmCIF 格式，添加此标志

## 输出文件

预处理完成后，会在 `output_dir` 生成：

1. **缓存文件**: `{pdb_name}.pt` - 每个 PDB 文件的缓存
2. **数据划分文件**:
   - `train.txt` - 训练集文件列表
   - `val.txt` - 验证集文件列表
   - `test.txt` - 测试集文件列表

## 使用缓存数据集

### 方法 1: 自动缓存（推荐）

```python
from data.dataset import ProteinStructureDataset

# 指定 cache_dir，会自动使用缓存
dataset = ProteinStructureDataset(
    pdb_files="data/pdb_files",
    cache_dir="data/cache"  # 缓存目录
)
```

### 方法 2: 仅使用缓存

```python
# 如果所有文件都已预处理，可以直接使用缓存目录
dataset = ProteinStructureDataset(
    pdb_files="data/cache",  # 缓存目录
    cache_dir="data/cache"   # 相同的缓存目录
)
```

## 性能对比

- **无缓存**: 每次训练都需要重新解析 PDB 文件（慢）
- **有缓存**: 第一次加载时解析并缓存，后续直接加载（快）

**建议**: 对于大规模数据集，务必使用预处理和缓存功能。

## 注意事项

1. **缓存文件格式**: 缓存文件是 PyTorch 的 `.pt` 格式，包含完整的数据字典
2. **缓存更新**: 如果 PDB 文件更新，需要删除对应的缓存文件或重新预处理
3. **磁盘空间**: 缓存文件会占用额外的磁盘空间（通常比原始 PDB 文件大）
4. **失败处理**: 解析失败的 PDB 文件会被跳过，不会生成缓存

## 故障排除

### Q: 预处理时某些文件失败
**A**: 检查失败原因，可能是：
- PDB 文件格式错误
- 缺少必要的骨架原子
- 文件损坏

### Q: 如何使用预处理的数据集划分
**A**: 预处理脚本会生成 `train.txt`, `val.txt`, `test.txt`，你可以：
1. 手动读取这些文件
2. 使用对应的文件列表创建数据集

### Q: 缓存文件太大
**A**: 这是正常的，因为缓存包含了处理后的张量数据。如果空间不足，可以考虑：
- 只缓存训练集
- 使用压缩存储（需要修改代码）

## 示例工作流

```bash
# 1. 下载 PDB 文件
mkdir -p data/pdb_files
cd data/pdb_files
curl -O https://files.rcsb.org/download/1CRN.pdb
curl -O https://files.rcsb.org/download/1UBQ.pdb
# ... 更多文件

# 2. 预处理
cd ../..
python scripts/preprocess_dataset.py \
    --pdb_dir data/pdb_files \
    --output_dir data/cache

# 3. 训练（使用缓存）
python train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --epochs 50
```
