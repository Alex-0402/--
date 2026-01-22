# 训练指南 - 使用下载的 CATH 数据

本指南说明如何使用从 CATH 下载的 PDB 文件进行模型训练。

## 完整流程概览

```
下载 PDB 文件 → 预处理为缓存格式 → 训练模型 → 评估模型
```

## 步骤 1: 预处理数据（必需）

将下载的 PDB 文件转换为训练所需的缓存格式。

### 基本用法

```bash
python scripts/preprocess_dataset.py \
    --pdb_dir raw_data \
    --output_dir data/cache
```

### 完整参数示例

```bash
python scripts/preprocess_dataset.py \
    --pdb_dir raw_data \
    --output_dir data/cache \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --num_workers 4
```

### 参数说明

- `--pdb_dir`: PDB 文件所在目录（默认：`raw_data`）
- `--output_dir`: 缓存文件输出目录（默认：`data/cache`）
- `--train_ratio`: 训练集比例（默认：0.8）
- `--val_ratio`: 验证集比例（默认：0.1）
- `--test_ratio`: 测试集比例（默认：0.1）
- `--num_workers`: 并行处理进程数（默认：CPU 核心数）

### 预处理输出

预处理完成后，会在 `data/cache/` 目录下生成：

- `{pdb_id}.pt`: 每个 PDB 文件的缓存（包含处理后的数据）
- `train.txt`: 训练集文件列表
- `val.txt`: 验证集文件列表
- `test.txt`: 测试集文件列表

### 示例输出

```
扫描 PDB 文件目录: raw_data
找到 47 个 PDB 文件
使用 8 个进程并行处理

开始处理 PDB 文件...
处理进度: 100%|████████████████| 47/47 [00:30<00:00,  1.5文件/s]

处理完成:
  成功: 45/47
  失败: 2/47

划分数据集...
数据集划分:
  训练集: 36 (80.0%)
  验证集: 4 (10.0%)
  测试集: 5 (10.0%)

文件列表已保存:
  data/cache/train.txt
  data/cache/val.txt
  data/cache/test.txt
```

## 步骤 2: 训练模型

使用预处理后的数据训练模型。

### 基本用法

```bash
python train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --epochs 50
```

### 完整参数示例

```bash
python train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --hidden_dim 256 \
    --mask_ratio 0.15 \
    --save_dir checkpoints
```

### 参数说明

#### 数据参数
- `--pdb_path`: 数据路径（使用缓存目录：`data/cache`）
- `--cache_dir`: 缓存目录（与 `pdb_path` 相同）
- `--batch_size`: 批次大小（默认：4）
- `--val_split`: 验证集比例（如果使用预处理的数据集划分，则忽略此参数）

#### 模型参数
- `--hidden_dim`: 隐藏层维度（默认：256）
- `--num_encoder_layers`: Encoder 层数（默认：3）
- `--num_decoder_layers`: Decoder 层数（默认：3）
- `--num_heads`: 注意力头数（默认：8）

#### 训练参数
- `--epochs`: 训练轮数（默认：50）
- `--learning_rate`: 学习率（默认：1e-4）
- `--weight_decay`: 权重衰减（默认：1e-5）
- `--mask_ratio`: 掩码比例（默认：0.15）
- `--masking_strategy`: 掩码策略，`random` 或 `block`（默认：random）

#### 其他参数
- `--save_dir`: 模型保存目录（默认：`checkpoints`）
- `--device`: 设备，`cuda` 或 `cpu`（默认：自动检测）

### 训练输出

训练过程中会显示：

```
======================================================================
蛋白质侧链设计模型 - 训练
======================================================================
设备: cpu
PDB 路径: data/cache
批次大小: 4
训练轮数: 50
掩码比例: 0.15
掩码策略: random
======================================================================

1. 加载数据集...
   数据集大小: 45
   使用缓存目录: data/cache
   训练集: 36, 验证集: 4

2. 初始化模型...
   Encoder 参数: 1,034,688
   Decoder 参数: 3,385,432
   总参数: 4,420,120

3. 初始化训练器...

4. 开始训练...
开始训练，设备: cpu
掩码策略: random, 掩码比例: 0.15

Epoch 1/50
------------------------------------------------------------
Training: 100%|████████████████| 9/9 [00:05<00:00,  1.6it/s, loss=6.6738, frag=2.6369, tors=4.0369]
Train Loss: 6.6738 (Fragment: 2.6369, Torsion: 4.0369)
⚠️  验证集为空，跳过验证

Epoch 2/50
...
```

### 模型保存

训练过程中会保存：

- `checkpoints/best_model.pt`: 验证集上表现最好的模型
- `checkpoints/checkpoint_epoch_{N}.pt`: 每 N 个 epoch 的检查点（默认每 10 个 epoch）

## 步骤 3: 评估模型（可选）

在测试集上评估训练好的模型。

### 基本用法

```bash
python test.py \
    --model_path checkpoints/best_model.pt \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_test_split
```

### 完整参数示例

```bash
python test.py \
    --model_path checkpoints/best_model.pt \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_test_split \
    --batch_size 4 \
    --mask_ratio 0.15
```

## 快速开始脚本

创建一个一键运行的脚本：

```bash
#!/bin/bash
# quick_train.sh

echo "步骤 1: 预处理数据..."
python scripts/preprocess_dataset.py \
    --pdb_dir raw_data \
    --output_dir data/cache \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1

echo ""
echo "步骤 2: 训练模型..."
python train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --epochs 50 \
    --batch_size 4 \
    --save_dir checkpoints

echo ""
echo "步骤 3: 评估模型..."
python test.py \
    --model_path checkpoints/best_model.pt \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_test_split

echo ""
echo "完成！"
```

## 常见问题

### Q1: 预处理失败怎么办？

**A**: 检查 PDB 文件是否完整，某些文件可能损坏。预处理脚本会自动跳过失败的文件。

### Q2: 训练时内存不足？

**A**: 尝试减小批次大小：
```bash
python train.py --batch_size 2 ...
```

### Q3: 训练很慢？

**A**: 
- 使用 GPU（如果可用）：`--device cuda`
- 减小模型大小：`--hidden_dim 128`
- 减少训练轮数：`--epochs 20`

### Q4: 如何继续训练？

**A**: 修改训练脚本加载检查点，或使用 `inference.py` 进行推理。

### Q5: 数据集太小怎么办？

**A**: 下载更多数据：
```bash
python scripts/download_cath_subset.py --limit 200
```

## 数据流程总结

```
raw_data/              # 原始 PDB 文件
  ├── 1ubq.pdb
  ├── 1crn.pdb
  └── ...

    ↓ 预处理

data/cache/            # 缓存文件
  ├── 1ubq.pt
  ├── 1crn.pt
  ├── train.txt
  ├── val.txt
  └── test.txt

    ↓ 训练

checkpoints/           # 模型检查点
  ├── best_model.pt
  └── checkpoint_epoch_10.pt
```

## 下一步

训练完成后，可以：

1. **使用模型进行推理**：参考 `inference.py`
2. **分析训练结果**：查看 `checkpoints/test_results.txt`
3. **调整超参数**：根据验证集表现调整学习率、模型大小等
4. **下载更多数据**：使用更大的数据集进行训练
