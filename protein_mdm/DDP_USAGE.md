# DDP 多GPU训练使用指南

## 概述

代码已支持分布式数据并行（DDP）训练，可以在多张GPU上并行训练，大幅提升训练速度。

## 使用方法

### 方法1：使用 torchrun（推荐）

`torchrun` 是 PyTorch 推荐的分布式训练启动方式，会自动处理进程管理和环境变量设置。

#### 使用所有可用GPU（8张）

```bash
torchrun --nproc_per_node=8 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --epochs 50 \
    --batch_size 8
```

#### 使用指定数量的GPU

```bash
# 使用4张GPU
torchrun --nproc_per_node=4 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --epochs 50 \
    --batch_size 8

# 使用2张GPU
torchrun --nproc_per_node=2 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --epochs 50 \
    --batch_size 8
```

#### 指定特定GPU

```bash
# 使用GPU 0, 1, 2, 3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --epochs 50 \
    --batch_size 8
```

### 方法2：使用 --ddp 参数（手动模式）

如果环境变量未设置，可以使用 `--ddp` 参数手动启用（需要手动设置环境变量）：

```bash
# 不推荐，建议使用 torchrun
python train.py \
    --ddp \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --epochs 50 \
    --batch_size 8
```

### 单GPU训练（向后兼容）

如果不使用 DDP，代码会自动回退到单GPU模式，使用方法与之前完全相同：

```bash
python train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --epochs 50 \
    --batch_size 4
```

## 重要参数说明

### 批次大小（batch_size）

- **单GPU模式**：`batch_size=4` 表示每个批次4个样本
- **DDP模式**：`batch_size=8` 表示**每个GPU**处理8个样本
  - 总有效批次大小 = `batch_size × GPU数量`
  - 例如：`batch_size=8`，8张GPU → 总批次大小 = 64

### 学习率调整

使用DDP时，总批次大小会增大，可能需要调整学习率：

- **线性缩放**：学习率 = 基础学习率 × GPU数量
  ```bash
  # 单GPU: lr=1e-4, batch_size=4
  # 8GPU: lr=8e-4, batch_size=4 (每GPU) → 总batch_size=32
  ```

- **平方根缩放**：学习率 = 基础学习率 × √(GPU数量)
  ```bash
  # 8GPU: lr=1e-4 × √8 ≈ 2.83e-4
  ```

- **保守方式**：保持学习率不变，让总批次大小自然增大

### 数据加载器优化

DDP模式下，代码会自动：
- 使用 `DistributedSampler` 分配数据
- 设置 `num_workers=4` 加速数据加载
- 启用 `pin_memory=True` 提升GPU传输速度

## 性能提升

使用8张RTX 2080 Ti GPU的预期加速比：

- **理论加速比**：接近8倍
- **实际加速比**：约7-8倍（考虑通信开销）
- **时间节省**：10小时训练 → 约1.2-1.4小时

## 注意事项

1. **模型保存**：只在 rank 0（主进程）保存模型，避免重复保存
2. **打印输出**：只在 rank 0 打印训练信息，避免输出混乱
3. **数据同步**：每个epoch会自动同步所有GPU的梯度
4. **验证集**：验证时也会使用分布式采样，确保数据一致性

## 故障排查

### 问题1：端口冲突

如果遇到端口占用错误，可以修改端口：

```bash
# 修改 train.py 中的 --master_port 默认值
# 或使用环境变量
export MASTER_PORT=12356
torchrun --nproc_per_node=8 train.py ...
```

### 问题2：内存不足

如果GPU内存不足，可以：
- 减小每GPU的 `batch_size`
- 减小模型大小（`--hidden_dim`）
- 使用更少的GPU

### 问题3：通信错误

如果遇到NCCL通信错误：
- 确保所有GPU在同一节点
- 检查NCCL环境变量设置
- 尝试使用 `NCCL_DEBUG=INFO` 查看详细日志

## 示例命令

### 完整训练示例（8GPU）

```bash
torchrun --nproc_per_node=8 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --hidden_dim 256 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --mask_ratio 0.15 \
    --save_dir checkpoints
```

### 快速测试（2GPU）

```bash
torchrun --nproc_per_node=2 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --epochs 5 \
    --batch_size 4 \
    --save_dir checkpoints_test
```

## 向后兼容性

- ✅ 单GPU训练完全兼容，无需修改
- ✅ 所有原有参数和功能保持不变
- ✅ 训练逻辑和模型结构完全一致
- ✅ 保存的模型格式兼容，可以正常加载
