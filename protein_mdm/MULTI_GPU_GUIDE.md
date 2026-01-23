# 多GPU训练使用指南

## 快速开始

### 方法1：使用 torchrun（推荐）

`torchrun` 是 PyTorch 推荐的分布式训练启动方式，会自动处理进程管理和环境变量设置。

#### 使用所有可用GPU（自动检测）
```bash
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --epochs 300 \
    --batch_size 4 \
    --use_discrete_diffusion \
    --num_diffusion_steps 1000 \
    --warmup_epochs 20 \
    --learning_rate 5e-4
```

#### 使用指定数量的GPU
```bash
# 使用4张GPU
torchrun --nproc_per_node=4 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --epochs 300 \
    --batch_size 4 \
    --use_discrete_diffusion \
    --num_diffusion_steps 1000

# 使用2张GPU
torchrun --nproc_per_node=2 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --epochs 300 \
    --batch_size 4
```

#### 使用指定的GPU
```bash
# 只使用GPU 0, 1, 2, 3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --epochs 300 \
    --batch_size 4
```

### 方法2：使用 python -m torch.distributed.launch（旧方法，不推荐）

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --epochs 300 \
    --batch_size 4
```

## 重要说明

### 批次大小
- **DDP模式**：`--batch_size=4` 表示**每个GPU**处理4个样本
- **总批次大小** = `batch_size × GPU数量`
- 例如：4个GPU，batch_size=4 → 总批次大小 = 16

### 学习率调整
使用多GPU时，总批次大小增大，通常需要按比例调整学习率：
- 单GPU: `lr = 5e-4`, `batch_size = 4`
- 4 GPU: `lr = 5e-4 × 4 = 2e-3`, `batch_size = 4` (每个GPU)

或者使用线性缩放规则：
```bash
# 4个GPU，学习率 = 5e-4 × sqrt(4) = 1e-3
torchrun --nproc_per_node=4 train.py \
    --learning_rate 1e-3 \
    --batch_size 4 \
    ...
```

### 检查GPU使用情况
训练时，在另一个终端运行：
```bash
watch -n 1 nvidia-smi
```

## 常见问题

### Q: 如何知道是否在使用多GPU？
A: 训练开始时，会显示：
```
DDP 模式: 启用 (world_size=4)
模型已包装为 DDP (device_id=0)
```

### Q: 单GPU训练会受影响吗？
A: 不会。如果不使用 `torchrun`，代码会自动回退到单GPU模式。

### Q: 如何只使用部分GPU？
A: 使用 `CUDA_VISIBLE_DEVICES` 环境变量：
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py ...
```

### Q: 训练速度提升多少？
A: 理论上，N个GPU应该接近N倍加速。实际加速比取决于：
- 模型大小
- 数据加载速度
- GPU间通信开销
- 通常能达到 0.8-0.95×N 的加速比

## 完整示例

### 8 GPU训练（推荐配置）
```bash
torchrun --nproc_per_node=8 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --epochs 300 \
    --batch_size 4 \
    --use_discrete_diffusion \
    --num_diffusion_steps 1000 \
    --warmup_epochs 20 \
    --learning_rate 5e-4 \
    --save_dir checkpoints
```

### 2 GPU训练（小规模）
```bash
torchrun --nproc_per_node=2 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --epochs 300 \
    --batch_size 8 \
    --use_discrete_diffusion \
    --num_diffusion_steps 1000
```
