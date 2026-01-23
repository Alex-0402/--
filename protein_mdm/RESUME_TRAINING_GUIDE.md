# 恢复训练使用指南

## 功能说明

现在支持从之前的检查点恢复训练，并且能够：
1. **恢复模型权重**：加载encoder和decoder的权重
2. **恢复优化器状态**：继续使用之前的学习率、动量等状态
3. **恢复训练历史**：加载之前的损失曲线数据，在原有基础上继续绘制
4. **恢复学习率调度**：继续使用之前的学习率调度状态

## 使用方法

### 基本用法

从最佳模型恢复训练：

```bash
python train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --epochs 300 \
    --resume checkpoints/best_model.pt
```

从特定epoch的检查点恢复：

```bash
python train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --epochs 300 \
    --resume checkpoints/checkpoint_epoch_175.pt
```

### 多GPU恢复训练

```bash
torchrun --nproc_per_node=8 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --epochs 300 \
    --resume checkpoints/best_model.pt \
    --batch_size 4
```

## 恢复训练的行为

### 1. Epoch编号
- 如果checkpoint是epoch 175，恢复后会从**epoch 176**开始训练
- 训练目标仍然是 `--epochs 300`，所以会训练到epoch 300

### 2. 损失曲线
- 会自动加载之前的 `train_losses` 和 `val_losses`
- 新的训练数据会**追加**到历史数据后面
- 可视化图表会显示完整的训练历史（包括恢复前和恢复后的数据）

### 3. 学习率调度
- 如果checkpoint包含学习率调度器状态，会恢复
- Warmup阶段：如果恢复时还在warmup阶段，会继续warmup
- CosineAnnealing阶段：会从checkpoint的位置继续

### 4. 最佳模型
- 会恢复 `best_val_loss`，只有新的验证损失更低时才会保存新的最佳模型

## 示例场景

### 场景1：训练中断后恢复

```bash
# 原始训练（训练到epoch 175时中断）
python train.py --epochs 300 ...

# 恢复训练（从epoch 175继续到300）
python train.py --epochs 300 --resume checkpoints/checkpoint_epoch_175.pt ...
```

### 场景2：在已有模型基础上继续训练更多epochs

```bash
# 之前训练了300个epochs，现在想再训练100个
python train.py --epochs 400 --resume checkpoints/best_model.pt ...
```

### 场景3：从最佳模型微调

```bash
# 使用最佳模型，继续训练但调整学习率
python train.py \
    --epochs 350 \
    --resume checkpoints/best_model.pt \
    --learning_rate 1e-5  # 使用更小的学习率微调
```

## 注意事项

1. **参数一致性**：
   - 模型架构参数（`hidden_dim`, `num_encoder_layers`等）必须与checkpoint一致
   - 如果checkpoint是用不同参数训练的，恢复可能会失败

2. **数据路径**：
   - `--pdb_path` 和 `--cache_dir` 应该与原始训练时一致
   - 确保数据路径正确，否则可能加载错误的数据

3. **学习率**：
   - 如果checkpoint包含学习率调度器状态，会使用checkpoint中的状态
   - 如果指定了新的 `--learning_rate`，可能会覆盖checkpoint的学习率（取决于调度器状态）

4. **DDP模式**：
   - 恢复训练时，所有GPU都会加载相同的checkpoint
   - 确保checkpoint文件对所有rank可访问

## 检查点文件内容

checkpoint文件包含：
- `epoch`: 训练到的epoch编号
- `encoder_state_dict`: Encoder模型权重
- `decoder_state_dict`: Decoder模型权重
- `optimizer_state_dict`: 优化器状态
- `cosine_scheduler_state_dict`: 学习率调度器状态（如果使用）
- `train_losses`: 训练损失历史（列表）
- `val_losses`: 验证损失历史（列表）
- `loss`: 最佳验证损失
- `warmup_epochs`: Warmup轮数
- `total_epochs`: 总训练轮数
- `max_lr`: 最大学习率

## 故障排查

### 问题1：KeyError 或 size mismatch

**原因**：模型架构参数与checkpoint不一致

**解决**：确保使用与训练时相同的参数：
```bash
python train.py \
    --hidden_dim 256 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --resume checkpoints/best_model.pt \
    ...
```

### 问题2：损失曲线没有延续

**原因**：checkpoint中没有保存训练历史

**解决**：使用较新的checkpoint（包含train_losses和val_losses）

### 问题3：学习率不正确

**原因**：checkpoint中的学习率调度器状态与当前设置不匹配

**解决**：检查checkpoint中的warmup_epochs和total_epochs是否与当前训练一致

## 完整示例

```bash
# 从epoch 175的checkpoint恢复，继续训练到300
torchrun --nproc_per_node=8 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --epochs 300 \
    --resume checkpoints/checkpoint_epoch_175.pt \
    --batch_size 4 \
    --use_discrete_diffusion \
    --num_diffusion_steps 1000 \
    --warmup_epochs 20 \
    --learning_rate 5e-4
```

恢复训练时，会显示：
```
从检查点恢复训练: checkpoints/checkpoint_epoch_175.pt
✅ 已加载检查点，从 epoch 176 继续训练
   训练历史: 175 个epoch
   验证历史: 175 个epoch
恢复训练: 从 epoch 176 继续，目标 300 epochs
```
