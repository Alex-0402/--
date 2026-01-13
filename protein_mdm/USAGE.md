# 使用指南

## 快速开始

### 1. 训练模型

```bash
# 基本训练
python train.py --pdb_path data/pdb_files --epochs 50 --batch_size 4

# 使用自定义参数
python train.py \
    --pdb_path data/pdb_files \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --mask_ratio 0.15 \
    --masking_strategy random \
    --save_dir checkpoints
```

**参数说明：**
- `--pdb_path`: PDB 文件路径或目录（必需）
- `--epochs`: 训练轮数（默认：50）
- `--batch_size`: 批次大小（默认：4）
- `--learning_rate`: 学习率（默认：1e-4）
- `--mask_ratio`: 掩码比例（默认：0.15）
- `--masking_strategy`: 掩码策略，`random` 或 `block`（默认：random）
- `--save_dir`: 模型保存目录（默认：checkpoints）

### 2. 生成侧链

```bash
# 使用训练好的模型生成侧链
python inference.py \
    --model_path checkpoints/best_model.pt \
    --pdb_path data/pdb_files/1CRN.pdb
```

**参数说明：**
- `--model_path`: 模型检查点路径（必需）
- `--pdb_path`: 输入 PDB 文件（骨架结构）（必需）
- `--output_path`: 输出路径（可选，待实现结构重建）
- `--hidden_dim`: 隐藏层维度（需与训练时一致，默认：256）

## 训练流程

1. **准备数据**
   ```bash
   # 下载 PDB 文件到 data/pdb_files/
   mkdir -p data/pdb_files
   cd data/pdb_files
   curl -O https://files.rcsb.org/download/1CRN.pdb
   curl -O https://files.rcsb.org/download/1UBQ.pdb
   ```

2. **开始训练**
   ```bash
   python train.py --pdb_path data/pdb_files --epochs 50
   ```

3. **监控训练**
   - 训练过程中会显示损失值
   - 最佳模型会自动保存到 `checkpoints/best_model.pt`
   - 每 10 个 epoch 保存一次检查点

4. **使用模型**
   ```bash
   python inference.py \
       --model_path checkpoints/best_model.pt \
       --pdb_path data/pdb_files/1CRN.pdb
   ```

## 训练技巧

### 调整批次大小
- 如果内存不足，减小 `--batch_size`
- 如果 GPU 内存充足，可以增大批次大小以加速训练

### 调整学习率
- 默认学习率：`1e-4`
- 如果损失不下降，尝试 `1e-3` 或 `1e-5`
- 学习率会自动调整（ReduceLROnPlateau）

### 掩码策略
- **random**: 随机掩码（推荐用于训练）
- **block**: 块状掩码（模拟连续片段缺失）

### 验证集
- 默认使用 10% 的数据作为验证集
- 可以通过 `--val_split` 调整比例
- 设置为 0 则不使用验证集

## 模型检查点

训练过程中会保存：
- `best_model.pt`: 验证损失最低的模型
- `checkpoint_epoch_N.pt`: 每 N 个 epoch 的检查点

检查点包含：
- 模型权重（encoder + decoder）
- 优化器状态
- 训练历史
- 训练轮数

## 常见问题

### Q: 训练时内存不足
**A**: 减小 `--batch_size`，或使用更小的模型（减小 `--hidden_dim`）

### Q: 损失不下降
**A**: 
- 检查数据是否正确加载
- 尝试调整学习率
- 检查掩码比例是否合理（建议 0.1-0.3）

### Q: 如何恢复训练
**A**: 修改训练脚本加载检查点（待实现）

### Q: 推理时维度不匹配
**A**: 确保 `--hidden_dim` 与训练时一致

## 下一步

- [ ] 实现结构重建功能（从片段和扭转角重建原子坐标）
- [ ] 实现评估指标（RMSD、准确率等）
- [ ] 实现自适应推理策略
- [ ] 添加数据增强功能
