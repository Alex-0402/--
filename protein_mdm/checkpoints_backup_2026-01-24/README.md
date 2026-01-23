# 模型备份 - 2026年1月24日

此文件夹包含在重新训练前保存的最新模型和损失图。

## 备份内容

### 模型文件
- **best_model.pt**: 验证集上表现最好的模型（52MB）
- **checkpoint_epoch_260.pt**: 第260轮的检查点模型（52MB）

### 损失图
- **training_curves_epoch_245.png**: 第245轮的训练曲线图（602KB）
- **val_loss_plateau_analysis.png**: 验证损失平台分析图（858KB）

## 备份时间
2026年1月24日 01:36

## 使用说明
如果需要恢复训练，可以使用以下命令：

```bash
python train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --epochs 300 \
    --resume checkpoints_backup_2026-01-24/best_model.pt
```

或者从特定epoch继续：

```bash
python train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --epochs 300 \
    --resume checkpoints_backup_2026-01-24/checkpoint_epoch_260.pt
```
