#!/bin/bash
# 重新开始训练脚本 - 使用更新后的 vocabulary
# 注意：此脚本从零开始训练，不使用任何旧的检查点

cd /home/Oliver-0402/--/protein_mdm

echo "=========================================="
echo "开始新的训练（使用更新后的 vocabulary）"
echo "=========================================="
echo ""
echo "训练配置："
echo "  - 8卡并行训练"
echo "  - 从零开始（不使用旧检查点）"
echo "  - 数据集: data/cache"
echo "  - 批次大小: 4 (每GPU)"
echo "  - 总批次大小: 32 (8 GPU × 4)"
echo "  - 学习率: 2e-4"
echo "  - 训练轮数: 500"
echo ""

# 设置环境变量
export NCCL_TIMEOUT=3600

# 开始训练（注意：没有 --resume 参数，从零开始）
torchrun --nproc_per_node=8 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --epochs 500 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --weight_decay 1e-4 \
    --warmup_epochs 20 \
    --early_stopping_patience 30 \
    --early_stopping_min_delta 0.001 \
    --num_diffusion_steps 1000 \
    --masking_strategy random \
    --save_dir checkpoints \
    --visualize \
    --plot_every 5

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
