#!/bin/bash
# 重启训练脚本 - 解决卡住问题

echo "正在清理卡住的训练进程..."

# 1. 杀死所有训练进程
pkill -f "train.py"
pkill -f "torchrun"

# 2. 等待进程完全退出
echo "等待进程退出..."
sleep 5

# 3. 检查是否还有残留进程
if ps aux | grep -E "(train.py|torchrun)" | grep -v grep > /dev/null; then
    echo "⚠️  仍有残留进程，强制杀死..."
    pkill -9 -f "train.py"
    pkill -9 -f "torchrun"
    sleep 2
fi

# 4. 确认清理完成
if ps aux | grep -E "(train.py|torchrun)" | grep -v grep > /dev/null; then
    echo "❌ 仍有进程在运行，请手动检查"
    ps aux | grep -E "(train.py|torchrun)" | grep -v grep
    exit 1
else
    echo "✅ 所有训练进程已清理"
fi

# 5. 设置环境变量
export NCCL_TIMEOUT=3600

# 6. 重新启动训练
echo ""
echo "重新启动训练..."
echo ""

cd /home/Oliver-0402/--/protein_mdm

torchrun --nproc_per_node=8 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --resume checkpoints/best_model.pt \
    --epochs 600 \
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
