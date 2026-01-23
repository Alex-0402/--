#!/bin/bash
# 紧急修复：杀死卡住的进程并重新启动（启用可视化记录损失曲线）

echo "🔧 紧急修复：清理卡住的进程..."

# 1. 杀死所有训练进程
pkill -9 -f "train.py"
pkill -9 -f "torchrun"
sleep 3

# 2. 确认清理完成
if ps aux | grep -E "(train.py|torchrun)" | grep -v grep > /dev/null; then
    echo "❌ 仍有进程在运行，请手动检查"
    ps aux | grep -E "(train.py|torchrun)" | grep -v grep
    exit 1
fi

echo "✅ 进程已清理"

# 3. 设置环境变量
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=WARN  # 减少日志输出

# 4. 重新启动训练（启用可视化记录损失曲线）
echo ""
echo "🚀 重新启动训练（已启用可视化记录损失曲线）..."
echo ""

cd /home/Oliver-0402/--/protein_mdm

# 使用最佳模型继续训练
RESUME_MODEL="checkpoints/best_model.pt"
if [ ! -f "$RESUME_MODEL" ]; then
    echo "⚠️  警告: best_model.pt 不存在，尝试使用最新的 checkpoint"
    RESUME_MODEL=$(ls -t checkpoints/checkpoint_epoch_*.pt 2>/dev/null | head -1)
    if [ -z "$RESUME_MODEL" ]; then
        echo "❌ 错误: 找不到任何模型文件"
        exit 1
    fi
fi

echo "使用模型: $RESUME_MODEL"

torchrun --nproc_per_node=8 train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --use_predefined_split \
    --resume "$RESUME_MODEL" \
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
    --plot_every 5  # 每5个epoch绘制一次图表
