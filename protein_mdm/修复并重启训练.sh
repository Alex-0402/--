#!/bin/bash
# 修复训练卡住问题并重启训练

echo "=========================================="
echo "修复训练卡住问题"
echo "=========================================="
echo ""

# 1. 停止当前训练进程
echo "1. 停止当前训练进程..."
pkill -f "train.py"
pkill -f "torchrun"
sleep 5

# 2. 检查是否还有残留进程
if ps aux | grep -E "(train.py|torchrun)" | grep -v grep > /dev/null; then
    echo "⚠️  仍有残留进程，强制杀死..."
    pkill -9 -f "train.py"
    pkill -9 -f "torchrun"
    sleep 2
fi

# 3. 确认清理完成
if ps aux | grep -E "(train.py|torchrun)" | grep -v grep > /dev/null; then
    echo "❌ 仍有进程在运行，请手动检查"
    ps aux | grep -E "(train.py|torchrun)" | grep -v grep
    exit 1
else
    echo "✅ 所有训练进程已清理"
fi

echo ""
echo "2. 已修复的问题："
echo "   ✅ 将 find_unused_parameters 改为 False（提高性能）"
echo "   ✅ 数据加载器已配置为 num_workers=0（避免死锁）"
echo "   ✅ 在多个关键位置添加 DDP barrier 同步："
echo "      - train_epoch 开始前"
echo "      - train_epoch 结束后"
echo "      - validate 开始前"
echo "      - validate 结束后"
echo "      - epoch 循环开始前"
echo "      - set_epoch 后"
echo "      - 训练完成后"
echo "   ✅ 禁用非 rank 0 进程的 tqdm 进度条（减少输出干扰）"
echo ""

# 4. 设置环境变量
export NCCL_TIMEOUT=3600

# 5. 重新启动训练
echo "3. 重新启动训练..."
echo ""

cd /home/Oliver-0402/--/protein_mdm

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
echo "训练完成或中断"
echo "=========================================="
