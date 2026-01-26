#!/bin/bash
# 8 GPU DDP 训练启动脚本（使用 GPU 0-7，共8张卡）

# 杀掉残留进程
echo "清理残留进程..."
pkill -f "train.py" || true
sleep 2

# 注意：不要在这里设置 CUDA_VISIBLE_DEVICES
# train.py 会通过 --gpu_ids 参数自动设置
# 如果在这里设置，torchrun 可能无法正确识别 GPU 数量

# 调试日志（可选）
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# NCCL 优化设置（train.py 内部也会设置，这里作为备用）
export NCCL_P2P_DISABLE=1  # 禁用 P2P 防止 2080Ti 可能出现的 P2P 死锁
export NCCL_BLOCKING_WAIT=1  # 阻塞等待，报错时提供更多信息
export NCCL_TIMEOUT=1800     # 30分钟超时

# 启动命令 (nproc_per_node=8，使用所有 8 张 GPU)
echo "启动 8 GPU DDP 训练..."
echo "使用 GPU: 0,1,2,3,4,5,6,7 (物理编号)"
echo "PyTorch 逻辑编号: 0,1,2,3,4,5,6,7 (通过 --gpu_ids 映射)"
echo ""

torchrun --nproc_per_node=8 --master_port=29506 train.py \
    --pdb_path data/cache_20000 \
    --cache_dir data/cache_20000 \
    --batch_size 4 \
    --epochs 600 \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    --ddp \
    --use_predefined_split \
    --num_diffusion_steps 1000 \
    --warmup_epochs 20 \
    --learning_rate 5e-4 \
    --early_stopping_patience 50 \
    --early_stopping_min_delta 0.001 \
    --dropout 0.3 \
    --save_dir checkpoints_20000
    #--resume checkpoints/best_model.pt  # 可选：从断点继续训练

echo ""
echo "训练完成或已中断"
