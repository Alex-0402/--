#!/bin/bash
# 8 GPU DDP 训练启动脚本（使用 GPU 0-7，共8张卡）

set -euo pipefail

# 切换到脚本所在目录，确保从任意路径启动都能找到 train.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 用法说明
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "用法: bash run_8gpu.sh [--debug_mode] [其他 train.py 参数]"
    echo "示例(稳定模式): bash run_8gpu.sh"
    echo "示例(调试模式): bash run_8gpu.sh --debug_mode"
    echo "示例(附加参数): bash run_8gpu.sh --debug_mode --resume checkpoints_20000/best_model.pt"
    echo "日志文件: 默认保存到 logs/train_8gpu_时间戳.log"
    exit 0
fi

# 参数解析：支持开关调试模式，并透传其余参数给 train.py
DEBUG_MODE=false
EXTRA_ARGS=()

# 恢复参数处理：默认从best_model恢复；若用户显式传入--resume则不再注入默认值
USER_PROVIDED_RESUME=false
SKIP_NEXT=false
for arg in "$@"; do
    if [[ "$SKIP_NEXT" == true ]]; then
        SKIP_NEXT=false
        continue
    fi

    if [[ "$arg" == "--debug_mode" ]]; then
        DEBUG_MODE=true
        continue
    fi

    if [[ "$arg" == "--resume" ]]; then
        USER_PROVIDED_RESUME=true
        SKIP_NEXT=true
        continue
    fi

    if [[ "$arg" == --resume=* ]]; then
        USER_PROVIDED_RESUME=true
        continue
    fi

    EXTRA_ARGS+=("$arg")
done

DEBUG_ARG=""
if [[ "$DEBUG_MODE" == true ]]; then
    DEBUG_ARG="--debug_mode"
fi

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
export TORCH_NCCL_BLOCKING_WAIT=1  # 阻塞等待，报错时提供更多信息
export NCCL_TIMEOUT=1800     # 30分钟超时

# 默认从上次最佳模型继续训练（除非用户显式指定--resume）
RESUME_ARG=""
if [[ "$USER_PROVIDED_RESUME" == false ]]; then
    RESUME_ARG="--resume checkpoints_20000/best_model.pt"
fi

# 启动命令 (nproc_per_node=8，使用所有 8 张 GPU)
echo "启动 8 GPU DDP 训练..."
echo "使用 GPU: 0,1,2,3,4,5,6,7 (物理编号)"
echo "PyTorch 逻辑编号: 0,1,2,3,4,5,6,7 (通过 --gpu_ids 映射)"
if [[ "$DEBUG_MODE" == true ]]; then
    echo "模式: 调试模式 (debug_mode=ON)"
else
    echo "模式: 稳定模式 (debug_mode=OFF)"
fi
echo ""

# 统一日志（stdout+stderr），便于 tmux 断开后追踪训练状态
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_8gpu_$(date +%F_%H-%M-%S).log"
echo "日志文件: $LOG_FILE"
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
    --save_dir checkpoints_20000_new \
    # ${RESUME_ARG} \
    ${DEBUG_ARG} \
    "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
    # --resume checkpoints/best_model.pt  # 可选：从断点继续训练

echo ""
echo "训练完成或已中断"
