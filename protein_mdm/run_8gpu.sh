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
    echo "示例(从头训练): bash run_8gpu.sh --no_resume"
    echo "可选环境变量: GPU_IDS, EPOCHS, BATCH_SIZE, SAVE_DIR, MASTER_PORT"
    echo "日志文件: 默认保存到 logs/trainlog_时间戳.log"
    exit 0
fi

# 参数解析：支持开关调试模式，并透传其余参数给 train.py
DEBUG_MODE=false
EXTRA_ARGS=()

# 恢复参数处理：默认从best_model恢复；若用户显式传入--resume则不再注入默认值
USER_PROVIDED_RESUME=false
DISABLE_RESUME=false
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

    if [[ "$arg" == "--no_resume" ]]; then
        DISABLE_RESUME=true
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
if [[ "$USER_PROVIDED_RESUME" == false && "$DISABLE_RESUME" == false ]]; then
    RESUME_ARG="--resume checkpoints_20000/best_model.pt"
fi

# 可配置运行参数（通过环境变量覆盖）
GPU_IDS="${GPU_IDS:-1,2,3,4,5,6,7}"
EPOCHS="${EPOCHS:-600}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAVE_DIR="${SAVE_DIR:-checkpoints_20000_0327}"
MASTER_PORT="${MASTER_PORT:-29506}"

# 从 GPU_IDS 自动计算进程数
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NPROC_PER_NODE="${#GPU_ARRAY[@]}"
if [[ "$NPROC_PER_NODE" -le 0 ]]; then
    echo "错误: GPU_IDS 解析失败，请检查格式（示例: 1,2,3,4,5,6,7）"
    exit 1
fi
PRIMARY_GPU="${GPU_ARRAY[0]}"

# 消除 Python 输出缓冲，保证实时写入日志
export PYTHONUNBUFFERED=1

# 启动命令（后台模式）
echo "启动 ${NPROC_PER_NODE} GPU DDP 训练 (后台模式)..."
echo "使用物理 GPU: ${GPU_IDS}"
echo "PyTorch 逻辑 RANK: 0 对应物理 GPU ${PRIMARY_GPU}"
if [[ "$DEBUG_MODE" == true ]]; then
    echo "模式: 调试模式 (debug_mode=ON)"
else
    echo "模式: 稳定模式 (debug_mode=OFF)"
fi
echo ""

# 统一日志（stdout+stderr），便于后台追踪训练状态
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/trainlog_$(date +%F_%H-%M-%S).log"
echo "日志文件已建立在: $LOG_FILE"
echo ""

nohup torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" train.py \
    --pdb_path data/cache_20000 \
    --cache_dir data/cache_20000 \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --gpu_ids "${GPU_IDS}" \
    --ddp \
    --use_predefined_split \
    --num_diffusion_steps 1000 \
    --warmup_epochs 20 \
    --learning_rate 5e-4 \
    --early_stopping_patience 50 \
    --early_stopping_min_delta 0.001 \
    --dropout 0.3 \
    --save_dir "${SAVE_DIR}" \
    ${RESUME_ARG} \
    ${DEBUG_ARG} \
    "${EXTRA_ARGS[@]}" </dev/null > "$LOG_FILE" 2>&1 &

# 获取后台进程 ID
bg_pid=$!
echo "✅ 训练已经在后台成功启动! 进程 PID: ${bg_pid}"
echo "📝 完整的控制台和训练指标均输出至: ${LOG_FILE}"
echo "🔍 实时查看训练进度请复制运行以下命令:"
echo "   tail -f ${LOG_FILE}"
echo ""
