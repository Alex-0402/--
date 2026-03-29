#!/bin/bash
# 单卡/CPU 推理启动脚本

set -euo pipefail

# 切换到脚本所在目录，确保从任意路径启动都能找到 inference.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 用法说明
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "用法: bash run_inference.sh [PDB文件路径] [其他 inference.py 参数]"
    echo "示例(默认推断): bash run_inference.sh data/pdb_files/1CRN.pdb"
    echo "示例(对比策略): bash run_inference.sh data/pdb_files/1CRN.pdb --strategy both"
    echo "示例(修改步数): bash run_inference.sh data/pdb_files/1CRN.pdb --num_iterations 15 --energy_beta 0.15"
    echo "可选环境变量: GPU_ID, MODEL_PATH, OUTPUT_DIR, NUM_ITER"
    echo "日志文件: 默认保存到 logs/inferencelog_时间戳.log"
    exit 0
fi

# 检查是否传入了PDB文件
if [[ $# -eq 0 || "${1:-}" == -* ]]; then
    # 如果没传参数，或者第一个参数就是一个flag（例如 --model_path）
    echo "============== 提示 =============="
    echo "用法: bash run_inference.sh [待推理的PDB骨架路径] [其他参数]"
    echo "示例: bash run_inference.sh data/raw/1CRN.pdb"
    echo "错误: 未提供明确的 PDB 路径，脚本退出。"
    exit 1
fi

PDB_PATH="$1"
shift # 将第一个参数弹出，剩下的参数后续透传给 inference.py

# 默认环境变量配置（可通过 export 覆盖）
GPU_ID="${GPU_ID:-0}"
# 默认使用2万数据集跑出来的最新best_model
MODEL_PATH="${MODEL_PATH:-checkpoints_20000_0327/best_model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-inference_outputs}"
NUM_ITER="${NUM_ITER:-12}"

# 验证模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "警告: 模型文件 $MODEL_PATH 不存在！请检查路径或使用 export MODEL_PATH=... 指定。"
fi

# 消除 Python 输出缓冲，保证实时写入日志
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# 统一日志（stdout+stderr），便于后续追踪状态
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# 从输入路径中提取出基础文件名（如 1CRN）
PDB_BASENAME=$(basename "$PDB_PATH" .pdb)
LOG_FILE="$LOG_DIR/inference_${PDB_BASENAME}_$(date +%F_%H-%M-%S).log"

# 前台启动命令，并使用 tee 将输出同时打到终端和日志文件
echo "============== 开始推理 =============="
echo "目标 PDB: $PDB_PATH"
echo "使用 GPU: CUDA_VISIBLE_DEVICES=$GPU_ID"
echo "加载模型: $MODEL_PATH"
echo "循环步数: $NUM_ITER"
echo "输出目录: $OUTPUT_DIR"
echo "附加参数: $@"
echo "日志文件: $LOG_FILE"
echo "------------------------------------"

python inference.py \
    --model_path "$MODEL_PATH" \
    --pdb_path "$PDB_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_iterations "$NUM_ITER" \
    --energy_beta 0.1 \
    --strategy both \
    "$@" | tee "$LOG_FILE"

echo "------------------------------------"
echo "✅ 推理任务已完成!"
echo "📝 生成结构的PDB文件已保存在: ${OUTPUT_DIR}/ 目录下"
echo "===================================="
