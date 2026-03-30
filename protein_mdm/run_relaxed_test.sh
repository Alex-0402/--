#!/bin/bash
# 单条PDB序列恢复率（包含Relaxed AAR模糊匹配）测试脚本
# 用于快速验证加入了降维容错机制后的新恢复率指标

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "checkpoints_20000/best_model.pt" ]; then
    echo "未找到 checkpoints_20000/best_model.pt 模型文件，请检查！"
    exit 1
fi

PDB_FILE=${1:-raw_data/152l.pdb}

if [ ! -f "$PDB_FILE" ]; then
    echo "未找到测试PDB: $PDB_FILE"
    echo "请提供一个存在的PDB路径作为第一个参数！"
    exit 1
fi

echo "====================================================="
echo "  运行松弛版序列恢复率 (Relaxed AAR) 评估演示 "
echo "====================================================="
echo "输入PDB: $PDB_FILE"
echo "使用模型: checkpoints_20000/best_model.pt"
echo "推理策略: adaptive"
echo "计算阈值: Threshold=0.65"
echo ""

# 使用 inference.py 进行推理；我们在内部已经挂载了 evaluate_relaxed 脚本
python inference.py \
    --model_path checkpoints_20000_0328/best_model.pt \
    --pdb_path "$PDB_FILE" \
    --strategy adaptive

echo "====================================================="
echo "✅ 测试完成！您可以在输出日志中查看： "
echo "  - Residue侧链严格一致率(Strict Exact)"
echo "  - Residue模糊投射恢复率(Relaxed AAR) ✨"
echo "-----------------------------------------------------"
echo "如果是要进行全量测试，请继续使用 ./run_testset_benchmark.sh，"
echo "该基准测试脚本已被更新，会自动搜集并统计新加入的 RlxAAR 字段！"
