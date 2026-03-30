#!/bin/bash
# ==============================================================================
# MDM 测试集全量基准评估与对比脚本 (Testset Benchmark Script)
# 
# 作用:
# 1. 读取 test.txt 中指定的 1018 个未见过的 PDB 数据
# 2. 对每个蛋白质分别运行 `random` 和 `adaptive` 双策略侧链组装
# 3. 统计并收集所有评价指标 (FragAcc, ResExact, RMSD, Clash)
# 4. 生成统计学分布报告及 paired-difference 制表 
# ==============================================================================

set -e

# =========================
# 配置参数
# =========================
MODEL_PATH="checkpoints_20000_0328/best_model.pt"
TEST_LIST="data/cache_20000/test.txt"
RAW_DATA_DIR="raw_data"
OUTPUT_CSV="testset_benchmark_final.csv"
MAX_SAMPLES=20  # 可修改为 20~50 用作小批测试，当前默认跑完所有

echo "======================================================================"
echo "📊 开始 MDM 测试集大规模基准测试"
echo "======================================================================"
echo "模型权重: $MODEL_PATH"
echo "测试名单: $TEST_LIST"
echo "原始数据: $RAW_DATA_DIR"
echo "======================================================================"

python3 scripts/benchmark_inference_strategies.py \
    --model_path "$MODEL_PATH" \
    --pdb_dir "$RAW_DATA_DIR" \
    --test_list "$TEST_LIST" \
    --output_csv "$OUTPUT_CSV" \
    --max_samples $MAX_SAMPLES \
    --seeds 42 \
    --num_iterations 12 \
    --min_commit_ratio 0.05 \
    --max_commit_ratio 0.20

echo ""
echo "🎉 全量推理评估完成！"
echo "📝 每条 PDB 两组策略的明细数据已保存在: $OUTPUT_CSV"
echo "建议将此 CSV 文件导入 Excel 或用于论文配图代码读取。"
