#!/bin/bash
# ==============================================================================
# 数据预处理与缓存生成完整流水线 (Pipelines for Data Preprocessing & Caching)
# 
# 作用:
# 1. 读取原始 PDB 文件夹 (如 raw_data/)
# 2. 结合白名单 (high_quality_pdbs.txt) 过滤不要的结构
# 3. 计算所有需要的特征 (包括修复后的 chi1~chi4 全级侧链角度和 Fragment Mappings)
# 4. 把张量数据序列化保存为 .pt 缓存，放置在独立的 Cache 目录
# 5. 按照定义好的比例划分训练(train.txt)、验证(val.txt)、测试(test.txt)数据
# ==============================================================================

# 中断时报错退出
set -e

# =========================
# 1. 目录与参数配置
# =========================
RAW_DATA_DIR="raw_data"                        # 原始 PDB 存放路径
OUTPUT_CACHE_DIR="data/cache_20000"            # 处理后的 .pt 文件目标路径
ALLOWLIST_FILE="data/high_quality_pdbs.txt"    # 高质量数据集筛选名单

NUM_WORKERS=16                                 # 并行处理进程数 (根据机器 CPU 调整)
TRAIN_RATIO=0.8                                # 训练集比例
VAL_RATIO=0.1                                  # 验证集比例
TEST_RATIO=0.1                                 # 测试集比例

echo "======================================================================"
echo "🚀 开始执行 MDM 数据流水线..."
echo "📂 源数据目录: $RAW_DATA_DIR"
echo "🗂️ 目标缓存目录: $OUTPUT_CACHE_DIR"
echo "✅ 数据过滤器: $ALLOWLIST_FILE"
echo "======================================================================"

# =========================
# 2. 创建或重置预处理输出目录
# =========================
if [ -d "$OUTPUT_CACHE_DIR" ]; then
    echo "⚠️ 检测到目标目录 $OUTPUT_CACHE_DIR 已经存在。"
    echo "💡 为了防止残留旧版缓存数据 (旧的单级 chi1 维度不匹配等)，推荐先将其清理。"
    # read -p "需要清理现有缓存吗？(y/n): " clear_cache
    # if [ "$clear_cache" = "y" ]; then
    #     rm -rf "$OUTPUT_CACHE_DIR"/*.pt
    # fi
    # (如果需要全自动化执行且无视旧文件，可以取消上述注释或加上自动 rm)
fi

mkdir -p "$OUTPUT_CACHE_DIR"

# =========================
# 3. 运行 Python 数据集转化引擎
# =========================
# （这里调用 scripts/preprocess_dataset.py 生成缓存和切割列表）
echo "⏳ 正在提取蛋白质 3D 特征进行预处理和打包，请耐心等待..."
echo "使用的进程数：$NUM_WORKERS"

python3 scripts/preprocess_dataset.py \
    --pdb_dir "$RAW_DATA_DIR" \
    --output_dir "$OUTPUT_CACHE_DIR" \
    --num_workers $NUM_WORKERS \
    --train_ratio $TRAIN_RATIO \
    --val_ratio $VAL_RATIO \
    --test_ratio $TEST_RATIO \
    --allowlist_txt "$ALLOWLIST_FILE"

# =========================
# 4. 后处理检查
# =========================
if [ -f "$OUTPUT_CACHE_DIR/train.txt" ] && [ -f "$OUTPUT_CACHE_DIR/val.txt" ]; then
    TRAIN_COUNT=$(wc -l < "$OUTPUT_CACHE_DIR/train.txt")
    VAL_COUNT=$(wc -l < "$OUTPUT_CACHE_DIR/val.txt")
    TEST_COUNT=$(wc -l < "$OUTPUT_CACHE_DIR/test.txt")
    echo "======================================================================"
    echo "🎉 数据缓存构建完毕！数据集分割信息如下："
    echo "   - 训练集 (train.txt) : $TRAIN_COUNT 条"
    echo "   - 验证集 (val.txt)   : $VAL_COUNT 条"
    echo "   - 测试集 (test.txt)  : $TEST_COUNT 条"
    echo ""
    echo "✅ 您现在可以使用指定该 cache 目录进行运行训练，例如："
    echo "   bash run_8gpu.sh"
    echo "======================================================================"
else
    echo "❌ 警告: 在 $OUTPUT_CACHE_DIR 下未找到正确生成 train.txt/val.txt 划分数据。"
    exit 1
fi
