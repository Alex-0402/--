#!/bin/bash
# 下载更多 CATH 数据的脚本

echo "=========================================="
echo "下载更多 CATH S40 数据"
echo "=========================================="

# 删除旧的列表文件（如果存在），强制重新下载完整的 CATH S40 列表
if [ -f "data/meta/cath_s40_list.txt" ]; then
    echo "删除旧的列表文件，准备下载完整的 CATH S40 列表..."
    rm data/meta/cath_s40_list.txt
fi

# 下载 200 个蛋白质结构
echo ""
echo "开始下载 200 个蛋白质结构..."
python scripts/download_cath_subset.py \
    --output_dir raw_data \
    --limit 200 \
    --random \
    --skip_existing

echo ""
echo "=========================================="
echo "下载完成！"
echo "=========================================="
echo ""
echo "下一步：预处理数据"
echo "python scripts/preprocess_dataset.py \\"
echo "    --pdb_dir raw_data \\"
echo "    --output_dir data/cache \\"
echo "    --train_ratio 0.8 \\"
echo "    --val_ratio 0.1 \\"
echo "    --test_ratio 0.1"
