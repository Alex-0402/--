#!/bin/bash
# 重新生成缓存文件（使用更新后的 vocabulary）
# 保留现有的 train.txt 和 val.txt 划分

cd /home/Oliver-0402/--/protein_mdm

echo "=========================================="
echo "重新生成缓存文件（使用更新后的 vocabulary）"
echo "=========================================="
echo ""

# 检查原始 PDB 文件目录
PDB_DIR="raw_data"
CACHE_DIR="data/cache"

if [ ! -d "$PDB_DIR" ]; then
    echo "❌ 错误: 找不到原始 PDB 文件目录: $PDB_DIR"
    echo "   请确保原始 PDB 文件在 $PDB_DIR 目录中"
    exit 1
fi

echo "1. 检查现有缓存..."
if [ -d "$CACHE_DIR" ]; then
    # 统计 .pt 文件数量
    PT_COUNT=$(find "$CACHE_DIR" -name "*.pt" -type f | wc -l)
    echo "   找到 $PT_COUNT 个缓存文件 (.pt)"
    
    # 检查是否有划分文件
    if [ -f "$CACHE_DIR/train.txt" ] && [ -f "$CACHE_DIR/val.txt" ]; then
        TRAIN_COUNT=$(wc -l < "$CACHE_DIR/train.txt")
        VAL_COUNT=$(wc -l < "$CACHE_DIR/val.txt")
        echo "   找到数据集划分文件:"
        echo "     - train.txt: $TRAIN_COUNT 个文件"
        echo "     - val.txt: $VAL_COUNT 个文件"
        echo ""
        echo "   ⚠️  将删除所有 .pt 缓存文件，但保留 train.txt 和 val.txt"
        read -p "   是否继续？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "   已取消"
            exit 1
        fi
        
        # 删除所有 .pt 文件，但保留划分文件
        echo ""
        echo "2. 删除旧的缓存文件..."
        find "$CACHE_DIR" -name "*.pt" -type f -delete
        echo "   ✅ 已删除所有 .pt 缓存文件"
    else
        echo "   ⚠️  未找到 train.txt 或 val.txt，将重新生成所有文件"
        read -p "   是否继续？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "   已取消"
            exit 1
        fi
        
        # 删除所有文件（包括划分文件）
        echo ""
        echo "2. 清理缓存目录..."
        rm -f "$CACHE_DIR"/*.pt "$CACHE_DIR"/train.txt "$CACHE_DIR"/val.txt "$CACHE_DIR"/test.txt 2>/dev/null
        echo "   ✅ 已清理缓存目录"
    fi
else
    echo "   缓存目录不存在，将创建: $CACHE_DIR"
    mkdir -p "$CACHE_DIR"
fi

echo ""
echo "3. 重新生成缓存文件..."
echo "   使用更新后的 vocabulary（修复了 SER, CYS, ILE 的片段映射）"
echo ""

# 检查是否有现有的划分文件
if [ -f "$CACHE_DIR/train.txt" ] && [ -f "$CACHE_DIR/val.txt" ]; then
    echo "   检测到现有划分文件，将保持相同的划分"
    echo "   注意: 预处理脚本会重新生成划分，但我们会手动处理"
    echo ""
    echo "   ⚠️  由于预处理脚本会重新划分，建议："
    echo "   1. 先备份 train.txt 和 val.txt"
    echo "   2. 运行预处理脚本重新生成缓存"
    echo "   3. 恢复 train.txt 和 val.txt"
    echo ""
    read -p "   是否继续使用预处理脚本？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   已取消"
        exit 1
    fi
    
    # 备份划分文件
    cp "$CACHE_DIR/train.txt" "$CACHE_DIR/train.txt.backup"
    cp "$CACHE_DIR/val.txt" "$CACHE_DIR/val.txt.backup"
    echo "   ✅ 已备份划分文件"
fi

# 运行预处理脚本
echo ""
echo "4. 运行预处理脚本..."
python scripts/preprocess_dataset.py \
    --pdb_dir "$PDB_DIR" \
    --output_dir "$CACHE_DIR" \
    --train_ratio 0.9 \
    --val_ratio 0.1 \
    --test_ratio 0.0 \
    --num_workers $(nproc)

# 如果之前有划分文件，询问是否恢复
if [ -f "$CACHE_DIR/train.txt.backup" ] && [ -f "$CACHE_DIR/val.txt.backup" ]; then
    echo ""
    echo "5. 恢复原始划分..."
    read -p "   是否恢复之前的 train.txt 和 val.txt 划分？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp "$CACHE_DIR/train.txt.backup" "$CACHE_DIR/train.txt"
        cp "$CACHE_DIR/val.txt.backup" "$CACHE_DIR/val.txt"
        echo "   ✅ 已恢复原始划分"
    else
        echo "   ℹ️  使用新的划分"
    fi
    # 删除备份文件
    rm -f "$CACHE_DIR/train.txt.backup" "$CACHE_DIR/val.txt.backup"
fi

echo ""
echo "=========================================="
echo "缓存重新生成完成！"
echo "=========================================="
echo ""
echo "下一步: 可以开始训练了"
echo "  bash 重新开始训练.sh"
