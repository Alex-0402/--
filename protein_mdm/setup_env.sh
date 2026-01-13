#!/bin/bash
# 快速设置虚拟环境的脚本

echo "=========================================="
echo "Protein MDM 环境设置脚本"
echo "=========================================="

# 检查是否已存在虚拟环境
if [ -d "venv" ]; then
    echo "⚠️  虚拟环境已存在，跳过创建步骤"
else
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        echo "✅ 虚拟环境创建成功"
    else
        echo "❌ 虚拟环境创建失败，请检查 Python 安装"
        exit 1
    fi
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 升级 pip
echo "⬆️  升级 pip..."
pip install --upgrade pip

# 安装依赖
echo "📥 安装项目依赖..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 环境设置完成！"
    echo "=========================================="
    echo ""
    echo "下次使用时，请运行："
    echo "  source venv/bin/activate"
    echo ""
    echo "退出虚拟环境："
    echo "  deactivate"
    echo ""
else
    echo "❌ 依赖安装失败"
    exit 1
fi
