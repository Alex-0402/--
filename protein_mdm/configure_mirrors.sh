#!/bin/bash
# 配置 Conda 和 pip 使用清华镜像源的脚本

echo "=========================================="
echo "配置 Conda 和 pip 使用清华镜像源"
echo "=========================================="

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "⚠️  conda 命令未找到，尝试初始化..."
    
    # 检查常见的 conda 安装位置
    if [ -f "$HOME/miniconda3/bin/conda" ]; then
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    elif [ -f "$HOME/anaconda3/bin/conda" ]; then
        eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
    else
        echo "❌ 未找到 conda，请先安装 Miniconda"
        exit 1
    fi
else
    eval "$(conda shell.bash hook)"
fi

# 配置 Conda 使用清华源
echo ""
echo "🌐 配置 Conda 使用清华镜像源..."
conda config --remove-key channels 2>/dev/null
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --set show_channel_urls yes

echo "✅ Conda 镜像源配置完成"
echo ""
echo "当前 Conda 镜像源配置："
conda config --show channels

# 配置 pip 使用清华源
echo ""
echo "🌐 配置 pip 使用清华镜像源..."

# 创建 pip 配置目录（如果不存在）
mkdir -p ~/.pip

# 创建或更新 pip 配置文件
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

echo "✅ pip 镜像源配置完成"
echo ""
echo "pip 配置文件位置: ~/.pip/pip.conf"
echo "配置内容："
cat ~/.pip/pip.conf

echo ""
echo "=========================================="
echo "✅ 镜像源配置完成！"
echo "=========================================="
echo ""
echo "现在可以使用以下命令测试："
echo "  conda search numpy"
echo "  pip install --upgrade pip"
echo ""
