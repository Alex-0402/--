#!/bin/bash
# 使用 Miniconda 配置环境的脚本

echo "=========================================="
echo "Protein MDM 环境设置脚本 (Miniconda)"
echo "=========================================="

# 检查 conda 是否已安装
if ! command -v conda &> /dev/null; then
    echo "⚠️  conda 命令未在 PATH 中找到"
    echo ""
    
    # 检查常见的 conda 安装位置
    CONDA_PATHS=(
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "/opt/conda"
        "/usr/local/miniconda3"
        "/usr/local/anaconda3"
    )
    
    CONDA_FOUND=false
    CONDA_BASE=""
    
    for path in "${CONDA_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/bin/conda" ]; then
            CONDA_FOUND=true
            CONDA_BASE="$path"
            echo "✅ 在 $path 找到 Miniconda/Anaconda"
            break
        fi
    done
    
    if [ "$CONDA_FOUND" = true ]; then
        echo ""
        echo "正在初始化 conda..."
        # 初始化 conda
        eval "$($CONDA_BASE/bin/conda shell.bash hook)"
        
        # 验证 conda 是否可用
        if command -v conda &> /dev/null; then
            echo "✅ Conda 初始化成功"
        else
            echo "❌ Conda 初始化失败"
            echo ""
            echo "请手动运行以下命令初始化 conda："
            echo "  $CONDA_BASE/bin/conda init bash"
            echo "  source ~/.bashrc"
            echo "然后重新运行此脚本"
            exit 1
        fi
    else
        echo "❌ 未找到 Miniconda/Anaconda 安装"
        echo ""
        echo "请先安装 Miniconda："
        echo "  1. 下载: https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "  2. 安装: bash Miniconda3-latest-Linux-x86_64.sh"
        echo "  3. 初始化: ~/miniconda3/bin/conda init bash"
        echo "  4. 重新加载: source ~/.bashrc"
        echo "  5. 重新运行此脚本"
        echo ""
        exit 1
    fi
else
    # conda 已在 PATH 中，初始化 shell hook
    eval "$(conda shell.bash hook)"
fi

# 配置 Conda 使用清华源
echo "🌐 配置 Conda 使用清华镜像源..."
# 清除现有 channels（避免重复）
conda config --remove-key channels 2>/dev/null || true
# 添加清华镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --set show_channel_urls yes
echo "✅ Conda 镜像源配置完成（使用清华源，下载速度更快）"

# 环境名称
ENV_NAME="protein_mdm"

# 检查是否已存在环境（默认位置）
ENV_EXISTS_DEFAULT=false
if conda env list | grep -q "^${ENV_NAME} "; then
    ENV_EXISTS_DEFAULT=true
fi

# 检查是否在项目目录下存在环境
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH_PROJECT="${PROJECT_DIR}/.conda_env"
ENV_EXISTS_PROJECT=false
if [ -d "${ENV_PATH_PROJECT}" ]; then
    ENV_EXISTS_PROJECT=true
fi

if [ "$ENV_EXISTS_DEFAULT" = true ] || [ "$ENV_EXISTS_PROJECT" = true ]; then
    echo "⚠️  Conda 环境已存在"
    if [ "$ENV_EXISTS_DEFAULT" = true ]; then
        echo "   位置: ${ENV_NAME} (默认位置)"
    fi
    if [ "$ENV_EXISTS_PROJECT" = true ]; then
        echo "   位置: ${ENV_PATH_PROJECT}"
    fi
    
    read -p "是否删除并重新创建？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  删除旧环境..."
        if [ "$ENV_EXISTS_DEFAULT" = true ]; then
            conda env remove -n ${ENV_NAME} -y 2>/dev/null
        fi
        if [ "$ENV_EXISTS_PROJECT" = true ]; then
            rm -rf "${ENV_PATH_PROJECT}"
        fi
    else
        echo "使用现有环境，跳过创建步骤"
        echo ""
        echo "激活环境："
        if [ "$ENV_EXISTS_DEFAULT" = true ]; then
            echo "  conda activate ${ENV_NAME}"
        else
            echo "  conda activate ${ENV_PATH_PROJECT}"
        fi
        exit 0
    fi
fi

# 创建 conda 环境
echo "📦 创建 Conda 环境 '${ENV_NAME}' (Python 3.10)..."

# 尝试在默认位置创建环境
if conda create -n ${ENV_NAME} python=3.10 -y 2>/dev/null; then
    echo "✅ 环境创建成功（在默认位置）"
    ENV_PATH="${ENV_NAME}"
else
    # 如果失败，尝试在项目目录下创建
    echo "⚠️  无法在默认位置创建环境，尝试在项目目录下创建..."
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ENV_PATH="${PROJECT_DIR}/.conda_env"
    
    # 配置 conda 使用项目目录下的缓存（如果可能）
    if [ -w "${PROJECT_DIR}" ]; then
        export CONDA_PKGS_DIRS="${PROJECT_DIR}/.conda_pkgs"
        mkdir -p "${CONDA_PKGS_DIRS}"
    fi
    
    if conda create --prefix ${ENV_PATH} python=3.10 -y 2>&1 | tee /tmp/conda_error.log; then
        echo "✅ 环境创建成功（在项目目录: ${ENV_PATH}）"
    else
        echo "❌ Conda 环境创建失败"
        echo ""
        echo "可能的原因："
        echo "  1. 权限不足（无法写入缓存或环境目录）"
        echo "  2. 磁盘空间不足"
        echo "  3. Conda 配置问题"
        echo ""
        echo "建议的解决方案："
        echo "  1. 检查权限：确保对项目目录有写权限"
        echo "  2. 手动初始化 conda："
        echo "     ~/miniconda3/bin/conda init bash"
        echo "     source ~/.bashrc"
        echo "  3. 或者使用 sudo 运行（不推荐）"
        echo ""
        echo "Conda 错误信息："
        tail -5 /tmp/conda_error.log 2>/dev/null || echo "无法获取详细错误信息"
        exit 1
    fi
fi

# 激活环境
echo "🔧 激活 Conda 环境..."
eval "$(conda shell.bash hook)"
if [ "$ENV_PATH" = "${ENV_NAME}" ]; then
    conda activate ${ENV_NAME}
else
    conda activate ${ENV_PATH}
fi

# 配置 pip 使用清华源
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
echo "✅ pip 镜像源配置完成（使用清华源，下载速度更快）"

# 升级 pip
echo "⬆️  升级 pip..."
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装依赖
echo "📥 安装项目依赖（使用清华源）..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 环境设置完成！"
    echo "=========================================="
    echo ""
    echo "下次使用时，请运行："
    if [ "$ENV_PATH" = "${ENV_NAME}" ]; then
        echo "  conda activate ${ENV_NAME}"
    else
        echo "  conda activate ${ENV_PATH}"
    fi
    echo ""
    echo "退出环境："
    echo "  conda deactivate"
    echo ""
else
    echo "❌ 依赖安装失败"
    exit 1
fi
