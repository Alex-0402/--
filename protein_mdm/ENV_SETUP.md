# 环境配置指南 (Miniconda)

本文档详细说明如何使用 Miniconda 配置 Protein MDM 项目的开发环境。

## 前置要求

- Miniconda 或 Anaconda（推荐 Miniconda，更轻量）
- 网络连接（用于下载依赖包）

## 快速开始

### 方法 1: 使用自动配置脚本（推荐）

1. **安装 Miniconda（如果尚未安装）**

   ```bash
   # 下载 Miniconda（使用清华镜像，速度更快）
   wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
   
   # 安装（按照提示操作）
   bash Miniconda3-latest-Linux-x86_64.sh
   
   # 重新加载 shell 配置
   source ~/.bashrc
   ```

2. **运行配置脚本**

   ```bash
   cd /home/Oliver-0402/--/protein_mdm
   bash setup_env.sh
   ```

   脚本会自动：
   - **配置 Conda 和 pip 使用清华镜像源**（加速下载）
   - 创建名为 `protein_mdm` 的 Conda 环境
   - 安装 Python 3.10
   - 升级 pip
   - 安装所有项目依赖

**注意**: 脚本已自动配置清华镜像源，下载速度会显著提升！

### 方法 2: 手动配置

如果自动脚本失败，可以按照以下步骤手动配置：

#### 步骤 1: 安装 Miniconda

```bash
# 下载 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh

# 重新加载 shell
source ~/.bashrc
```

#### 步骤 2: 创建 Conda 环境

```bash
cd /home/Oliver-0402/--/protein_mdm
conda create -n protein_mdm python=3.10 -y
```

#### 步骤 3: 激活环境

```bash
conda activate protein_mdm
```

激活后，终端提示符前会显示 `(protein_mdm)`。

#### 步骤 4: 升级 pip

```bash
pip install --upgrade pip
```

#### 步骤 5: 安装项目依赖

```bash
pip install -r requirements.txt
```

## 验证安装

安装完成后，可以运行以下命令验证环境：

```bash
# 检查 Python 版本
python --version

# 检查 Conda 环境
conda env list

# 检查 PyTorch 是否安装
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# 检查其他关键依赖
python -c "import numpy, Bio, einops; print('核心依赖安装成功')"
```

## 使用 Conda 环境

### 激活环境

每次使用项目前，需要激活 Conda 环境：

```bash
conda activate protein_mdm
```

### 退出环境

```bash
conda deactivate
```

### 在环境中运行项目

```bash
# 激活环境后
python main.py --mode test
python train.py
python inference.py
```

## 依赖说明

项目的主要依赖包括：

- **PyTorch** (>=2.0.0): 深度学习框架
- **torch-geometric** (>=2.3.0): 图神经网络库（必需）
- **BioPython** (>=1.79): PDB 文件解析
- **NumPy** (>=1.21.0,<2.0.0): 数值计算
- **einops** (>=0.6.0): 张量操作
- **tqdm** (>=4.64.0): 进度条
- **requests** (>=2.28.0): HTTP 请求

## 常见问题

### 问题 1: 未检测到 conda 命令

**错误信息**: `未检测到 conda`

**解决方案**:
```bash
# 检查 conda 是否在 PATH 中
which conda

# 如果未找到，重新加载 shell 配置
source ~/.bashrc
# 或
source ~/.zshrc
```

### 问题 2: pip 安装失败

**可能原因**:
- 网络连接问题
- pip 版本过旧
- 镜像源未配置

**解决方案**:
```bash
# 手动配置 pip 使用清华源（如果自动配置失败）
bash configure_mirrors.sh

# 或手动配置
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 问题 3: torch-geometric 安装失败

**解决方案**:
```bash
# 先安装 PyTorch
pip install torch

# 然后安装 torch-geometric（需要根据 PyTorch 版本选择）
pip install torch-geometric
```

### 问题 4: Conda 环境激活失败

**解决方案**:
```bash
# 初始化 conda（如果首次使用）
conda init bash
# 或
conda init zsh

# 重新加载 shell
source ~/.bashrc
```

### 问题 5: 环境已存在但想重新创建

```bash
# 删除旧环境
conda env remove -n protein_mdm

# 重新运行配置脚本
bash setup_env.sh
```

## 镜像源配置

项目已自动配置使用**清华镜像源**，大幅提升下载速度：

- **Conda 镜像**: 清华 Anaconda 镜像
- **pip 镜像**: 清华 PyPI 镜像

### 手动配置镜像源

如果自动配置失败，可以手动运行：

```bash
bash configure_mirrors.sh
```

### 验证镜像源配置

```bash
# 查看 Conda 镜像源
conda config --show channels

# 查看 pip 镜像源
cat ~/.pip/pip.conf
```

## 更新依赖

如果需要更新依赖：

```bash
# 激活环境
conda activate protein_mdm

# 升级所有包（自动使用清华源）
pip install --upgrade -r requirements.txt
```

## 导出和共享环境

### 导出环境配置

```bash
# 导出 conda 环境
conda env export > environment.yml

# 导出 pip 依赖
pip freeze > requirements-frozen.txt
```

### 从配置文件创建环境

```bash
# 从 environment.yml 创建
conda env create -f environment.yml

# 或从 requirements.txt 安装
conda create -n protein_mdm python=3.10 -y
conda activate protein_mdm
pip install -r requirements.txt
```

## 重新创建环境

如果需要完全重新创建环境：

```bash
# 删除旧环境
conda env remove -n protein_mdm

# 重新运行配置脚本
bash setup_env.sh
```

## Conda vs Venv 的优势

使用 Conda 的优势：
- ✅ 更好的包管理（包括系统级依赖）
- ✅ 可以管理不同 Python 版本
- ✅ 更好的科学计算包支持
- ✅ 可以安装非 Python 依赖（如 CUDA 工具包）
- ✅ 环境隔离更彻底

## 下一步

环境配置完成后，可以：

1. 运行测试：`python test_all.py`
2. 查看使用指南：`cat USAGE.md`
3. 开始训练：`python train.py --help`

---

**最后更新**: 2024
