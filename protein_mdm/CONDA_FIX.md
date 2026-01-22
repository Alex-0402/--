# Conda 初始化修复指南

如果运行 `setup_env.sh` 时遇到权限错误，请按照以下步骤修复：

## 问题症状

- 错误信息：`NoWritableEnvsDirError` 或 `CondaToSPermissionError`
- Conda 已安装但无法创建环境

## 解决方案

### 方法 1: 初始化 Conda（推荐）

```bash
# 1. 初始化 conda（如果尚未初始化）
~/miniconda3/bin/conda init bash

# 2. 重新加载 shell 配置
source ~/.bashrc

# 3. 验证 conda 是否可用
conda --version

# 4. 重新运行配置脚本
cd /home/Oliver-0402/--/protein_mdm
bash setup_env.sh
```

### 方法 2: 手动创建环境

如果自动脚本仍然失败，可以手动创建环境：

```bash
# 1. 初始化 conda（如果尚未初始化）
~/miniconda3/bin/conda init bash
source ~/.bashrc

# 2. 配置镜像源（使用清华源，加速下载）
bash configure_mirrors.sh

# 3. 创建环境
conda create -n protein_mdm python=3.10 -y

# 4. 激活环境
conda activate protein_mdm

# 5. 安装依赖（自动使用清华源）
pip install --upgrade pip
pip install -r requirements.txt
```

### 方法 3: 在项目目录下创建环境

如果无法在默认位置创建环境，可以在项目目录下创建：

```bash
# 1. 初始化 conda
~/miniconda3/bin/conda init bash
source ~/.bashrc

# 2. 配置镜像源（使用清华源）
cd /home/Oliver-0402/--/protein_mdm
bash configure_mirrors.sh

# 3. 在项目目录下创建环境
conda create --prefix ./.conda_env python=3.10 -y

# 4. 激活环境
conda activate ./.conda_env

# 5. 安装依赖（自动使用清华源）
pip install --upgrade pip
pip install -r requirements.txt
```

## 验证安装

安装完成后，验证环境：

```bash
# 检查 Python 版本
python --version

# 检查 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 检查其他依赖
python -c "import numpy, Bio, einops; print('依赖安装成功')"
```

## 常见问题

### Q: conda init 后仍然无法使用 conda

**A**: 确保重新加载了 shell 配置：
```bash
source ~/.bashrc
# 或
source ~/.zshrc
```

### Q: 权限错误仍然存在

**A**: 检查 conda 缓存目录权限：
```bash
# 检查 conda 配置
conda config --show

# 如果需要，可以设置自定义缓存目录
mkdir -p ~/conda_cache
conda config --add pkgs_dirs ~/conda_cache
```

### Q: 环境创建成功但激活失败

**A**: 确保 conda 已正确初始化：
```bash
# 重新初始化
~/miniconda3/bin/conda init bash
source ~/.bashrc

# 验证
conda env list
```

---

**提示**: 如果问题仍然存在，请检查系统日志或联系系统管理员。
