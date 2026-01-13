# Protein Masked Diffusion Model

基于原子级片段自适应推理掩码扩散模型的蛋白质设计研究

## 项目概述

本项目实现了一个创新的蛋白质侧链设计方法，核心思想是：
- 将侧链视为**刚性化学片段 (Rigid Chemical Fragments)** 的组合
- 使用**掩码扩散模型 (Masked Diffusion Model)** 进行训练
- 采用**自适应推理策略 (Adaptive Inference)** 进行生成

## 项目结构

```
protein_mdm/
├── data/                      # 数据处理模块
│   ├── vocabulary.py         # ⭐ 核心：片段词汇表和映射规则
│   ├── geometry.py           # ⭐ 核心：扭转角计算和离散化
│   └── dataset.py            # ⭐ 核心：PDB 数据集加载器
├── models/                    # 模型架构模块
│   ├── encoder.py            # ⭐ 几何 GNN 骨架编码器
│   └── decoder.py            # ⭐ Transformer 片段解码器
├── utils/                     # 工具函数模块
│   └── protein_utils.py      # Biopython 辅助函数
├── main.py                    # 项目主入口
├── test_all.py               # ⭐ 综合测试脚本
├── PROJECT_STRUCTURE.md      # 详细项目结构说明
└── TESTING_GUIDE.md          # 测试指南
```

**详细说明**: 查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 了解每个文件的作用

## 核心特性

### 1. 片段分词 (Fragment Tokenization)

`data/vocabulary.py` 实现了将 20 种标准氨基酸映射到化学片段序列的功能：

- **特殊 Token**: `[PAD]`, `[MASK]`, `[BOS]`, `[EOS]`
- **化学片段**: `METHYL`, `METHYLENE`, `HYDROXYL`, `PHENYL`, `AMINE`, `CARBOXYL`, `AMIDE`, `GUANIDINE`, `IMIDAZOLE`, `INDOLE`, `THIOL`, `BRANCH_CH`

示例：
- `ALA` → `['METHYL']`
- `PHE` → `['METHYLENE', 'PHENYL']`
- `VAL` → `['BRANCH_CH', 'METHYL', 'METHYL']`

### 2. 几何处理 (Geometry)

`data/geometry.py` 提供了扭转角计算和离散化功能：

- 使用 BioPython 计算二面角 (dihedral angles)
- 将连续角度离散化为 bins (默认 72 bins = 5度分辨率)
- 支持离散化与反离散化转换

### 3. 数据集加载 (Dataset)

`data/dataset.py` 实现了 PDB 文件加载和预处理：

- 提取骨架坐标 (N, CA, C, O)
- 转换为片段序列和扭转角序列
- 支持批处理和填充

## 安装

### 1. 创建虚拟环境（推荐）

使用虚拟环境可以隔离项目依赖，避免与系统 Python 或其他项目冲突：

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate

# Windows:
# venv\Scripts\activate
```

### 2. 安装依赖

```bash
# 确保虚拟环境已激活，然后安装核心依赖
pip install --upgrade pip
pip install -r requirements.txt

# 可选：安装高级功能依赖（GVP、torch-geometric 等）
# pip install -r requirements-optional.txt
```

**注意**：
- 所有核心依赖都在 `requirements.txt` 中
- `torch-geometric` 是必需的（用于几何 GNN 编码器）
- 确保已安装所有依赖后再运行代码

### 3. 退出虚拟环境

```bash
deactivate
```

**注意**：每次使用项目时，记得先激活虚拟环境！

## 使用方法

### 🚀 训练模型

```bash
# 基本训练
python train.py --pdb_path data/pdb_files --epochs 50 --batch_size 4

# 详细参数说明见 USAGE.md
```

### 🔮 生成侧链

```bash
# 使用训练好的模型
python inference.py \
    --model_path checkpoints/best_model.pt \
    --pdb_path data/pdb_files/1CRN.pdb
```

### 🧪 测试项目

```bash
# 测试所有模块
python test_all.py

# 测试模型架构
python test_models.py

# 测试主程序
python main.py --mode test
```

**详细使用指南**: 查看 [USAGE.md](USAGE.md)  
**测试指南**: 查看 [TESTING_GUIDE.md](TESTING_GUIDE.md)

### 加载 PDB 文件

```python
from protein_mdm.data import ProteinStructureDataset, get_vocab

# 加载数据集
dataset = ProteinStructureDataset("path/to/protein.pdb")

# 获取样本
sample = dataset[0]
print(f"Backbone shape: {sample['backbone_coords'].shape}")
print(f"Fragment tokens: {sample['fragment_token_ids']}")
```

## 开发状态

### ✅ 已完成
- ✅ 片段词汇表和映射规则
- ✅ 扭转角计算和离散化
- ✅ 数据集加载器
- ✅ 几何 GNN 骨架编码器（基于 torch_geometric）
- ✅ Transformer 片段解码器

### ⏳ 待完成
- ⏳ 结构重建功能（从片段和扭转角重建原子坐标）
- ⏳ 评估指标（RMSD、准确率等）
- ⏳ 自适应推理策略优化

## 技术栈

- Python 3.9+
- PyTorch 2.0+
- BioPython (PDB 解析)
- torch-geometric (几何图神经网络)
- NumPy (数值计算)

## 作者

Research Team, 2024

## 许可证

Academic Use Only
