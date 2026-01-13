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
├── data/
│   ├── vocabulary.py      # 片段 Token 词汇表和映射规则
│   ├── geometry.py        # 扭转角计算与离散化工具
│   └── dataset.py         # PDB 数据集加载器
├── models/
│   ├── encoder.py         # 骨架编码器 (GVP 占位符)
│   └── decoder.py         # 片段和扭转角预测器
├── utils/
│   └── protein_utils.py   # Biopython 辅助函数
└── main.py                # 项目入口
```

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
- 核心功能（词汇表、几何工具、数据集加载）只需要 `requirements.txt` 中的依赖
- `gvp-pytorch` 目前是可选的，因为编码器还是占位符实现
- 如果需要 GVP，可以从 GitHub 安装：
  ```bash
  pip install git+https://github.com/drorlab/gvp-pytorch.git
  ```

### 3. 退出虚拟环境

```bash
deactivate
```

**注意**：每次使用项目时，记得先激活虚拟环境！

## 使用方法

### 测试核心功能

```bash
# 测试词汇表
python -m protein_mdm.data.vocabulary

# 测试几何工具
python -m protein_mdm.data.geometry

# 运行主程序
python protein_mdm/main.py --mode test
```

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

- ✅ 片段词汇表和映射规则
- ✅ 扭转角计算和离散化
- ✅ 数据集加载器
- ⏳ GVP 骨架编码器 (占位符)
- ⏳ 掩码扩散训练循环
- ⏳ 自适应推理策略

## 技术栈

- Python 3.9+
- PyTorch 2.0+
- BioPython (PDB 解析)
- torch_geometric (图数据处理)
- gvp-pytorch (SE(3) 等变性编码)

## 作者

Research Team, 2024

## 许可证

Academic Use Only
