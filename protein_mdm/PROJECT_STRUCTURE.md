# 项目结构详细说明

本文档详细说明项目的文件结构和每个模块的作用。

## 📁 项目目录树

```
protein_mdm/
├── data/                      # 数据处理模块
│   ├── __init__.py           # 模块导出
│   ├── vocabulary.py         # ⭐ 核心：片段词汇表和映射规则
│   ├── geometry.py           # ⭐ 核心：扭转角计算和离散化
│   └── dataset.py             # ⭐ 核心：PDB 数据集加载器
│
├── models/                    # 模型架构模块
│   ├── __init__.py           # 模块导出
│   ├── encoder.py            # 骨架编码器（占位符，待集成 GVP）
│   └── decoder.py            # 片段和扭转角预测器
│
├── utils/                     # 工具函数模块
│   ├── __init__.py           # 模块导出
│   └── protein_utils.py      # Biopython 辅助函数
│
├── venv/                      # Python 虚拟环境（不提交到 Git）
│
├── __init__.py                # 包初始化文件
├── main.py                    # 项目主入口
├── test_all.py               # ⭐ 综合测试脚本
│
├── requirements.txt          # 核心依赖列表
├── requirements-optional.txt # 可选依赖列表
├── setup_env.sh              # 环境设置脚本
├── .gitignore                # Git 忽略文件
├── README.md                  # 项目说明文档
└── PROJECT_STRUCTURE.md       # 本文档
```

---

## 📄 核心文件详解

### 🎯 `data/vocabulary.py` - 片段词汇表（核心）

**作用**: 实现氨基酸到化学片段的映射，这是整个项目的基石。

**核心类**: `FragmentVocab`

**主要功能**:
1. **定义化学片段 Token**:
   - 特殊 Token: `[PAD]`, `[MASK]`, `[BOS]`, `[EOS]`
   - 化学片段: `METHYL`, `METHYLENE`, `HYDROXYL`, `PHENYL`, `AMINE`, `CARBOXYL`, `AMIDE`, `GUANIDINE`, `IMIDAZOLE`, `INDOLE`, `THIOL`, `BRANCH_CH`

2. **硬编码 20 种标准氨基酸映射**:
   ```python
   'ALA' -> ['METHYL']
   'PHE' -> ['METHYLENE', 'PHENYL']
   'VAL' -> ['BRANCH_CH', 'METHYL', 'METHYL']
   ```

3. **提供转换函数**:
   - `residue_to_fragments(res_name)`: 氨基酸名 → 片段列表
   - `fragments_to_indices(fragments)`: 片段列表 → Token 索引
   - `indices_to_fragments(indices)`: Token 索引 → 片段列表

**为什么重要**: 
- 这是项目的核心创新点：将侧链视为刚性化学片段的组合
- 所有后续处理都依赖这个映射关系

---

### 📐 `data/geometry.py` - 几何工具（核心）

**作用**: 处理蛋白质结构中的扭转角（torsion angles）。

**主要函数**:

1. **`calculate_dihedrals(coords, atom_indices)`**:
   - 计算二面角（dihedral angles）
   - 输入: 原子坐标和四原子索引
   - 输出: 角度（弧度，范围 [-π, π]）
   - 用途: 从 PDB 结构中提取侧链的扭转角

2. **`discretize_angle(angle_rad, num_bins=72)`**:
   - 将连续角度离散化为 bin 索引
   - 默认 72 bins = 5度分辨率
   - 用途: 将角度转换为分类任务的类别

3. **`undiscretize_angle(bin_idx, num_bins=72)`**:
   - 将 bin 索引还原为连续角度
   - 用途: 模型预测后重建角度

4. **向量化版本**: `discretize_angles()`, `undiscretize_angles()`

**为什么重要**:
- 扭转角决定侧链的空间构象
- 离散化是神经网络处理角度的标准方法

---

### 📦 `data/dataset.py` - 数据集加载器（核心）

**作用**: 从 PDB 文件加载蛋白质结构并转换为模型可用的格式。

**核心类**: `ProteinStructureDataset`

**处理流程**:
1. **解析 PDB 文件** (使用 BioPython)
2. **提取骨架坐标**: N, CA, C, O 原子坐标 → `[L, 4, 3]`
3. **提取侧链信息**:
   - 使用 `vocabulary.py` 转换为片段序列
   - 使用 `geometry.py` 计算扭转角并离散化
4. **返回字典**:
   ```python
   {
       'backbone_coords': Tensor[L, 4, 3],
       'fragment_token_ids': Tensor[M],  # M = 总片段数
       'torsion_bins': Tensor[K],        # K = 扭转角数量
       'residue_types': List[str],
       'sequence_length': Tensor[1]
   }
   ```

**辅助函数**: `collate_fn()`
- 用于批处理不同长度的序列
- 自动填充到相同长度

**为什么重要**:
- 连接原始数据和模型
- 处理数据格式转换和批处理

---

### 🧠 `models/encoder.py` - 骨架编码器

**作用**: 将骨架坐标编码为节点嵌入（node embeddings）。

**核心类**: `BackboneEncoder`

**当前状态**: ⚠️ **占位符实现**
- 使用简单的 MLP 作为占位符
- **TODO**: 集成 GVP (Geometric Vector Perceptron) 以实现 SE(3) 等变性

**输入/输出**:
- 输入: `backbone_coords` [batch_size, L, 4, 3]
- 输出: `node_embeddings` [batch_size, L, hidden_dim]

**为什么需要 SE(3) 等变性**:
- 蛋白质的预测应该与全局旋转/平移无关
- GVP 可以保证这一点

---

### 🎨 `models/decoder.py` - 片段预测器

**作用**: 从节点嵌入预测片段类型和扭转角。

**核心类**: `FragmentDecoder`

**输出**:
- `fragment_logits`: [batch_size, M, vocab_size] - 片段类型预测
- `torsion_logits`: [batch_size, K, num_torsion_bins] - 扭转角预测

**当前状态**: 基础实现，后续需要：
- 处理变长片段序列
- 与掩码扩散模型集成

---

### 🛠️ `utils/protein_utils.py` - 辅助工具

**作用**: 提供常用的蛋白质结构处理函数。

**主要函数**:
- `load_pdb_structure()`: 加载 PDB 文件
- `get_backbone_atoms()`: 提取骨架原子
- `get_sidechain_atoms()`: 提取侧链原子
- `calculate_rmsd()`: 计算 RMSD
- `align_structures()`: 结构对齐

**为什么重要**: 提供可复用的工具函数，避免重复代码。

---

### 🚀 `main.py` - 项目入口

**作用**: 主程序入口，用于测试和演示。

**功能**:
1. 初始化词汇表和模型
2. 测试前向传播
3. 加载数据集（如果提供 PDB 路径）

**使用方式**:
```bash
python main.py --mode test
python main.py --mode test --pdb_path path/to/protein.pdb
```

---

### ✅ `test_all.py` - 综合测试脚本

**作用**: 全面测试所有核心模块。

**测试内容**:
1. ✅ 词汇表模块（20种氨基酸映射）
2. ✅ 几何工具模块（角度离散化）
3. ✅ 模型前向传播（Encoder + Decoder）
4. ⚠️ 数据集加载（需要 PDB 文件）

**使用方式**:
```bash
python test_all.py
python test_all.py --pdb_path path/to/protein.pdb
```

---

## 🔄 数据流

```
PDB 文件
    ↓
[dataset.py] 解析 PDB
    ↓
提取骨架坐标 + 侧链信息
    ↓
[vocabulary.py] 侧链 → 片段序列
[geometry.py] 侧链 → 扭转角序列
    ↓
模型输入: {backbone_coords, fragment_tokens, torsion_bins}
    ↓
[encoder.py] 骨架坐标 → 节点嵌入
    ↓
[decoder.py] 节点嵌入 → 片段预测 + 扭转角预测
```

---

## 📊 模块依赖关系

```
vocabulary.py (独立)
    ↑
    ├── dataset.py
    └── decoder.py (需要 vocab_size)

geometry.py (独立)
    ↑
    └── dataset.py

dataset.py
    ├── vocabulary.py
    └── geometry.py

encoder.py (独立，待集成 GVP)
    ↓
decoder.py
    ├── encoder.py (需要 input_dim)
    └── vocabulary.py (需要 vocab_size)
```

---

## 🎯 下一步开发重点

1. **集成 GVP**: 在 `encoder.py` 中替换占位符
2. **实现掩码扩散**: 训练循环和损失函数
3. **自适应推理**: 实现锚点优先的生成策略
4. **完善数据集**: 改进扭转角提取逻辑

---

## 💡 使用建议

1. **首次使用**: 运行 `test_all.py` 验证环境
2. **开发新功能**: 参考现有模块的结构和注释
3. **调试**: 使用 `main.py` 进行快速测试
4. **数据处理**: 优先使用 `utils/protein_utils.py` 中的函数

---

**最后更新**: 2024
