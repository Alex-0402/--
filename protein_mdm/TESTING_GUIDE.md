# 测试指南

本指南说明如何测试项目的各个模块。

## 🚀 快速开始

### 1. 激活虚拟环境

```bash
cd /Users/ljh/Desktop/毕设/protein_mdm
source venv/bin/activate
```

### 2. 运行综合测试（推荐）

```bash
# 测试所有核心模块（不需要 PDB 文件）
python test_all.py

# 如果有多余的 PDB 文件，可以测试数据集加载
python test_all.py --pdb_path path/to/protein.pdb
```

---

## 📋 分模块测试

### 测试 1: 词汇表模块

```bash
python data/vocabulary.py
```

**预期输出**:
- 显示词汇表大小和片段数量
- 显示所有 Token 映射
- 显示 20 种氨基酸的片段映射

**验证点**:
- ✅ 所有 20 种氨基酸都能正确映射
- ✅ Token 索引从 0 开始（特殊 Token）到 15（BRANCH_CH）

---

### 测试 2: 几何工具模块

```bash
python data/geometry.py
```

**预期输出**:
- 显示二面角计算示例
- 显示角度离散化测试（72 bins）
- 显示向量化操作测试

**验证点**:
- ✅ 角度离散化/反离散化误差 < 5度
- ✅ 向量化操作正常工作

**注意**: 如果 BioPython 未安装，二面角计算会失败，但不影响其他测试。

---

### 测试 3: 模型前向传播

```bash
python main.py --mode test
```

**预期输出**:
- 词汇表初始化信息
- 模型初始化信息
- 前向传播的形状信息

**验证点**:
- ✅ Encoder 输出形状: [batch_size, seq_len, hidden_dim]
- ✅ Decoder 输出形状正确
- ✅ 无运行时错误

---

### 测试 4: 数据集加载（需要 PDB 文件）

```bash
# 如果有 PDB 文件
python main.py --mode test --pdb_path path/to/protein.pdb
```

**预期输出**:
- 数据集大小
- 样本信息（骨架形状、片段数量等）
- 批处理信息

**验证点**:
- ✅ 能正确解析 PDB 文件
- ✅ 提取的骨架坐标形状正确
- ✅ 片段序列和扭转角序列正确生成

**如果没有 PDB 文件**: 可以跳过此测试，不影响其他功能。

---

## 🧪 测试检查清单

运行 `test_all.py` 后，检查以下项目：

### ✅ 必须通过的测试

- [ ] 词汇表模块：20/20 氨基酸映射成功
- [ ] 几何工具模块：角度离散化误差 < 5度
- [ ] 模型前向传播：无错误，形状正确
- [ ] 梯度计算：可以反向传播

### ⚠️ 可选测试（需要额外资源）

- [ ] 数据集加载：需要 PDB 文件
- [ ] 二面角计算：需要 BioPython

---

## 🐛 常见问题排查

### 问题 1: "ModuleNotFoundError: No module named 'data'"

**原因**: 当前目录不对，或未安装包

**解决**:
```bash
# 确保在项目根目录
cd /Users/ljh/Desktop/毕设/protein_mdm

# 或者使用模块方式运行
python -m data.vocabulary
```

### 问题 2: "ImportError: cannot import name 'calc_dihedral'"

**原因**: BioPython 未安装或版本不对

**解决**:
```bash
pip install biopython
```

### 问题 3: "CUDA out of memory" 或 PyTorch 相关错误

**原因**: GPU 内存不足或 PyTorch 未正确安装

**解决**:
```bash
# 检查 PyTorch 安装
python -c "import torch; print(torch.__version__)"

# 如果使用 CPU，确保安装了 CPU 版本
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 问题 4: 数据集加载失败

**原因**: PDB 文件格式问题或缺少原子

**解决**:
- 检查 PDB 文件是否完整
- 确保文件包含完整的骨架原子（N, CA, C, O）
- 查看错误信息中的具体提示

---

## 📊 测试输出解读

### 成功的测试输出示例

```
============================================================
测试 1: 词汇表模块 (FragmentVocab)
============================================================
✅ 词汇表初始化成功
   - 词汇表大小: 16
   - 片段数量: 12
   - 特殊 Token: ['[PAD]', '[MASK]', '[BOS]', '[EOS]']

   测试 20 种标准氨基酸映射:
   ✓ ALA -> ['METHYL'] -> [4]
   ✓ PHE -> ['METHYLENE', 'PHENYL'] -> [5, 7]
   ...

   ✅ 成功映射 20/20 种氨基酸
   ✅ 错误处理测试通过
```

### 失败的测试输出示例

```
❌ 词汇表测试失败: KeyError: 'XXX'
Traceback (most recent call last):
  ...
```

**处理**: 查看错误堆栈，定位问题代码行。

---

## 🎯 下一步

测试通过后，可以：

1. **开始开发**: 参考 `PROJECT_STRUCTURE.md` 了解项目结构
2. **集成 GVP**: 在 `models/encoder.py` 中替换占位符
3. **实现训练循环**: 添加掩码扩散模型的训练代码
4. **准备数据**: 收集或下载 PDB 文件用于训练

---

**提示**: 如果所有测试都通过，说明项目基础功能正常，可以开始下一步开发！
