# PDB文件格式详解

## 什么是PDB文件？

PDB（Protein Data Bank）文件是存储蛋白质、核酸等生物大分子三维结构信息的标准格式。它使用文本格式，包含原子坐标、序列信息、实验方法等数据。

## PDB文件的主要组成部分

### 1. 文件头部信息（Header Section）

#### HEADER（第1行）
```
HEADER    OXYGEN TRANSPORT                        08-DEC-97   1A00
```
- **格式**：记录类型、分类、日期、PDB ID
- **说明**：文件的基本标识信息
- **1A00**：这是该蛋白质的PDB ID

#### TITLE
```
TITLE     HEMOGLOBIN (VAL BETA1 MET, TRP BETA37 TYR) MUTANT
```
- 描述蛋白质的标题信息

#### COMPND（化合物信息）
```
COMPND    MOL_ID: 1;
COMPND   2 MOLECULE: HEMOGLOBIN (ALPHA CHAIN);
COMPND   3 CHAIN: A, C;
```
- **MOL_ID**：分子编号
- **MOLECULE**：分子名称
- **CHAIN**：链标识符（A、B、C、D等）
- **MUTATION**：是否有突变

#### SOURCE（来源信息）
```
SOURCE    MOL_ID: 1;
SOURCE   2 ORGANISM_SCIENTIFIC: HOMO SAPIENS;
SOURCE   3 ORGANISM_COMMON: HUMAN;
```
- 描述蛋白质的来源：生物体、组织、细胞等

#### EXPDTA（实验方法）
```
EXPDTA    X-RAY DIFFRACTION
```
- 结构测定方法：X射线衍射、NMR、电子显微镜等

#### REMARK（备注）
- **REMARK 1**：参考文献
- **REMARK 2**：分辨率（如：2.00 Å）
- **REMARK 3**：精修信息（R值、B因子等）
- **REMARK 280**：结晶条件
- **REMARK 290**：晶体学对称性

### 2. 晶体学信息

#### CRYST1（晶体参数）
```
CRYST1   84.100  112.000   63.800  90.00  90.00  90.00 P 21 21 21    8
```
- **格式**：a b c α β γ 空间群 晶胞中的分子数
- **a, b, c**：晶胞边长（Å）
- **α, β, γ**：晶胞角度（度）
- **P 21 21 21**：空间群符号

### 3. 原子坐标记录（最重要的部分）

#### ATOM记录（标准原子）
```
ATOM      1  N   VAL A   1     101.601  38.534  -1.962  1.00 53.29           N
```
**字段说明（从左到右）：**
1. **ATOM**：记录类型（固定宽度7列）
2. **原子序号**（7-11列）：1, 2, 3...
3. **原子名称**（13-16列）：N, CA, C, O等
   - **N**：主链氮原子
   - **CA**：α碳原子
   - **C**：羰基碳原子
   - **O**：羰基氧原子
   - **CB, CG等**：侧链原子
4. **残基名称**（18-20列）：VAL（缬氨酸）、LEU（亮氨酸）等
5. **链标识符**（22列）：A, B, C, D
6. **残基序号**（23-26列）：1, 2, 3...
7. **X坐标**（31-38列）：101.601（单位：Å）
8. **Y坐标**（39-46列）：38.534
9. **Z坐标**（47-54列）：-1.962
10. **占有率**（55-60列）：1.00（0-1之间）
11. **B因子**（61-66列）：53.29（温度因子，反映原子运动程度）
12. **元素符号**（77-78列）：N, C, O等

#### HETATM记录（非标准原子）
```
HETATM 4387  CHA HEM A 142      94.094  61.877  -3.344  1.00 12.77           C
```
- **HETATM**：异质原子（非标准氨基酸/核苷酸）
- 常见例子：**HEM**（血红素）、**HOH**（水分子）、配体等
- 格式与ATOM相同

### 4. 连接信息

#### CONECT（连接记录）
```
CONECT 4538 4537 4539 4541
```
- 定义原子之间的化学键连接
- 第一个数字是中心原子序号，后面是连接的原子序号

### 5. 文件结束

#### END
```
END
```
- 标记PDB文件的结束

## 实际例子解析

让我们看一个具体的ATOM记录：
```
ATOM      1  N   VAL A   1     101.601  38.534  -1.962  1.00 53.29           N
```

**解读：**
- 这是第1个原子
- 原子类型：N（氮原子）
- 属于：VAL（缬氨酸）残基
- 在链A上
- 是第1个残基
- 三维坐标：(101.601, 38.534, -1.962) Å
- 占有率：100%
- B因子：53.29（较高，说明这个原子可能比较灵活）

## 关于1A00这个文件

根据文件内容，1A00是：
- **蛋白质**：血红蛋白（Hemoglobin）
- **突变体**：VAL BETA1 MET, TRP BETA37 TYR
- **分辨率**：2.00 Å（高分辨率）
- **链数**：4条链（A, B, C, D）
- **实验方法**：X射线晶体学
- **来源**：人类（Homo sapiens）

## 如何读取PDB文件

### Python示例代码
```python
def parse_pdb_atom_line(line):
    """解析ATOM记录行"""
    return {
        'record': line[0:6].strip(),
        'atom_num': int(line[6:11].strip()),
        'atom_name': line[12:16].strip(),
        'residue': line[17:20].strip(),
        'chain': line[21:22].strip(),
        'residue_num': int(line[22:26].strip()),
        'x': float(line[30:38].strip()),
        'y': float(line[38:46].strip()),
        'z': float(line[46:54].strip()),
        'occupancy': float(line[54:60].strip()),
        'b_factor': float(line[60:66].strip()),
        'element': line[76:78].strip()
    }

# 读取PDB文件
with open('1a00.pdb', 'r') as f:
    atoms = []
    for line in f:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom = parse_pdb_atom_line(line)
            atoms.append(atom)
```

## 常用工具

1. **PyMOL**：可视化PDB结构
2. **Biopython**：Python库，用于解析PDB文件
3. **Bio3D**：R语言包，用于结构分析
4. **ChimeraX**：现代的结构可视化工具

## 注意事项

1. **固定宽度格式**：PDB使用固定列宽，不是空格分隔
2. **坐标单位**：通常是埃（Å），1 Å = 0.1 nm
3. **链标识符**：可以是字母或数字
4. **残基编号**：可能不连续（有缺失的残基）
5. **B因子**：值越大，原子越灵活/不确定

## 使用BioPython解析PDB文件（推荐方法）

虽然可以手动解析PDB文件，但使用**BioPython**库更加方便和可靠。BioPython是处理生物信息学数据的标准Python库。

### 安装BioPython
```bash
pip install biopython
```

### 基本用法

#### 1. 加载PDB文件
```python
from Bio.PDB import PDBParser

# 创建解析器
parser = PDBParser(QUIET=True)

# 解析PDB文件
structure = parser.get_structure('protein', '1a00.pdb')

# 结构层次：Structure -> Model -> Chain -> Residue -> Atom
# Structure[0] 是第一个模型（通常只有一个）
model = structure[0]

# 获取所有链
chains = list(model.get_chains())
print(f"链数: {len(chains)}")  # 对于1a00，应该是4条链

# 获取第一条链
chain = chains[0]
print(f"链ID: {chain.id}")  # 例如 'A'
```

#### 2. 提取主链原子坐标
```python
import numpy as np

def get_backbone_coords(chain):
    """提取主链原子坐标 (N, CA, C, O)"""
    backbone_atoms = ['N', 'CA', 'C', 'O']
    coords = []
    
    for residue in chain:
        residue_coords = []
        for atom_name in backbone_atoms:
            try:
                atom = residue[atom_name]
                residue_coords.append(atom.get_coord())
            except KeyError:
                # 如果缺少某个原子，跳过这个残基
                break
        else:
            # 所有原子都存在，添加到列表
            coords.append(residue_coords)
    
    return np.array(coords)  # 形状: [L, 4, 3]，L是残基数

# 使用示例
chain = list(structure[0].get_chains())[0]
backbone = get_backbone_coords(chain)
print(f"主链坐标形状: {backbone.shape}")  # 例如 (141, 4, 3)
```

#### 3. 提取残基序列
```python
def get_sequence(chain):
    """提取氨基酸序列"""
    sequence = []
    for residue in chain:
        res_name = residue.get_resname()
        sequence.append(res_name)
    return sequence

# 使用示例
chain = list(structure[0].get_chains())[0]
sequence = get_sequence(chain)
print(f"序列长度: {len(sequence)}")
print(f"前5个残基: {sequence[:5]}")  # 例如 ['VAL', 'LEU', 'SER', 'PRO', 'ALA']
```

#### 4. 访问特定原子
```python
# 获取第一个残基的CA原子
first_residue = list(chain.get_residues())[0]
ca_atom = first_residue['CA']

# 获取坐标
ca_coord = ca_atom.get_coord()
print(f"CA坐标: {ca_coord}")  # 例如 [103.062  38.513  -2.159]

# 获取B因子
b_factor = ca_atom.get_bfactor()
print(f"B因子: {b_factor}")  # 例如 47.99
```

### 在本项目中的实际应用

这个项目（protein_mdm）已经实现了完整的PDB解析功能。主要代码在：

1. **`protein_mdm/utils/protein_utils.py`**：提供基础工具函数
   ```python
   from protein_mdm.utils.protein_utils import load_pdb_structure, get_backbone_atoms
   
   # 加载结构
   structure = load_pdb_structure('1a00.pdb')
   
   # 获取第一个残基的主链原子
   chain = list(structure[0].get_chains())[0]
   first_residue = list(chain.get_residues())[0]
   backbone_coords = get_backbone_atoms(first_residue)
   print(backbone_coords.shape)  # (4, 3) - N, CA, C, O的坐标
   ```

2. **`protein_mdm/data/dataset.py`**：完整的Dataset类
   ```python
   from protein_mdm.data.dataset import ProteinStructureDataset
   
   # 创建数据集
   dataset = ProteinStructureDataset('protein_mdm/raw_data/1a00.pdb')
   
   # 获取处理后的数据
   data = dataset[0]
   print(data.keys())
   # dict_keys(['backbone_coords', 'fragment_token_ids', 'torsion_bins', 
   #            'residue_types', 'sequence_length', 'pdb_path'])
   
   # backbone_coords: [L, 4, 3] - 主链坐标
   # fragment_token_ids: [M] - 片段token序列
   # torsion_bins: [K] - 扭转角离散化
   ```

### 完整的解析示例

```python
from Bio.PDB import PDBParser
import numpy as np

# 解析1a00.pdb
parser = PDBParser(QUIET=True)
structure = parser.get_structure('1a00', 'protein_mdm/raw_data/1a00.pdb')

# 遍历所有模型、链、残基和原子
for model in structure:
    print(f"模型: {model.id}")
    for chain in model:
        print(f"  链: {chain.id}, 残基数: {len(list(chain.get_residues()))}")
        for residue in chain:
            res_name = residue.get_resname()
            res_num = residue.id[1]
            print(f"    残基 {res_num}: {res_name}")
            
            # 获取CA原子坐标
            if 'CA' in residue:
                ca = residue['CA']
                coord = ca.get_coord()
                b_factor = ca.get_bfactor()
                print(f"      CA坐标: {coord}, B因子: {b_factor:.2f}")
            break  # 只打印第一个残基作为示例
        break  # 只打印第一条链作为示例
    break  # 只打印第一个模型
```

## 总结

PDB文件是结构生物学的标准格式，包含：
- ✅ 原子坐标（最重要的信息）
- ✅ 序列信息
- ✅ 实验条件
- ✅ 结构质量指标（分辨率、R值等）

### 学习建议

1. **理解格式**：先理解PDB文件的文本格式和字段含义
2. **使用工具**：使用BioPython等库来解析，避免手动解析的复杂性
3. **实践操作**：
   - 打开1a00.pdb文件，查看实际的ATOM记录
   - 使用BioPython加载并提取坐标
   - 尝试可视化结构（使用PyMOL或ChimeraX）
4. **结合项目**：查看`protein_mdm/data/dataset.py`了解如何在实际项目中使用

理解PDB格式对于结构生物学、药物设计、蛋白质工程等领域至关重要！
