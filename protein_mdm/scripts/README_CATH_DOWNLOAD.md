# CATH S40 数据集子集下载工具

## 简介

`download_cath_subset.py` 是一个用于自动下载 CATH S40 非冗余数据集子集的工具脚本。它可以帮助你快速获取少量蛋白质结构文件用于开发和测试，而不需要下载整个巨大的数据集。

## 功能特性

- ✅ 自动从 CATH 官方下载最新的 S40 非冗余列表
- ✅ 支持随机采样或顺序选择
- ✅ 自动从 RCSB PDB 下载对应的 PDB 文件
- ✅ 显示下载进度条
- ✅ 错误处理和重试机制
- ✅ 跳过已存在的文件（可选）

## 安装依赖

确保已安装所需依赖：

```bash
pip install requests tqdm
```

或者安装所有项目依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

下载默认 50 个蛋白质：

```bash
python scripts/download_cath_subset.py
```

### 下载指定数量

下载 100 个蛋白质：

```bash
python scripts/download_cath_subset.py --limit 100
```

### 指定输出目录

```bash
python scripts/download_cath_subset.py --limit 50 --output_dir data/pdb_files
```

### 随机采样

使用随机采样（默认使用固定种子 42）：

```bash
python scripts/download_cath_subset.py --limit 50 --random
```

使用自定义随机种子：

```bash
python scripts/download_cath_subset.py --limit 50 --random --seed 123
```

### 跳过已存在的文件

如果某些文件已下载，跳过它们：

```bash
python scripts/download_cath_subset.py --limit 50 --skip_existing
```

### 使用现有的列表文件

如果已经下载了 CATH 列表文件：

```bash
python scripts/download_cath_subset.py --list_file data/meta/cath_s40_list.txt
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--limit` | int | 50 | 下载的蛋白质数量 |
| `--output_dir` | str | `raw_data` | PDB 文件输出目录 |
| `--list_file` | str | None | CATH 列表文件路径（如果已存在） |
| `--random` | flag | False | 是否使用随机采样 |
| `--seed` | int | 42 | 随机种子（仅在 `--random` 时有效） |
| `--skip_existing` | flag | False | 跳过已存在的 PDB 文件 |

## 输出文件

### 目录结构

```
protein_mdm/
├── data/
│   └── meta/
│       └── cath_s40_list.txt    # CATH S40 列表文件（自动下载）
└── raw_data/                     # 默认输出目录
    ├── 1oai.pdb
    ├── 2abc.pdb
    ├── ...
    └── downloaded_files.txt      # 下载成功的文件列表
```

### 文件说明

- **cath_s40_list.txt**: CATH S40 非冗余数据集完整列表（从 CATH 官方下载）
- **{pdb_id}.pdb**: 下载的 PDB 结构文件
- **downloaded_files.txt**: 成功下载的文件列表（每行一个 PDB ID）

## 工作流程

1. **下载列表文件**：从 CATH 官方下载 S40 非冗余列表
2. **解析列表**：解析列表文件，提取 CATH ID
3. **采样子集**：根据参数选择指定数量的 ID（随机或顺序）
4. **提取 PDB ID**：从 CATH ID 中提取 4 位 PDB ID
5. **下载文件**：从 RCSB PDB 下载对应的 PDB 文件
6. **保存结果**：将文件保存到指定目录，并生成文件列表

## CATH ID 格式说明

CATH ID 通常是 7 位字符，例如：
- `1oaiA00`: 前 4 位是 PDB ID (`1oai`)，第 5 位是 Chain ID (`A`)
- 脚本会自动提取前 4 位作为 PDB ID 用于下载

## 注意事项

1. **网络连接**：确保网络连接正常，能够访问 CATH 和 RCSB PDB 网站
2. **下载速度**：脚本在每次下载之间添加了 0.1 秒延迟，避免请求过快
3. **文件验证**：下载的文件会进行基本验证（文件大小 > 100 字节）
4. **错误处理**：如果某个文件下载失败，会显示警告但不会中断整个流程

## 示例输出

```
======================================================================
CATH S40 数据集子集下载工具
======================================================================
下载数量: 50
输出目录: raw_data
随机采样: False
======================================================================

正在下载 CATH S40 列表文件...
  URL: http://download.cathdb.info/cath/releases/latest-release/...
  保存到: data/meta/cath_s40_list.txt
  ✅ CATH S40 列表文件下载成功

2. 解析列表文件...
  找到 12345 个 CATH ID

3. 选择子集...
  选择了前 50 个 ID
  提取到 50 个唯一的 PDB ID

4. 下载 PDB 文件到: raw_data
下载进度: 100%|████████████████| 50/50 [00:15<00:00,  3.2文件/s]

======================================================================
下载完成！
======================================================================
成功: 48
失败: 2
总计: 50

文件保存在: raw_data
======================================================================

已保存文件列表到: raw_data/downloaded_files.txt
```

## 故障排除

### 问题：无法下载列表文件

**解决方案**：
- 检查网络连接
- 确认 CATH 网站可访问
- 尝试手动访问 URL 验证

### 问题：某些 PDB 文件下载失败

**解决方案**：
- 这是正常的，某些 PDB ID 可能已失效或不存在
- 脚本会自动跳过失败的下载
- 可以重新运行脚本，使用 `--skip_existing` 跳过已下载的文件

### 问题：文件下载很慢

**解决方案**：
- 这是正常的，RCSB PDB 服务器可能有速率限制
- 脚本已添加延迟以避免请求过快
- 可以分批下载，每次下载少量文件

## 后续步骤

下载完成后，可以使用预处理脚本将 PDB 文件转换为缓存格式：

```bash
python scripts/preprocess_dataset.py \
    --pdb_dir raw_data \
    --output_dir data/cache \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

然后使用训练脚本进行训练：

```bash
python train.py \
    --pdb_path data/cache \
    --cache_dir data/cache \
    --epochs 50
```
