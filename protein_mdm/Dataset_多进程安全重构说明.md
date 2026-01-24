# Dataset 多进程安全重构说明

## 🔍 问题分析

### 原始问题
在 DDP 训练中，Epoch 2 开始时出现死锁，进程卡在获取第一个 batch 数据的地方。

### 可能原因
1. **文件句柄在 `__init__` 中打开**：如果 Dataset 在 `__init__` 中打开文件句柄（如 h5py、lmdb、open()），在 DataLoader 使用 `num_workers > 0` 时，fork 进程会复制文件句柄，导致死锁。
2. **Parser 对象在 `__init__` 中创建**：虽然 BioPython 的 PDBParser 不直接持有文件句柄，但在 fork 时复制对象可能导致问题。

## ✅ 重构方案

### 1. 移除 `__init__` 中的文件打开操作

**修复前**：
```python
def __init__(self, ...):
    self.parser = PDBParser(QUIET=True)  # ❌ 在 __init__ 中创建
    # 其他初始化...
```

**修复后**：
```python
def __init__(self, ...):
    self.use_mmcif = use_mmcif
    self._parser = None  # ✅ 懒加载：在需要时创建
    self._parser_lock = threading.Lock()  # 线程锁，确保线程安全
    # 其他初始化...
```

### 2. 在 `__getitem__` 中实现懒加载

**关键改进**：
- 使用 `_get_parser()` 方法按需创建 parser
- 每次调用都打开和关闭文件，不持有文件句柄
- 使用线程锁确保线程安全

```python
def _get_parser(self):
    """懒加载 parser：按需创建，确保多进程安全"""
    if self._parser is None:
        with self._parser_lock:
            if self._parser is None:
                # ✅ 懒加载：在需要时才创建 parser
                # 这确保每个 worker 进程在 fork 后创建自己的 parser
                self._parser = MMCIFParser(QUIET=True) if self.use_mmcif else PDBParser(QUIET=True)
    return self._parser
```

### 3. 确保对多进程（num_workers > 0）安全

**关键设计**：
1. **懒加载 parser**：每个 worker 进程在 fork 后创建自己的 parser 实例
2. **文件操作懒加载**：所有文件操作（`torch.load()`, `torch.save()`, `parser.get_structure()`）都在 `__getitem__` 中进行
3. **不持有文件句柄**：每次调用都打开和关闭文件，不持有文件句柄
4. **线程安全**：使用线程锁确保多线程环境下的安全性

## 📋 修改详情

### 修改的方法

1. **`__init__`**：
   - ✅ 移除 `self.parser` 的创建
   - ✅ 添加 `self._parser = None`（懒加载）
   - ✅ 添加 `self._parser_lock`（线程锁）
   - ✅ 只收集文件路径，不打开文件

2. **`_get_parser()`**（新增）：
   - ✅ 懒加载 parser，按需创建
   - ✅ 使用线程锁确保线程安全
   - ✅ 确保每个 worker 进程有自己的 parser 实例

3. **`_load_from_cache()`**：
   - ✅ 每次调用都打开和关闭文件（已实现）
   - ✅ 使用 `map_location='cpu'` 确保数据加载到 CPU

4. **`_save_to_cache()`**：
   - ✅ 每次调用都打开和关闭文件（已实现）

5. **`_parse_pdb_file()`**：
   - ✅ 使用 `_get_parser()` 获取 parser（懒加载）
   - ✅ `parser.get_structure()` 每次调用都打开和关闭文件

6. **`__getitem__()`**：
   - ✅ 所有文件操作都在这里进行（懒加载）
   - ✅ 每次调用都打开和关闭文件，不持有文件句柄

## 🧪 测试建议

### 1. 单进程测试
```python
# 测试单进程加载
dataset = ProteinStructureDataset(...)
sample = dataset[0]  # 应该正常工作
```

### 2. 多进程测试（num_workers > 0）
```python
# 测试多进程加载
loader = DataLoader(dataset, batch_size=4, num_workers=4)
for batch in loader:
    # 应该正常工作，不会死锁
    pass
```

### 3. DDP 测试
```python
# 测试 DDP 训练
# 应该不会在 Epoch 2 开始时死锁
torchrun --nproc_per_node=8 train.py ...
```

## ⚠️ 注意事项

1. **性能考虑**：
   - 懒加载可能导致首次访问稍慢（需要创建 parser）
   - 但避免了多进程死锁问题，这是值得的

2. **线程安全**：
   - 使用 `threading.Lock()` 确保多线程环境下的安全性
   - 虽然每个 worker 进程通常只有一个线程，但为了安全起见还是添加了锁

3. **文件操作**：
   - 所有文件操作都在 `__getitem__` 中进行
   - 每次调用都打开和关闭文件，不持有文件句柄
   - 这确保多进程安全，但可能略微影响性能

## 📝 总结

✅ **已完成的修复**：
1. 移除 `__init__` 中的 parser 创建
2. 实现懒加载 parser（`_get_parser()`）
3. 确保所有文件操作都在 `__getitem__` 中进行
4. 添加线程锁确保线程安全
5. 优化打印输出，避免多进程输出混乱

✅ **多进程安全保证**：
- 每个 worker 进程在 fork 后创建自己的 parser 实例
- 所有文件操作都是懒加载，不持有文件句柄
- 使用线程锁确保线程安全

现在 Dataset 类应该可以在 `num_workers > 0` 的情况下安全使用，不会导致 DDP 训练死锁。
