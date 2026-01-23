# 推理使用指南

## 快速开始

### 基本用法

使用训练好的模型对单个PDB文件进行推理：

```bash
python inference.py \
    --model_path checkpoints/best_model.pt \
    --pdb_path data/pdb_files/1CRN.pdb
```

### 完整参数示例

```bash
python inference.py \
    --model_path checkpoints/best_model.pt \
    --pdb_path data/pdb_files/1CRN.pdb \
    --hidden_dim 256 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --num_heads 8 \
    --num_iterations 10 \
    --device cuda \
    --output_path output.pdb
```

## 参数说明

### 必需参数

- `--model_path`: 训练好的模型检查点路径（如 `checkpoints/best_model.pt`）
- `--pdb_path`: 输入的PDB文件路径（包含骨架结构）

### 可选参数

- `--hidden_dim`: 隐藏层维度（默认256，需与训练时一致）
- `--num_encoder_layers`: Encoder层数（默认3，需与训练时一致）
- `--num_decoder_layers`: Decoder层数（默认3，需与训练时一致）
- `--num_heads`: 注意力头数（默认8，需与训练时一致）
- `--num_iterations`: 自适应迭代推理的迭代轮数（默认10）
  - 更多迭代轮数通常能获得更好的结果，但推理时间更长
  - 建议范围：5-20
- `--device`: 设备（`cuda` 或 `cpu`，默认自动检测）
- `--output_path`: 输出文件路径（可选，目前仅打印结果）

## 推理过程说明

### 自适应迭代推理（MaskGIT风格）

推理过程采用自适应迭代解码：

1. **初始化**：所有片段位置设置为 `[MASK]`
2. **迭代循环**（默认10轮）：
   - 模型预测所有位置的概率分布
   - 计算每个位置的置信度（最大概率）
   - **自适应Mask**：保留置信度最高的 `(k/K)` 比例的Token
   - 将低置信度位置重新Mask，输入下一轮
3. **最终输出**：使用所有预测作为最终结果

### 输出信息

推理完成后会显示：
- 生成的片段序列（前10个）
- 生成的扭转角（前10个，以度为单位）
- 总片段数和扭转角数

## 使用示例

### 示例1：使用默认参数

```bash
python inference.py \
    --model_path checkpoints/best_model.pt \
    --pdb_path data/pdb_files/1CRN.pdb
```

### 示例2：增加迭代次数以获得更好结果

```bash
python inference.py \
    --model_path checkpoints/best_model.pt \
    --pdb_path data/pdb_files/1CRN.pdb \
    --num_iterations 20
```

### 示例3：使用CPU推理

```bash
python inference.py \
    --model_path checkpoints/best_model.pt \
    --pdb_path data/pdb_files/1CRN.pdb \
    --device cpu
```

### 示例4：批量推理（使用脚本）

创建一个批量推理脚本 `batch_inference.sh`：

```bash
#!/bin/bash

MODEL_PATH="checkpoints/best_model.pt"
PDB_DIR="data/pdb_files"
OUTPUT_DIR="inference_results"

mkdir -p $OUTPUT_DIR

for pdb_file in $PDB_DIR/*.pdb; do
    filename=$(basename "$pdb_file" .pdb)
    echo "Processing $filename..."
    
    python inference.py \
        --model_path $MODEL_PATH \
        --pdb_path "$pdb_file" \
        --num_iterations 10 \
        > "$OUTPUT_DIR/${filename}_output.txt"
    
    echo "Completed $filename"
done
```

运行：
```bash
chmod +x batch_inference.sh
./batch_inference.sh
```

## 注意事项

1. **模型参数一致性**：
   - `--hidden_dim`, `--num_encoder_layers`, `--num_decoder_layers`, `--num_heads` 必须与训练时一致
   - 如果不确定，可以查看训练时的输出或检查点文件

2. **输入PDB文件要求**：
   - 必须包含完整的骨架结构（N, CA, C, O原子）
   - 侧链可以缺失（会被预测）

3. **内存和速度**：
   - GPU推理速度远快于CPU（推荐使用GPU）
   - 长序列（>500残基）可能需要更多内存

4. **迭代次数**：
   - 更多迭代通常能获得更好的结果
   - 但超过20轮后收益递减
   - 建议根据序列长度调整：短序列（<200残基）用5-10轮，长序列用10-20轮

## 故障排查

### 问题1：模型加载失败

**错误**：`KeyError` 或 `size mismatch`

**解决**：检查模型参数是否与训练时一致
```bash
# 查看训练时的参数
python train.py --help  # 查看默认参数
```

### 问题2：CUDA内存不足

**错误**：`RuntimeError: CUDA out of memory`

**解决**：
- 使用CPU推理：`--device cpu`
- 或减小batch size（如果支持批量推理）

### 问题3：PDB文件格式错误

**错误**：`KeyError` 或 `IndexError`

**解决**：
- 确保PDB文件包含完整的骨架原子
- 检查文件是否损坏

## 输出结果说明

推理输出包括：

1. **片段序列**：预测的化学片段Token序列
2. **扭转角**：预测的侧链扭转角（弧度转角度）

目前结构重建功能（将片段和扭转角转换为3D坐标）尚未实现，输出为文本格式。

## 下一步

- [ ] 实现结构重建功能（片段+扭转角 → 3D坐标）
- [ ] 支持批量推理
- [ ] 添加结果可视化
- [ ] 支持输出标准PDB格式
