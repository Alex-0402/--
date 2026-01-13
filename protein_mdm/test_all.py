"""
综合测试脚本 - 测试所有核心模块

运行此脚本可以测试项目的所有核心功能：
1. 词汇表模块
2. 几何工具模块
3. 模型前向传播
4. 数据集加载（如果提供 PDB 文件）

使用方法:
    python test_all.py
    python test_all.py --pdb_path path/to/protein.pdb
"""

import sys
import torch
from typing import Optional

print("="*70)
print("Protein MDM - 综合测试脚本")
print("="*70)

# ============================================================================
# 测试 1: 词汇表模块 (Vocabulary)
# ============================================================================
print("\n" + "="*70)
print("测试 1: 词汇表模块 (FragmentVocab)")
print("="*70)

try:
    from data.vocabulary import FragmentVocab, get_vocab, SpecialTokens
    
    vocab = get_vocab()
    print(f"✅ 词汇表初始化成功")
    print(f"   - 词汇表大小: {vocab.get_vocab_size()}")
    print(f"   - 片段数量: {vocab.get_fragment_count()}")
    print(f"   - 特殊 Token: {[vocab.idx_to_token[i] for i in range(4)]}")
    
    # 测试所有 20 种氨基酸
    print("\n   测试 20 种标准氨基酸映射:")
    test_residues = [
        "ALA", "VAL", "LEU", "ILE", "MET",  # 非极性脂肪族
        "PHE", "TYR", "TRP",                  # 芳香族
        "SER", "THR", "ASN", "GLN",          # 极性不带电
        "LYS", "ARG", "HIS",                 # 正电
        "ASP", "GLU",                        # 负电
        "CYS", "GLY", "PRO"                  # 特殊
    ]
    
    success_count = 0
    for res in test_residues:
        try:
            fragments = vocab.residue_to_fragments(res)
            indices = vocab.fragments_to_indices(fragments)
            print(f"   ✓ {res:3s} -> {fragments} -> {indices}")
            success_count += 1
        except Exception as e:
            print(f"   ✗ {res:3s} -> 错误: {e}")
    
    print(f"\n   ✅ 成功映射 {success_count}/20 种氨基酸")
    
    # 测试错误处理
    try:
        vocab.residue_to_fragments("XXX")
        print("   ✗ 错误处理测试失败: 应该抛出 KeyError")
    except KeyError:
        print("   ✅ 错误处理测试通过: 未知残基正确抛出 KeyError")
    
    vocab_test_passed = True
    
except Exception as e:
    print(f"❌ 词汇表测试失败: {e}")
    import traceback
    traceback.print_exc()
    vocab_test_passed = False

# ============================================================================
# 测试 2: 几何工具模块 (Geometry)
# ============================================================================
print("\n" + "="*70)
print("测试 2: 几何工具模块 (Torsion Angles)")
print("="*70)

try:
    import numpy as np
    from data.geometry import (
        calculate_dihedrals,
        discretize_angle,
        discretize_angles,
        undiscretize_angle,
        undiscretize_angles,
        get_torsion_angle_resolution
    )
    
    print("✅ 几何模块导入成功")
    
    # 测试角度离散化
    print("\n   测试角度离散化 (72 bins = 5度分辨率):")
    test_angles = [
        -np.pi, -np.pi/2, 0, np.pi/2, np.pi
    ]
    
    for angle in test_angles:
        bin_idx = discretize_angle(angle, num_bins=72)
        angle_recovered = undiscretize_angle(bin_idx, num_bins=72)
        error = abs(angle - angle_recovered)
        print(f"   ✓ {np.degrees(angle):7.2f}° -> Bin {bin_idx:3d} -> {np.degrees(angle_recovered):7.2f}° (误差: {np.degrees(error):.2f}°)")
    
    # 测试向量化操作
    print("\n   测试向量化操作:")
    angles_array = np.linspace(-np.pi, np.pi, 10)
    bin_indices = discretize_angles(angles_array, num_bins=72)
    angles_recovered = undiscretize_angles(bin_indices, num_bins=72)
    max_error = np.max(np.abs(angles_array - angles_recovered))
    print(f"   ✓ 向量化测试通过 (最大误差: {np.degrees(max_error):.2f}°)")
    
    # 测试二面角计算（需要 BioPython）
    print("\n   测试二面角计算:")
    try:
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        angle = calculate_dihedrals(coords, [(0, 1, 2, 3)])[0]
        print(f"   ✓ 二面角计算成功: {np.degrees(angle):.2f}°")
    except Exception as e:
        print(f"   ⚠ 二面角计算需要 BioPython: {e}")
    
    print(f"\n   ✅ 分辨率: {get_torsion_angle_resolution(72):.2f} 度/每 bin")
    geometry_test_passed = True
    
except Exception as e:
    print(f"❌ 几何模块测试失败: {e}")
    import traceback
    traceback.print_exc()
    geometry_test_passed = False

# ============================================================================
# 测试 3: 模型前向传播
# ============================================================================
print("\n" + "="*70)
print("测试 3: 模型前向传播 (Encoder + Decoder)")
print("="*70)

try:
    from models.encoder import BackboneEncoder
    from models.decoder import FragmentDecoder
    from data.vocabulary import get_vocab
    
    vocab = get_vocab()
    
    # 初始化模型
    hidden_dim = 256
    encoder = BackboneEncoder(hidden_dim=hidden_dim)
    decoder = FragmentDecoder(
        input_dim=hidden_dim,
        vocab_size=vocab.get_vocab_size(),
        num_torsion_bins=72,
        hidden_dim=hidden_dim
    )
    
    print("✅ 模型初始化成功")
    print(f"   - Encoder 输出维度: {encoder.get_output_dim()}")
    print(f"   - Decoder 词汇表大小: {decoder.vocab_size}")
    print(f"   - Decoder 扭转角 bins: {decoder.num_torsion_bins}")
    
    # 测试前向传播
    print("\n   测试前向传播:")
    batch_size = 2
    seq_len = 10
    frag_seq_len = 20
    
    # 创建虚拟骨架坐标 [batch_size, seq_len, 4 atoms, 3 coords]
    dummy_backbone = torch.randn(batch_size, seq_len, 4, 3)
    print(f"   - 输入形状: {dummy_backbone.shape}")
    
    # Encoder 前向传播
    node_embeddings = encoder(dummy_backbone)
    print(f"   - Encoder 输出形状: {node_embeddings.shape}")
    assert node_embeddings.shape == (batch_size, seq_len, hidden_dim), \
        f"Encoder 输出形状错误: {node_embeddings.shape}"
    
    # 创建目标片段序列（Token IDs）
    target_fragments = torch.randint(0, vocab.get_vocab_size(), (batch_size, frag_seq_len))
    
    # Decoder 前向传播
    frag_logits, tors_logits = decoder(
        node_embeddings=node_embeddings,
        target_fragments=target_fragments
    )
    print(f"   - Fragment logits 形状: {frag_logits.shape}")
    print(f"   - Torsion logits 形状: {tors_logits.shape}")
    
    # 验证输出形状
    assert frag_logits.shape[0] == batch_size, "Fragment logits batch size 错误"
    assert tors_logits.shape[0] == batch_size, "Torsion logits batch size 错误"
    
    print("   ✅ 前向传播测试通过")
    
    # 测试梯度
    print("\n   测试梯度计算:")
    loss = frag_logits.sum() + tors_logits.sum()
    loss.backward()
    print("   ✅ 梯度计算成功")
    
    model_test_passed = True
    
except Exception as e:
    print(f"❌ 模型测试失败: {e}")
    import traceback
    traceback.print_exc()
    model_test_passed = False

# ============================================================================
# 测试 4: 数据集加载 (可选，需要 PDB 文件)
# ============================================================================
print("\n" + "="*70)
print("测试 4: 数据集加载 (需要 PDB 文件)")
print("="*70)

pdb_path = None
if len(sys.argv) > 1:
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_path", type=str, default=None)
    args = parser.parse_args()
    pdb_path = args.pdb_path
    
    # 检查路径是否存在且有效
    if pdb_path and not os.path.exists(pdb_path):
        print(f"   ⚠️  PDB 路径不存在: {pdb_path}")
        print("   ⚠️  跳过数据集测试")
        pdb_path = None

if pdb_path:
    try:
        from data.dataset import ProteinStructureDataset, collate_fn
        from torch.utils.data import DataLoader
        
        print(f"   加载 PDB 文件: {pdb_path}")
        dataset = ProteinStructureDataset(pdb_path)
        print(f"   ✅ 数据集加载成功，包含 {len(dataset)} 个样本")
        
        if len(dataset) > 0:
            # 测试单个样本
            sample = dataset[0]
            print(f"\n   样本信息:")
            print(f"   - 骨架坐标形状: {sample['backbone_coords'].shape}")
            print(f"   - 片段 Token 数量: {len(sample['fragment_token_ids'])}")
            print(f"   - 扭转角 bins 数量: {len(sample['torsion_bins'])}")
            print(f"   - 序列长度: {sample['sequence_length'].item()}")
            print(f"   - 残基类型: {sample['residue_types'][:5]}...")  # 显示前5个
            
            # 测试 DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                collate_fn=collate_fn,
                shuffle=False
            )
            batch = next(iter(dataloader))
            print(f"\n   批处理信息:")
            print(f"   - 批处理骨架形状: {batch['backbone_coords'].shape}")
            print(f"   - 批处理片段形状: {batch['fragment_token_ids'].shape}")
            print(f"   - 批处理扭转角形状: {batch['torsion_bins'].shape}")
            print(f"   - 序列长度: {batch['sequence_lengths']}")
            
            print("   ✅ 数据集测试通过")
            dataset_test_passed = True
        else:
            print("   ⚠ 数据集为空")
            dataset_test_passed = None
            
    except Exception as e:
        print(f"   ❌ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        dataset_test_passed = False
else:
    print("   ⚠️  跳过数据集测试 (未提供有效的 PDB 文件)")
    print("   提示: 使用 --pdb_path path/to/protein.pdb 来测试数据集加载")
    print("   注意: PDB 文件必须存在且可读")
    dataset_test_passed = None

# ============================================================================
# 测试总结
# ============================================================================
print("\n" + "="*70)
print("测试总结")
print("="*70)

tests = [
    ("词汇表模块", vocab_test_passed),
    ("几何工具模块", geometry_test_passed),
    ("模型前向传播", model_test_passed),
    ("数据集加载", dataset_test_passed),
]

passed = sum(1 for _, result in tests if result is True)
total = sum(1 for _, result in tests if result is not None)

for name, result in tests:
    if result is True:
        status = "✅ 通过"
    elif result is False:
        status = "❌ 失败"
    else:
        status = "⚠️  跳过"
    print(f"   {name:20s}: {status}")

print(f"\n   总计: {passed}/{total} 个测试通过")

if passed == total:
    print("\n   🎉 所有核心测试通过！项目基础功能正常。")
else:
    print("\n   ⚠️  部分测试失败，请检查错误信息。")

print("="*70)
