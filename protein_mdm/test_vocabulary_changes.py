"""
测试 vocabulary.py 的修改

验证：
1. 化学拓扑错误修复（SER, CYS, ILE）
2. FRAGMENT_FEATURES 字典是否正确
3. get_physicochemical_embedding 函数是否正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from data.vocabulary import FragmentVocab, get_vocab

print("="*70)
print("测试 Vocabulary 修改")
print("="*70)

# ============================================================================
# 测试 1: 验证化学拓扑修复
# ============================================================================
print("\n" + "="*70)
print("测试 1: 验证化学拓扑修复")
print("="*70)

vocab = get_vocab()

# 检查关键残基的片段映射
expected_mappings = {
    "SER": ["METHYLENE", "HYDROXYL"],  # 应该包含 CB (METHYLENE)
    "CYS": ["METHYLENE", "THIOL"],     # 应该包含 CB (METHYLENE)
    "ILE": ["BRANCH_CH", "METHYL", "METHYLENE", "METHYL"],  # β-支链结构
}

all_passed = True
for res_name, expected_fragments in expected_mappings.items():
    actual_fragments = vocab.residue_to_fragments(res_name)
    if actual_fragments == expected_fragments:
        print(f"✅ {res_name}: {actual_fragments}")
    else:
        print(f"❌ {res_name}: 期望 {expected_fragments}, 实际 {actual_fragments}")
        all_passed = False

# ============================================================================
# 测试 2: 验证 FRAGMENT_FEATURES 字典
# ============================================================================
print("\n" + "="*70)
print("测试 2: 验证 FRAGMENT_FEATURES 字典")
print("="*70)

# 检查所有片段都有特征定义
fragment_tokens = vocab.FRAGMENT_TOKENS
fragment_features = vocab.FRAGMENT_FEATURES

print(f"片段数量: {len(fragment_tokens)}")
print(f"特征定义数量: {len(fragment_features)}")

missing_fragments = []
for fragment in fragment_tokens:
    if fragment not in fragment_features:
        missing_fragments.append(fragment)
        print(f"❌ 缺少特征定义: {fragment}")
    else:
        features = fragment_features[fragment]
        if len(features) != 5:
            print(f"❌ {fragment}: 特征维度错误 (期望5, 实际{len(features)})")
            all_passed = False
        else:
            print(f"✅ {fragment:12s}: 疏水性={features[0]:6.2f}, 电荷={features[1]:4.1f}, "
                  f"分子量={features[2]:6.2f}, H供体={features[3]}, H受体={features[4]}")

if missing_fragments:
    all_passed = False
    print(f"\n❌ 缺少 {len(missing_fragments)} 个片段的特征定义")
else:
    print(f"\n✅ 所有片段都有特征定义")

# ============================================================================
# 测试 3: 验证 get_physicochemical_embedding 函数
# ============================================================================
print("\n" + "="*70)
print("测试 3: 验证 get_physicochemical_embedding 函数")
print("="*70)

try:
    # 测试单个 token
    print("\n   测试单个 Token ID:")
    test_token_ids = [
        (4, "METHYL"),
        (5, "METHYLENE"),
        (6, "HYDROXYL"),
        (8, "AMINE"),
        (9, "CARBOXYL"),
    ]
    
    for token_id, fragment_name in test_token_ids:
        token_tensor = torch.tensor([token_id])
        features = vocab.get_physicochemical_embedding(token_tensor, normalize=False)
        expected_features = vocab.FRAGMENT_FEATURES[fragment_name]
        
        # 比较特征（允许小的浮点误差）
        features_list = features.squeeze().tolist()
        match = all(abs(f - e) < 1e-5 for f, e in zip(features_list, expected_features))
        
        if match:
            print(f"   ✅ Token {token_id:2d} ({fragment_name:12s}): {features_list}")
        else:
            print(f"   ❌ Token {token_id:2d} ({fragment_name:12s}):")
            print(f"      期望: {expected_features}")
            print(f"      实际: {features_list}")
            all_passed = False
    
    # 测试批量 token
    print("\n   测试批量 Token IDs:")
    batch_token_ids = torch.tensor([[4, 5, 6], [8, 9, 4]])  # [batch_size=2, seq_len=3]
    features = vocab.get_physicochemical_embedding(batch_token_ids, normalize=False)
    
    if features.shape == (2, 3, 5):
        print(f"   ✅ 批量特征形状正确: {features.shape}")
        
        # 验证第一个样本的第一个token (METHYL)
        first_token_features = features[0, 0].tolist()
        expected = vocab.FRAGMENT_FEATURES["METHYL"]
        if all(abs(f - e) < 1e-5 for f, e in zip(first_token_features, expected)):
            print(f"   ✅ 批量特征值正确")
        else:
            print(f"   ❌ 批量特征值错误")
            all_passed = False
    else:
        print(f"   ❌ 批量特征形状错误: 期望 (2, 3, 5), 实际 {features.shape}")
        all_passed = False
    
    # 测试特殊 token (应该返回零向量)
    print("\n   测试特殊 Token (PAD, MASK, BOS, EOS):")
    special_tokens = torch.tensor([0, 1, 2, 3])  # PAD, MASK, BOS, EOS
    features = vocab.get_physicochemical_embedding(special_tokens, normalize=False)
    zero_vector = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    all_zero = all(all(abs(f) < 1e-5 for f in feat.tolist()) for feat in features)
    if all_zero:
        print(f"   ✅ 特殊 Token 返回零向量")
    else:
        print(f"   ❌ 特殊 Token 应该返回零向量，但实际: {features}")
        all_passed = False
    
    # 测试归一化
    print("\n   测试归一化:")
    token_tensor = torch.tensor([4])  # METHYL
    features_normalized = vocab.get_physicochemical_embedding(token_tensor, normalize=True)
    features_unnormalized = vocab.get_physicochemical_embedding(token_tensor, normalize=False)
    
    if not torch.allclose(features_normalized, features_unnormalized):
        print(f"   ✅ 归一化正常工作 (归一化后与未归一化不同)")
    else:
        print(f"   ⚠️  归一化可能未生效")
    
    # 测试实际氨基酸序列
    print("\n   测试实际氨基酸序列:")
    test_residues = ["SER", "CYS", "ILE"]
    for res in test_residues:
        fragments = vocab.residue_to_fragments(res)
        indices = vocab.fragments_to_indices(fragments)
        token_tensor = torch.tensor([indices])
        features = vocab.get_physicochemical_embedding(token_tensor, normalize=False)
        print(f"   {res:3s} -> {fragments} -> 特征形状: {features.shape}")
    
except Exception as e:
    print(f"❌ get_physicochemical_embedding 测试失败: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*70)
if all_passed:
    print("✅ 所有测试通过！")
    print("\n下一步建议:")
    print("1. 运行完整测试: python test_all.py")
    print("2. 如果之前有训练好的模型，建议重新训练（因为 vocabulary 已改变）")
    print("3. 检查模型是否使用了 get_physicochemical_embedding，确保兼容性")
else:
    print("❌ 部分测试失败，请检查上述错误信息")
print("="*70)
