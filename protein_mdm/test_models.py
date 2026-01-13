"""
测试新的模型架构（GVP Encoder + Transformer Decoder）

运行此脚本验证模型是否能正常工作。
"""

import torch
from models.encoder import BackboneEncoder
from models.decoder import FragmentDecoder
from data.vocabulary import get_vocab

print("="*70)
print("模型架构测试")
print("="*70)

# 初始化词汇表
print("\n1. 初始化词汇表...")
vocab = get_vocab()
vocab_size = vocab.get_vocab_size()
print(f"   词汇表大小: {vocab_size}")

# 测试 Encoder
print("\n2. 测试 BackboneEncoder...")
try:
    encoder = BackboneEncoder(
        hidden_dim=256,
        num_layers=3,
        k_neighbors=30
    )
    
    # 创建虚拟输入
    batch_size = 2
    seq_len = 10
    dummy_coords = torch.randn(batch_size, seq_len, 4, 3)
    
    # 前向传播
    print(f"   输入形状: {dummy_coords.shape}")
    node_embeddings = encoder(dummy_coords)
    print(f"   输出形状: {node_embeddings.shape}")
    print(f"   ✅ Encoder 测试通过")
    
except Exception as e:
    print(f"   ❌ Encoder 测试失败: {e}")
    import traceback
    traceback.print_exc()
    node_embeddings = None

# 测试 Decoder
print("\n3. 测试 FragmentDecoder...")
try:
    if node_embeddings is not None:
        decoder = FragmentDecoder(
            input_dim=256,
            vocab_size=vocab_size,
            num_torsion_bins=72,
            hidden_dim=256,
            num_layers=3,
            num_heads=8
        )
        
        # 创建目标片段序列（Token IDs）
        frag_seq_len = 20
        target_fragments = torch.randint(0, vocab_size, (batch_size, frag_seq_len))
        
        # 前向传播
        print(f"   节点嵌入形状: {node_embeddings.shape}")
        print(f"   目标片段形状: {target_fragments.shape}")
        frag_logits, tors_logits = decoder(
            node_embeddings=node_embeddings,
            target_fragments=target_fragments
        )
        print(f"   Fragment logits 形状: {frag_logits.shape}")
        print(f"   Torsion logits 形状: {tors_logits.shape}")
        print(f"   ✅ Decoder 测试通过")
    else:
        print("   ⚠️  跳过 Decoder 测试（Encoder 失败）")
        
except Exception as e:
    print(f"   ❌ Decoder 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试端到端
print("\n4. 测试端到端流程...")
try:
    if node_embeddings is not None:
        # 模拟训练时的前向传播
        # 1. Encoder
        backbone_coords = torch.randn(1, 15, 4, 3)
        node_emb = encoder(backbone_coords)
        
        # 2. Decoder
        target_frags = torch.randint(0, vocab_size, (1, 25))
        frag_logits, tors_logits = decoder(node_emb, target_frags)
        
        # 3. 计算损失（示例）
        target_frag_ids = torch.randint(0, vocab_size, (1, 25))
        frag_loss = torch.nn.functional.cross_entropy(
            frag_logits.reshape(-1, vocab_size),
            target_frag_ids.reshape(-1)
        )
        
        target_tors_bins = torch.randint(0, 72, (1, 25))
        tors_loss = torch.nn.functional.cross_entropy(
            tors_logits.reshape(-1, 72),
            target_tors_bins.reshape(-1)
        )
        
        print(f"   Fragment loss: {frag_loss.item():.4f}")
        print(f"   Torsion loss: {tors_loss.item():.4f}")
        print(f"   ✅ 端到端测试通过")
    else:
        print("   ⚠️  跳过端到端测试")
        
except Exception as e:
    print(f"   ❌ 端到端测试失败: {e}")
    import traceback
    traceback.print_exc()

# 模型大小统计
print("\n5. 模型参数统计...")
try:
    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total_params = encoder_params + decoder_params
    
    print(f"   Encoder 参数: {encoder_params:,}")
    print(f"   Decoder 参数: {decoder_params:,}")
    print(f"   总参数: {total_params:,}")
    
except Exception as e:
    print(f"   ⚠️  参数统计失败: {e}")

print("\n" + "="*70)
print("测试完成！")
print("="*70)
