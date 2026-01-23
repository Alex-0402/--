"""
推理脚本

使用方法:
    python inference.py --model_path checkpoints/best_model.pt --pdb_path data/pdb_files/1CRN.pdb
"""

import argparse
import torch
import numpy as np

from models.encoder import BackboneEncoder
from models.decoder import FragmentDecoder
from data.dataset import ProteinStructureDataset
from data.vocabulary import get_vocab
from data.geometry import undiscretize_angles


def main():
    parser = argparse.ArgumentParser(
        description="使用训练好的模型生成蛋白质侧链"
    )
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型检查点路径")
    parser.add_argument("--pdb_path", type=str, required=True,
                       help="PDB 文件路径（骨架结构）")
    parser.add_argument("--output_path", type=str, default=None,
                       help="输出路径（可选）")
    parser.add_argument("--device", type=str, default=None,
                       help="设备 (cuda/cpu)")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="隐藏层维度（需与训练时一致）")
    parser.add_argument("--num_encoder_layers", type=int, default=3,
                       help="Encoder 层数（需与训练时一致）")
    parser.add_argument("--num_decoder_layers", type=int, default=3,
                       help="Decoder 层数（需与训练时一致）")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="注意力头数（需与训练时一致）")
    parser.add_argument("--num_iterations", type=int, default=10,
                       help="自适应迭代推理的迭代轮数（默认10）")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("="*70)
    print("蛋白质侧链生成 - 推理")
    print("="*70)
    print(f"设备: {device}")
    print(f"模型: {args.model_path}")
    print(f"输入 PDB: {args.pdb_path}")
    print("="*70)
    
    # 加载模型
    print("\n1. 加载模型...")
    vocab = get_vocab()
    
    encoder = BackboneEncoder(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_encoder_layers
    )
    decoder = FragmentDecoder(
        input_dim=args.hidden_dim,
        vocab_size=vocab.get_vocab_size(),
        num_torsion_bins=72,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_decoder_layers,
        num_heads=args.num_heads
    )
    
    checkpoint = torch.load(args.model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    print(f"   模型加载成功 (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # 加载骨架结构
    print("\n2. 加载骨架结构...")
    dataset = ProteinStructureDataset(args.pdb_path)
    sample = dataset[0]
    
    backbone_coords = sample['backbone_coords'].unsqueeze(0).to(device)
    sequence_length = sample['sequence_length'].item()
    
    print(f"   序列长度: {sequence_length}")
    print(f"   骨架形状: {backbone_coords.shape}")
    
    # 生成侧链（自适应迭代推理）
    print("\n3. 生成侧链（自适应迭代推理）...")
    num_iterations = args.num_iterations
    print(f"   迭代轮数: {num_iterations}")
        
        with torch.no_grad():
            # Encoder
            residue_types = sample.get('residue_types', None)
            if residue_types is not None:
                residue_types = [residue_types]  # 包装成batch格式
            node_embeddings = encoder(backbone_coords, residue_types=residue_types)
            
            # 创建初始片段序列（全 MASK）
            frag_seq_len = len(sample['fragment_token_ids'])
            mask_token_id = vocab.token_to_idx["[MASK]"]
            target_fragments = torch.full(
                (1, frag_seq_len),
                mask_token_id,
                dtype=torch.long,
                device=device
            )
            
            # 迭代解码循环（MaskGIT风格）
            for k in range(num_iterations):
                # 当前保留比例（逐渐增加）
                keep_ratio = k / num_iterations  # 从0到1
                
                # Decoder预测
                frag_logits, tors_logits = decoder(
                    node_embeddings=node_embeddings,
                    target_fragments=target_fragments
                )
                
                # 计算每个位置的置信度（使用最大概率作为置信度）
                frag_probs = torch.softmax(frag_logits, dim=-1)  # [1, M, vocab_size]
                max_probs, _ = torch.max(frag_probs, dim=-1)  # [1, M] - 每个位置的最大概率
                confidence_scores = max_probs[0]  # [M]
                
                # 计算需要保留的Token数量
                num_to_keep = int(frag_seq_len * keep_ratio)
                
                if k < num_iterations - 1:
                    # 不是最后一轮：根据置信度保留高置信度的Token，其余重新Mask
                    if num_to_keep > 0:
                        # 获取置信度最高的位置
                        _, top_indices = torch.topk(confidence_scores, k=num_to_keep, largest=True)
                        
                        # 更新target_fragments：保留高置信度位置的预测，其余重新Mask
                        predicted_tokens = torch.argmax(frag_logits, dim=-1)[0]  # [M]
                        target_fragments[0, top_indices] = predicted_tokens[top_indices]
                        # 其余位置重新Mask
                        mask_positions = torch.ones(frag_seq_len, dtype=torch.bool, device=device)
                        mask_positions[top_indices] = False
                        target_fragments[0, mask_positions] = mask_token_id
                    else:
                        # 第一轮：全部保持Mask
                        pass
                    
                    if (k + 1) % 2 == 0 or k == 0:
                        print(f"   迭代 {k+1}/{num_iterations}: 保留 {num_to_keep}/{frag_seq_len} 个Token "
                              f"(平均置信度: {confidence_scores.mean().item():.3f})")
                else:
                    # 最后一轮：使用所有预测
                    fragment_predictions = torch.argmax(frag_logits, dim=-1)[0]
                    torsion_predictions = torch.argmax(tors_logits, dim=-1)[0]
                    print(f"   迭代 {k+1}/{num_iterations}: 完成生成")
    
    # 处理结果
    print("\n4. 结果:")
    predicted_fragments = vocab.indices_to_fragments(
        fragment_predictions.cpu().tolist()
    )
    torsion_angles = undiscretize_angles(
        torsion_predictions.cpu().numpy(),
        num_bins=72
    )
    
    print(f"   生成片段数: {len(predicted_fragments)}")
    print(f"   生成扭转角数: {len(torsion_angles)}")
    print(f"\n   前 10 个预测片段:")
    for i, frag in enumerate(predicted_fragments[:10]):
        print(f"      {i+1:3d}. {frag}")
    
    print(f"\n   前 10 个扭转角:")
    for i, angle in enumerate(torsion_angles[:10]):
        print(f"      {i+1:3d}. {np.degrees(angle):7.2f}°")
    
    if args.output_path:
        print(f"\n5. 保存结果到 {args.output_path}...")
        # TODO: 实现结构重建和保存
        print("   (结构重建功能待实现)")
    
    print("\n" + "="*70)
    print("推理完成！")
    print("="*70)


if __name__ == "__main__":
    main()
