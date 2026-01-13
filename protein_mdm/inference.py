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
    
    encoder = BackboneEncoder(hidden_dim=args.hidden_dim, num_layers=3)
    decoder = FragmentDecoder(
        input_dim=args.hidden_dim,
        vocab_size=vocab.get_vocab_size(),
        num_torsion_bins=72,
        hidden_dim=args.hidden_dim
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
    
    # 生成侧链
    print("\n3. 生成侧链...")
    with torch.no_grad():
        # Encoder
        node_embeddings = encoder(backbone_coords)
        
        # 创建初始片段序列（全 MASK）
        frag_seq_len = len(sample['fragment_token_ids'])
        target_fragments = torch.full(
            (1, frag_seq_len),
            vocab.token_to_idx["[MASK]"],
            dtype=torch.long,
            device=device
        )
        
        # Decoder（迭代生成）
        # 简化版本：一次性生成所有片段
        frag_logits, tors_logits = decoder(
            node_embeddings=node_embeddings,
            target_fragments=target_fragments
        )
        
        # 获取预测
        fragment_predictions = torch.argmax(frag_logits, dim=-1)[0]
        torsion_predictions = torch.argmax(tors_logits, dim=-1)[0]
    
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
