"""
推理脚本

使用方法:
    python inference.py --model_path checkpoints/best_model.pt --pdb_path data/pdb_files/1CRN.pdb
"""

import argparse
import torch
import numpy as np
import os

from models.encoder import BackboneEncoder
from models.decoder import FragmentDecoder
from data.dataset import ProteinStructureDataset
from data.vocabulary import get_vocab
from data.geometry import undiscretize_angles
from utils.protein_utils import (
    build_sidechain_pseudo_atoms,
    write_backbone_sidechain_pdb,
    compute_clash_score,
    extract_sidechain_centroids_from_pdb,
    compute_centroid_rmsd,
    split_fragments_by_residue,
)


def main():
    parser = argparse.ArgumentParser(
        description="使用训练好的模型生成蛋白质侧链"
    )
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型检查点路径")
    parser.add_argument("--pdb_path", type=str, required=True,
                       help="PDB 文件路径（骨架结构）")
    parser.add_argument("--output_path", type=str, default=None,
                       help="输出PDB路径（可选）")
    parser.add_argument("--output_dir", type=str, default="inference_outputs",
                       help="输出目录（默认: inference_outputs）")
    parser.add_argument("--reference_pdb", type=str, default=None,
                       help="参考真值PDB路径（可选，默认使用输入PDB）")
    parser.add_argument("--device", type=str, default=None,
                       help="设备 (cuda/cpu)")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="隐藏层维度（需与训练时一致）")
    parser.add_argument("--num_iterations", type=int, default=12,
                       help="自适应推理迭代轮数")
    parser.add_argument("--min_commit_ratio", type=float, default=0.08,
                       help="每轮最小提交比例（相对当前未确定Token）")
    parser.add_argument("--max_commit_ratio", type=float, default=0.40,
                       help="每轮最大提交比例（相对当前未确定Token）")
    parser.add_argument("--strategy", type=str, default="adaptive",
                       choices=["adaptive", "random", "both"],
                       help="推理策略: adaptive(自适应) / random(随机顺序) / both(两者对比)")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机策略种子")
    
    args = parser.parse_args()

    if not (0.0 < args.min_commit_ratio <= 1.0 and 0.0 < args.max_commit_ratio <= 1.0):
        raise ValueError("min_commit_ratio 和 max_commit_ratio 必须在 (0, 1] 范围内")
    if args.min_commit_ratio > args.max_commit_ratio:
        raise ValueError("min_commit_ratio 不能大于 max_commit_ratio")
    
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

    strategies = [args.strategy] if args.strategy != "both" else ["random", "adaptive"]

    os.makedirs(args.output_dir, exist_ok=True)

    def _build_output_path(strategy_name: str) -> str:
        if args.output_path:
            # 如果只给了文件名（无目录），自动写入 output_dir
            if os.path.dirname(args.output_path) == "":
                base_output = os.path.join(args.output_dir, args.output_path)
            else:
                base_output = args.output_path
            if len(strategies) == 1:
                return base_output
            root, ext = os.path.splitext(base_output)
            ext = ext if ext else ".pdb"
            return f"{root}_{strategy_name}{ext}"
        base_name = os.path.splitext(os.path.basename(args.pdb_path))[0]
        return os.path.join(args.output_dir, f"{base_name}_{strategy_name}_predicted_sidechain.pdb")

    def decode_with_strategy(strategy_name: str, node_embeddings: torch.Tensor):
        frag_seq_len = len(sample['fragment_token_ids'])
        mask_token_id = vocab.token_to_idx["[MASK]"]
        target_fragments = torch.full(
            (1, frag_seq_len),
            mask_token_id,
            dtype=torch.long,
            device=device
        )

        fragment_predictions = None
        torsion_predictions = None

        print(f"\n3.{1 if strategy_name == 'random' else 2 if len(strategies) > 1 else 1} 推理策略: {strategy_name}")
        print(f"   迭代轮数: {args.num_iterations}")

        for k in range(args.num_iterations):
            frag_logits, tors_logits, _ = decoder(
                node_embeddings=node_embeddings,
                target_fragments=target_fragments
            )
            frag_probs = torch.softmax(frag_logits, dim=-1)
            predicted_tokens = torch.argmax(frag_logits, dim=-1)[0]

            masked_positions = (target_fragments[0] == mask_token_id)
            remaining = int(masked_positions.sum().item())
            if remaining == 0:
                fragment_predictions = target_fragments[0].clone()
                torsion_predictions = torch.argmax(tors_logits, dim=-1)[0]
                print(f"   迭代 {k+1}/{args.num_iterations}: 已全部确定，提前结束")
                break

            progress = (k + 1) / args.num_iterations
            base_ratio = args.min_commit_ratio + (args.max_commit_ratio - args.min_commit_ratio) * progress
            base_ratio = float(np.clip(base_ratio, args.min_commit_ratio, args.max_commit_ratio))

            masked_indices = torch.where(masked_positions)[0]

            if strategy_name == "adaptive":
                top2_vals, _ = torch.topk(frag_probs[0], k=2, dim=-1)
                margins = top2_vals[:, 0] - top2_vals[:, 1]
                masked_margins = margins[masked_positions]
                avg_margin = masked_margins.mean().item() if masked_margins.numel() > 0 else 0.0
                avg_uncertainty = 1.0 - avg_margin

                adaptive_ratio = base_ratio * (1.0 - 0.6 * avg_uncertainty)
                adaptive_ratio = float(np.clip(adaptive_ratio, args.min_commit_ratio, args.max_commit_ratio))
                if k == args.num_iterations - 1:
                    num_to_commit = remaining
                else:
                    num_to_commit = max(1, int(np.ceil(remaining * adaptive_ratio)))
                    num_to_commit = min(num_to_commit, remaining)

                commit_local_idx = torch.topk(masked_margins, k=num_to_commit, largest=True).indices
                commit_indices = masked_indices[commit_local_idx]
            else:
                if k == args.num_iterations - 1:
                    num_to_commit = remaining
                else:
                    num_to_commit = max(1, int(np.ceil(remaining * base_ratio)))
                    num_to_commit = min(num_to_commit, remaining)

                perm = torch.randperm(masked_indices.numel(), device=masked_indices.device)
                commit_indices = masked_indices[perm[:num_to_commit]]
                avg_margin = float('nan')
                avg_uncertainty = float('nan')

            target_fragments[0, commit_indices] = predicted_tokens[commit_indices]

            if (k + 1) % 2 == 0 or k == 0 or k == args.num_iterations - 1:
                msg = (
                    f"   迭代 {k+1}/{args.num_iterations}: 新提交 {num_to_commit}/{remaining} | "
                    f"剩余Mask {int((target_fragments[0] == mask_token_id).sum().item())}"
                )
                if strategy_name == "adaptive":
                    msg += f" | avg_margin={avg_margin:.3f} | avg_uncertainty={avg_uncertainty:.3f}"
                print(msg)

            fragment_predictions = torch.argmax(frag_logits, dim=-1)[0]
            torsion_predictions = torch.argmax(tors_logits, dim=-1)[0]

        remaining_masked = (target_fragments[0] == mask_token_id)
        if remaining_masked.any() and fragment_predictions is not None:
            target_fragments[0, remaining_masked] = fragment_predictions[remaining_masked]

        frag_logits, tors_logits, _ = decoder(
            node_embeddings=node_embeddings,
            target_fragments=target_fragments
        )
        fragment_predictions = torch.argmax(frag_logits, dim=-1)[0]
        torsion_predictions = torch.argmax(tors_logits, dim=-1)[0]

        return fragment_predictions, torsion_predictions

    def evaluate_and_report(strategy_name: str, fragment_predictions: torch.Tensor, torsion_predictions: torch.Tensor):
        print(f"\n4. 结果 ({strategy_name})")
        predicted_fragments = vocab.indices_to_fragments(fragment_predictions.cpu().tolist())
        torsion_angles = undiscretize_angles(torsion_predictions.cpu().numpy(), num_bins=72)

        print(f"   生成片段数: {len(predicted_fragments)}")
        print(f"   生成扭转角数: {len(torsion_angles)}")

        gt_fragment_ids = sample['fragment_token_ids'].cpu().tolist()
        pred_fragment_ids = fragment_predictions.cpu().tolist()
        min_frag_len = min(len(gt_fragment_ids), len(pred_fragment_ids))
        frag_token_acc = np.mean(
            np.array(pred_fragment_ids[:min_frag_len]) == np.array(gt_fragment_ids[:min_frag_len])
        ) if min_frag_len > 0 else float('nan')

        gt_torsion_bins = sample['torsion_bins'].cpu().numpy()
        pred_torsion_bins = torsion_predictions.cpu().numpy()
        min_tors_len = min(len(gt_torsion_bins), len(pred_torsion_bins))
        if min_tors_len > 0:
            torsion_bin_acc = np.mean(pred_torsion_bins[:min_tors_len] == gt_torsion_bins[:min_tors_len])
            gt_angles = undiscretize_angles(gt_torsion_bins[:min_tors_len], num_bins=72)
            pred_angles = undiscretize_angles(pred_torsion_bins[:min_tors_len], num_bins=72)
            diff = np.arctan2(np.sin(pred_angles - gt_angles), np.cos(pred_angles - gt_angles))
            torsion_mae_deg = np.degrees(np.mean(np.abs(diff)))
        else:
            torsion_bin_acc = float('nan')
            torsion_mae_deg = float('nan')

        gt_fragments = vocab.indices_to_fragments(gt_fragment_ids)
        residue_types = sample.get('residue_types', [])
        pred_per_res = split_fragments_by_residue(
            predicted_fragments=predicted_fragments,
            residue_types=residue_types,
            residue_to_fragments_map=vocab._residue_to_fragments_map,
        )
        gt_per_res = split_fragments_by_residue(
            predicted_fragments=gt_fragments,
            residue_types=residue_types,
            residue_to_fragments_map=vocab._residue_to_fragments_map,
        )
        per_res_match = np.array([int(p == g) for p, g in zip(pred_per_res, gt_per_res)], dtype=np.int32)
        residue_type_match_rate = float(per_res_match.mean()) if len(per_res_match) > 0 else float('nan')

        print(f"   Fragment Token Acc: {frag_token_acc:.4f}")
        print(f"   Residue侧链类型一致率(Exact): {residue_type_match_rate:.4f}")
        print(f"   Torsion Bin Acc: {torsion_bin_acc:.4f}")
        print(f"   Torsion Circular MAE: {torsion_mae_deg:.2f}°")

        print(f"\n5. 结构重建与评估 ({strategy_name})")
        backbone_np = sample['backbone_coords'].cpu().numpy()
        pseudo_atoms = build_sidechain_pseudo_atoms(
            backbone_coords=backbone_np,
            residue_types=residue_types,
            predicted_fragments=predicted_fragments,
            torsion_angles=torsion_angles,
            residue_to_fragments_map=vocab._residue_to_fragments_map,
        )
        sidechain_coords = np.asarray([a['coord'] for a in pseudo_atoms], dtype=np.float32) if len(pseudo_atoms) > 0 else np.zeros((0, 3), dtype=np.float32)
        clash_score = compute_clash_score(sidechain_coords, threshold=2.0)
        print(f"   伪原子数量: {len(pseudo_atoms)}")
        print(f"   Clash score (<2.0Å): {clash_score:.4f}")

        centroid_rmsd = float('nan')
        matched_rmsd = float('nan')
        coverage = 0.0
        ref_pdb = args.reference_pdb if args.reference_pdb else args.pdb_path
        try:
            ref_centroids = extract_sidechain_centroids_from_pdb(ref_pdb, residue_types)
            pred_centroids = []
            for ridx in range(1, len(residue_types) + 1):
                coords = [a['coord'] for a in pseudo_atoms if int(a['res_idx']) == ridx]
                if len(coords) == 0:
                    coords = [backbone_np[ridx - 1, 1]] if ridx - 1 < backbone_np.shape[0] else []
                if len(coords) > 0:
                    pred_centroids.append(np.mean(np.asarray(coords), axis=0))
            pred_centroids = np.asarray(pred_centroids, dtype=np.float32) if len(pred_centroids) > 0 else np.zeros((0, 3), dtype=np.float32)
            centroid_rmsd = compute_centroid_rmsd(pred_centroids, ref_centroids)
            print(f"   侧链中心点 RMSD(对齐后): {centroid_rmsd:.4f} Å")

            n_match = min(len(per_res_match), len(pred_centroids), len(ref_centroids))
            matched_idx = np.where(per_res_match[:n_match] == 1)[0]
            coverage = (len(matched_idx) / n_match) if n_match > 0 else 0.0
            if len(matched_idx) >= 3:
                matched_pred = pred_centroids[matched_idx]
                matched_ref = ref_centroids[matched_idx]
                matched_rmsd = compute_centroid_rmsd(matched_pred, matched_ref)
                print(f"   条件RMSD(类型匹配子集): {matched_rmsd:.4f} Å")
                print(f"   条件RMSD覆盖率: {coverage:.4f} ({len(matched_idx)}/{n_match})")
            else:
                print(f"   条件RMSD: 样本不足（匹配残基 {len(matched_idx)}/{n_match}）")
        except Exception as e:
            print(f"   ⚠️  无法计算 RMSD: {e}")

        output_path = _build_output_path(strategy_name)
        write_backbone_sidechain_pdb(
            backbone_coords=backbone_np,
            residue_types=residue_types,
            pseudo_atoms=pseudo_atoms,
            output_path=output_path,
        )
        print(f"   已输出重建结构: {output_path}")

        return {
            'strategy': strategy_name,
            'frag_token_acc': float(frag_token_acc),
            'residue_exact': float(residue_type_match_rate),
            'torsion_bin_acc': float(torsion_bin_acc),
            'torsion_mae_deg': float(torsion_mae_deg),
            'clash': float(clash_score),
            'rmsd_all': float(centroid_rmsd),
            'rmsd_matched': float(matched_rmsd),
            'coverage': float(coverage),
            'output_path': output_path,
        }

    # 固定随机种子（random 策略可复现）
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with torch.no_grad():
        residue_types = sample.get('residue_types', None)
        encoder_residue_types = [residue_types] if residue_types is not None else None
        node_embeddings = encoder(backbone_coords, residue_types=encoder_residue_types)

        all_results = []
        for strategy_name in strategies:
            fragment_predictions, torsion_predictions = decode_with_strategy(strategy_name, node_embeddings)
            result = evaluate_and_report(strategy_name, fragment_predictions, torsion_predictions)
            all_results.append(result)

        if len(all_results) > 1:
            print("\n6. 策略对比汇总")
            print("   " + "-" * 64)
            for r in all_results:
                print(
                    f"   {r['strategy']:<8} | FragAcc={r['frag_token_acc']:.4f} | "
                    f"ResExact={r['residue_exact']:.4f} | Coverage={r['coverage']:.4f} | "
                    f"RMSD_all={r['rmsd_all']:.4f} | RMSD_matched={r['rmsd_matched']:.4f} | "
                    f"Clash={r['clash']:.4f}"
                )
    
    print("\n" + "="*70)
    print("推理完成！")
    print("="*70)


if __name__ == "__main__":
    main()
