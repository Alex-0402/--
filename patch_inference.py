import re

with open('protein_mdm/inference.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Edit 1: decode_with_strategy return
content = content.replace(
'''        fragment_predictions = torch.argmax(frag_logits, dim=-1)[0]
        torsion_predictions = torch.argmax(tors_logits, dim=-1)[0]

        return fragment_predictions, torsion_predictions''',
'''        fragment_predictions = torch.argmax(frag_logits, dim=-1)[0]
        torsion_predictions = torch.argmax(tors_logits, dim=-1)[0]
        
        # 获取与选定 torsion 对应的 offset
        if _ is not None:
            # _ 是 offset_logits [batch_size, seq_len, num_bins]
            offset_logits = _[0]  # [seq_len, num_bins]
            # 根据 torsion_predictions 选中对应的 offset
            offset_predictions = offset_logits.gather(dim=-1, index=torsion_predictions.unsqueeze(-1)).squeeze(-1)
        else:
            offset_predictions = torch.zeros_like(torsion_predictions, dtype=torch.float32)

        return fragment_predictions, torsion_predictions, offset_predictions'''
)

# Edit 2: evaluate_and_report signature and init computation
content = content.replace(
'''    def evaluate_and_report(strategy_name: str, fragment_predictions: torch.Tensor, torsion_predictions: torch.Tensor):
        print(f"\\n4. 结果 ({strategy_name})")
        predicted_fragments = vocab.indices_to_fragments(fragment_predictions.cpu().tolist())
        torsion_angles = undiscretize_angles(torsion_predictions.cpu().numpy(), num_bins=72)''',
'''    def evaluate_and_report(strategy_name: str, fragment_predictions: torch.Tensor, torsion_predictions: torch.Tensor, offset_predictions: torch.Tensor):
        print(f"\\n4. 结果 ({strategy_name})")
        predicted_fragments = vocab.indices_to_fragments(fragment_predictions.cpu().tolist())
        torsion_angles_raw = undiscretize_angles(torsion_predictions.cpu().numpy(), num_bins=72)
        import numpy as np
        torsion_angles = torsion_angles_raw + offset_predictions.cpu().numpy() * (2 * np.pi / 72)'''
)

# Edit 3: mae computation
content = content.replace(
'''            torsion_bin_acc = np.mean(pred_bins_valid == gt_bins_valid)
            gt_angles = undiscretize_angles(gt_bins_valid, num_bins=72)
            pred_angles = undiscretize_angles(pred_bins_valid, num_bins=72)
            diff = np.arctan2(np.sin(pred_angles - gt_angles), np.cos(pred_angles - gt_angles))
            torsion_mae_deg = np.degrees(np.mean(np.abs(diff)))''',
'''            torsion_bin_acc = np.mean(pred_bins_valid == gt_bins_valid)
            
            # Use raw ground truth angles if available
            torsion_raw = sample.get('torsion_raw', None)
            if torsion_raw is not None:
                gt_angles = torsion_raw.cpu().numpy()[:min_tors_len][valid_mask]
            else:
                gt_angles = undiscretize_angles(gt_bins_valid, num_bins=72)
                
            pred_angles_raw = undiscretize_angles(pred_bins_valid, num_bins=72)
            pred_offsets = offset_predictions.cpu().numpy()[:min_tors_len][valid_mask]
            pred_angles = pred_angles_raw + pred_offsets * (2 * np.pi / 72)
            
            diff = np.arctan2(np.sin(pred_angles - gt_angles), np.cos(pred_angles - gt_angles))
            torsion_mae_deg = np.degrees(np.mean(np.abs(diff)))'''
)

# Edit 4: usage calls
content = content.replace(
'''    rand_frag, rand_tors = decode_with_strategy(sample, strategy='random')
    evaluate_and_report('Random', rand_frag, rand_tors)

    adap_frag, adap_tors = decode_with_strategy(sample, strategy='adaptive')
    evaluate_and_report('Adaptive', adap_frag, adap_tors)''',
'''    rand_frag, rand_tors, rand_offset = decode_with_strategy(sample, strategy='random')
    evaluate_and_report('Random', rand_frag, rand_tors, rand_offset)

    adap_frag, adap_tors, adap_offset = decode_with_strategy(sample, strategy='adaptive')
    evaluate_and_report('Adaptive', adap_frag, adap_tors, adap_offset)'''
)

with open('protein_mdm/inference.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched!")
