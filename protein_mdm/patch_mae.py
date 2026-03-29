import re

with open('inference.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = """        if min_tors_len > 0 and np.sum(valid_mask) > 0:
            gt_bins_valid = gt_torsion_bins[:min_tors_len][valid_mask]
            pred_bins_valid = pred_torsion_bins[:min_tors_len][valid_mask]
            
            torsion_bin_acc = np.mean(pred_bins_valid == gt_bins_valid)
            
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
            torsion_mae_deg = np.degrees(np.mean(np.abs(diff)))
        else:
            torsion_bin_acc = float('nan')
            torsion_mae_deg = float('nan')"""

new_block = """        if min_tors_len > 0 and np.sum(valid_mask) > 0:
            gt_bins_valid = gt_torsion_bins[:min_tors_len][valid_mask]
            pred_bins_valid = pred_torsion_bins[:min_tors_len][valid_mask]
            
            torsion_bin_acc = np.mean(pred_bins_valid == gt_bins_valid)
            
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
            torsion_mae_deg = np.degrees(np.mean(np.abs(diff)))

            # CONDITIONAL MAE (Only correctly predicted fragments)
            gt_frag_arr = np.array(gt_fragment_ids[:min_tors_len])
            pred_frag_arr = np.array(pred_fragment_ids[:min_tors_len])
            frag_correct_mask = (gt_frag_arr == pred_frag_arr)
            cond_valid_mask = valid_mask & frag_correct_mask

            if np.sum(cond_valid_mask) > 0:
                if torsion_raw is not None:
                    c_gt_angles = torsion_raw.cpu().numpy()[:min_tors_len][cond_valid_mask]
                else:
                    c_gt_bins = gt_torsion_bins[:min_tors_len][cond_valid_mask]
                    c_gt_angles = undiscretize_angles(c_gt_bins, num_bins=72)
                    
                c_pred_bins = pred_torsion_bins[:min_tors_len][cond_valid_mask]
                c_pred_angles_raw = undiscretize_angles(c_pred_bins, num_bins=72)
                c_pred_offsets = offset_predictions.cpu().numpy()[:min_tors_len][cond_valid_mask]
                c_pred_angles = c_pred_angles_raw + c_pred_offsets * (2 * np.pi / 72)
                
                c_diff = np.arctan2(np.sin(c_pred_angles - c_gt_angles), np.cos(c_pred_angles - c_gt_angles))
                cond_torsion_mae_deg = np.degrees(np.mean(np.abs(c_diff)))
            else:
                cond_torsion_mae_deg = float('nan')

            # MAE by Chi Level (0 to 4 based on fragment_levels)
            chi_mae_dict = {}
            if 'fragment_levels' in sample:
                levels = sample['fragment_levels'].cpu().numpy()[:min_tors_len]
                for lvl in range(5): # Expected 0, 1, 2, 3, 4
                    lvl_mask = valid_mask & (levels == lvl) & frag_correct_mask # Evaluated on correct fragments!
                    if np.sum(lvl_mask) > 0:
                        if torsion_raw is not None:
                            l_gt = torsion_raw.cpu().numpy()[:min_tors_len][lvl_mask]
                        else:
                            l_gt_bins = gt_torsion_bins[:min_tors_len][lvl_mask]
                            l_gt = undiscretize_angles(l_gt_bins, num_bins=72)
                        l_pred_bins = pred_torsion_bins[:min_tors_len][lvl_mask]
                        l_pred_raw = undiscretize_angles(l_pred_bins, num_bins=72)
                        l_pred_off = offset_predictions.cpu().numpy()[:min_tors_len][lvl_mask]
                        l_pred = l_pred_raw + l_pred_off * (2 * np.pi / 72)
                        l_diff = np.arctan2(np.sin(l_pred - l_gt), np.cos(l_pred - l_gt))
                        chi_mae_dict[f'chi_{lvl+1}'] = np.degrees(np.mean(np.abs(l_diff)))

        else:
            torsion_bin_acc = float('nan')
            torsion_mae_deg = float('nan')
            cond_torsion_mae_deg = float('nan')
            chi_mae_dict = {}"""

if old_block in content:
    content = content.replace(old_block, new_block)
    print("Block replaced successfully!")
else:
    print("Error: Block not found.")

def replace_print_block():
    old_print = """        print(f"   Fragment Token Acc: {frag_token_acc:.4f}")
        print(f"   Residue侧链类型一致率(Exact): {residue_type_match_rate:.4f}")
        print(f"   Torsion Bin Acc: {torsion_bin_acc:.4f}")
        print(f"   Torsion Circular MAE: {torsion_mae_deg:.2f}°")"""
        
    new_print = """        print(f"   Fragment Token Acc: {frag_token_acc:.4f}")
        print(f"   Residue侧链类型一致率(Exact): {residue_type_match_rate:.4f}")
        print(f"   Torsion Bin Acc: {torsion_bin_acc:.4f}")
        print(f"   全局 Torsion MAE (含错片): {torsion_mae_deg:.2f}°")
        print(f"   条件 Torsion MAE (仅对片): {cond_torsion_mae_deg:.2f}°")
        if chi_mae_dict:
            chi_strs = [f"{k}={v:.1f}°" for k, v in chi_mae_dict.items()]
            print(f"   分层条件 MAE (Chi-1~5): " + " | ".join(chi_strs))"""
            
    return old_print, new_print

old_p, new_p = replace_print_block()
if old_p in content:
    content = content.replace(old_p, new_p)
    print("Print block replaced successfully!")
else:
    print("Error: Print block not found.")
    
# Third replacement: returning metrics
old_ret = """        return {
            'strategy': strategy_name,
            'frag_token_acc': frag_token_acc,
            'residue_exact': residue_type_match_rate,
            'coverage': residue_type_match_rate,
            'rmsd_all': rmsd_all,
            'rmsd_matched': rmsd_matched,
            'clash': clash_score
        }"""

new_ret = """        return {
            'strategy': strategy_name,
            'frag_token_acc': frag_token_acc,
            'residue_exact': residue_type_match_rate,
            'coverage': residue_type_match_rate,
            'rmsd_all': rmsd_all,
            'rmsd_matched': rmsd_matched,
            'clash': clash_score,
            'cond_mae': cond_torsion_mae_deg,
        }"""

if old_ret in content:
    content = content.replace(old_ret, new_ret)
    print("Return block replaced successfully!")
else:
    print("Error: Return block not found.")
    
with open('inference.py', 'w', encoding='utf-8') as f:
    f.write(content)
    
