"""
评价脚本：模糊匹配/松弛恢复率计算 (Relaxed Sequence Recovery)
用于解决离散片段预测时因为微小误差导致整条侧链被判错的问题。
"""
import difflib
from typing import List, Dict, Tuple

def decode_relaxed_amino_acid(
    predicted_fragments: List[str],
    vocab_residue_map: Dict[str, List[str]],
    threshold: float = 0.75
) -> Tuple[str, float]:
    """
    将一串预测的片段，通过最长公共子序列（编辑距离相似度）映射回最可能的氨基酸。
    
    参数:
        predicted_fragments: 该残基位置预测出的片段列表，例如 ["METHYLENE", "METHYLENE", "AMIDE"]
        vocab_residue_map: 标准的 20 种氨基酸映射表，来自 FragmentVocab._residue_to_fragments_map
        threshold: 相似度接受阈值。如果最高相似度不足此值，则判为未知("UNK")
        
    返回:
        best_aa: 最匹配的氨基酸类型（三字母代号）
        best_ratio: 相似度分数 (0.0 到 1.0)
    """
    best_aa = "UNK"
    best_ratio = 0.0
    
    # 遍历20种标准氨基酸
    for aa, standard_fragments in vocab_residue_map.items():
        # 如果长度都为0（例如 GLY），则完美匹配
        if len(standard_fragments) == 0 and len(predicted_fragments) == 0:
            return aa, 1.0
            
        # 计算相似度 ratio (范围 0.0 到 1.0)
        # 例如: ["METHYLENE", "METHYLENE", "CARBOXYL"] (GLU)
        #      ["METHYLENE", "METHYLENE", "AMIDE"] (GLN)
        # 匹配长度为2，总长度为 3+3=6，ratio = 4 / 6 = 0.66
        sm = difflib.SequenceMatcher(None, predicted_fragments, standard_fragments)
        ratio = sm.ratio()
        
        # 记录最接近的标准氨基酸
        if ratio > best_ratio:
            best_ratio = ratio
            best_aa = aa
            
    # 根据阈值决定是否接受此匹配
    if best_ratio >= threshold:
        return best_aa, best_ratio
    else:
        return "UNK", best_ratio

def calculate_relaxed_aar(
    pred_per_res: List[List[str]],
    gt_per_res: List[List[str]], 
    true_residue_types: List[str],
    vocab_residue_map: Dict[str, List[str]],
    threshold: float = 0.70
) -> dict:
    """
    计算完整序列在“严格精确匹配”和“松弛特征映射”下的恢复率。
    """
    total = len(true_residue_types)
    strict_correct = 0
    relaxed_correct = 0
    
    details = []
    
    for i in range(total):
        pred_frags = pred_per_res[i]
        true_frags = gt_per_res[i]
        true_aa = true_residue_types[i]
        
        # 1. 传统的 Strict Exact Match
        if pred_frags == true_frags:
            strict_correct += 1
            is_strict = True
        else:
            is_strict = False
            
        # 2. 新型 Relaxed Match (降维模糊映射)
        projected_aa, sim_ratio = decode_relaxed_amino_acid(pred_frags, vocab_residue_map, threshold)
        
        # 如果模型吐出来的预测序列，最贴近（且超过阈值贴近）的地雷就是它本来的真实氨基酸
        if projected_aa == true_aa:
            relaxed_correct += 1
            is_relaxed = True
        else:
            is_relaxed = False
            
        details.append({
            'res_idx': i,
            'true_aa': true_aa,
            'true_frags': true_frags,
            'pred_frags': pred_frags,
            'projected_aa': projected_aa,
            'similarity': sim_ratio,
            'is_strict': is_strict,
            'is_relaxed': is_relaxed
        })
        
    return {
        'total': total,
        'strict_acc': strict_correct / total if total > 0 else 0.0,
        'relaxed_acc': relaxed_correct / total if total > 0 else 0.0,
        'strict_correct': strict_correct,
        'relaxed_correct': relaxed_correct,
        'details': details
    }
