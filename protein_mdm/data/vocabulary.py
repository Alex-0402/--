"""
Fragment Tokenization Vocabulary for Protein Side-chain Design

This module defines the mapping from amino acid residues to rigid chemical fragments.
The core idea is to decompose side-chains into atomic-level fragments (tokens) that can be
assembled like building blocks, rather than predicting entire residues atom-by-atom.

Author: Research Team
Date: 2024
"""

from typing import List, Dict, Optional, Tuple
from enum import IntEnum
import torch
import numpy as np


class SpecialTokens(IntEnum):
    """Special tokens for sequence processing"""
    PAD = 0
    MASK = 1
    BOS = 2  # Beginning of Sequence
    EOS = 3  # End of Sequence


class FragmentVocab:
    """
    Vocabulary for mapping amino acid residues to chemical fragment sequences.
    
    Each fragment represents a rigid chemical group that can be used as a building block
    for constructing side-chains. This approach enables:
    1. Modular side-chain generation
    2. Better handling of chemical constraints
    3. Adaptive inference strategies
    """
    
    # Fragment token definitions (starting from index 4, after special tokens)
    FRAGMENT_TOKENS = [
        "METHYL",           # -CH3
        "METHYLENE",        # -CH2-
        "HYDROXYL",         # -OH
        "PHENYL",           # 苯环 (C6H5-)
        "AMINE",            # -NH3+
        "CARBOXYL",         # -COO-
        "AMIDE",            # -CONH2
        "GUANIDINE",        # -NH-C(=NH)-NH2 (精氨酸特征基团)
        "IMIDAZOLE",        # 咪唑环 (组氨酸特征基团)
        "INDOLE",           # 吲哚环 (色氨酸特征基团)
        "THIOL",            # -SH
        "BRANCH_CH",        # >CH- (分叉点，用于支链氨基酸)
    ]
    
    # Fragment Token 的理化特征定义
    # 特征向量包括: [疏水性(Hydropathy), 电荷(Charge), 分子量(MW), 氢键供体数(H-donors), 氢键受体数(H-acceptors)]
    # 这些特征直接定义每个Fragment Token本身的理化属性，而不是基于氨基酸的平均值
    FRAGMENT_FEATURES: Dict[str, List[float]] = {
        # 疏水片段：高疏水性, 0电荷, 0氢键
        "METHYL": [3.0, 0.0, 15.03, 0, 0],           # -CH3: 疏水, 中性, 无氢键
        "METHYLENE": [1.5, 0.0, 14.03, 0, 0],        # -CH2-: 中等疏水, 中性, 无氢键
        "PHENYL": [2.5, 0.0, 77.10, 0, 0],           # C6H5-: 疏水芳香, 中性, 无氢键
        
        # 亲水片段：亲水性, 0电荷, 有氢键能力
        "HYDROXYL": [-1.0, 0.0, 17.01, 1, 1],        # -OH: 亲水, 中性, 1供体1受体
        "AMIDE": [-2.5, 0.0, 44.03, 2, 2],           # -CONH2: 亲水, 中性, 2供体2受体
        
        # 正电片段：亲水性, +1电荷
        "AMINE": [-2.0, 1.0, 17.03, 2, 1],           # -NH3+: 亲水, 正电, 2供体1受体
        "GUANIDINE": [-3.0, 1.0, 59.08, 4, 1],       # -NH-C(=NH)-NH2: 亲水, 正电, 4供体1受体
        
        # 负电片段：亲水性, -1电荷
        "CARBOXYL": [-2.5, -1.0, 45.02, 0, 2],      # -COO-: 亲水, 负电, 0供体2受体
        
        # 特殊片段
        "THIOL": [2.0, 0.0, 33.08, 1, 0],            # -SH: 疏水, 中性, 1供体0受体
        "IMIDAZOLE": [-2.0, 0.5, 68.08, 2, 2],       # 咪唑环: 亲水, 弱正电, 2供体2受体
        "INDOLE": [-0.5, 0.0, 117.15, 1, 1],         # 吲哚环: 弱亲水, 中性, 1供体1受体
        "BRANCH_CH": [2.0, 0.0, 13.02, 0, 0],        # >CH-: 疏水支链点, 中性, 无氢键
    }
    
    # 20种标准氨基酸的理化特征（保留用于其他用途）
    # 特征向量包括: [疏水性(Hydropathy), 电荷(Charge), 分子量(MW), 氢键供体数(H-donors), 氢键受体数(H-acceptors)]
    # 数据来源: Kyte-Doolittle疏水性指数, 标准分子量, 氢键能力
    PHYSICOCHEMICAL_FEATURES: Dict[str, List[float]] = {
        # Non-polar aliphatic
        "ALA": [1.8, 0.0, 89.09, 0, 0],      # 疏水, 中性, 小分子
        "VAL": [4.2, 0.0, 117.15, 0, 0],     # 疏水, 中性
        "LEU": [3.8, 0.0, 131.17, 0, 0],     # 疏水, 中性
        "ILE": [4.5, 0.0, 131.17, 0, 0],     # 疏水, 中性
        "MET": [1.9, 0.0, 149.21, 0, 1],     # 疏水, 中性, 含S
        
        # Aromatic
        "PHE": [2.8, 0.0, 165.19, 0, 0],     # 疏水, 中性, 芳香
        "TYR": [-1.3, 0.0, 181.19, 1, 1],    # 亲水, 中性, 芳香, 含OH
        "TRP": [-0.9, 0.0, 204.23, 1, 1],    # 亲水, 中性, 芳香, 含N
        
        # Polar uncharged
        "SER": [-0.8, 0.0, 105.09, 1, 1],    # 亲水, 中性, 含OH
        "THR": [-0.7, 0.0, 119.12, 1, 1],    # 亲水, 中性, 含OH
        "ASN": [-3.5, 0.0, 132.12, 2, 2],    # 亲水, 中性, 含酰胺
        "GLN": [-3.5, 0.0, 146.15, 2, 2],    # 亲水, 中性, 含酰胺
        
        # Positively charged
        "LYS": [-3.9, 1.0, 146.19, 2, 1],    # 亲水, 正电, 含NH3+
        "ARG": [-4.5, 1.0, 174.20, 4, 1],    # 亲水, 正电, 含胍基
        "HIS": [-3.2, 0.5, 155.16, 2, 2],    # 亲水, 弱正电, 含咪唑
        
        # Negatively charged
        "ASP": [-3.5, -1.0, 133.10, 1, 2],   # 亲水, 负电, 含COO-
        "GLU": [-3.5, -1.0, 147.13, 1, 2],   # 亲水, 负电, 含COO-
        
        # Special cases
        "CYS": [2.5, 0.0, 121.16, 1, 0],     # 疏水, 中性, 含SH
        "GLY": [-0.4, 0.0, 75.07, 0, 0],     # 亲水, 中性, 最小
        "PRO": [-1.6, 0.0, 115.13, 0, 0],    # 亲水, 中性, 环状
    }
    
    # 特征归一化参数（用于标准化）
    FEATURE_STATS = {
        'hydropathy': {'mean': 0.0, 'std': 2.5},      # 归一化到 [-1, 1] 范围
        'charge': {'mean': 0.0, 'std': 0.5},          # 归一化到 [-2, 2] 范围
        'molecular_weight': {'mean': 130.0, 'std': 30.0},  # 归一化到 [0, 200] 范围
        'h_donors': {'mean': 1.0, 'std': 1.2},       # 归一化到 [0, 4] 范围
        'h_acceptors': {'mean': 0.8, 'std': 0.9},     # 归一化到 [0, 2] 范围
    }
    
    def __init__(self):
        """Initialize the vocabulary with token-to-index mappings"""
        # Build token to index mapping
        self.token_to_idx: Dict[str, int] = {
            "[PAD]": SpecialTokens.PAD,
            "[MASK]": SpecialTokens.MASK,
            "[BOS]": SpecialTokens.BOS,
            "[EOS]": SpecialTokens.EOS,
        }
        
        # Add fragment tokens (starting from index 4)
        for idx, token in enumerate(self.FRAGMENT_TOKENS, start=4):
            self.token_to_idx[token] = idx
        
        # Build reverse mapping
        self.idx_to_token: Dict[int, str] = {v: k for k, v in self.token_to_idx.items()}
        
        # Vocabulary size
        self.vocab_size = len(self.token_to_idx)
        
        # Hardcoded mapping from 20 standard amino acids to fragment sequences
        # Note: Fragment order follows depth-first traversal of side-chain structure
        self._residue_to_fragments_map: Dict[str, List[str]] = {
            # Non-polar aliphatic
            "ALA": ["METHYL"],  # CH3
            "VAL": ["BRANCH_CH", "METHYL", "METHYL"],  # CH(CH3)2
            "LEU": ["METHYLENE", "BRANCH_CH", "METHYL", "METHYL"],  # CH2-CH(CH3)2
            
            # [修正] ILE: 去掉开头的 METHYLENE，改为直接分支
            # 结构: CA -> CH(CH3)(CH2CH3)
            "ILE": ["BRANCH_CH", "METHYL", "METHYLENE", "METHYL"],  
            
            "MET": ["METHYLENE", "METHYLENE", "THIOL", "METHYL"],  # CH2-CH2-S-CH3
            
            # Aromatic
            "PHE": ["METHYLENE", "PHENYL"],  # CH2-C6H5
            "TYR": ["METHYLENE", "PHENYL", "HYDROXYL"],  # CH2-C6H4-OH
            "TRP": ["METHYLENE", "INDOLE"],  # CH2-Indole
            
            # Polar uncharged
            # [修正] SER: 增加 CB (METHYLENE)
            "SER": ["METHYLENE", "HYDROXYL"],  # CH2-OH
            
            "THR": ["BRANCH_CH", "HYDROXYL", "METHYL"],  # CH(OH)CH3
            "ASN": ["METHYLENE", "AMIDE"],  # CH2-CONH2
            "GLN": ["METHYLENE", "METHYLENE", "AMIDE"],  # CH2-CH2-CONH2
            
            # Positively charged
            "LYS": ["METHYLENE", "METHYLENE", "METHYLENE", "METHYLENE", "AMINE"],  # (CH2)4-NH3+
            "ARG": ["METHYLENE", "METHYLENE", "METHYLENE", "GUANIDINE"],  # (CH2)3-Guanidine
            "HIS": ["METHYLENE", "IMIDAZOLE"],  # CH2-Imidazole
            
            # Negatively charged
            "ASP": ["METHYLENE", "CARBOXYL"],  # CH2-COO-
            "GLU": ["METHYLENE", "METHYLENE", "CARBOXYL"],  # CH2-CH2-COO-
            
            # Special cases
            # [修正] CYS: 增加 CB (METHYLENE)
            "CYS": ["METHYLENE", "THIOL"],  # CH2-SH
            
            "GLY": [],  # No side chain
            "PRO": ["METHYLENE", "METHYLENE", "METHYLENE"],  # Cyclic simplified
        }
    
    def residue_to_fragments(self, res_name: str) -> List[str]:
        """
        Convert a residue name to its fragment token sequence.
        
        Args:
            res_name: Three-letter amino acid code (e.g., 'ALA', 'PHE')
            
        Returns:
            List of fragment token strings representing the side-chain structure
            
        Raises:
            KeyError: If residue name is not in the vocabulary
        """
        res_name_upper = res_name.upper()
        if res_name_upper not in self._residue_to_fragments_map:
            raise KeyError(
                f"Unknown residue name: {res_name}. "
                f"Supported residues: {list(self._residue_to_fragments_map.keys())}"
            )
        return self._residue_to_fragments_map[res_name_upper].copy()
    
    def fragments_to_indices(self, fragments: List[str]) -> List[int]:
        """
        Convert a list of fragment tokens to their index representation.
        
        Args:
            fragments: List of fragment token strings
            
        Returns:
            List of token indices
        """
        return [self.token_to_idx[token] for token in fragments]
    
    def indices_to_fragments(self, indices: List[int]) -> List[str]:
        """
        Convert a list of token indices back to fragment tokens.
        
        Args:
            indices: List of token indices
            
        Returns:
            List of fragment token strings
        """
        return [self.idx_to_token[idx] for idx in indices]
    
    def get_vocab_size(self) -> int:
        """Return the vocabulary size (including special tokens)"""
        return self.vocab_size
    
    def get_fragment_count(self) -> int:
        """Return the number of fragment tokens (excluding special tokens)"""
        return len(self.FRAGMENT_TOKENS)
    
    def __len__(self) -> int:
        """Return vocabulary size"""
        return self.vocab_size
    
    def __repr__(self) -> str:
        return f"FragmentVocab(vocab_size={self.vocab_size}, fragments={len(self.FRAGMENT_TOKENS)})"
    
    def get_physicochemical_embedding(
        self,
        token_ids: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        将Token ID映射为理化特征向量
        
        此方法直接使用 FRAGMENT_FEATURES 字典查询每个Fragment Token本身的理化属性，
        而不是基于氨基酸的平均值计算。这提供了更准确和科学的特征表示。
        
        Args:
            token_ids: Token ID张量 [batch_size, seq_len] 或 [seq_len]
            normalize: 是否归一化特征（默认True）
        
        Returns:
            理化特征向量 [batch_size, seq_len, 5] 或 [seq_len, 5]
                特征维度: [疏水性, 电荷, 分子量, 氢键供体数, 氢键受体数]
        """
        # 确保是2D张量
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        # 构建Token ID到Fragment名称的映射
        # 索引: PAD=0, MASK=1, BOS=2, EOS=3, 然后片段从4开始
        # 片段顺序与 FRAGMENT_TOKENS 列表一致
        idx_to_fragment = {}
        for idx, fragment in enumerate(self.FRAGMENT_TOKENS, start=4):
            idx_to_fragment[idx] = fragment
        
        # 使用向量化操作构建特征矩阵（更高效）
        token_ids_flat = token_ids.flatten().cpu().numpy()
        features_list = []
        for token_id in token_ids_flat:
            token_id_int = int(token_id)
            
            # 特殊Token -> 零向量
            if token_id_int in [SpecialTokens.PAD, SpecialTokens.MASK, 
                               SpecialTokens.BOS, SpecialTokens.EOS]:
                feat = [0.0, 0.0, 0.0, 0.0, 0.0]
            # 片段Token -> 直接从 FRAGMENT_FEATURES 字典查询
            elif token_id_int in idx_to_fragment:
                fragment_name = idx_to_fragment[token_id_int]
                if fragment_name in self.FRAGMENT_FEATURES:
                    feat = self.FRAGMENT_FEATURES[fragment_name].copy()
                else:
                    # 未知片段，使用零向量
                    feat = [0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                # 未知Token，使用零向量
                feat = [0.0, 0.0, 0.0, 0.0, 0.0]
            
            features_list.append(feat)
        
        # 转换为张量
        features = torch.tensor(features_list, dtype=torch.float32, device=device)  # [batch_size * seq_len, 5]
        features = features.view(batch_size, seq_len, 5)  # [batch_size, seq_len, 5]
        
        # 归一化（可选）
        if normalize:
            # 对每个特征维度进行归一化
            stats = self.FEATURE_STATS
            features[:, :, 0] = (features[:, :, 0] - stats['hydropathy']['mean']) / stats['hydropathy']['std']  # 疏水性
            features[:, :, 1] = (features[:, :, 1] - stats['charge']['mean']) / stats['charge']['std']  # 电荷
            features[:, :, 2] = (features[:, :, 2] - stats['molecular_weight']['mean']) / stats['molecular_weight']['std']  # 分子量
            features[:, :, 3] = (features[:, :, 3] - stats['h_donors']['mean']) / stats['h_donors']['std']  # 氢键供体
            features[:, :, 4] = (features[:, :, 4] - stats['h_acceptors']['mean']) / stats['h_acceptors']['std']  # 氢键受体
        
        if squeeze_output:
            features = features.squeeze(0)
        
        return features


# Global vocabulary instance (singleton pattern)
_vocab_instance: Optional[FragmentVocab] = None


def get_vocab() -> FragmentVocab:
    """
    Get the global vocabulary instance (singleton).
    
    Returns:
        FragmentVocab instance
    """
    global _vocab_instance
    if _vocab_instance is None:
        _vocab_instance = FragmentVocab()
    return _vocab_instance


# Example usage and testing
if __name__ == "__main__":
    vocab = FragmentVocab()
    
    print("Fragment Vocabulary:")
    print(f"Vocabulary size: {vocab.get_vocab_size()}")
    print(f"Fragment tokens: {vocab.get_fragment_count()}")
    print("\nToken mappings:")
    for token, idx in sorted(vocab.token_to_idx.items(), key=lambda x: x[1]):
        print(f"  {idx:3d}: {token}")
    
    print("\n" + "="*60)
    print("Residue to Fragment Mappings:")
    print("="*60)
    
    # Test all 20 standard amino acids
    test_residues = [
        "ALA", "VAL", "LEU", "ILE", "MET",
        "PHE", "TYR", "TRP",
        "SER", "THR", "ASN", "GLN",
        "LYS", "ARG", "HIS",
        "ASP", "GLU",
        "CYS", "GLY", "PRO"
    ]
    
    for res in test_residues:
        fragments = vocab.residue_to_fragments(res)
        indices = vocab.fragments_to_indices(fragments)
        print(f"{res:3s} -> {fragments} -> {indices}")
