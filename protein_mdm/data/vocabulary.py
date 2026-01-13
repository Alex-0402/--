"""
Fragment Tokenization Vocabulary for Protein Side-chain Design

This module defines the mapping from amino acid residues to rigid chemical fragments.
The core idea is to decompose side-chains into atomic-level fragments (tokens) that can be
assembled like building blocks, rather than predicting entire residues atom-by-atom.

Author: Research Team
Date: 2024
"""

from typing import List, Dict, Optional
from enum import IntEnum


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
            "LEU": ["METHYLENE", "BRANCH_CH", "METHYL", "METHYL"],  # CH2CH(CH3)2
            "ILE": ["METHYLENE", "BRANCH_CH", "METHYL", "METHYLENE", "METHYL"],  # CH(CH3)CH2CH3
            "MET": ["METHYLENE", "METHYLENE", "THIOL", "METHYL"],  # CH2CH2SCH3
            
            # Aromatic
            "PHE": ["METHYLENE", "PHENYL"],  # CH2-C6H5
            "TYR": ["METHYLENE", "PHENYL", "HYDROXYL"],  # CH2-C6H4-OH
            "TRP": ["METHYLENE", "INDOLE"],  # CH2-吲哚环
            
            # Polar uncharged
            "SER": ["HYDROXYL"],  # CH2OH
            "THR": ["BRANCH_CH", "HYDROXYL", "METHYL"],  # CH(OH)CH3
            "ASN": ["METHYLENE", "AMIDE"],  # CH2CONH2
            "GLN": ["METHYLENE", "METHYLENE", "AMIDE"],  # CH2CH2CONH2
            
            # Positively charged
            "LYS": ["METHYLENE", "METHYLENE", "METHYLENE", "METHYLENE", "AMINE"],  # (CH2)4NH3+
            "ARG": ["METHYLENE", "METHYLENE", "METHYLENE", "GUANIDINE"],  # (CH2)3-NH-C(=NH)-NH2
            "HIS": ["METHYLENE", "IMIDAZOLE"],  # CH2-咪唑环
            
            # Negatively charged
            "ASP": ["METHYLENE", "CARBOXYL"],  # CH2COO-
            "GLU": ["METHYLENE", "METHYLENE", "CARBOXYL"],  # CH2CH2COO-
            
            # Special cases
            "CYS": ["THIOL"],  # CH2SH
            "GLY": [],  # 无侧链 (仅H原子)
            "PRO": ["METHYLENE", "METHYLENE", "METHYLENE"],  # 形成环状结构，简化处理为链状
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
