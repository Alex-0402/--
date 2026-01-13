"""
Fragment and Torsion Angle Decoder

This module implements the decoder that predicts fragment tokens and torsion angles
from backbone node embeddings.

The decoder should be designed to work with the masked diffusion model framework,
predicting both fragment types and their associated torsion angles.

Author: Research Team
Date: 2024
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class FragmentDecoder(nn.Module):
    """
    Decoder for predicting fragment tokens and torsion angles.
    
    This decoder takes node embeddings from the backbone encoder and predicts:
    1. Fragment token types (classification)
    2. Torsion angle bins (classification)
    
    Input: Node embeddings [batch_size, L, hidden_dim]
    Output: 
        - Fragment logits [batch_size, M, vocab_size] where M is total fragments
        - Torsion logits [batch_size, K, num_bins] where K is number of torsion angles
    """
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        num_torsion_bins: int = 72,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize the fragment decoder.
        
        Args:
            input_dim: Dimension of input node embeddings
            vocab_size: Size of fragment vocabulary
            num_torsion_bins: Number of bins for torsion angle discretization
            hidden_dim: Hidden dimension for decoder layers
            num_layers: Number of decoder layers
            dropout: Dropout rate
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_torsion_bins = num_torsion_bins
        
        # Projection layers
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Decoder layers
        decoder_layers = []
        for _ in range(num_layers):
            decoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Output heads
        self.fragment_head = nn.Linear(hidden_dim, vocab_size)
        self.torsion_head = nn.Linear(hidden_dim, num_torsion_bins)
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        fragment_positions: Optional[torch.Tensor] = None,
        torsion_positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict fragment tokens and torsion angles.
        
        Args:
            node_embeddings: Node embeddings [batch_size, L, input_dim]
            fragment_positions: Optional indices mapping fragments to residues
                               [batch_size, M] where M is number of fragments
            torsion_positions: Optional indices mapping torsions to residues
                              [batch_size, K] where K is number of torsions
        
        Returns:
            Tuple of:
            - fragment_logits: [batch_size, M, vocab_size]
            - torsion_logits: [batch_size, K, num_torsion_bins]
        """
        batch_size, seq_len, _ = node_embeddings.shape
        
        # Project input
        x = self.input_proj(node_embeddings)
        x = self.decoder(x)
        
        # For now, use all node embeddings for prediction
        # In practice, you would use fragment_positions and torsion_positions
        # to select the appropriate node embeddings
        
        # Fragment prediction: aggregate per residue and predict
        # This is simplified - in practice, you need to handle variable fragments per residue
        fragment_features = x.mean(dim=1)  # [batch_size, hidden_dim]
        fragment_logits = self.fragment_head(fragment_features)  # [batch_size, vocab_size]
        fragment_logits = fragment_logits.unsqueeze(1)  # [batch_size, 1, vocab_size]
        
        # Torsion prediction: similar approach
        torsion_features = x.mean(dim=1)  # [batch_size, hidden_dim]
        torsion_logits = self.torsion_head(torsion_features)  # [batch_size, num_torsion_bins]
        torsion_logits = torsion_logits.unsqueeze(1)  # [batch_size, 1, num_torsion_bins]
        
        return fragment_logits, torsion_logits


# Example usage
if __name__ == "__main__":
    # Test the decoder
    vocab_size = 16  # 4 special tokens + 12 fragments
    decoder = FragmentDecoder(
        input_dim=256,
        vocab_size=vocab_size,
        num_torsion_bins=72
    )
    
    # Create dummy input
    dummy_embeddings = torch.randn(2, 10, 256)  # batch_size=2, seq_len=10
    
    # Forward pass
    frag_logits, tors_logits = decoder(dummy_embeddings)
    print(f"Input shape: {dummy_embeddings.shape}")
    print(f"Fragment logits shape: {frag_logits.shape}")
    print(f"Torsion logits shape: {tors_logits.shape}")
