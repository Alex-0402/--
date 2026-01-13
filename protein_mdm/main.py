"""
Main Entry Point for Protein Masked Diffusion Model

This script provides the main entry point for training and inference
of the protein side-chain design model.

Author: Research Team
Date: 2024
"""

import argparse
import torch
from torch.utils.data import DataLoader

from data.vocabulary import FragmentVocab, get_vocab
from data.dataset import ProteinStructureDataset, collate_fn
from models.encoder import BackboneEncoder
from models.decoder import FragmentDecoder


def main():
    """Main function for training and inference"""
    parser = argparse.ArgumentParser(
        description="Protein Side-chain Design with Masked Diffusion Model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference", "test"],
        default="test",
        help="Mode: train, inference, or test"
    )
    parser.add_argument(
        "--pdb_path",
        type=str,
        default=None,
        help="Path to PDB file or directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for encoder/decoder"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Protein Masked Diffusion Model")
    print("="*60)
    
    # Initialize vocabulary
    print("\n1. Initializing vocabulary...")
    vocab = get_vocab()
    print(f"   Vocabulary size: {vocab.get_vocab_size()}")
    print(f"   Fragment tokens: {vocab.get_fragment_count()}")
    
    # Test vocabulary with example residues
    print("\n2. Testing vocabulary mappings:")
    test_residues = ["ALA", "PHE", "VAL", "ARG"]
    for res in test_residues:
        fragments = vocab.residue_to_fragments(res)
        print(f"   {res} -> {fragments}")
    
    # Initialize models
    print("\n3. Initializing models...")
    encoder = BackboneEncoder(hidden_dim=args.hidden_dim)
    decoder = FragmentDecoder(
        input_dim=args.hidden_dim,
        vocab_size=vocab.get_vocab_size(),
        num_torsion_bins=72,
        hidden_dim=args.hidden_dim
    )
    print(f"   Encoder output dim: {encoder.get_output_dim()}")
    print(f"   Decoder vocab size: {decoder.vocab_size}")
    
    # Test with dummy data
    print("\n4. Testing model forward pass...")
    dummy_backbone = torch.randn(1, 10, 4, 3)  # batch_size=1, seq_len=10
    node_embeddings = encoder(dummy_backbone)
    frag_logits, tors_logits = decoder(node_embeddings)
    print(f"   Input backbone shape: {dummy_backbone.shape}")
    print(f"   Node embeddings shape: {node_embeddings.shape}")
    print(f"   Fragment logits shape: {frag_logits.shape}")
    print(f"   Torsion logits shape: {tors_logits.shape}")
    
    # Load dataset if PDB path is provided
    if args.pdb_path:
        print(f"\n5. Loading dataset from: {args.pdb_path}")
        try:
            dataset = ProteinStructureDataset(args.pdb_path)
            print(f"   Dataset size: {len(dataset)}")
            
            if len(dataset) > 0:
                # Load first sample
                sample = dataset[0]
                print(f"   Sample backbone shape: {sample['backbone_coords'].shape}")
                print(f"   Sample fragment tokens: {len(sample['fragment_token_ids'])}")
                print(f"   Sample torsion bins: {len(sample['torsion_bins'])}")
                print(f"   Sequence length: {sample['sequence_length'].item()}")
                
                # Test dataloader
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    collate_fn=collate_fn,
                    shuffle=False
                )
                print(f"\n6. Testing dataloader...")
                batch = next(iter(dataloader))
                print(f"   Batch backbone shape: {batch['backbone_coords'].shape}")
                print(f"   Batch fragment shape: {batch['fragment_token_ids'].shape}")
                print(f"   Batch torsion shape: {batch['torsion_bins'].shape}")
        except Exception as e:
            print(f"   Error loading dataset: {e}")
    
    print("\n" + "="*60)
    print("Initialization complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Implement GVP-based encoder for SE(3) equivariance")
    print("  2. Implement masked diffusion training loop")
    print("  3. Implement adaptive inference strategy")
    print("  4. Train on protein structure dataset")


if __name__ == "__main__":
    main()
