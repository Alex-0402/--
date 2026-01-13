"""
Protein Structure Dataset Loader

This module implements the ProteinStructureDataset class for loading and processing
PDB files into a format suitable for training the masked diffusion model.

The dataset extracts:
1. Backbone coordinates (N, CA, C, O) for each residue
2. Side-chain fragment token sequences
3. Torsion angle bins for flexible regions

Author: Research Team
Date: 2024
"""

import os
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

from .vocabulary import FragmentVocab, get_vocab
from .geometry import calculate_dihedrals, discretize_angles


# Suppress BioPython warnings for cleaner output
warnings.simplefilter('ignore', PDBConstructionWarning)


class ProteinStructureDataset(Dataset):
    """
    Dataset for loading protein structures from PDB files.
    
    Each sample contains:
    - backbone_coords: Tensor of shape [L, 4, 3] where L is sequence length
                      and 4 represents N, CA, C, O atoms
    - fragment_token_ids: Tensor of shape [M] where M is total fragment count
    - torsion_bins: Tensor of shape [K] where K is number of torsion angles
    - residue_types: List of residue names for reference
    
    Attributes:
        pdb_files: List of paths to PDB or mmCIF files
        vocab: FragmentVocab instance for tokenization
        parser: BioPython parser (PDBParser or MMCIFParser)
    """
    
    # Standard backbone atom names
    BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']
    
    def __init__(
        self,
        pdb_files: Union[str, List[str]],
        vocab: Optional[FragmentVocab] = None,
        use_mmcif: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            pdb_files: Path to a single PDB file, directory containing PDB files,
                      or list of PDB file paths
            vocab: FragmentVocab instance (if None, uses global singleton)
            use_mmcif: If True, use MMCIFParser instead of PDBParser
        """
        self.vocab = vocab if vocab is not None else get_vocab()
        self.parser = MMCIFParser(QUIET=True) if use_mmcif else PDBParser(QUIET=True)
        
        # Collect PDB files
        if isinstance(pdb_files, str):
            if os.path.isfile(pdb_files):
                self.pdb_files = [pdb_files]
            elif os.path.isdir(pdb_files):
                # Find all PDB/mmCIF files in directory
                extensions = ['.pdb', '.cif'] if use_mmcif else ['.pdb']
                self.pdb_files = [
                    os.path.join(pdb_files, f)
                    for f in os.listdir(pdb_files)
                    if any(f.endswith(ext) for ext in extensions)
                ]
            else:
                raise ValueError(f"Invalid path: {pdb_files}")
        else:
            self.pdb_files = pdb_files
        
        if len(self.pdb_files) == 0:
            raise ValueError("No PDB files found!")
        
        print(f"Initialized dataset with {len(self.pdb_files)} PDB files")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.pdb_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and process a single protein structure.
        
        Args:
            idx: Index of the PDB file to load
        
        Returns:
            Dictionary containing:
            - 'backbone_coords': Tensor [L, 4, 3]
            - 'fragment_token_ids': Tensor [M]
            - 'torsion_bins': Tensor [K]
            - 'residue_types': List of residue names (for debugging)
            - 'sequence_length': Scalar tensor with sequence length L
        """
        pdb_path = self.pdb_files[idx]
        
        try:
            # Parse PDB file
            structure = self.parser.get_structure('protein', pdb_path)
            
            # Extract data from first model and first chain
            model = structure[0]
            chain = list(model.get_chains())[0]
            
            # Extract backbone coordinates and side-chain information
            backbone_coords, fragment_tokens, torsion_angles, residue_types = \
                self._extract_structure_data(chain)
            
            # Convert to tensors
            backbone_tensor = torch.tensor(backbone_coords, dtype=torch.float32)
            
            # Convert fragment tokens to indices
            fragment_indices = []
            for tokens in fragment_tokens:
                fragment_indices.extend(self.vocab.fragments_to_indices(tokens))
            fragment_tensor = torch.tensor(fragment_indices, dtype=torch.long)
            
            # Discretize torsion angles
            if len(torsion_angles) > 0:
                torsion_bins = discretize_angles(np.array(torsion_angles), num_bins=72)
                torsion_tensor = torch.tensor(torsion_bins, dtype=torch.long)
            else:
                torsion_tensor = torch.tensor([], dtype=torch.long)
            
            return {
                'backbone_coords': backbone_tensor,
                'fragment_token_ids': fragment_tensor,
                'torsion_bins': torsion_tensor,
                'residue_types': residue_types,
                'sequence_length': torch.tensor(len(backbone_coords), dtype=torch.long),
                'pdb_path': pdb_path
            }
        
        except Exception as e:
            print(f"Error loading {pdb_path}: {e}")
            # Return empty tensors on error (could also raise or skip)
            return {
                'backbone_coords': torch.zeros((0, 4, 3), dtype=torch.float32),
                'fragment_token_ids': torch.tensor([], dtype=torch.long),
                'torsion_bins': torch.tensor([], dtype=torch.long),
                'residue_types': [],
                'sequence_length': torch.tensor(0, dtype=torch.long),
                'pdb_path': pdb_path
            }
    
    def _extract_structure_data(
        self,
        chain
    ) -> Tuple[np.ndarray, List[List[str]], List[float], List[str]]:
        """
        Extract backbone coordinates, fragment tokens, and torsion angles from a chain.
        
        Args:
            chain: BioPython Chain object
        
        Returns:
            Tuple of:
            - backbone_coords: Array [L, 4, 3]
            - fragment_tokens: List of lists, each inner list contains fragment tokens for a residue
            - torsion_angles: List of torsion angles in radians
            - residue_types: List of residue names
        """
        backbone_coords = []
        fragment_tokens = []
        torsion_angles = []
        residue_types = []
        
        residues = list(chain.get_residues())
        
        for residue in residues:
            res_name = residue.get_resname()
            
            # Skip non-standard residues and water
            if res_name not in self.vocab._residue_to_fragments_map:
                continue
            
            # Extract backbone atoms (N, CA, C, O)
            backbone_atoms = []
            for atom_name in self.BACKBONE_ATOMS:
                try:
                    atom = residue[atom_name]
                    coords = atom.get_coord()
                    backbone_atoms.append(coords)
                except KeyError:
                    # Missing backbone atom - skip this residue
                    break
            
            if len(backbone_atoms) != 4:
                # Incomplete backbone - skip this residue
                continue
            
            backbone_coords.append(np.array(backbone_atoms))
            residue_types.append(res_name)
            
            # Get fragment tokens for this residue
            fragments = self.vocab.residue_to_fragments(res_name)
            fragment_tokens.append(fragments)
            
            # Extract torsion angles from side-chain
            # Note: This is a simplified version. In practice, you would need
            # to identify the specific torsion angles for each residue type.
            # For now, we extract chi angles (side-chain dihedrals) if available.
            residue_torsions = self._extract_residue_torsions(residue)
            torsion_angles.extend(residue_torsions)
        
        if len(backbone_coords) == 0:
            return np.zeros((0, 4, 3)), [], [], []
        
        backbone_coords = np.array(backbone_coords)
        return backbone_coords, fragment_tokens, torsion_angles, residue_types
    
    def _extract_residue_torsions(self, residue) -> List[float]:
        """
        Extract torsion angles (chi angles) from a residue's side-chain.
        
        This is a simplified implementation. In practice, you would need to:
        1. Identify the specific chi angles for each residue type
        2. Extract the appropriate atom coordinates
        3. Calculate dihedral angles
        
        Args:
            residue: BioPython Residue object
        
        Returns:
            List of torsion angles in radians
        """
        res_name = residue.get_resname()
        torsion_angles = []
        
        # Simplified: Extract chi1 angle for residues with side-chains
        # Chi1 is typically defined as: N-CA-CB-CG (or similar)
        try:
            # Get backbone atoms
            n_atom = residue['N']
            ca_atom = residue['CA']
            
            # Try to get side-chain atoms (this is residue-specific)
            if 'CB' in residue:
                cb_atom = residue['CB']
                
                # Try to find next atom in side-chain
                next_atom = None
                for atom_name in ['CG', 'CG1', 'OG', 'OG1', 'SG']:
                    if atom_name in residue:
                        next_atom = residue[atom_name]
                        break
                
                if next_atom is not None:
                    # Calculate chi1 angle
                    coords = np.array([
                        n_atom.get_coord(),
                        ca_atom.get_coord(),
                        cb_atom.get_coord(),
                        next_atom.get_coord()
                    ])
                    angle = calculate_dihedrals(coords, [(0, 1, 2, 3)])[0]
                    torsion_angles.append(angle)
        except (KeyError, IndexError):
            # Missing atoms - skip this torsion
            pass
        
        return torsion_angles


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching variable-length protein sequences.
    
    This function pads sequences to the same length for batch processing.
    
    Args:
        batch: List of samples from the dataset
    
    Returns:
        Batched dictionary with padded tensors
    """
    # Find maximum sequence length
    max_seq_len = max(item['sequence_length'].item() for item in batch)
    max_fragments = max(len(item['fragment_token_ids']) for item in batch)
    max_torsions = max(len(item['torsion_bins']) for item in batch)
    
    batch_size = len(batch)
    
    # Initialize batched tensors
    backbone_batch = torch.zeros(
        (batch_size, max_seq_len, 4, 3),
        dtype=torch.float32
    )
    fragment_batch = torch.zeros(
        (batch_size, max_fragments),
        dtype=torch.long
    )
    torsion_batch = torch.zeros(
        (batch_size, max_torsions),
        dtype=torch.long
    )
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill in the batch
    for i, item in enumerate(batch):
        seq_len = item['sequence_length'].item()
        frag_len = len(item['fragment_token_ids'])
        tors_len = len(item['torsion_bins'])
        
        backbone_batch[i, :seq_len] = item['backbone_coords']
        fragment_batch[i, :frag_len] = item['fragment_token_ids']
        torsion_batch[i, :tors_len] = item['torsion_bins']
        seq_lengths[i] = seq_len
    
    return {
        'backbone_coords': backbone_batch,
        'fragment_token_ids': fragment_batch,
        'torsion_bins': torsion_batch,
        'sequence_lengths': seq_lengths,
        'residue_types': [item['residue_types'] for item in batch],
        'pdb_paths': [item['pdb_path'] for item in batch]
    }


# Example usage
if __name__ == "__main__":
    # Example: Create dataset from a single PDB file
    # dataset = ProteinStructureDataset("path/to/protein.pdb")
    # sample = dataset[0]
    # print(f"Backbone shape: {sample['backbone_coords'].shape}")
    # print(f"Fragment tokens: {sample['fragment_token_ids']}")
    # print(f"Torsion bins: {sample['torsion_bins']}")
    pass
