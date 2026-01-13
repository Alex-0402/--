"""
Protein Utilities

Helper functions for working with protein structures using BioPython.

This module provides convenient wrappers and utilities for common protein
structure operations.

Author: Research Team
Date: 2024
"""

from typing import List, Tuple, Optional
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

warnings.simplefilter('ignore', PDBConstructionWarning)


def load_pdb_structure(pdb_path: str, use_mmcif: bool = False):
    """
    Load a protein structure from a PDB or mmCIF file.
    
    Args:
        pdb_path: Path to PDB or mmCIF file
        use_mmcif: If True, use MMCIFParser instead of PDBParser
    
    Returns:
        BioPython Structure object
    """
    parser = MMCIFParser(QUIET=True) if use_mmcif else PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    return structure


def get_backbone_atoms(residue) -> Optional[np.ndarray]:
    """
    Extract backbone atom coordinates (N, CA, C, O) from a residue.
    
    Args:
        residue: BioPython Residue object
    
    Returns:
        Array of shape [4, 3] containing coordinates of N, CA, C, O atoms,
        or None if any atom is missing
    """
    backbone_atoms = ['N', 'CA', 'C', 'O']
    coords = []
    
    for atom_name in backbone_atoms:
        try:
            atom = residue[atom_name]
            coords.append(atom.get_coord())
        except KeyError:
            return None
    
    return np.array(coords)


def get_sidechain_atoms(residue) -> List[Tuple[str, np.ndarray]]:
    """
    Extract side-chain atom coordinates from a residue.
    
    Args:
        residue: BioPython Residue object
    
    Returns:
        List of tuples (atom_name, coordinates) for all side-chain atoms
    """
    backbone_atoms = {'N', 'CA', 'C', 'O'}
    sidechain_atoms = []
    
    for atom in residue:
        if atom.get_name() not in backbone_atoms:
            sidechain_atoms.append((atom.get_name(), atom.get_coord()))
    
    return sidechain_atoms


def get_residue_sequence(chain) -> List[str]:
    """
    Extract the amino acid sequence from a chain.
    
    Args:
        chain: BioPython Chain object
    
    Returns:
        List of three-letter residue codes
    """
    sequence = []
    for residue in chain:
        res_name = residue.get_resname()
        sequence.append(res_name)
    return sequence


def calculate_center_of_mass(coords: np.ndarray) -> np.ndarray:
    """
    Calculate the center of mass of a set of coordinates.
    
    Args:
        coords: Array of shape [N, 3] containing N points in 3D space
    
    Returns:
        Center of mass as array of shape [3]
    """
    return np.mean(coords, axis=0)


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Calculate Root Mean Square Deviation (RMSD) between two sets of coordinates.
    
    Args:
        coords1: Array of shape [N, 3]
        coords2: Array of shape [N, 3]
    
    Returns:
        RMSD value
    """
    if coords1.shape != coords2.shape:
        raise ValueError("Coordinate arrays must have the same shape")
    
    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return rmsd


def align_structures(
    coords1: np.ndarray,
    coords2: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Align two structures using Kabsch algorithm (simplified version).
    
    This function centers both structures and rotates coords2 to minimize RMSD
    with coords1.
    
    Args:
        coords1: Reference coordinates [N, 3]
        coords2: Coordinates to align [N, 3]
    
    Returns:
        Tuple of (aligned_coords2, rmsd_after_alignment)
    """
    # Center both structures
    center1 = calculate_center_of_mass(coords1)
    center2 = calculate_center_of_mass(coords2)
    
    coords1_centered = coords1 - center1
    coords2_centered = coords2 - center2
    
    # Calculate rotation matrix using SVD (Kabsch algorithm)
    H = coords2_centered.T @ coords1_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation and translation
    coords2_aligned = (coords2_centered @ R.T) + center1
    
    # Calculate RMSD
    rmsd = calculate_rmsd(coords1, coords2_aligned)
    
    return coords2_aligned, rmsd


# Example usage
if __name__ == "__main__":
    print("Protein utilities module loaded successfully.")
    print("Available functions:")
    print("  - load_pdb_structure()")
    print("  - get_backbone_atoms()")
    print("  - get_sidechain_atoms()")
    print("  - get_residue_sequence()")
    print("  - calculate_center_of_mass()")
    print("  - calculate_rmsd()")
    print("  - align_structures()")
