"""
Protein Utilities

Helper functions for working with protein structures using BioPython.

This module provides convenient wrappers and utilities for common protein
structure operations.

Author: Research Team
Date: 2024
"""

from typing import List, Tuple, Optional, Dict
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


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Safely normalize a vector."""
    n = np.linalg.norm(v)
    if n < eps:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def split_fragments_by_residue(
    predicted_fragments: List[str],
    residue_types: List[str],
    residue_to_fragments_map: Dict[str, List[str]]
) -> List[List[str]]:
    """
    Split flattened predicted fragment list into residue-wise lists.

    We use the expected fragment count of each residue type as a stable splitter.
    """
    per_residue: List[List[str]] = []
    cursor = 0
    for res_name in residue_types:
        expected = len(residue_to_fragments_map.get(res_name, []))
        if expected <= 0:
            per_residue.append([])
            continue
        frag_slice = predicted_fragments[cursor: cursor + expected]
        if len(frag_slice) < expected:
            frag_slice = frag_slice + ["METHYLENE"] * (expected - len(frag_slice))
        per_residue.append(frag_slice)
        cursor += expected
    return per_residue


def build_sidechain_pseudo_atoms(
    backbone_coords: np.ndarray,
    residue_types: List[str],
    predicted_fragments: List[str],
    torsion_angles: np.ndarray,
    residue_to_fragments_map: Dict[str, List[str]]
) -> List[Dict[str, object]]:
    """
    Build pseudo side-chain atoms from predicted fragments/torsions.

    NOTE: This is an approximate reconstruction for evaluation and visualization.
    """
    fragment_lengths = {
        "METHYL": 1.53,
        "METHYLENE": 1.53,
        "HYDROXYL": 1.43,
        "PHENYL": 1.39,
        "AMINE": 1.47,
        "CARBOXYL": 1.25,
        "AMIDE": 1.34,
        "GUANIDINE": 1.33,
        "IMIDAZOLE": 1.37,
        "INDOLE": 1.38,
        "THIOL": 1.81,
        "BRANCH_CH": 1.53,
    }

    residue_fragments = split_fragments_by_residue(
        predicted_fragments=predicted_fragments,
        residue_types=residue_types,
        residue_to_fragments_map=residue_to_fragments_map,
    )

    pseudo_atoms: List[Dict[str, object]] = []
    torsion_cursor = 0

    for i, res_name in enumerate(residue_types):
        if i >= backbone_coords.shape[0]:
            break

        n_atom = backbone_coords[i, 0]
        ca_atom = backbone_coords[i, 1]
        c_atom = backbone_coords[i, 2]

        # Local orthonormal frame at CA
        e1 = _normalize(c_atom - ca_atom)
        n_vec = _normalize(n_atom - ca_atom)
        e3 = _normalize(np.cross(e1, n_vec))
        if np.linalg.norm(e3) < 1e-6:
            e3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        e2 = _normalize(np.cross(e3, e1))

        prev = ca_atom.astype(np.float32)
        frags = residue_fragments[i]

        for j, frag in enumerate(frags):
            bond_len = fragment_lengths.get(frag, 1.50)

            # Use predicted torsion when available, otherwise deterministic fallback
            if torsion_cursor < len(torsion_angles):
                phi = float(torsion_angles[torsion_cursor])
            else:
                phi = 2.0 * np.pi * ((j + 1) % 6) / 6.0
            torsion_cursor += 1

            theta = np.deg2rad(70.0)
            direction = (
                np.cos(theta) * e1
                + np.sin(theta) * (np.cos(phi) * e2 + np.sin(phi) * e3)
            )
            direction = _normalize(direction)
            pos = prev + bond_len * direction

            pseudo_atoms.append({
                "res_idx": i + 1,
                "res_name": res_name,
                "atom_name": f"SC{j + 1}"[:4],
                "coord": pos.astype(np.float32),
            })
            prev = pos

    return pseudo_atoms


def write_backbone_sidechain_pdb(
    backbone_coords: np.ndarray,
    residue_types: List[str],
    pseudo_atoms: List[Dict[str, object]],
    output_path: str,
):
    """Write backbone + reconstructed pseudo side-chain atoms to PDB."""
    lines: List[str] = []
    serial = 1
    atom_order = ["N", "CA", "C", "O"]

    for i, res_name in enumerate(residue_types):
        if i >= backbone_coords.shape[0]:
            break
        for a_idx, atom_name in enumerate(atom_order):
            x, y, z = backbone_coords[i, a_idx]
            lines.append(
                f"ATOM  {serial:5d} {atom_name:<4}{res_name:>4} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {atom_name[0]:>2}"
            )
            serial += 1

    for atom in pseudo_atoms:
        x, y, z = atom["coord"]
        atom_name = str(atom["atom_name"])
        res_name = str(atom["res_name"])
        res_idx = int(atom["res_idx"])
        element = atom_name[1] if len(atom_name) > 1 else "C"
        lines.append(
            f"ATOM  {serial:5d} {atom_name:<4}{res_name:>4} A{res_idx:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element:>2}"
        )
        serial += 1

    lines.append("END")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def compute_clash_score(coords: np.ndarray, threshold: float = 2.0) -> float:
    """Compute fraction of atom pairs with distance < threshold (Å)."""
    if coords.shape[0] < 2:
        return 0.0
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
    iu = np.triu_indices(coords.shape[0], k=1)
    pair_dist = dist[iu]
    if pair_dist.size == 0:
        return 0.0
    return float(np.mean(pair_dist < threshold))


def extract_sidechain_centroids_from_pdb(
    pdb_path: str,
    residue_types: List[str],
    use_mmcif: bool = False
) -> np.ndarray:
    """Extract side-chain centroid for each valid residue in the first chain."""
    structure = load_pdb_structure(pdb_path, use_mmcif=use_mmcif)
    model = structure[0]
    chain = list(model.get_chains())[0]

    centroids: List[np.ndarray] = []
    backbone_atoms = {'N', 'CA', 'C', 'O'}

    # Keep same ordering logic as dataset: standard residue + complete backbone
    valid_residue_names: List[str] = []
    for residue in chain:
        res_name = residue.get_resname()
        if len(valid_residue_names) >= len(residue_types):
            break
        if res_name not in residue_types:
            continue
        has_backbone = all(atom in residue for atom in ['N', 'CA', 'C', 'O'])
        if not has_backbone:
            continue
        valid_residue_names.append(res_name)

        sidechain_coords = [atom.get_coord() for atom in residue if atom.get_name() not in backbone_atoms]
        if len(sidechain_coords) == 0 and 'CA' in residue:
            sidechain_coords = [residue['CA'].get_coord()]
        if len(sidechain_coords) == 0:
            continue
        centroids.append(np.mean(np.asarray(sidechain_coords), axis=0))

    if len(centroids) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(centroids, dtype=np.float32)


def compute_centroid_rmsd(pred_centroids: np.ndarray, ref_centroids: np.ndarray) -> float:
    """Compute aligned RMSD between predicted and reference side-chain centroids."""
    n = min(len(pred_centroids), len(ref_centroids))
    if n == 0:
        return float('nan')
    pred = pred_centroids[:n]
    ref = ref_centroids[:n]
    pred_aligned, rmsd = align_structures(ref, pred)
    _ = pred_aligned
    return float(rmsd)


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
