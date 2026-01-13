"""
Geometric Utilities for Torsion Angle Calculation and Discretization

This module provides functions for:
1. Calculating dihedral (torsion) angles from atomic coordinates
2. Discretizing continuous angles into bins for neural network processing
3. Converting discrete bins back to continuous angles

Torsion angles are crucial for side-chain conformation prediction, as they determine
the spatial arrangement of atoms in flexible side-chain regions.

Author: Research Team
Date: 2024
"""

import numpy as np
from typing import Tuple, List, Optional
from Bio.PDB.vectors import Vector, calc_dihedral


def calculate_dihedrals(
    coords: np.ndarray,
    atom_indices: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    """
    Calculate dihedral (torsion) angles from atomic coordinates.
    
    A dihedral angle is defined by four atoms: A-B-C-D, where the angle
    measures the rotation around the B-C bond as viewed along that bond.
    
    Args:
        coords: Atomic coordinates array of shape [N, 3] where N is the number of atoms
        atom_indices: List of tuples (i, j, k, l) where each tuple represents
                     the indices of four atoms defining a dihedral angle (A-B-C-D)
    
    Returns:
        Array of dihedral angles in radians, shape [M] where M is len(atom_indices)
        Angles are in range [-pi, pi]
    
    Example:
        >>> coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        >>> angles = calculate_dihedrals(coords, [(0, 1, 2, 3)])
        >>> print(f"Angle: {np.degrees(angles[0]):.2f} degrees")
    """
    if len(atom_indices) == 0:
        return np.array([])
    
    angles = []
    for i, j, k, l in atom_indices:
        # Extract coordinates for the four atoms
        p1 = Vector(coords[i])
        p2 = Vector(coords[j])
        p3 = Vector(coords[k])
        p4 = Vector(coords[l])
        
        # Calculate dihedral angle using BioPython's calc_dihedral
        # This function returns angle in radians, range [-pi, pi]
        angle = calc_dihedral(p1, p2, p3, p4)
        angles.append(angle)
    
    return np.array(angles)


def calculate_dihedral_from_coords(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray
) -> float:
    """
    Calculate dihedral angle from four 3D coordinate points.
    
    This is a convenience function for calculating a single dihedral angle
    without needing to index into a coordinate array.
    
    Args:
        p1, p2, p3, p4: Four 3D coordinate arrays, each of shape [3]
                       representing atoms A, B, C, D in the dihedral A-B-C-D
    
    Returns:
        Dihedral angle in radians, range [-pi, pi]
    """
    v1 = Vector(p1)
    v2 = Vector(p2)
    v3 = Vector(p3)
    v4 = Vector(p4)
    return calc_dihedral(v1, v2, v3, v4)


def discretize_angle(angle_rad: float, num_bins: int = 72) -> int:
    """
    Discretize a continuous angle (in radians) into a bin index.
    
    The angle range [-pi, pi] is divided into num_bins equally-spaced bins.
    This discretization is necessary for neural network classification tasks.
    
    Args:
        angle_rad: Angle in radians, should be in range [-pi, pi]
        num_bins: Number of discrete bins (default: 72, giving 5-degree resolution)
    
    Returns:
        Bin index in range [0, num_bins-1]
    
    Note:
        Angles exactly equal to pi are mapped to bin 0 (wrapping around)
        to ensure consistent binning at the boundary.
    """
    # Normalize angle to [0, 2*pi) range
    angle_normalized = angle_rad + np.pi  # Shift to [0, 2*pi)
    
    # Handle edge case: angle exactly equal to pi
    if angle_normalized >= 2 * np.pi:
        angle_normalized = 0.0
    
    # Calculate bin index
    bin_width = 2 * np.pi / num_bins
    bin_idx = int(angle_normalized / bin_width)
    
    # Clamp to valid range [0, num_bins-1]
    bin_idx = max(0, min(bin_idx, num_bins - 1))
    
    return bin_idx


def discretize_angles(angles_rad: np.ndarray, num_bins: int = 72) -> np.ndarray:
    """
    Discretize an array of continuous angles into bin indices.
    
    Vectorized version of discretize_angle for efficiency.
    
    Args:
        angles_rad: Array of angles in radians, shape [N]
        num_bins: Number of discrete bins (default: 72)
    
    Returns:
        Array of bin indices, shape [N], dtype=int
    """
    # Normalize angles to [0, 2*pi) range
    angles_normalized = angles_rad + np.pi
    
    # Handle edge cases
    angles_normalized = np.where(
        angles_normalized >= 2 * np.pi,
        0.0,
        angles_normalized
    )
    
    # Calculate bin indices
    bin_width = 2 * np.pi / num_bins
    bin_indices = (angles_normalized / bin_width).astype(int)
    
    # Clamp to valid range
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    return bin_indices


def undiscretize_angle(bin_idx: int, num_bins: int = 72) -> float:
    """
    Convert a discrete bin index back to a continuous angle (in radians).
    
    The bin index is converted to the center angle of the corresponding bin.
    This is used during inference to reconstruct continuous angles from
    discrete predictions.
    
    Args:
        bin_idx: Bin index in range [0, num_bins-1]
        num_bins: Number of discrete bins (default: 72)
    
    Returns:
        Angle in radians, range [-pi, pi]
    
    Note:
        The returned angle represents the center of the bin, which is
        appropriate for most reconstruction tasks.
    """
    # Clamp bin index to valid range
    bin_idx = max(0, min(bin_idx, num_bins - 1))
    
    # Calculate bin width
    bin_width = 2 * np.pi / num_bins
    
    # Convert to angle in [0, 2*pi) range (center of bin)
    angle_normalized = (bin_idx + 0.5) * bin_width
    
    # Shift back to [-pi, pi] range
    angle_rad = angle_normalized - np.pi
    
    return angle_rad


def undiscretize_angles(bin_indices: np.ndarray, num_bins: int = 72) -> np.ndarray:
    """
    Convert an array of discrete bin indices back to continuous angles.
    
    Vectorized version of undiscretize_angle for efficiency.
    
    Args:
        bin_indices: Array of bin indices, shape [N], dtype=int
        num_bins: Number of discrete bins (default: 72)
    
    Returns:
        Array of angles in radians, shape [N]
    """
    # Clamp bin indices to valid range
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    # Calculate bin width
    bin_width = 2 * np.pi / num_bins
    
    # Convert to angles in [0, 2*pi) range (center of bin)
    angles_normalized = (bin_indices.astype(float) + 0.5) * bin_width
    
    # Shift back to [-pi, pi] range
    angles_rad = angles_normalized - np.pi
    
    return angles_rad


def get_torsion_angle_resolution(num_bins: int = 72) -> float:
    """
    Get the angular resolution (in degrees) for a given number of bins.
    
    Args:
        num_bins: Number of discrete bins
    
    Returns:
        Angular resolution in degrees
    """
    return 360.0 / num_bins


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("Torsion Angle Calculation and Discretization Test")
    print("="*60)
    
    # Test dihedral angle calculation
    print("\n1. Dihedral Angle Calculation:")
    coords = np.array([
        [0.0, 0.0, 0.0],  # Atom 0
        [1.0, 0.0, 0.0],  # Atom 1
        [1.0, 1.0, 0.0],  # Atom 2
        [0.0, 1.0, 0.0],  # Atom 3
    ])
    
    angle = calculate_dihedrals(coords, [(0, 1, 2, 3)])[0]
    print(f"   Calculated angle: {np.degrees(angle):.2f} degrees ({angle:.4f} radians)")
    
    # Test discretization
    print("\n2. Angle Discretization (72 bins = 5-degree resolution):")
    test_angles = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    for angle in test_angles:
        bin_idx = discretize_angle(angle, num_bins=72)
        angle_recovered = undiscretize_angle(bin_idx, num_bins=72)
        print(f"   Angle: {np.degrees(angle):7.2f}° -> Bin: {bin_idx:3d} -> Recovered: {np.degrees(angle_recovered):7.2f}°")
    
    # Test vectorized operations
    print("\n3. Vectorized Operations:")
    angles_array = np.linspace(-np.pi, np.pi, 10)
    bin_indices = discretize_angles(angles_array, num_bins=72)
    angles_recovered = undiscretize_angles(bin_indices, num_bins=72)
    
    print("   Original -> Bin -> Recovered:")
    for orig, bin_idx, rec in zip(angles_array, bin_indices, angles_recovered):
        print(f"   {np.degrees(orig):7.2f}° -> {bin_idx:3d} -> {np.degrees(rec):7.2f}°")
    
    print(f"\n4. Resolution: {get_torsion_angle_resolution(72):.2f} degrees per bin")
