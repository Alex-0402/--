import sys
import os
import copy
from rdkit import Chem

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.vocabulary import FragmentVocab

vocab = FragmentVocab()

def get_backbone_indices(mol: Chem.Mol):
    patt = Chem.MolFromSmarts("[NX3,NX4+;!$(NC=N)]-[CX4H](-*)-[CX3](=[OX1])-[OX2]")
    matches = mol.GetSubstructMatches(patt)
    
    if not matches and mol.GetNumAtoms() > 0:
        patt2 = Chem.MolFromSmarts("[N]-[C]-[C](=[O])-[O]")
        matches = mol.GetSubstructMatches(patt2)
        
    backbone_idx = []
    if matches:
        backbone_idx = list(matches[0]) 
    
    for atom_idx in copy.copy(backbone_idx):
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'H' and neighbor.GetIdx() not in backbone_idx:
                backbone_idx.append(neighbor.GetIdx())
                
    return backbone_idx

def partition_fragments(mol: Chem.Mol, aa_name: str):
    fragments_expected = vocab.residue_to_fragments(aa_name.upper())
    
    if not fragments_expected: 
        return []

    bb_indices = get_backbone_indices(mol)
    sidechain_atoms = [i for i in range(mol.GetNumAtoms()) if i not in bb_indices]
    
    if len(fragments_expected) == 1:
        return [{"token": fragments_expected[0], "atoms": sidechain_atoms, "parent_hook": bb_indices[1]}]
        
    partitions = []
    chunk_size = len(sidechain_atoms) // len(fragments_expected)
    if chunk_size == 0: chunk_size = 1
    
    for i, token in enumerate(fragments_expected):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < len(fragments_expected) - 1 else len(sidechain_atoms)
        partitions.append({
            "token": token,
            "atoms": sidechain_atoms[start_idx:end_idx],
            "parent_hook": bb_indices[1] if i == 0 else sidechain_atoms[start_idx-1]
        })
        
    return partitions
