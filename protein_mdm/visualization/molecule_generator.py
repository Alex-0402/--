import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

AA_SMILES = {
    "ALA": "C[C@H](N)C(=O)O",
    "ARG": "N=C(N)NCCC[C@H](N)C(=O)O",
    "ASN": "NC(=O)C[C@H](N)C(=O)O",
    "ASP": "O=C(O)C[C@H](N)C(=O)O",
    "CYS": "SC[C@H](N)C(=O)O",
    "GLN": "NC(=O)CC[C@H](N)C(=O)O",
    "GLU": "O=C(O)CC[C@H](N)C(=O)O",
    "GLY": "NCC(=O)O",
    "HIS": "C1=C(NC=N1)C[C@H](N)C(=O)O",
    "ILE": "CC[C@H](C)[C@H](N)C(=O)O",
    "LEU": "CC(C)C[C@H](N)C(=O)O",
    "LYS": "NCCCC[C@H](N)C(=O)O",
    "MET": "CSCC[C@H](N)C(=O)O",
    "PHE": "C1=CC=C(C=C1)C[C@H](N)C(=O)O",
    "PRO": "O=C(O)[C@@H]1CCCN1",
    "SER": "OC[C@H](N)C(=O)O",
    "THR": "C[C@@H](O)[C@H](N)C(=O)O",
    "TRP": "C1=CC=C2C(=C1)C(=CN2)C[C@H](N)C(=O)O",
    "TYR": "OC1=CC=C(C[C@H](N)C(=O)O)C=C1",
    "VAL": "CC(C)[C@H](N)C(=O)O"
}

def generate_aa_conformer(aa_name: str) -> Chem.Mol:
    smiles = AA_SMILES.get(aa_name.upper())
    if not smiles:
        raise ValueError(f"Unknown amino acid {aa_name}")
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    
    return mol

def get_pdb_string(mol: Chem.Mol) -> str:
    return Chem.MolToPDBBlock(mol)
