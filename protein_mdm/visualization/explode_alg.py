import numpy as np
from rdkit import Chem

def get_atom_to_part_map(num_atoms, partitions):
    """建立 原子 -> block ID 的反向映射字典。如果不在任意 partition 中，返回 -1 (代表主链等)"""
    mapping = {}
    for part_idx, part in enumerate(partitions):
        for atom_idx in part["atoms"]:
            mapping[atom_idx] = part_idx
    
    # 未被映射的默认为 -1 (保留在原始区域，比如主链骨架)
    for i in range(num_atoms):
        if i not in mapping:
            mapping[i] = -1
    return mapping

def apply_explosion(mol: Chem.Mol, partitions: list, explode_factor: float = 1.5) -> Chem.Mol:
    """
    修改原子的3D坐标以拉开距离，并物理上切断不同 Fragment (以及主链) 之间的化学键。
    这样 py3Dmol 在显示时才不会画出“长长的牵红线”。
    """
    if not partitions:
        return mol

    # 1. 物理断开不同区域的跨界化学键
    emw = Chem.EditableMol(mol)
    atom_mapping = get_atom_to_part_map(mol.GetNumAtoms(), partitions)
    
    bonds_to_remove = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        # 如果键的两端属于不同的“组” (譬如从主链-1 到 fragment 0，或 fragment 0 到 1)
        if atom_mapping[a1] != atom_mapping[a2]:
            bonds_to_remove.append((a1, a2))
            
    # 执行删键
    for a1, a2 in bonds_to_remove:
        emw.RemoveBond(a1, a2)
        
    exploded_mol = emw.GetMol()
    
    # 获取可编辑分子的新构象坐标引用
    conf = exploded_mol.GetConformer()
    ref_positions = mol.GetConformer().GetPositions() # 用原分子的坐标计算
    
    # 取整体几何中心作为发散原点
    center = np.mean(ref_positions, axis=0)

    # 2. 修改剩余原子的3D坐标 (拉开碎片)
    for part_idx, part in enumerate(partitions):
        if not part["atoms"]: continue
        
        part_positions = [ref_positions[i] for i in part["atoms"]]
        part_center = np.mean(part_positions, axis=0)
        
        vector = part_center - center
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector_normalized = vector / norm
        else:
            vector_normalized = np.array([0.0, 1.0, 0.0])
            
        trans_distance = explode_factor * (part_idx * 0.7 + 1.0)
        translation = vector_normalized * trans_distance
        
        # 将拉开后的位移写回新产生的 exploded_mol 的 conformer 中
        for atom_idx in part["atoms"]:
            pos = conf.GetAtomPosition(atom_idx)
            new_pos = pos + translation
            conf.SetAtomPosition(atom_idx, new_pos)
            
    # 让主链微调(可选), 这里保持主链不动，仅爆开侧链
            
    return exploded_mol
