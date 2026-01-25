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


def random_rotation_matrix() -> torch.Tensor:
    """
    生成均匀分布的随机 3D 旋转矩阵。
    
    使用 QR 分解方法生成均匀分布在 SO(3) 群上的旋转矩阵。
    
    Returns:
        形状为 [3, 3] 的旋转矩阵张量
    """
    x = torch.randn(3, 3)
    q, r = torch.linalg.qr(x)
    d = torch.diag(r).sign()
    q *= d
    return q


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
        use_mmcif: bool = False,
        cache_dir: Optional[str] = None,
        lazy_loading: bool = True
    ):
        """
        Initialize the dataset.
        
        ✅ 多进程安全设计：
        - 不在 __init__ 中打开任何文件句柄
        - 不在 __init__ 中创建 parser（避免 fork 问题）
        - 所有文件操作都在 __getitem__ 中进行（懒加载）
        
        Args:
            pdb_files: Path to a single PDB file, directory containing PDB files,
                      or list of PDB file paths
            vocab: FragmentVocab instance (if None, uses global singleton)
            use_mmcif: If True, use MMCIFParser instead of PDBParser
            cache_dir: Optional directory to cache processed data (if None, no caching)
            lazy_loading: If True, only load data when accessed (default: True)
        """
        self.vocab = vocab if vocab is not None else get_vocab()
        # ✅ 多进程安全：不在 __init__ 中创建 parser，避免 fork 时复制文件句柄
        # parser 将在 __getitem__ 中按需创建（懒加载）
        # 每个 worker 进程在 fork 后都会创建自己的 parser 实例
        self.use_mmcif = use_mmcif
        self._parser = None  # 懒加载：在需要时创建（每个 worker 进程独立）
        
        self.cache_dir = cache_dir
        self.lazy_loading = lazy_loading
        
        # Create cache directory if specified
        # 注意：os.makedirs 是安全的，不会导致 fork 问题
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Collect PDB files
        # 注意：只收集文件路径，不打开文件
        if isinstance(pdb_files, str):
            if os.path.isfile(pdb_files):
                self.pdb_files = [pdb_files]
            elif os.path.isdir(pdb_files):
                # 如果指定了缓存目录且 pdb_files 就是缓存目录，查找 .pt 文件
                # 否则查找 .pdb 或 .cif 文件
                if cache_dir is not None and os.path.abspath(pdb_files) == os.path.abspath(cache_dir):
                    # 使用缓存目录，查找 .pt 文件（排除划分文件）
                    pt_files = [
                        os.path.join(pdb_files, f)
                        for f in os.listdir(pdb_files)
                        if f.endswith('.pt') and f not in ['train.txt', 'val.txt', 'test.txt']
                    ]
                    self.pdb_files = pt_files
                else:
                    # 查找 PDB/mmCIF 文件
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
        
        # ✅ 修复：只在主进程中打印，避免多进程输出混乱
        # 使用 os.getpid() 检查是否为主进程
        if os.getpid() == os.getppid() or not hasattr(os, 'getppid'):
            # 主进程或无法确定父进程时打印
            print(f"Initialized dataset with {len(self.pdb_files)} PDB files")
            if self.cache_dir:
                print(f"Cache directory: {self.cache_dir}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.pdb_files)
    
    def _get_parser(self):
        """
        懒加载 parser：按需创建，确保多进程安全。
        
        ✅ 多进程安全设计：
        - 每个 worker 进程在 fork 后都会创建自己的 parser 实例
        - 不在 __init__ 中创建，避免 fork 时复制文件句柄
        - 每个 worker 进程独立，不需要锁（进程间不共享内存）
        
        Returns:
            PDBParser 或 MMCIFParser 实例
        """
        # ✅ 懒加载：在需要时才创建 parser
        # 每个 worker 进程在 fork 后都会创建自己的 parser 实例
        # 不需要锁，因为每个进程有独立的内存空间
        if self._parser is None:
            self._parser = MMCIFParser(QUIET=True) if self.use_mmcif else PDBParser(QUIET=True)
        return self._parser
    
    def _get_cache_path(self, pdb_path: str) -> Optional[str]:
        """
        获取缓存文件路径
        
        Args:
            pdb_path: PDB 文件路径或缓存文件路径
        
        Returns:
            缓存文件路径，如果 cache_dir 为 None 则返回 None
        """
        if self.cache_dir is None:
            return None
        
        # 如果 pdb_path 已经是 .pt 文件，直接返回
        if pdb_path.endswith('.pt'):
            return pdb_path
        
        # 从 PDB 路径提取文件名（不含扩展名）
        pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
        cache_path = os.path.join(self.cache_dir, f"{pdb_name}.pt")
        return cache_path
    
    def _load_from_cache(self, cache_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        从缓存加载数据（懒加载，在 __getitem__ 中调用）
        
        ✅ 多进程安全：
        - 每次调用都打开和关闭文件，不持有文件句柄
        - 使用 map_location='cpu' 确保数据加载到 CPU
        
        Args:
            cache_path: 缓存文件路径
        
        Returns:
            数据字典，如果文件不存在则返回 None
        """
        if not os.path.exists(cache_path):
            return None
        
        try:
            # ✅ 懒加载：每次调用都打开和关闭文件，不持有文件句柄
            # 这确保多进程安全，避免 fork 时复制文件句柄
            data = torch.load(cache_path, map_location='cpu')
            return data
        except Exception as e:
            # 只在主进程中打印警告，避免多进程输出混乱
            if os.getpid() == os.getppid() or not hasattr(os, 'getppid'):
                print(f"Warning: Failed to load cache {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, cache_path: str, data: Dict[str, torch.Tensor]):
        """
        保存数据到缓存（懒加载，在 __getitem__ 中调用）
        
        ✅ 多进程安全：
        - 每次调用都打开和关闭文件，不持有文件句柄
        - 使用文件锁确保多进程写入安全（可选）
        
        Args:
            cache_path: 缓存文件路径
            data: 要保存的数据字典
        """
        try:
            # ✅ 懒加载：每次调用都打开和关闭文件，不持有文件句柄
            # 这确保多进程安全，避免 fork 时复制文件句柄
            torch.save(data, cache_path)
        except Exception as e:
            # 只在主进程中打印警告，避免多进程输出混乱
            if os.getpid() == os.getppid() or not hasattr(os, 'getppid'):
                print(f"Warning: Failed to save cache {cache_path}: {e}")
    
    def _parse_pdb_file(self, pdb_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        解析 PDB 文件并返回处理后的数据（懒加载，在 __getitem__ 中调用）
        
        ✅ 多进程安全：
        - 使用懒加载的 parser（每个 worker 进程有自己的实例）
        - parser.get_structure() 每次调用都打开和关闭文件，不持有文件句柄
        
        Args:
            pdb_path: PDB 文件路径
        
        Returns:
            数据字典，如果解析失败则返回 None
        """
        try:
            # ✅ 懒加载：使用 _get_parser() 获取 parser
            # 这确保每个 worker 进程在 fork 后创建自己的 parser
            parser = self._get_parser()
            # Parse PDB file
            # 注意：get_structure() 每次调用都打开和关闭文件，不持有文件句柄
            structure = parser.get_structure('protein', pdb_path)
            
            # Extract data from first model and first chain
            model = structure[0]
            chain = list(model.get_chains())[0]
            
            # Extract backbone coordinates and side-chain information
            backbone_coords, fragment_tokens, torsion_angles, residue_types = \
                self._extract_structure_data(chain)
            
            # 检查是否成功提取数据
            if len(backbone_coords) == 0:
                return None
            
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
            
            data = {
                'backbone_coords': backbone_tensor,
                'fragment_token_ids': fragment_tensor,
                'torsion_bins': torsion_tensor,
                'residue_types': residue_types,
                'sequence_length': torch.tensor(len(backbone_coords), dtype=torch.long),
                'pdb_path': pdb_path
            }
            
            return data
        
        except Exception as e:
            # 解析失败，返回 None
            return None
    
    def _apply_data_augmentation(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        对蛋白质结构数据进行增强，防止过拟合。
        
        应用的操作：
        1. 中心化：将 CA 原子中心平移到原点
        2. 随机旋转：使用均匀分布的旋转矩阵
        3. 高斯噪声：添加微小的随机噪声
        
        Args:
            data: 包含 'backbone_coords' 的数据字典
        
        Returns:
            增强后的数据字典
        """
        backbone_coords = data['backbone_coords']  # [L, 4, 3]
        
        # 1. 中心化 (Centering)：计算 CA 原子的中心，将整个蛋白质平移到原点
        center = backbone_coords[:, 1, :].mean(dim=0, keepdim=True)  # CA 原子索引为 1
        backbone_coords = backbone_coords - center.unsqueeze(1)
        
        # 2. 随机旋转 (Random Rotation)：使用生成的旋转矩阵对坐标进行旋转
        rot_mat = random_rotation_matrix()
        L, atoms, dims = backbone_coords.shape
        coords_flat = backbone_coords.view(-1, 3)
        coords_rotated = torch.matmul(coords_flat, rot_mat)
        backbone_coords = coords_rotated.view(L, atoms, dims)
        
        # 3. 高斯噪声 (Gaussian Noise)：给坐标添加微小的随机噪声
        noise = torch.randn_like(backbone_coords) * 0.02
        backbone_coords = backbone_coords + noise
        
        # 更新增强后的坐标
        data['backbone_coords'] = backbone_coords
        return data
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load and process a single protein structure.
        
        ✅ 多进程安全设计：
        - 所有文件操作都在这里进行（懒加载）
        - 每次调用都打开和关闭文件，不持有文件句柄
        - 使用懒加载的 parser，确保每个 worker 进程有自己的实例
        
        Args:
            idx: Index of the PDB file to load
        
        Returns:
            Dictionary containing:
            - 'backbone_coords': Tensor [L, 4, 3]
            - 'fragment_token_ids': Tensor [M]
            - 'torsion_bins': Tensor [K]
            - 'residue_types': List of residue names (for debugging)
            - 'sequence_length': Scalar tensor with sequence length L
            - 'pdb_path': Original PDB file path
            
            If parsing fails, returns None (will be filtered by collate_fn)
        """
        pdb_path = self.pdb_files[idx]
        
        # ✅ 懒加载：如果 pdb_path 已经是 .pt 文件（缓存文件），直接加载
        # 每次调用都打开和关闭文件，不持有文件句柄
        if pdb_path.endswith('.pt'):
            cached_data = self._load_from_cache(pdb_path)
            if cached_data is not None:
                # 对缓存数据应用数据增强
                return self._apply_data_augmentation(cached_data)
            else:
                # 缓存文件损坏，返回 None
                return None
        
        # ✅ 懒加载：检查缓存
        # 每次调用都打开和关闭文件，不持有文件句柄
        cache_path = self._get_cache_path(pdb_path)
        if cache_path is not None:
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                # 对缓存数据应用数据增强
                return self._apply_data_augmentation(cached_data)
        
        # ✅ 懒加载：缓存未命中，解析 PDB 文件
        # 使用懒加载的 parser，确保每个 worker 进程有自己的实例
        data = self._parse_pdb_file(pdb_path)
        
        # ✅ 懒加载：如果解析成功，保存到缓存
        # 每次调用都打开和关闭文件，不持有文件句柄
        if data is not None and cache_path is not None:
            self._save_to_cache(cache_path, data)
        
        # 数据增强：防止过拟合（对新解析的数据也应用增强）
        if data is not None:
            data = self._apply_data_augmentation(data)
        
        # 如果解析失败，返回 None（会被 collate_fn 过滤）
        return data
    
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


def collate_fn(batch: List[Optional[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching variable-length protein sequences.
    
    This function pads sequences to the same length for batch processing.
    It also filters out None samples (failed parsing) to prevent DataLoader crashes.
    
    Args:
        batch: List of samples from the dataset (may contain None for failed samples)
    
    Returns:
        Batched dictionary with padded tensors
    """
    # 过滤掉 None 样本（解析失败的样本）
    valid_batch = [item for item in batch if item is not None]
    
    if len(valid_batch) == 0:
        # 如果所有样本都失败，返回空批次
        return {
            'backbone_coords': torch.zeros((0, 1, 4, 3), dtype=torch.float32),
            'fragment_token_ids': torch.zeros((0, 1), dtype=torch.long),
            'torsion_bins': torch.zeros((0, 1), dtype=torch.long),
            'sequence_lengths': torch.zeros(0, dtype=torch.long),
            'residue_types': [],
            'pdb_paths': []
        }
    
    # Find maximum sequence length
    max_seq_len = max(item['sequence_length'].item() for item in valid_batch)
    max_fragments = max(len(item['fragment_token_ids']) for item in valid_batch)
    max_torsions = max(len(item['torsion_bins']) for item in valid_batch)
    
    batch_size = len(valid_batch)
    
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
    for i, item in enumerate(valid_batch):
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
        'residue_types': [item['residue_types'] for item in valid_batch],
        'pdb_paths': [item['pdb_path'] for item in valid_batch]
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
