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
        lazy_loading: bool = True,
        augment: bool = False,
        allowlist_txt: Optional[str] = None
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
            augment: If True, apply data augmentation (random rotation and noise).
                    Should be False for validation set to ensure stable metrics.
        """
        self.vocab = vocab if vocab is not None else get_vocab()
        # ✅ 多进程安全：不在 __init__ 中创建 parser，避免 fork 时复制文件句柄
        # parser 将在 __getitem__ 中按需创建（懒加载）
        # 每个 worker 进程在 fork 后都会创建自己的 parser 实例
        self.use_mmcif = use_mmcif
        self._parser = None  # 懒加载：在需要时创建（每个 worker 进程独立）
        
        self.cache_dir = cache_dir
        self.lazy_loading = lazy_loading
        self.augment = augment
        
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

        # 🔥 添加对高质量 PDB 列表的过滤
        if allowlist_txt is not None:
            if os.path.exists(allowlist_txt):
                with open(allowlist_txt, 'r') as f:
                    allowed_sets = {line.strip().lower() for line in f if line.strip()}
                
                filtered_files = []
                for p in self.pdb_files:
                    # 提取文件名（不含扩展名）转小写作为标识符
                    base_name = os.path.basename(p).split('.')[0].lower()
                    if base_name in allowed_sets:
                        filtered_files.append(p)
                
                # 仅当主进程时打印过滤信息
                if os.getpid() == os.getppid() or not hasattr(os, 'getppid'):
                    print(f"Applied high_quality_pdbs.txt filter: kept {len(filtered_files)} / {len(self.pdb_files)} files.")
                self.pdb_files = filtered_files
            else:
                if os.getpid() == os.getppid() or not hasattr(os, 'getppid'):
                    print(f"⚠️ 警告: allowlist_txt '{allowlist_txt}' 不存在，跳过过滤。")
        
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

    def _upgrade_cached_data(self, data: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], bool]:
        """
        为旧缓存补齐新字段，避免新逻辑依赖字段缺失。

        兼容字段：
        - fragment_levels
        - fragment_residue_idx

        Returns:
            (data, upgraded)
        """
        upgraded = False

        frag_tokens = data.get('fragment_token_ids', torch.tensor([], dtype=torch.long))
        frag_len = int(frag_tokens.shape[0]) if isinstance(frag_tokens, torch.Tensor) else 0
        residue_types = data.get('residue_types', [])

        def _fit_length(values: List[int], target_len: int, pad_value: int = 0) -> List[int]:
            if len(values) < target_len:
                values = values + [pad_value] * (target_len - len(values))
            elif len(values) > target_len:
                values = values[:target_len]
            return values

        # 1) 补齐 fragment_levels
        need_levels = ('fragment_levels' not in data)
        if not need_levels:
            levels_obj = data['fragment_levels']
            need_levels = (not isinstance(levels_obj, torch.Tensor)) or (int(levels_obj.shape[0]) != frag_len)

        if need_levels:
            inferred_levels: List[int] = []
            for res_name in residue_types:
                try:
                    inferred_levels.extend(self.vocab.residue_to_fragment_levels(res_name))
                except Exception:
                    # 未知残基回退：按该残基片段数补 0
                    try:
                        n_frag = len(self.vocab.residue_to_fragments(res_name))
                    except Exception:
                        n_frag = 0
                    inferred_levels.extend([0] * n_frag)

            inferred_levels = _fit_length(inferred_levels, frag_len, pad_value=0)
            data['fragment_levels'] = torch.tensor(inferred_levels, dtype=torch.long)
            upgraded = True

        # 2) 补齐 fragment_residue_idx
        need_res_idx = ('fragment_residue_idx' not in data)
        if not need_res_idx:
            ridx_obj = data['fragment_residue_idx']
            need_res_idx = (not isinstance(ridx_obj, torch.Tensor)) or (int(ridx_obj.shape[0]) != frag_len)

        if need_res_idx:
            inferred_ridx: List[int] = []
            for i_res, res_name in enumerate(residue_types):
                try:
                    n_frag = len(self.vocab.residue_to_fragments(res_name))
                except Exception:
                    n_frag = 0
                inferred_ridx.extend([i_res] * n_frag)

            inferred_ridx = _fit_length(inferred_ridx, frag_len, pad_value=0)
            data['fragment_residue_idx'] = torch.tensor(inferred_ridx, dtype=torch.long)
            upgraded = True

        # 3) 补齐 fragment_parents
        need_parents = ('fragment_parents' not in data)
        if not need_parents:
            parents_obj = data['fragment_parents']
            need_parents = (not isinstance(parents_obj, torch.Tensor)) or (int(parents_obj.shape[0]) != frag_len)

        if need_parents:
            inferred_parents: List[int] = []
            for res_name in residue_types:
                try:
                    inferred_parents.extend(self.vocab.residue_to_fragment_parents(res_name))
                except Exception:
                    try:
                        n_frag = len(self.vocab.residue_to_fragments(res_name))
                    except Exception:
                        n_frag = 0
                    inferred_parents.extend([-1] * n_frag)

            inferred_parents = _fit_length(inferred_parents, frag_len, pad_value=-1)
            data['fragment_parents'] = torch.tensor(inferred_parents, dtype=torch.long)
            upgraded = True

        return data, upgraded
    
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
            backbone_coords, fragment_tokens, torsion_angles, residue_types, torsion_angles_by_residue = \
                self._extract_structure_data(chain)
            
            # 检查是否成功提取数据
            if len(backbone_coords) == 0:
                return None
            
            # Convert to tensors
            backbone_tensor = torch.tensor(backbone_coords, dtype=torch.float32)
            
            # Convert fragment tokens to indices and extract hierarchical levels/parents
            fragment_indices = []
            fragment_levels = []
            fragment_parents = []
            fragment_residue_idx = []
            for i_res, (res_name, tokens) in enumerate(zip(residue_types, fragment_tokens)):
                indices = self.vocab.fragments_to_indices(tokens)
                fragment_indices.extend(indices)
                fragment_residue_idx.extend([i_res] * len(indices))
                try:
                    levels = self.vocab.residue_to_fragment_levels(res_name)
                    fragment_levels.extend(levels)
                    parents = self.vocab.residue_to_fragment_parents(res_name)
                    fragment_parents.extend(parents)
                except ValueError:
                    # Fallback for unknown residue types just in case
                    fragment_levels.extend([0] * len(tokens))
                    fragment_parents.extend([-1] * len(tokens))
            
            fragment_tensor = torch.tensor(fragment_indices, dtype=torch.long)
            fragment_levels_tensor = torch.tensor(fragment_levels, dtype=torch.long)
            fragment_parents_tensor = torch.tensor(fragment_parents, dtype=torch.long)
            fragment_residue_idx_tensor = torch.tensor(fragment_residue_idx, dtype=torch.long)
            
            # 构建“与片段序列对齐”的 torsion 监督：长度与 fragment_token_ids 一致
            # 策略：每个残基的第 i 个片段位置对应监督其第 i 个 chi 角（若存在），支持最多chi1~chi4的完整侧链构象
            aligned_torsion_raw = []
            aligned_torsion_valid = []
            for frags, chis in zip(fragment_tokens, torsion_angles_by_residue):
                frag_n = len(frags)
                if frag_n == 0:
                    continue
                
                # 遍历分配每个片段的二面角（多片段长侧链自然展开）
                for i in range(frag_n):
                    if chis is not None and i < len(chis) and chis[i] is not None:
                        aligned_torsion_raw.append(float(chis[i]))
                        aligned_torsion_valid.append(True)
                    else:
                        aligned_torsion_raw.append(0.0)
                        aligned_torsion_valid.append(False)
                        
            if len(aligned_torsion_raw) > 0:
                torsion_bins = discretize_angles(np.array(aligned_torsion_raw), num_bins=72)
                torsion_tensor = torch.tensor(torsion_bins, dtype=torch.long)
                torsion_raw_tensor = torch.tensor(aligned_torsion_raw, dtype=torch.float32)
                torsion_valid_tensor = torch.tensor(aligned_torsion_valid, dtype=torch.bool)
            else:
                torsion_tensor = torch.tensor([], dtype=torch.long)
                torsion_raw_tensor = torch.tensor([], dtype=torch.float32)
                torsion_valid_tensor = torch.tensor([], dtype=torch.bool)
            
            data = {
                'backbone_coords': backbone_tensor,
                'fragment_token_ids': fragment_tensor,
                'fragment_levels': fragment_levels_tensor,
                'fragment_parents': fragment_parents_tensor,
                'fragment_residue_idx': fragment_residue_idx_tensor,
                'torsion_bins': torsion_tensor,
                'torsion_raw': torsion_raw_tensor,  # 原始浮点角度值（弧度）
                'torsion_valid_mask': torsion_valid_tensor,
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
        1. 中心化：将 CA 原子中心平移到原点（始终执行）
        2. 随机旋转：使用均匀分布的旋转矩阵（仅在 augment=True 时执行）
        3. 高斯噪声：添加微小的随机噪声（仅在 augment=True 时执行）
        
        Args:
            data: 包含 'backbone_coords' 的数据字典
        
        Returns:
            增强后的数据字典
        """
        backbone_coords = data['backbone_coords']  # [L, 4, 3]
        
        # 1. 中心化 (Centering)：计算 CA 原子的中心，将整个蛋白质平移到原点
        # 注意：中心化应该对所有数据都执行（无论是否增强），以确保数据一致性
        center = backbone_coords[:, 1, :].mean(dim=0, keepdim=True)  # CA 原子索引为 1
        backbone_coords = backbone_coords - center.unsqueeze(1)
        
        # 2. 随机旋转和高斯噪声：仅在 augment=True 时执行
        if self.augment:
            # 2.1 随机旋转 (Random Rotation)：使用生成的旋转矩阵对坐标进行旋转
            rot_mat = random_rotation_matrix()
            L, atoms, dims = backbone_coords.shape
            coords_flat = backbone_coords.view(-1, 3)
            coords_rotated = torch.matmul(coords_flat, rot_mat)
            backbone_coords = coords_rotated.view(L, atoms, dims)
            
            # 2.2 高斯噪声 (Gaussian Noise)：给坐标添加微小的随机噪声
            # 减小噪声强度以缓解过拟合，避免破坏原子间的长程依赖
            noise = torch.randn_like(backbone_coords) * 0.01
            backbone_coords = backbone_coords + noise
        
        # 更新处理后的坐标
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
                cached_data, upgraded = self._upgrade_cached_data(cached_data)
                if upgraded:
                    self._save_to_cache(pdb_path, cached_data)
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
                cached_data, upgraded = self._upgrade_cached_data(cached_data)
                if upgraded:
                    self._save_to_cache(cache_path, cached_data)
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
    ) -> Tuple[np.ndarray, List[List[str]], List[float], List[str], List[Optional[float]]]:
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
        torsion_angles_by_residue: List[List[Optional[float]]] = []
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
            
            # 解封完整的 CHI1-CHI4 多级复杂侧链计算
            residue_torsions = self._extract_residue_torsions(residue)
            torsion_angles.extend([t for t in residue_torsions if t is not None])
            torsion_angles_by_residue.append(residue_torsions)
            
        if len(backbone_coords) == 0:
            return np.zeros((0, 4, 3)), [], [], [], []
        
        backbone_coords = np.array(backbone_coords)
        return backbone_coords, fragment_tokens, torsion_angles, residue_types, torsion_angles_by_residue
    
    def _extract_residue_torsions(self, residue) -> List[Optional[float]]:
        """
        Extract complete torsion angles (chi1 - chi4) from a residue's side-chain.
        
        Args:
            residue: BioPython Residue object
        
        Returns:
            List of torsion angles in radians
        """
        res_name = residue.get_resname()
        torsion_angles = []
        
        # 完整氨基酸侧链二面角定义字典 (\chi 1 - 4)
        CHI_ATOMS = {
            'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
            'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
            'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'], ['CB', 'CG', 'SD', 'CE']],
            'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']],
            'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']],
            'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
            'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']], # 有时定义不同，取一种主流的
            'PRO': [['N', 'CA', 'CB', 'CG']],
            'THR': [['N', 'CA', 'CB', 'OG1']],
            'VAL': [['N', 'CA', 'CB', 'CG1']],
            'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
            'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
            'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
            'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
            'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
            'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
            'CYS': [['N', 'CA', 'CB', 'SG']],
            'SER': [['N', 'CA', 'CB', 'OG']]
        }

        # 如果没有 \chi 角的残基 (ALA, GLY)，直接返回空序列
        if res_name not in CHI_ATOMS:
            return []

        # 尝试依级计算每个 \chi 角
        for atom_names in CHI_ATOMS[res_name]:
            try:
                coords = np.array([residue[an].get_coord() for an in atom_names])
                angle = calculate_dihedrals(coords, [(0, 1, 2, 3)])[0]
                torsion_angles.append(angle)
            except (KeyError, IndexError):
                # 缺失原子导致该层级及以后无法计算，置空填充占位
                torsion_angles.append(None)
                
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
            'fragment_levels': torch.zeros((0, 1), dtype=torch.long),
            'fragment_residue_idx': torch.zeros((0, 1), dtype=torch.long),
            'torsion_bins': torch.zeros((0, 1), dtype=torch.long),
            'torsion_raw': torch.zeros((0, 1), dtype=torch.float32),
            'torsion_valid_mask': torch.zeros((0, 1), dtype=torch.bool),
            'sequence_lengths': torch.zeros(0, dtype=torch.long),
            'fragment_lengths': torch.zeros(0, dtype=torch.long),
            'torsion_lengths': torch.zeros(0, dtype=torch.long),
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
    fragment_levels_batch = torch.zeros(
        (batch_size, max_fragments),
        dtype=torch.long
    )
    fragment_parents_batch = torch.zeros(
        (batch_size, max_fragments),
        dtype=torch.long
    )
    fragment_parents_batch.fill_(-1)  # -1 for backbone root default
    fragment_residue_idx_batch = torch.zeros(
        (batch_size, max_fragments),
        dtype=torch.long
    )
    torsion_batch = torch.zeros(
        (batch_size, max_torsions),
        dtype=torch.long
    )
    torsion_raw_batch = torch.zeros(
        (batch_size, max_torsions),
        dtype=torch.float32
    )
    torsion_valid_batch = torch.zeros(
        (batch_size, max_torsions),
        dtype=torch.bool
    )
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)
    fragment_lengths = torch.zeros(batch_size, dtype=torch.long)
    torsion_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill in the batch
    for i, item in enumerate(valid_batch):
        seq_len = item['sequence_length'].item()
        frag_len = len(item['fragment_token_ids'])
        tors_len = len(item['torsion_bins'])
        
        backbone_batch[i, :seq_len] = item['backbone_coords']
        fragment_batch[i, :frag_len] = item['fragment_token_ids']
        if 'fragment_levels' in item:
            fragment_levels_batch[i, :frag_len] = item['fragment_levels']
        if 'fragment_parents' in item:
            fragment_parents_batch[i, :frag_len] = item['fragment_parents']
        if 'fragment_residue_idx' in item:
            fragment_residue_idx_batch[i, :frag_len] = item['fragment_residue_idx']
        torsion_batch[i, :tors_len] = item['torsion_bins']
        # torsion_raw 使用 0 填充（后续会被 mask 掉）
        if tors_len > 0 and 'torsion_raw' in item:
            torsion_raw_batch[i, :tors_len] = item['torsion_raw']
        else:
            # 如果缺少 torsion_raw 字段，用 0 填充
            torsion_raw_batch[i, :tors_len] = 0

        # 优先使用样本提供的有效掩码（新格式）；旧缓存回退到“前 tors_len 全有效”
        if tors_len > 0 and 'torsion_valid_mask' in item and item['torsion_valid_mask'] is not None:
            valid_mask = item['torsion_valid_mask']
            vlen = min(tors_len, int(valid_mask.shape[0]))
            torsion_valid_batch[i, :vlen] = valid_mask[:vlen].bool()
        else:
            torsion_valid_batch[i, :tors_len] = True
        seq_lengths[i] = seq_len
        fragment_lengths[i] = frag_len
        torsion_lengths[i] = tors_len
    
    return {
        'backbone_coords': backbone_batch,
        'fragment_token_ids': fragment_batch,
        'fragment_levels': fragment_levels_batch,
        'fragment_parents': fragment_parents_batch,
        'fragment_residue_idx': fragment_residue_idx_batch,
        'torsion_bins': torsion_batch,
        'torsion_raw': torsion_raw_batch,  # 原始浮点角度值（弧度）
        'torsion_valid_mask': torsion_valid_batch,
        'sequence_lengths': seq_lengths,
        'fragment_lengths': fragment_lengths,
        'torsion_lengths': torsion_lengths,
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
