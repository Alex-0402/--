"""
基于几何图神经网络的蛋白质骨架编码器

本模块实现了一个基于 torch_geometric 的几何图神经网络编码器，用于处理蛋白质骨架的 3D 结构。
编码器能够理解蛋白质的几何特征（二面角、距离、方向），并生成节点嵌入。

核心思想：
1. 基于 C-alpha 原子构建 k-NN 图
2. 计算几何特征（二面角、距离、方向向量）
3. 使用 GNN 层进行消息传递和特征更新
4. 输出节点嵌入用于下游任务

Author: Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

# 导入 torch_geometric（必需）
try:
    from torch_geometric.nn import MessagePassing
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    raise ImportError(
        "torch_geometric is required for BackboneEncoder. "
        "Please install it with: pip install torch-geometric"
    )


class DihedralAngleEncoder(nn.Module):
    """
    二面角编码器：计算并编码蛋白质骨架的二面角 (Phi, Psi, Omega)
    
    二面角的物理意义：
    - Phi (φ): C-N-CA-C 之间的二面角，描述主链的局部构象
    - Psi (ψ): N-CA-C-N 之间的二面角，描述主链的局部构象
    - Omega (ω): CA-C-N-CA 之间的二面角，通常接近 180°（平面结构）
    
    使用 sin/cos 编码可以保持角度的周期性（0° = 360°）
    """
    
    def __init__(self, output_dim: int = 64):
        """
        Args:
            output_dim: 输出特征维度（每个角度用 sin/cos 编码，3个角度共 6 维，然后投影）
        """
        super().__init__()
        self.output_dim = output_dim
        # 3个角度 × 2 (sin/cos) = 6 维，然后投影到 output_dim
        self.proj = nn.Linear(6, output_dim)
    
    def compute_dihedral(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor
    ) -> torch.Tensor:
        """
        计算四个点之间的二面角（弧度）
        
        Args:
            p1, p2, p3, p4: 四个 3D 点的坐标 [..., 3]
        
        Returns:
            二面角（弧度）[...]
        """
        # 计算向量
        v1 = p2 - p1  # 向量 p1->p2
        v2 = p3 - p2  # 向量 p2->p3
        v3 = p4 - p3  # 向量 p3->p4
        
        # 计算法向量
        n1 = torch.cross(v1, v2, dim=-1)  # 平面 1 的法向量
        n2 = torch.cross(v2, v3, dim=-1)  # 平面 2 的法向量
        
        # 归一化
        n1_norm = n1 / (torch.norm(n1, dim=-1, keepdim=True) + 1e-8)
        n2_norm = n2 / (torch.norm(n2, dim=-1, keepdim=True) + 1e-8)
        
        # 计算角度
        cos_angle = torch.sum(n1_norm * n2_norm, dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle = torch.acos(cos_angle)
        
        # 确定符号（使用右手定则）
        sign = torch.sign(torch.sum(n1_norm * v3, dim=-1))
        angle = angle * sign
        
        return angle
    
    def forward(self, backbone_coords: torch.Tensor) -> torch.Tensor:
        """
        计算并编码二面角特征
        
        Args:
            backbone_coords: 骨架坐标 [batch_size, L, 4, 3]
                           4 个原子：N, CA, C, O
        
        Returns:
            编码后的二面角特征 [batch_size, L, output_dim]
        """
        batch_size, seq_len, _, _ = backbone_coords.shape
        
        # 提取原子坐标
        # 注意：索引顺序是 N(0), CA(1), C(2), O(3)
        N = backbone_coords[:, :, 0, :]   # [batch, L, 3]
        CA = backbone_coords[:, :, 1, :]  # [batch, L, 3]
        C = backbone_coords[:, :, 2, :]   # [batch, L, 3]
        O = backbone_coords[:, :, 3, :]   # [batch, L, 3]
        
        # 计算 Phi 角：C(i-1) - N(i) - CA(i) - C(i)
        # 对于第一个残基，使用前一个残基的 C（如果存在）
        phi_angles = []
        for i in range(seq_len):
            if i == 0:
                # 第一个残基，使用当前残基的 C 作为近似
                phi = self.compute_dihedral(
                    C[:, i, :].unsqueeze(1),  # 使用当前 C 作为近似
                    N[:, i, :].unsqueeze(1),
                    CA[:, i, :].unsqueeze(1),
                    C[:, i, :].unsqueeze(1)
                )
            else:
                phi = self.compute_dihedral(
                    C[:, i-1, :].unsqueeze(1),
                    N[:, i, :].unsqueeze(1),
                    CA[:, i, :].unsqueeze(1),
                    C[:, i, :].unsqueeze(1)
                )
            phi_angles.append(phi)
        phi = torch.stack(phi_angles, dim=1).squeeze(2)  # [batch, L]
        
        # 计算 Psi 角：N(i) - CA(i) - C(i) - N(i+1)
        psi_angles = []
        for i in range(seq_len):
            if i == seq_len - 1:
                # 最后一个残基，使用当前残基的 N 作为近似
                psi = self.compute_dihedral(
                    N[:, i, :].unsqueeze(1),
                    CA[:, i, :].unsqueeze(1),
                    C[:, i, :].unsqueeze(1),
                    N[:, i, :].unsqueeze(1)  # 使用当前 N 作为近似
                )
            else:
                psi = self.compute_dihedral(
                    N[:, i, :].unsqueeze(1),
                    CA[:, i, :].unsqueeze(1),
                    C[:, i, :].unsqueeze(1),
                    N[:, i+1, :].unsqueeze(1)
                )
            psi_angles.append(psi)
        psi = torch.stack(psi_angles, dim=1).squeeze(2)  # [batch, L]
        
        # 计算 Omega 角：CA(i) - C(i) - N(i+1) - CA(i+1)
        omega_angles = []
        for i in range(seq_len):
            if i == seq_len - 1:
                # 最后一个残基，使用当前残基的 CA 作为近似
                omega = self.compute_dihedral(
                    CA[:, i, :].unsqueeze(1),
                    C[:, i, :].unsqueeze(1),
                    CA[:, i, :].unsqueeze(1),  # 使用当前 CA 作为近似
                    CA[:, i, :].unsqueeze(1)
                )
            else:
                omega = self.compute_dihedral(
                    CA[:, i, :].unsqueeze(1),
                    C[:, i, :].unsqueeze(1),
                    N[:, i+1, :].unsqueeze(1),
                    CA[:, i+1, :].unsqueeze(1)
                )
            omega_angles.append(omega)
        omega = torch.stack(omega_angles, dim=1).squeeze(2)  # [batch, L]
        
        # Sin/Cos 编码（保持周期性）
        angles = torch.stack([phi, psi, omega], dim=-1)  # [batch, L, 3]
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)
        
        # 拼接：6 维特征
        encoded = torch.cat([sin_angles, cos_angles], dim=-1)  # [batch, L, 6]
        
        # 投影到目标维度
        encoded = self.proj(encoded)  # [batch, L, output_dim]
        
        return encoded


class RBFEncoder(nn.Module):
    """
    径向基函数 (RBF) 编码器：用于编码节点间的距离
    
    物理意义：蛋白质中原子间的距离是重要的结构信息，RBF 编码可以将连续距离
    映射到高维特征空间，便于神经网络学习。
    """
    
    def __init__(self, num_rbf: int = 16, cutoff: float = 20.0):
        """
        Args:
            num_rbf: RBF 基函数的数量
            cutoff: 距离截断值（埃），超过此距离的边将被忽略
        """
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        
        # RBF 中心点（均匀分布在 [0, cutoff] 区间）
        self.register_buffer('centers', torch.linspace(0, cutoff, num_rbf))
        # RBF 宽度（标准差）
        self.register_buffer('width', torch.tensor(cutoff / num_rbf))
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        将距离编码为 RBF 特征
        
        Args:
            distances: 距离张量 [..., 1] 或 [...]
        
        Returns:
            RBF 编码特征 [..., num_rbf]
        """
        # 确保 distances 是 [..., 1] 形状
        if distances.dim() == 1:
            distances = distances.unsqueeze(-1)
        
        # 计算 RBF：exp(-(d - c)^2 / (2 * width^2))
        # distances: [..., 1], centers: [num_rbf]
        diff = distances - self.centers  # [..., num_rbf]
        rbf = torch.exp(-0.5 * (diff / self.width) ** 2)
        
        return rbf


class GNNLayer(MessagePassing):
    """
    图神经网络层：基于消息传递机制更新节点特征
    
    消息传递过程：
    1. 消息生成：基于源节点特征和边特征生成消息
    2. 消息聚合：聚合来自邻居节点的消息
    3. 节点更新：结合自身特征和聚合消息更新节点
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        """
        Args:
            node_dim: 节点特征维度
            edge_dim: 边特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout 率
        """
        super().__init__(aggr='add')  # 使用加法聚合
        
        # 消息生成网络
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 节点更新网络
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, node_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_dim]
        
        Returns:
            更新后的节点特征 [num_nodes, hidden_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        生成消息：从源节点 j 到目标节点 i
        
        Args:
            x_i: 目标节点特征 [num_edges, node_dim]
            x_j: 源节点特征 [num_edges, node_dim]
            edge_attr: 边特征 [num_edges, edge_dim]
        
        Returns:
            消息 [num_edges, hidden_dim]
        """
        # 拼接源节点特征和边特征
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.msg_mlp(msg_input)
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        更新节点特征
        
        Args:
            aggr_out: 聚合后的消息 [num_nodes, hidden_dim]
            x: 原始节点特征 [num_nodes, node_dim]
        
        Returns:
            更新后的节点特征 [num_nodes, hidden_dim]
        """
        # 拼接原始特征和聚合消息
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)


class BackboneEncoder(nn.Module):
    """
    基于几何图神经网络的蛋白质骨架编码器
    
    架构流程：
    1. 构图：基于 C-alpha 原子构建 k-NN 图
    2. 特征提取：计算节点特征（二面角 + 理化特征）和边特征（距离、方向）
    3. GNN 层：多层消息传递更新节点特征
    4. 输出：节点嵌入 [batch_size, L, hidden_dim]
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        k_neighbors: int = 30,
        node_feat_dim: int = 64,
        edge_feat_dim: int = 32,
        dropout: float = 0.1,
        use_physicochemical: bool = True,
        physicochemical_dim: int = 5
    ):
        """
        Args:
            hidden_dim: 隐藏层维度（输出维度）
            num_layers: GNN 层数
            k_neighbors: k-NN 图中的 k 值
            node_feat_dim: 节点特征维度（二面角编码后的维度）
            edge_feat_dim: 边特征维度（RBF 编码后的维度）
            dropout: Dropout 率
            use_physicochemical: 是否使用理化特征（默认True）
            physicochemical_dim: 理化特征维度（默认5：疏水性、电荷、分子量、氢键供体、氢键受体）
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors
        self.use_physicochemical = use_physicochemical
        self.physicochemical_dim = physicochemical_dim
        
        # 二面角编码器
        self.dihedral_encoder = DihedralAngleEncoder(output_dim=node_feat_dim)
        
        # RBF 编码器（用于距离编码）
        self.rbf_encoder = RBFEncoder(num_rbf=edge_feat_dim, cutoff=20.0)
        
        # 理化特征投影层（将理化特征映射到隐藏维度）
        if use_physicochemical:
            self.physicochemical_proj = nn.Linear(physicochemical_dim, hidden_dim)
        
        # 节点特征投影（从二面角特征到隐藏维度）
        # 如果使用理化特征，需要融合，所以投影维度可能需要调整
        geometric_dim = node_feat_dim
        self.node_proj = nn.Linear(geometric_dim, hidden_dim)
        
        # GNN 层堆叠
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                GNNLayer(
                    node_dim=hidden_dim if i > 0 else hidden_dim,
                    edge_dim=edge_feat_dim + 3,  # RBF 特征 + 方向向量（3维）
                    hidden_dim=hidden_dim,
                    dropout=dropout
                )
            )
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
    
    def build_graph(
        self,
        ca_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        基于 C-alpha 坐标构建 k-NN 图
        
        注意：对于批处理数据，我们为每个样本单独构建图，确保不同批次之间的节点不会连接。
        
        Args:
            ca_coords: C-alpha 坐标 [batch_size, L, 3]
            mask: 可选掩码 [batch_size, L]，True 表示有效位置
        
        Returns:
            edge_index: 边索引 [2, num_edges]
            edge_dist: 边距离 [num_edges, 1]
            edge_vec: 边方向向量 [num_edges, 3]
        """
        batch_size, seq_len, _ = ca_coords.shape
        device = ca_coords.device
        
        # 为每个批次单独构建图（确保不同批次之间不连接）
        all_edge_indices = []
        all_edge_dists = []
        all_edge_vecs = []
        
        for b in range(batch_size):
            # 当前批次的坐标
            ca_batch = ca_coords[b]  # [L, 3]
            
            # 如果提供了掩码，只考虑有效位置
            if mask is not None:
                valid_mask = mask[b]  # [L]
                if not valid_mask.all():
                    # 只使用有效位置
                    ca_batch = ca_batch[valid_mask]  # [L_valid, 3]
                    valid_indices = torch.where(valid_mask)[0]
                else:
                    valid_indices = torch.arange(seq_len, device=device)
            else:
                valid_indices = torch.arange(seq_len, device=device)
            
            L_valid = ca_batch.shape[0]
            
            if L_valid == 0:
                continue
            
            # 计算距离矩阵（只对当前批次）
            dist_matrix = torch.cdist(ca_batch, ca_batch)  # [L_valid, L_valid]
            
            # 排除自身（距离为 0）
            dist_matrix.fill_diagonal_(float('inf'))
            
            # 获取 k 个最近邻
            k = min(self.k_neighbors, L_valid - 1)  # 减1因为排除了自身
            if k > 0:
                _, topk_indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)
                
                # 构建边索引（需要加上批次偏移）
                batch_offset = b * seq_len
                num_edges = L_valid * k
                
                # 源节点：每个节点重复 k 次
                src_nodes = (valid_indices.unsqueeze(1) + batch_offset).repeat(1, k).flatten()
                # 目标节点：对应的最近邻
                dst_nodes = (valid_indices[topk_indices] + batch_offset).flatten()
                
                # 计算边特征
                src_coords = ca_batch.repeat_interleave(k, dim=0)  # [num_edges, 3]
                dst_coords = ca_batch[topk_indices.flatten()]  # [num_edges, 3]
                
                edge_vec = dst_coords - src_coords  # [num_edges, 3]
                edge_dist = torch.norm(edge_vec, dim=1, keepdim=True)  # [num_edges, 1]
                edge_vec_norm = edge_vec / (edge_dist + 1e-8)
                
                all_edge_indices.append(torch.stack([src_nodes, dst_nodes], dim=0))
                all_edge_dists.append(edge_dist)
                all_edge_vecs.append(edge_vec_norm)
        
        # 合并所有批次的边
        if all_edge_indices:
            edge_index = torch.cat(all_edge_indices, dim=1)  # [2, total_edges]
            edge_dist = torch.cat(all_edge_dists, dim=0)  # [total_edges, 1]
            edge_vec = torch.cat(all_edge_vecs, dim=0)  # [total_edges, 3]
        else:
            # 如果没有边，返回空张量
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_dist = torch.empty((0, 1), dtype=torch.float, device=device)
            edge_vec = torch.empty((0, 3), dtype=torch.float, device=device)
        
        return edge_index, edge_dist, edge_vec
    
    def forward(
        self,
        backbone_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        residue_types: Optional[List[List[str]]] = None
    ) -> torch.Tensor:
        """
        编码骨架坐标为节点嵌入
        
        Args:
            backbone_coords: 骨架坐标 [batch_size, L, 4, 3]
                           4 个原子：N, CA, C, O
            mask: 可选掩码 [batch_size, L]，True 表示有效位置
            residue_types: 可选残基类型列表 [batch_size, L]，每个元素是3字母残基代码（如'ALA'）
        
        Returns:
            节点嵌入 [batch_size, L, hidden_dim]
        """
        batch_size, seq_len, _, _ = backbone_coords.shape
        device = backbone_coords.device
        
        # 提取 C-alpha 坐标（索引 1）
        ca_coords = backbone_coords[:, :, 1, :]  # [batch_size, L, 3]
        
        # 1. 计算几何特征（二面角编码）
        geometric_features = self.dihedral_encoder(backbone_coords)  # [batch_size, L, node_feat_dim]
        geometric_features = self.node_proj(geometric_features)  # [batch_size, L, hidden_dim]
        
        # 2. 计算理化特征（如果启用）
        if self.use_physicochemical:
            # 从vocabulary模块导入理化特征
            from data.vocabulary import get_vocab
            vocab = get_vocab()
            
            if residue_types is not None:
                # 根据残基类型获取理化特征
                physicochemical_features = []
                for b in range(batch_size):
                    batch_features = []
                    for l in range(seq_len):
                        if l < len(residue_types[b]):
                            res_name = residue_types[b][l]
                            if res_name in vocab.PHYSICOCHEMICAL_FEATURES:
                                feat = vocab.PHYSICOCHEMICAL_FEATURES[res_name]
                            else:
                                # 未知残基，使用零向量
                                feat = [0.0, 0.0, 0.0, 0.0, 0.0]
                        else:
                            # 超出序列长度，使用零向量
                            feat = [0.0, 0.0, 0.0, 0.0, 0.0]
                        batch_features.append(feat)
                    physicochemical_features.append(batch_features)
                
                # 转换为张量并归一化
                pc_features = torch.tensor(physicochemical_features, dtype=torch.float32, device=device)  # [batch_size, L, 5]
                
                # 归一化
                stats = vocab.FEATURE_STATS
                pc_features[:, :, 0] = (pc_features[:, :, 0] - stats['hydropathy']['mean']) / stats['hydropathy']['std']
                pc_features[:, :, 1] = (pc_features[:, :, 1] - stats['charge']['mean']) / stats['charge']['std']
                pc_features[:, :, 2] = (pc_features[:, :, 2] - stats['molecular_weight']['mean']) / stats['molecular_weight']['std']
                pc_features[:, :, 3] = (pc_features[:, :, 3] - stats['h_donors']['mean']) / stats['h_donors']['std']
                pc_features[:, :, 4] = (pc_features[:, :, 4] - stats['h_acceptors']['mean']) / stats['h_acceptors']['std']
                
                # 投影到hidden_dim
                pc_features = self.physicochemical_proj(pc_features)  # [batch_size, L, hidden_dim]
                
                # 融合几何特征和理化特征（使用加法）
                node_features = geometric_features + pc_features  # [batch_size, L, hidden_dim]
            else:
                # 如果没有提供残基类型，只使用几何特征
                node_features = geometric_features
        else:
            # 不使用理化特征
            node_features = geometric_features
        
        # 2. 构建图
        edge_index, edge_dist, edge_vec = self.build_graph(ca_coords, mask)
        
        # 3. 编码边特征
        edge_rbf = self.rbf_encoder(edge_dist)  # [num_edges, edge_feat_dim]
        edge_features = torch.cat([edge_rbf, edge_vec], dim=-1)  # [num_edges, edge_feat_dim + 3]
        
        # 4. 展平节点特征用于 GNN 处理
        node_features_flat = node_features.view(-1, self.hidden_dim)  # [batch * L, hidden_dim]
        
        # 5. 通过 GNN 层
        x = node_features_flat
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            # 残差连接
            x_residual = x
            x = gnn_layer(x, edge_index, edge_features)
            x = layer_norm(x + x_residual)  # 残差连接 + 层归一化
        
        # 7. 恢复批处理维度
        node_embeddings = x.view(batch_size, seq_len, self.hidden_dim)  # [batch_size, L, hidden_dim]
        
        # 8. 应用掩码（如果有）
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(node_embeddings)
            node_embeddings = node_embeddings * mask_expanded.float()
        
        return node_embeddings
    
    def get_output_dim(self) -> int:
        """返回输出嵌入维度"""
        return self.hidden_dim


# 测试代码
if __name__ == "__main__":
    # 测试编码器
    encoder = BackboneEncoder(hidden_dim=256, num_layers=3, k_neighbors=30)
    
    # 创建虚拟输入
    batch_size = 2
    seq_len = 10
    dummy_coords = torch.randn(batch_size, seq_len, 4, 3)
    
    # 前向传播
    embeddings = encoder(dummy_coords)
    
    print("="*60)
    print("BackboneEncoder 测试")
    print("="*60)
    print(f"输入形状: {dummy_coords.shape}")
    print(f"输出形状: {embeddings.shape}")
    print(f"输出维度: {encoder.get_output_dim()}")
    print("="*60)
