"""
基于 Transformer 的片段解码器

本模块实现了一个 Transformer Decoder，用于根据骨架节点嵌入预测被 Mask 掉的片段序列和扭转角。

核心思想：
1. Fragment Embedding: 将离散的 Token ID 映射为连续向量
2. Positional Encoding: 添加位置信息
3. Self-Attention: 处理片段序列之间的依赖关系
4. Cross-Attention: 关注 Encoder 输出的结构特征（节点嵌入）
5. Prediction Heads: 预测片段类型和扭转角

Author: Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    位置编码：为序列添加位置信息
    
    使用标准的正弦/余弦位置编码，使模型能够理解片段在序列中的位置。
    这对于理解片段的顺序依赖关系很重要。
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度（嵌入维度）
            max_len: 最大序列长度
            dropout: Dropout 率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为 buffer（不参与梯度更新）
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入嵌入 [batch_size, seq_len, d_model]
        
        Returns:
            添加位置编码后的嵌入 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        max_len = self.pe.size(1)
        
        # 如果序列长度超过预设的最大长度，动态扩展位置编码
        if seq_len > max_len:
            # 计算需要扩展的长度
            device = x.device
            d_model = self.pe.size(2)
            
            # 生成扩展的位置编码
            position = torch.arange(max_len, seq_len, device=device).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * 
                               (-math.log(10000.0) / d_model))
            
            pe_extension = torch.zeros(seq_len - max_len, d_model, device=device)
            pe_extension[:, 0::2] = torch.sin(position * div_term)
            pe_extension[:, 1::2] = torch.cos(position * div_term)
            
            # 扩展位置编码
            pe_extended = torch.cat([self.pe.squeeze(0), pe_extension], dim=0).unsqueeze(0)
        else:
            pe_extended = self.pe
        
        # pe_extended: [1, seq_len, d_model]
        # x: [batch_size, seq_len, d_model]
        # 检查位置编码是否有 NaN
        if torch.isnan(pe_extended).any() or torch.isinf(pe_extended).any():
            # 如果位置编码有 NaN，只使用输入（不加位置编码）
            return self.dropout(x)
        
        x = x + pe_extended[:, :seq_len, :]
        
        # 检查相加后是否有 NaN
        if torch.isnan(x).any() or torch.isinf(x).any():
            # 如果相加后产生 NaN，只返回输入
            return self.dropout(x - pe_extended[:, :seq_len, :])
        
        return self.dropout(x)


class SpatialGeometricBias(nn.Module):
    """
    动态空间几何偏置 (Dynamic Spatial Geometric Bias)
    将三维骨架几何信息和 Mask 状态注入到一维 Transformer Attention 中
    """
    def __init__(self, num_heads: int, num_rbf_bins: int = 16, chunk_size: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.num_rbf_bins = num_rbf_bins
        self.chunk_size = chunk_size
        # 特征维度: RBF (bins) + 余弦相似度(1) + Mask状态i(1) + Mask状态j(1) = bins + 3
        feature_dim = num_rbf_bins + 3
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_heads)
        )
        
    def _rbf(self, D: torch.Tensor, D_min=0.0, D_max=20.0):
        # 计算径向基函数 (Radial Basis Function)
        D_mu = torch.linspace(D_min, D_max, self.num_rbf_bins, device=D.device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / self.num_rbf_bins
        D_expand = D.unsqueeze(-1)
        rbf = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return rbf
        
    def forward(self, backbone_coords, fragment_residue_idx, fragment_mask):
        B, M = fragment_residue_idx.shape
        device = backbone_coords.device
        
        # 提取 N, CA, C 坐标 [B, M, 3]
        max_idx = backbone_coords.shape[1] - 1
        # 防止因 padding 产生的无效残基索引越界
        safe_idx = torch.clamp(fragment_residue_idx, min=0, max=max_idx)
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, M)
        
        n_coords = backbone_coords[batch_indices, safe_idx, 0, :]
        ca_coords = backbone_coords[batch_indices, safe_idx, 1, :]
        c_coords = backbone_coords[batch_indices, safe_idx, 2, :]
        
        # NaN 防护：如果残基数据损坏导致坐标含 NaN，将其置零以防止梯度大爆炸
        n_coords = torch.nan_to_num(n_coords, nan=0.0)
        ca_coords = torch.nan_to_num(ca_coords, nan=0.0)
        c_coords = torch.nan_to_num(c_coords, nan=0.0)
        
        # 近似侧链指向向量
        n_ca = torch.nn.functional.normalize(n_coords - ca_coords, dim=-1)
        c_ca = torch.nn.functional.normalize(c_coords - ca_coords, dim=-1)
        v_dir = torch.nn.functional.normalize(n_ca + c_ca + 1e-6, dim=-1)
        
        # Mask 状态 [B, M, M]
        if fragment_mask is None:
            fragment_mask = torch.zeros((B, M), dtype=torch.bool, device=device)
        mask_f = fragment_mask.float()
        
        # 仅保留最终注意力偏置，按行分块计算，避免提前构建 D_mat/cos_theta 的全量 O(M^2) 中间张量。
        bias = torch.zeros(B, self.num_heads, M, M, dtype=n_coords.dtype, device=device)
        for i in range(0, M, self.chunk_size):
            end_idx = min(i + self.chunk_size, M)
            
            # [B, chunk, M]
            ca_chunk = ca_coords[:, i:end_idx, :].unsqueeze(2)
            D_chunk = torch.norm(ca_chunk - ca_coords.unsqueeze(1), dim=-1)

            v_chunk = v_dir[:, i:end_idx, :]
            cos_chunk = torch.einsum('bcd,bmd->bcm', v_chunk, v_dir).unsqueeze(-1)
            
            # Mask chunks: m_i is [B, chunk, 1], m_j is [B, 1, M] expanded to [B, chunk, M]
            mi_chunk = mask_f[:, i:end_idx].unsqueeze(-1).unsqueeze(-1).expand(B, end_idx - i, M, 1)
            mj_chunk = mask_f.unsqueeze(1).unsqueeze(-1).expand(B, end_idx - i, M, 1)
            
            # [B, chunk, M, bins]
            rbf_chunk = self._rbf(D_chunk)
            
            feat_chunk = torch.cat([rbf_chunk, cos_chunk, mi_chunk, mj_chunk], dim=-1)
            chunk_bias = self.mlp(feat_chunk)  # [B, chunk, M, Heads]
            bias[:, :, i:end_idx, :] = chunk_bias.permute(0, 3, 1, 2)
        
        # PyTorch attention mask 形状要求为：[B * Heads, M, M] 
        # 当它为 FloatTensor 且传给 tgt_mask 时，它会被直接相加到 Attention Scores 上
        bias = bias.reshape(B * self.num_heads, M, M)
        return bias

class FragmentDecoder(nn.Module):
    """
    基于 Transformer 的片段解码器
    
    架构：
    1. Fragment Embedding + Positional Encoding
    2. Transformer Decoder Layers (Self-Attention + Cross-Attention)
    3. Type Head: 预测片段 Token ID
    4. Torsion Head: 预测扭转角 Bin ID
    """
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        num_torsion_bins: int = 72,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        """
        Args:
            input_dim: Encoder 输出的节点嵌入维度
            vocab_size: 片段词汇表大小
            num_torsion_bins: 扭转角离散化的 bin 数量
            hidden_dim: Transformer 隐藏层维度
            num_layers: Transformer 层数
            num_heads: 多头注意力的头数
            dropout: Dropout 率
            max_seq_len: 最大序列长度（用于位置编码）
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_torsion_bins = num_torsion_bins
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        # 提高空间偏置阈值：适配单卡11G显存+BatchSize=4计算极限
        # M在3200以内可保持不OOM，绝大部分长链(2000-3000)均可纳入三维空间计算
        self.max_spatial_bias_tokens = 2048
        self._spatial_bias_warned = False
        
        # Fragment Embedding: 将 Token ID 映射为向量
        self.fragment_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Transformer Decoder Layers
        # 使用 PyTorch 原生的 TransformerDecoderLayer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # FFN 维度通常是 d_model 的 4 倍
            dropout=dropout,
            activation='gelu',
            batch_first=True  # 使用 batch_first=True 以便处理 [batch, seq, dim] 格式
        )
        self.transformer_layers = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # 注入动态空间几何偏置模块
        self.spatial_bias = SpatialGeometricBias(num_heads=num_heads)
        
        # Encoder 输出的投影层（将 input_dim 投影到 hidden_dim）
        self.encoder_proj = nn.Linear(input_dim, hidden_dim)
        
        # 预测头
        # Type Head: 预测片段类型
        self.type_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Torsion Head: 预测扭转角 bin
        self.torsion_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_torsion_bins)
        )
        
        # Offset Head: 预测扭转角的微调偏移量（用于提升精度）
        # 输出维度为 num_torsion_bins，每个 bin 预测一个 offset
        self.offset_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_torsion_bins)
        )
        
        # 确保在 eval 模式下 Dropout 被禁用
        self._dropout_rate = dropout
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        # 使用 Xavier 初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def create_padding_mask(
        self,
        seq_len: int,
        batch_size: int,
        valid_lengths: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        创建填充掩码（用于忽略 padding 位置）
        
        Args:
            seq_len: 序列长度
            batch_size: 批次大小
            valid_lengths: 每个序列的有效长度 [batch_size]，如果为 None 则所有位置都有效
            device: 设备
        
        Returns:
            掩码 [batch_size, seq_len]，True 表示有效位置，False 表示 padding
        """
        if valid_lengths is None:
            # 所有位置都有效
            return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) < valid_lengths.unsqueeze(1)
        return mask
    
    def create_causal_mask(self, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        创建因果掩码（用于自注意力，防止看到未来信息）
        
        Args:
            seq_len: 序列长度
            device: 设备
        
        Returns:
            掩码 [seq_len, seq_len]，True 表示不能关注（被屏蔽）
        """
        # PyTorch Transformer 的布尔掩码语义：True 表示该位置会被屏蔽
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        target_fragments: torch.Tensor,
        fragment_mask: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
        backbone_coords: Optional[torch.Tensor] = None,
        fragment_residue_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            node_embeddings: Encoder 输出的节点嵌入 [batch_size, L, input_dim]
                            L 是残基序列长度
            target_fragments: 目标片段序列（可能包含 MASK token）[batch_size, M, vocab_size]
                            M 是片段序列长度
            fragment_mask: 片段掩码 [batch_size, M]，True 表示需要预测的位置
            sequence_lengths: 每个序列的有效长度 [batch_size]
        
        Returns:
            Tuple of:
            - fragment_logits: [batch_size, M, vocab_size] - 片段类型预测
            - torsion_logits: [batch_size, M, num_torsion_bins] - 扭转角预测
            - offset_logits: [batch_size, M, num_torsion_bins] - 扭转角偏移量预测
        """
        batch_size, seq_len, _ = node_embeddings.shape
        frag_seq_len = target_fragments.shape[1] if target_fragments.dim() > 1 else 1
        device = node_embeddings.device
        
        # 1. 处理 Encoder 输出（Memory）
        # 投影到 hidden_dim
        memory = self.encoder_proj(node_embeddings)  # [batch_size, L, hidden_dim]
        
        # 2. Fragment Embedding
        # 如果 target_fragments 是 token IDs，需要转换为嵌入
        if target_fragments.dim() == 2 and target_fragments.dtype == torch.long:
            # target_fragments 是 token IDs [batch_size, M]
            tgt_emb = self.fragment_embedding(target_fragments)  # [batch_size, M, hidden_dim]
        else:
            # target_fragments 已经是嵌入向量
            tgt_emb = target_fragments
        
        # 3. 添加位置编码
        tgt_emb = self.pos_encoder(tgt_emb)  # [batch_size, M, hidden_dim]
        
        # 4. 创建掩码
        # Padding mask for memory (encoder output)
        memory_mask = self.create_padding_mask(seq_len, batch_size, sequence_lengths, device)
        
        # 4. 构建动态空间偏置 (替代了简单的 Causal Mask)
        # 在标准的 Masked Language Model 或是 MDM 中，我们实际上需要双向注意力（可以看其余未被掩码的词）
        # 但是为了注入物理约束，我们使用 spatial_tgt_mask 加在注意力分数上
        spatial_tgt_mask = None
        if (
            backbone_coords is not None
            and fragment_residue_idx is not None
            and frag_seq_len <= self.max_spatial_bias_tokens
        ):
            spatial_tgt_mask = self.spatial_bias(backbone_coords, fragment_residue_idx, fragment_mask)
        else:
            # 大序列时回退，避免 O(M^2) 注意力偏置导致显存峰值失控
            if (
                backbone_coords is not None
                and fragment_residue_idx is not None
                and frag_seq_len > self.max_spatial_bias_tokens
                and not self._spatial_bias_warned
            ):
                print(
                    f"  ⚠️  序列长度 M={frag_seq_len} 超过空间偏置阈值 "
                    f"{self.max_spatial_bias_tokens}，已回退到因果掩码以避免 OOM"
                )
                self._spatial_bias_warned = True
            spatial_tgt_mask = self.create_causal_mask(frag_seq_len, device)

        # 5. Transformer Decoder
        # memory: [batch_size, L, hidden_dim] - Encoder 输出
        # tgt: [batch_size, M, hidden_dim] - 目标序列嵌入
        # memory_key_padding_mask: [batch_size, L] - Memory 的 padding 掩码
        # tgt_mask: [M, M] - 目标序列的因果掩码
        # tgt_key_padding_mask: [batch_size, M] - 目标序列的 padding 掩码
        
        # 检查输入是否有 NaN
        if torch.isnan(memory).any() or torch.isinf(memory).any():
            raise ValueError("Memory (encoder output) contains NaN/Inf")
        if torch.isnan(tgt_emb).any() or torch.isinf(tgt_emb).any():
            raise ValueError("Target embeddings contain NaN/Inf")
        
        # 注意：PyTorch 的 TransformerDecoder 需要 memory_key_padding_mask 和 tgt_key_padding_mask
        # 格式是：False 表示有效位置，True 表示需要忽略的位置（与我们的掩码相反）
        memory_key_padding_mask = ~memory_mask  # 反转：False=有效，True=padding
        tgt_key_padding_mask = None  # 暂时不使用
        
        decoder_output = self.transformer_layers(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=spatial_tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [batch_size, M, hidden_dim]
        decoder_output = torch.nan_to_num(decoder_output, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 6. 预测头
        fragment_logits = self.type_head(decoder_output)  # [batch_size, M, vocab_size]
        torsion_logits = self.torsion_head(decoder_output)  # [batch_size, M, num_torsion_bins]
        offset_logits = self.offset_head(decoder_output)  # [batch_size, M, num_torsion_bins]

        fragment_logits = torch.nan_to_num(fragment_logits, nan=0.0, posinf=0.0, neginf=0.0)
        torsion_logits = torch.nan_to_num(torsion_logits, nan=0.0, posinf=0.0, neginf=0.0)
        offset_logits = torch.nan_to_num(offset_logits, nan=0.0, posinf=0.0, neginf=0.0)
        
        return fragment_logits, torsion_logits, offset_logits


# 测试代码
if __name__ == "__main__":
    # 测试解码器
    vocab_size = 16  # 4 个特殊 token + 12 个片段
    decoder = FragmentDecoder(
        input_dim=256,
        vocab_size=vocab_size,
        num_torsion_bins=72,
        hidden_dim=256,
        num_layers=3,
        num_heads=8
    )
    
    # 创建虚拟输入
    batch_size = 2
    seq_len = 10  # 残基序列长度
    frag_seq_len = 20  # 片段序列长度
    
    # Encoder 输出（节点嵌入）
    node_embeddings = torch.randn(batch_size, seq_len, 256)
    
    # 目标片段序列（Token IDs，包含 MASK token）
    target_fragments = torch.randint(0, vocab_size, (batch_size, frag_seq_len))
    
    # 前向传播
    frag_logits, tors_logits, offset_logits = decoder(
        node_embeddings=node_embeddings,
        target_fragments=target_fragments
    )
    
    print("="*60)
    print("FragmentDecoder 测试")
    print("="*60)
    print(f"节点嵌入形状: {node_embeddings.shape}")
    print(f"目标片段形状: {target_fragments.shape}")
    print(f"Fragment logits 形状: {frag_logits.shape}")
    print(f"Torsion logits 形状: {tors_logits.shape}")
    print(f"Offset logits 形状: {offset_logits.shape}")
    print("="*60)
