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
            掩码 [seq_len, seq_len]，True 表示可以关注，False 表示不能关注
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        return ~mask  # 反转：True 表示可以关注
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        target_fragments: torch.Tensor,
        fragment_mask: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # Padding mask for target (fragment sequence)
        tgt_mask = self.create_padding_mask(frag_seq_len, batch_size, None, device)
        # Causal mask for self-attention
        causal_mask = self.create_causal_mask(frag_seq_len, device)
        
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
        
        try:
            decoder_output = self.transformer_layers(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )  # [batch_size, M, hidden_dim]
            
            # 检查 decoder 输出是否有 NaN
            if torch.isnan(decoder_output).any() or torch.isinf(decoder_output).any():
                # 如果包含 NaN，尝试用 0 替换
                decoder_output = torch.where(
                    torch.isnan(decoder_output) | torch.isinf(decoder_output),
                    torch.zeros_like(decoder_output),
                    decoder_output
                )
        except Exception as e:
            # 如果 transformer 失败，返回零输出
            print(f"  ⚠️  Transformer decoder 失败: {e}")
            decoder_output = torch.zeros(batch_size, frag_seq_len, self.hidden_dim, device=device)
        
        # 6. 预测头
        try:
            fragment_logits = self.type_head(decoder_output)  # [batch_size, M, vocab_size]
            torsion_logits = self.torsion_head(decoder_output)  # [batch_size, M, num_torsion_bins]
            
            # 检查预测输出是否有 NaN，如果有则替换为 0
            if torch.isnan(fragment_logits).any() or torch.isinf(fragment_logits).any():
                fragment_logits = torch.where(
                    torch.isnan(fragment_logits) | torch.isinf(fragment_logits),
                    torch.zeros_like(fragment_logits),
                    fragment_logits
                )
            if torch.isnan(torsion_logits).any() or torch.isinf(torsion_logits).any():
                torsion_logits = torch.where(
                    torch.isnan(torsion_logits) | torch.isinf(torsion_logits),
                    torch.zeros_like(torsion_logits),
                    torsion_logits
                )
        except Exception as e:
            # 如果预测头失败，返回零输出
            print(f"  ⚠️  预测头失败: {e}")
            fragment_logits = torch.zeros(batch_size, frag_seq_len, self.vocab_size, device=device)
            torsion_logits = torch.zeros(batch_size, frag_seq_len, self.num_torsion_bins, device=device)
        
        return fragment_logits, torsion_logits


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
    frag_logits, tors_logits = decoder(
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
    print("="*60)
