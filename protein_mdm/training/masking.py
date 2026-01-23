"""
掩码策略模块

用于训练时创建掩码，模拟掩码扩散模型的训练过程。
支持离散扩散模型的时间步调度（Cosine Schedule）。
"""

import torch
import numpy as np
from typing import Optional
from data.vocabulary import SpecialTokens


def cosine_schedule(t: torch.Tensor) -> torch.Tensor:
    """
    Cosine Schedule: 根据时间步 t (0 -> 1) 计算当前的 mask_ratio
    
    Args:
        t: 归一化时间步 [batch_size] 或标量，范围 [0, 1]
           t=0 时 mask_ratio=0，t=1 时 mask_ratio=1
    
    Returns:
        mask_ratio: [batch_size] 或标量，范围 [0, 1]
    """
    # Cosine schedule: mask_ratio = 1 - cos(π * t / 2)
    # 当 t=0 时，mask_ratio=0；当 t=1 时，mask_ratio=1
    mask_ratio = 1.0 - torch.cos(np.pi * t / 2.0)
    return mask_ratio


def create_masks(
    fragment_token_ids: torch.Tensor,
    strategy: str = "random",
    timesteps: Optional[torch.Tensor] = None,
    use_cosine_schedule: bool = True,
    **kwargs
) -> torch.Tensor:
    """
    创建片段掩码（离散扩散模式）
    
    Args:
        fragment_token_ids: 片段 Token IDs [batch_size, M]
        strategy: 掩码策略 ("random", "block")
        timesteps: 时间步 [batch_size] 或标量，范围 [0, 1]（用于离散扩散）
        use_cosine_schedule: 是否使用 Cosine Schedule（默认True）
        **kwargs: 其他策略参数
    
    Returns:
        布尔掩码 [batch_size, M]，True 表示需要预测的位置
    """
    batch_size, num_fragments = fragment_token_ids.shape
    device = fragment_token_ids.device
    
    # 使用离散扩散模式：根据时间步计算动态 mask_ratio
    if timesteps is not None and use_cosine_schedule:
        if timesteps.dim() == 0:
            # 标量时间步，扩展到batch
            timesteps = timesteps.expand(batch_size)
        # 计算每个样本的 mask_ratio
        dynamic_mask_ratios = cosine_schedule(timesteps)  # [batch_size]
    else:
        # 如果没有提供时间步，使用默认值（不应该发生，但保留作为fallback）
        raise ValueError("离散扩散模式需要提供 timesteps 参数")
    
    masks = []
    
    for i in range(batch_size):
        # 获取当前样本的 mask_ratio
        current_mask_ratio = dynamic_mask_ratios[i].item() if isinstance(dynamic_mask_ratios, torch.Tensor) else dynamic_mask_ratios
        
        if strategy == "random":
            # 随机掩码
            num_masked = int(num_fragments * current_mask_ratio)
            mask = torch.zeros(num_fragments, dtype=torch.bool, device=device)
            if num_masked > 0:
                indices = torch.randperm(num_fragments, device=device)[:num_masked]
                mask[indices] = True
        
        elif strategy == "block":
            # 块状掩码
            num_blocks = kwargs.get("num_blocks", 3)
            block_size = kwargs.get("block_size", 5)
            mask = torch.zeros(num_fragments, dtype=torch.bool, device=device)
            
            for _ in range(num_blocks):
                start = torch.randint(0, max(1, num_fragments - block_size), (1,)).item()
                end = min(start + block_size, num_fragments)
                mask[start:end] = True
        
        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")
        
        masks.append(mask)
    
    return torch.stack(masks)


def apply_masks(
    fragment_token_ids: torch.Tensor,
    masks: torch.Tensor,
    mask_token_id: int = SpecialTokens.MASK
) -> torch.Tensor:
    """
    应用掩码到片段序列
    
    Args:
        fragment_token_ids: 原始片段 Token IDs [batch_size, M]
        masks: 布尔掩码 [batch_size, M]
        mask_token_id: MASK token 的 ID
    
    Returns:
        掩码后的片段 Token IDs [batch_size, M]
    """
    masked_tokens = fragment_token_ids.clone()
    masked_tokens[masks] = mask_token_id
    return masked_tokens
