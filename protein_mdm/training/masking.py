"""
掩码策略模块

用于训练时创建掩码，模拟掩码扩散模型的训练过程。
"""

import torch
from typing import Optional
from ..data.vocabulary import SpecialTokens


def create_masks(
    fragment_token_ids: torch.Tensor,
    strategy: str = "random",
    mask_ratio: float = 0.15,
    **kwargs
) -> torch.Tensor:
    """
    创建片段掩码
    
    Args:
        fragment_token_ids: 片段 Token IDs [batch_size, M]
        strategy: 掩码策略 ("random", "block")
        mask_ratio: 掩码比例（0.0 到 1.0）
        **kwargs: 其他策略参数
    
    Returns:
        布尔掩码 [batch_size, M]，True 表示需要预测的位置
    """
    batch_size, num_fragments = fragment_token_ids.shape
    device = fragment_token_ids.device
    
    masks = []
    
    for i in range(batch_size):
        if strategy == "random":
            # 随机掩码
            num_masked = int(num_fragments * mask_ratio)
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
