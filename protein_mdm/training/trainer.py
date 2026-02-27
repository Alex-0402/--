"""
训练器模块

实现训练循环、损失计算、模型保存等功能。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict
import os
import numpy as np
# ✅ 修复：移除 tqdm，避免 I/O 阻塞导致死锁
# from tqdm import tqdm

# DDP 支持
try:
    import torch.distributed as dist
    DIST_AVAILABLE = True
except ImportError:
    DIST_AVAILABLE = False

from models.encoder import BackboneEncoder
from models.decoder import FragmentDecoder
from data.vocabulary import get_vocab, SpecialTokens
from training.masking import create_masks, apply_masks

# 可选导入可视化模块（如果matplotlib未安装，可视化功能将被禁用）
try:
    from training.visualization import plot_training_curves, plot_loss_comparison
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    # 定义占位函数以避免后续调用错误
    def plot_training_curves(*args, **kwargs):
        pass
    def plot_loss_comparison(*args, **kwargs):
        pass


class Trainer:
    """
    模型训练器
    
    功能：
    - 训练循环
    - 验证
    - 模型检查点保存
    - 学习率调度
    """
    
    def __init__(
        self,
        encoder: BackboneEncoder,
        decoder: FragmentDecoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-5,
        masking_strategy: str = "random",
        num_diffusion_steps: int = 1000,
        warmup_epochs: int = 20,
        total_epochs: int = 300,
        ddp_enabled: bool = False,
        rank: int = 0,
        world_size: int = 1,
        train_sampler: Optional[object] = None,
        val_sampler: Optional[object] = None,
        debug_mode: bool = False,
        label_smoothing: float = 0.1
    ):
        """
        初始化训练器
        
        Args:
            encoder: BackboneEncoder 实例
            decoder: FragmentDecoder 实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            device: 设备（默认：自动检测）
            learning_rate: 学习率
            weight_decay: 权重衰减
            masking_strategy: 掩码策略
            num_diffusion_steps: 扩散模型的时间步数
            warmup_epochs: Warmup 轮数
            total_epochs: 总训练轮数
            ddp_enabled: 是否启用 DDP
            rank: 当前进程的 rank（DDP 模式）
            world_size: 总进程数（DDP 模式）
            train_sampler: 训练集的 DistributedSampler（DDP 模式）
            val_sampler: 验证集的 DistributedSampler（DDP 模式）
            debug_mode: 是否启用详细调试日志（默认False）
            label_smoothing: 交叉熵标签平滑系数（默认0.1）
        """
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.masking_strategy = masking_strategy
        self.num_diffusion_steps = num_diffusion_steps
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_lr = learning_rate
        
        # DDP 相关
        self.ddp_enabled = ddp_enabled and DIST_AVAILABLE
        self.rank = rank
        self.world_size = world_size
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.debug_mode = debug_mode
        self.label_smoothing = float(max(0.0, min(1.0, label_smoothing)))
        
        # 优化器
        self.optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器：LinearWarmupCosineAnnealingLR
        # 先使用CosineAnnealingLR作为基础
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-6
        )
        # Warmup阶段会手动处理
        self.current_epoch = 0
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def _debug_print(self, message: str):
        """仅在主进程且开启调试时打印详细日志。"""
        if self.rank == 0 and self.debug_mode:
            print(message)
    
    def compute_loss(
        self,
        fragment_logits: torch.Tensor,
        torsion_logits: torch.Tensor,
        fragment_targets: torch.Tensor,
        torsion_targets: torch.Tensor,
        fragment_mask: Optional[torch.Tensor] = None,
        torsion_valid_mask: Optional[torch.Tensor] = None,
        offset_logits: Optional[torch.Tensor] = None,
        torsion_raw: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            fragment_logits: 片段预测 logits [batch_size, M, vocab_size]
            torsion_logits: 扭转角预测 logits [batch_size, M, num_bins]
            fragment_targets: 片段目标 [batch_size, M]
            torsion_targets: 扭转角目标 [batch_size, M]
            fragment_mask: 片段掩码 [batch_size, M]
            torsion_valid_mask: 扭转角有效位置掩码 [batch_size, M]
            offset_logits: Offset 预测 logits [batch_size, M, ...] (可选)
        
        Returns:
            损失字典
        """
        # 片段损失（交叉熵）
        # 有效片段位置（排除 PAD）
        valid_fragment_mask = fragment_targets != int(SpecialTokens.PAD)

        if fragment_mask is not None:
            # 只计算“被掩码且有效（非 PAD）”位置的损失
            effective_fragment_mask = fragment_mask & valid_fragment_mask
            masked_logits = fragment_logits[effective_fragment_mask]
            masked_targets = fragment_targets[effective_fragment_mask]
            if len(masked_logits) > 0:
                # 检查是否有无效值
                if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                    fragment_loss = torch.tensor(float('nan'), device=fragment_logits.device)
                else:
                    try:
                        fragment_loss = nn.functional.cross_entropy(
                            masked_logits.view(-1, fragment_logits.shape[-1]),
                            masked_targets.view(-1),
                            label_smoothing=self.label_smoothing
                        )
                        # 检查结果是否为 NaN
                        if torch.isnan(fragment_loss) or torch.isinf(fragment_loss):
                            fragment_loss = torch.tensor(0.0, device=fragment_logits.device)
                    except Exception:
                        fragment_loss = torch.tensor(0.0, device=fragment_logits.device)
            else:
                fragment_loss = torch.tensor(0.0, device=fragment_logits.device)
        else:
            # 没有提供掩码时，只计算非 PAD 位置
            masked_logits = fragment_logits[valid_fragment_mask]
            masked_targets = fragment_targets[valid_fragment_mask]

            # 检查是否有无效值
            if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                fragment_loss = torch.tensor(float('nan'), device=fragment_logits.device)
            else:
                try:
                    fragment_loss = nn.functional.cross_entropy(
                        masked_logits.reshape(-1, fragment_logits.shape[-1]),
                        masked_targets.reshape(-1),
                        label_smoothing=self.label_smoothing
                    )
                    # 检查结果是否为 NaN
                    if torch.isnan(fragment_loss) or torch.isinf(fragment_loss):
                        fragment_loss = torch.tensor(0.0, device=fragment_logits.device)
                except Exception:
                    fragment_loss = torch.tensor(0.0, device=fragment_logits.device)
        
        # 扭转角损失（交叉熵）
        # 注意：torsion_targets 的长度可能小于 torsion_logits 的第二维
        # 因为每个残基的片段数量可能大于扭转角数量
        batch_size, frag_seq_len, num_bins = torsion_logits.shape
        
        # 确保 torsion_targets 是二维的
        if torsion_targets.dim() == 1:
            # 如果是一维，假设是 [batch_size * seq_len] 格式
            torsion_targets = torsion_targets.view(batch_size, -1)
        
        tors_target_len = torsion_targets.shape[1]
        
        # 取较小的长度，确保对齐
        min_len = min(frag_seq_len, tors_target_len)
        
        if min_len > 0:
            # 只对前 min_len 个位置计算损失
            torsion_logits_trimmed = torsion_logits[:, :min_len, :]  # [batch_size, min_len, num_bins]
            torsion_targets_trimmed = torsion_targets[:, :min_len]   # [batch_size, min_len]

            # 扭转角有效位置掩码（避免把 padding 当作真实监督）
            if torsion_valid_mask is not None:
                torsion_valid_mask_trimmed = torsion_valid_mask[:, :min_len]
            else:
                torsion_valid_mask_trimmed = torch.ones_like(torsion_targets_trimmed, dtype=torch.bool)
            
            # 检查是否有无效值
            if torch.isnan(torsion_logits_trimmed).any() or torch.isinf(torsion_logits_trimmed).any():
                torsion_loss = torch.tensor(0.0, device=torsion_logits.device)
            elif (torsion_targets_trimmed < 0).any() or (torsion_targets_trimmed >= num_bins).any():
                # 目标值超出有效范围，过滤掉无效值 + padding位置
                valid_mask = (
                    (torsion_targets_trimmed >= 0)
                    & (torsion_targets_trimmed < num_bins)
                    & torsion_valid_mask_trimmed
                )
                if valid_mask.sum() > 0:
                    torsion_logits_valid = torsion_logits_trimmed[valid_mask]
                    torsion_targets_valid = torsion_targets_trimmed[valid_mask]
                    try:
                        torsion_loss = nn.functional.cross_entropy(
                            torsion_logits_valid.reshape(-1, num_bins),
                            torsion_targets_valid.reshape(-1),
                            label_smoothing=self.label_smoothing
                        )
                        if torch.isnan(torsion_loss) or torch.isinf(torsion_loss):
                            torsion_loss = torch.tensor(0.0, device=torsion_logits.device)
                    except Exception:
                        torsion_loss = torch.tensor(0.0, device=torsion_logits.device)
                else:
                    # 所有值都无效，损失为 0
                    torsion_loss = torch.tensor(0.0, device=torsion_logits.device)
            else:
                try:
                    # 仅在有效位置计算损失
                    valid_mask = torsion_valid_mask_trimmed
                    if valid_mask.sum() == 0:
                        torsion_loss = torch.tensor(0.0, device=torsion_logits.device)
                    else:
                        torsion_logits_valid = torsion_logits_trimmed[valid_mask]
                        torsion_targets_valid = torsion_targets_trimmed[valid_mask]
                        torsion_loss = nn.functional.cross_entropy(
                            torsion_logits_valid.reshape(-1, num_bins),
                            torsion_targets_valid.reshape(-1),
                            label_smoothing=self.label_smoothing
                        )
                        # 检查结果是否为 NaN
                        if torch.isnan(torsion_loss) or torch.isinf(torsion_loss):
                            torsion_loss = torch.tensor(0.0, device=torsion_logits.device)
                except Exception as e:
                    # 如果计算失败，返回 0 而不是 NaN
                    torsion_loss = torch.tensor(0.0, device=torsion_logits.device)
        else:
            # 如果没有有效的扭转角，损失为 0
            torsion_loss = torch.tensor(0.0, device=torsion_logits.device)
        
        # 3. 物理约束Loss：圆周几何一致性（预测分布 vs 真实扭转角）
        # 说明：不再对相邻 token 施加错误的“平滑”假设，而是直接约束
        # 预测扭转角分布的圆周期望角与 torsion_raw 在圆周空间的一致性。
        structure_loss = torch.tensor(0.0, device=fragment_logits.device)
        if torsion_raw is not None and torsion_logits is not None:
            try:
                batch_size, frag_seq_len, num_bins = torsion_logits.shape

                if torsion_raw.dim() == 1:
                    torsion_raw = torsion_raw.view(batch_size, -1)
                if torsion_targets.dim() == 1:
                    torsion_targets = torsion_targets.view(batch_size, -1)

                min_len = min(frag_seq_len, torsion_raw.shape[1], torsion_targets.shape[1])
                if min_len > 0:
                    torsion_logits_trimmed = torsion_logits[:, :min_len, :]  # [B, T, K]
                    torsion_raw_trimmed = torsion_raw[:, :min_len]  # [B, T]
                    torsion_targets_trimmed = torsion_targets[:, :min_len]  # [B, T]

                    if torsion_valid_mask is not None:
                        valid_mask = torsion_valid_mask[:, :min_len]
                    else:
                        valid_mask = torch.ones_like(torsion_targets_trimmed, dtype=torch.bool)

                    valid_mask = (
                        valid_mask
                        & (torsion_targets_trimmed >= 0)
                        & (torsion_targets_trimmed < num_bins)
                    )

                    if valid_mask.any():
                        probs = torch.softmax(torsion_logits_trimmed, dim=-1)  # [B, T, K]
                        bin_width = 2.0 * np.pi / num_bins
                        bin_centers = (
                            (torch.arange(num_bins, device=torsion_logits.device, dtype=torch.float32) + 0.5)
                            * bin_width - np.pi
                        )  # [K]

                        sin_expect = torch.sum(probs * torch.sin(bin_centers).view(1, 1, -1), dim=-1)  # [B, T]
                        cos_expect = torch.sum(probs * torch.cos(bin_centers).view(1, 1, -1), dim=-1)  # [B, T]
                        expected_angles = torch.atan2(sin_expect, cos_expect)  # [B, T]

                        circular_diff = torch.atan2(
                            torch.sin(expected_angles - torsion_raw_trimmed),
                            torch.cos(expected_angles - torsion_raw_trimmed),
                        )
                        loss_per_pos = circular_diff ** 2

                        valid_float = valid_mask.float()
                        denom = valid_float.sum().clamp(min=1.0)
                        structure_loss = (loss_per_pos * valid_float).sum() / denom

                        if torch.isnan(structure_loss) or torch.isinf(structure_loss):
                            structure_loss = torch.tensor(0.0, device=fragment_logits.device)
            except Exception:
                structure_loss = torch.tensor(0.0, device=fragment_logits.device)
        
        # 4. Offset Loss：真实的回归损失（Regression Loss）
        # 计算预测的 offset 与真实 offset 之间的 MSE Loss
        if offset_logits is not None and torsion_raw is not None:
            # 检查是否有无效值
            if torch.isnan(offset_logits).any() or torch.isinf(offset_logits).any():
                offset_loss = torch.tensor(0.0, device=fragment_logits.device)
            else:
                # 获取 batch 和序列维度信息
                batch_size, frag_seq_len, num_bins = torsion_logits.shape
                
                # 确保 torsion_targets 和 torsion_raw 的维度正确
                if torsion_targets.dim() == 1:
                    torsion_targets = torsion_targets.view(batch_size, -1)
                if torsion_raw.dim() == 1:
                    torsion_raw = torsion_raw.view(batch_size, -1)
                
                # 对齐长度（与 torsion_loss 计算保持一致）
                tors_target_len = torsion_targets.shape[1]
                tors_raw_len = torsion_raw.shape[1]
                min_len = min(frag_seq_len, tors_target_len, tors_raw_len)
                
                if min_len > 0:
                    # 截取有效长度
                    torsion_targets_trimmed = torsion_targets[:, :min_len]  # [batch_size, min_len]
                    torsion_raw_trimmed = torsion_raw[:, :min_len]  # [batch_size, min_len]
                    offset_logits_trimmed = offset_logits[:, :min_len, :]  # [batch_size, min_len, num_bins]

                    if torsion_valid_mask is not None:
                        torsion_valid_mask_trimmed = torsion_valid_mask[:, :min_len]
                    else:
                        torsion_valid_mask_trimmed = torch.ones_like(torsion_targets_trimmed, dtype=torch.bool)
                    
                    # 过滤无效的 torsion_targets（超出范围的值）
                    valid_mask = (
                        (torsion_targets_trimmed >= 0)
                        & (torsion_targets_trimmed < num_bins)
                        & torsion_valid_mask_trimmed
                    )
                    
                    if valid_mask.sum() > 0:
                        # 计算 bin 宽度和中心
                        bin_width = 2 * np.pi / num_bins
                        
                        # 将 bin 索引转换为 bin 中心角度（弧度）
                        # bin 中心 = (bin_idx + 0.5) * bin_width - π
                        bin_indices_float = torsion_targets_trimmed.float()  # [batch_size, min_len]
                        bin_centers = (bin_indices_float + 0.5) * bin_width - np.pi  # [batch_size, min_len]
                        
                        # 计算目标 offset：真实角度相对于 bin 中心的偏移（归一化到 bin 宽度）
                        # diff = (torsion_raw - bin_center) / bin_width
                        diff = (torsion_raw_trimmed - bin_centers) / bin_width  # [batch_size, min_len]
                        
                        # 周期性边界处理：确保 offset 在 [-0.5, 0.5] 范围内
                        # 使用公式: (diff + 0.5) % 1 - 0.5
                        target_offsets = (diff + 0.5) % 1.0 - 0.5  # [batch_size, min_len]
                        
                        # 从 offset_logits 中根据 torsion_targets 选择对应的 offset 预测值
                        # offset_logits: [batch_size, min_len, num_bins]
                        # torsion_targets_trimmed: [batch_size, min_len]
                        # 使用 gather 选择每个位置对应 bin 的 offset 预测
                        bin_indices = torsion_targets_trimmed.long().unsqueeze(-1)  # [batch_size, min_len, 1]
                        predicted_offsets = torch.gather(
                            offset_logits_trimmed, 
                            dim=2, 
                            index=bin_indices
                        ).squeeze(-1)  # [batch_size, min_len]
                        
                        # 计算 MSE Loss（只对有效位置）
                        loss_fn = nn.MSELoss(reduction='none')
                        loss_per_element = loss_fn(predicted_offsets, target_offsets)  # [batch_size, min_len]
                        
                        # 应用有效掩码
                        masked_loss = loss_per_element * valid_mask.float()  # [batch_size, min_len]
                        
                        # 计算平均损失（只对有效位置）
                        num_valid = valid_mask.sum().float()
                        if num_valid > 0:
                            offset_loss = 10.0 * masked_loss.sum() / num_valid
                        else:
                            offset_loss = torch.tensor(0.0, device=fragment_logits.device)
                        
                        # 检查结果是否为 NaN
                        if torch.isnan(offset_loss) or torch.isinf(offset_loss):
                            offset_loss = torch.tensor(0.0, device=fragment_logits.device)
                    else:
                        # 所有值都无效，损失为 0
                        offset_loss = torch.tensor(0.0, device=fragment_logits.device)
                else:
                    # 没有有效的扭转角，损失为 0
                    offset_loss = torch.tensor(0.0, device=fragment_logits.device)
        else:
            # 如果没有提供 torsion_raw，回退到 L2 正则化（保持向后兼容）
            if offset_logits is not None:
                if torch.isnan(offset_logits).any() or torch.isinf(offset_logits).any():
                    offset_loss = torch.tensor(0.0, device=fragment_logits.device)
                else:
                    # L2 正则化：惩罚大的 offset 值
                    offset_loss = 100.0 * torch.mean(offset_logits ** 2)
                    if torch.isnan(offset_loss) or torch.isinf(offset_loss):
                        offset_loss = torch.tensor(0.0, device=fragment_logits.device)
            else:
                offset_loss = torch.tensor(0.0, device=fragment_logits.device)
        
        # DDP 安全防护：当某个分支在当前 batch 没有有效监督时，
        # 上面可能返回“纯常数 0.0”损失，导致对应参数无梯度，触发
        # "Expected to have finished reduction..." 错误。
        # 这里将零损失与对应 logits 建立零系数连接，保证每轮都有参数参与反传。
        if not fragment_loss.requires_grad:
            fragment_loss = fragment_loss + fragment_logits.sum() * 0.0
        if not torsion_loss.requires_grad:
            torsion_loss = torsion_loss + torsion_logits.sum() * 0.0
        if not structure_loss.requires_grad:
            structure_loss = structure_loss + torsion_logits.sum() * 0.0
        if not offset_loss.requires_grad:
            if offset_logits is not None:
                offset_loss = offset_loss + offset_logits.sum() * 0.0
            else:
                offset_loss = offset_loss + torsion_logits.sum() * 0.0

        # 总损失（加权组合，缓解 Torsion Loss 过拟合）
        # - fragment_loss: 权重 1.0（主要损失）
        # - torsion_loss: 权重 0.5（降低权重，缓解过拟合）
        # - structure_loss: 权重 0.1（平滑性约束）
        # - offset_loss: 权重 1.0（正则化）
        total_loss = fragment_loss + 0.5 * torsion_loss + 0.1 * structure_loss + offset_loss
        
        return {
            'total_loss': total_loss,
            'fragment_loss': fragment_loss,
            'torsion_loss': torsion_loss,
            'structure_loss': structure_loss,
            'offset_loss': offset_loss
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            epoch: 当前 epoch 编号（从 1 开始）
        
        Returns:
            训练指标字典
        """
        # DDP 模式：在 epoch 开始时设置 sampler 的 epoch
        # 这确保每个 epoch 的数据顺序不同（如果 shuffle=True）
        if self.ddp_enabled and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0.0
        total_fragment_loss = 0.0
        total_torsion_loss = 0.0
        total_structure_loss = 0.0
        total_offset_loss = 0.0
        num_batches = 0
        
        batch_idx = 0
        total_batches_expected = len(self.train_loader)
        
        try:
            for batch in self.train_loader:
                # 移动到设备
                backbone_coords = batch['backbone_coords'].to(self.device)
                fragment_token_ids = batch['fragment_token_ids'].to(self.device)
                torsion_bins = batch['torsion_bins'].to(self.device)
                torsion_raw = batch.get('torsion_raw', None)
                if torsion_raw is not None:
                    torsion_raw = torsion_raw.to(self.device)
                sequence_lengths = batch['sequence_lengths'].to(self.device)
                fragment_lengths = batch.get('fragment_lengths', None)
                torsion_lengths = batch.get('torsion_lengths', None)
                torsion_valid_mask = batch.get('torsion_valid_mask', None)
                if fragment_lengths is not None:
                    fragment_lengths = fragment_lengths.to(self.device)
                if torsion_lengths is not None:
                    torsion_lengths = torsion_lengths.to(self.device)
                if torsion_valid_mask is not None:
                    torsion_valid_mask = torsion_valid_mask.to(self.device)

                # 有效位置掩码（避免 padding 污染监督信号）
                if fragment_lengths is not None:
                    fragment_valid_mask = torch.arange(
                        fragment_token_ids.shape[1], device=self.device
                    ).unsqueeze(0) < fragment_lengths.unsqueeze(1)
                else:
                    fragment_valid_mask = fragment_token_ids != int(SpecialTokens.PAD)

                # torsion 有效位置掩码：
                # 1) 优先使用数据集提供的精确掩码（新格式）
                # 2) 回退到旧逻辑（按长度左对齐）以兼容旧缓存
                if torsion_valid_mask is None:
                    if torsion_lengths is not None:
                        torsion_valid_mask = torch.arange(
                            torsion_bins.shape[1], device=self.device
                        ).unsqueeze(0) < torsion_lengths.unsqueeze(1)
                    else:
                        torsion_valid_mask = torch.ones_like(torsion_bins, dtype=torch.bool)
                
                # 创建掩码（离散扩散模式：采样时间步）
                # 为每个样本随机采样时间步 t ∈ [1, T]（归一化到 [0, 1]）
                batch_size = fragment_token_ids.shape[0]
                # 采样整数时间步 [1, T]
                timesteps = torch.randint(
                    1, self.num_diffusion_steps + 1,
                    size=(batch_size,),
                    device=self.device
                ).float() / self.num_diffusion_steps  # 归一化到 [0, 1]
                
                # 计算动态mask_ratio（用于监控）
                from training.masking import cosine_schedule
                dynamic_mask_ratios = cosine_schedule(timesteps)
                current_mask_ratio = dynamic_mask_ratios.mean().item()
                
                fragment_masks = create_masks(
                    fragment_token_ids,
                    strategy=self.masking_strategy,
                    timesteps=timesteps,
                    valid_mask=fragment_valid_mask,
                    use_cosine_schedule=True
                )
                
                # 应用掩码
                masked_fragments = apply_masks(fragment_token_ids, fragment_masks)
                
                # 前向传播
                self.optimizer.zero_grad()
                
                # Encoder（传递残基类型以支持理化特征）
                residue_types = batch.get('residue_types', None)
                node_embeddings = self.encoder(backbone_coords, mask=None, residue_types=residue_types)
                
                # Decoder
                frag_logits, tors_logits, offset_logits = self.decoder(
                    node_embeddings=node_embeddings,
                    target_fragments=masked_fragments,
                    sequence_lengths=sequence_lengths
                )
                
                # 计算损失（传入 offset_logits 和 torsion_raw 以计算真实的回归损失）
                losses = self.compute_loss(
                    frag_logits, tors_logits,
                    fragment_token_ids, torsion_bins,
                    fragment_mask=fragment_masks,
                    torsion_valid_mask=torsion_valid_mask,
                    offset_logits=offset_logits,
                    torsion_raw=torsion_raw
                )
                
                # 检查损失是否为 NaN/Inf（DDP 下必须全局一致地跳过，避免 rank 间步数不一致）
                local_invalid = bool(torch.isnan(losses['total_loss']) or torch.isinf(losses['total_loss']))
                if self.ddp_enabled:
                    invalid_tensor = torch.tensor(
                        [1 if local_invalid else 0],
                        device=self.device,
                        dtype=torch.int
                    )
                    dist.all_reduce(invalid_tensor, op=dist.ReduceOp.MAX)
                    global_invalid = int(invalid_tensor.item()) == 1
                else:
                    global_invalid = local_invalid

                if global_invalid:
                    if self.rank == 0:
                        print(f"  ⚠️  警告: 训练批次 {batch_idx} 存在 NaN/Inf，已全局同步跳过")
                    self.optimizer.zero_grad()
                    continue
                
                # 反向传播
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    max_norm=1.0
                )
                self.optimizer.step()
                
                # 更新指标
                total_loss += losses['total_loss'].item()
                total_fragment_loss += losses['fragment_loss'].item()
                total_torsion_loss += losses['torsion_loss'].item()
                total_structure_loss += losses.get('structure_loss', torch.tensor(0.0)).item()
                total_offset_loss += losses.get('offset_loss', torch.tensor(0.0)).item()
                num_batches += 1
                
                # 每 10 个 batch 打印一次（只在 rank 0 打印）
                batch_idx += 1
                if batch_idx % 10 == 0 and self.rank == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(
                        f"Batch {batch_idx}/{total_batches_expected} | "
                        f"loss={losses['total_loss'].item():.4f} | "
                        f"frag={losses['fragment_loss'].item():.4f} | "
                        f"tors={losses['torsion_loss'].item():.4f} | "
                        f"struct={losses.get('structure_loss', torch.tensor(0.0)).item():.4f} | "
                        f"offset={losses.get('offset_loss', torch.tensor(0.0)).item():.4f} | "
                        f"mask_ratio={current_mask_ratio:.3f} | "
                        f"lr={current_lr:.2e}"
                    )
        except Exception as e:
            if self.rank == 0:
                print(f"⚠️  错误: 训练迭代过程中出现异常: {e}")
                if self.debug_mode:
                    import traceback
                    traceback.print_exc()
            raise
        
        # DDP 模式：聚合所有进程的损失
        # 关键：所有 rank 必须无条件参与 all_reduce，避免 collective 不对齐导致死锁
        if self.ddp_enabled:
            stats_tensor = torch.tensor(
                [
                    total_loss,
                    total_fragment_loss,
                    total_torsion_loss,
                    total_structure_loss,
                    total_offset_loss,
                    float(num_batches)
                ],
                device=self.device,
                dtype=torch.float32
            )
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

            total_loss = stats_tensor[0].item()
            total_fragment_loss = stats_tensor[1].item()
            total_torsion_loss = stats_tensor[2].item()
            total_structure_loss = stats_tensor[3].item()
            total_offset_loss = stats_tensor[4].item()
            num_batches = int(stats_tensor[5].item())
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'fragment_loss': total_fragment_loss / num_batches if num_batches > 0 else 0.0,
            'torsion_loss': total_torsion_loss / num_batches if num_batches > 0 else 0.0,
            'structure_loss': total_structure_loss / num_batches if num_batches > 0 else 0.0,
            'offset_loss': total_offset_loss / num_batches if num_batches > 0 else 0.0
        }
    
    def validate(self) -> Dict[str, float]:
        """
        验证模型
        
        Returns:
            验证指标字典
        """
        if self.val_loader is None:
            return {}
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0.0
        total_fragment_loss = 0.0
        total_torsion_loss = 0.0
        total_structure_loss = 0.0
        total_offset_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            val_batch_idx = 0
            total_val_batches = len(self.val_loader)
            self._debug_print(f"开始验证，共 {total_val_batches} 个批次")
            for batch in self.val_loader:
                val_batch_idx += 1
                if val_batch_idx % 5 == 0:
                    self._debug_print(f"验证进度: {val_batch_idx}/{total_val_batches}")
                try:
                    backbone_coords = batch['backbone_coords'].to(self.device)
                    fragment_token_ids = batch['fragment_token_ids'].to(self.device)
                    torsion_bins = batch['torsion_bins'].to(self.device)
                    torsion_raw = batch.get('torsion_raw', None)
                    if torsion_raw is not None:
                        torsion_raw = torsion_raw.to(self.device)
                    sequence_lengths = batch['sequence_lengths'].to(self.device)
                    fragment_lengths = batch.get('fragment_lengths', None)
                    torsion_lengths = batch.get('torsion_lengths', None)
                    torsion_valid_mask = batch.get('torsion_valid_mask', None)
                    if fragment_lengths is not None:
                        fragment_lengths = fragment_lengths.to(self.device)
                    if torsion_lengths is not None:
                        torsion_lengths = torsion_lengths.to(self.device)
                    if torsion_valid_mask is not None:
                        torsion_valid_mask = torsion_valid_mask.to(self.device)

                    # 有效位置掩码（避免 padding 污染监督信号）
                    if fragment_lengths is not None:
                        fragment_valid_mask = torch.arange(
                            fragment_token_ids.shape[1], device=self.device
                        ).unsqueeze(0) < fragment_lengths.unsqueeze(1)
                    else:
                        fragment_valid_mask = fragment_token_ids != int(SpecialTokens.PAD)

                    # torsion 有效位置掩码：
                    # 1) 优先使用数据集提供的精确掩码（新格式）
                    # 2) 回退到旧逻辑（按长度左对齐）以兼容旧缓存
                    if torsion_valid_mask is None:
                        if torsion_lengths is not None:
                            torsion_valid_mask = torch.arange(
                                torsion_bins.shape[1], device=self.device
                            ).unsqueeze(0) < torsion_lengths.unsqueeze(1)
                        else:
                            torsion_valid_mask = torch.ones_like(torsion_bins, dtype=torch.bool)
                    
                    # 检查输入数据
                    if torch.isnan(backbone_coords).any() or torch.isinf(backbone_coords).any():
                        if self.rank == 0:
                            print(f"  ⚠️  警告: backbone_coords 包含 NaN/Inf，跳过此批次")
                        continue
                    
                    # 创建掩码（验证时也使用离散扩散，采样时间步）
                    # 为每个样本随机采样时间步 t ∈ [1, T]（归一化到 [0, 1]）
                    batch_size = fragment_token_ids.shape[0]
                    timesteps = torch.randint(
                        1, self.num_diffusion_steps + 1,
                        size=(batch_size,),
                        device=self.device
                    ).float() / self.num_diffusion_steps
                    
                    fragment_masks = create_masks(
                        fragment_token_ids,
                        strategy=self.masking_strategy,
                        timesteps=timesteps,
                        valid_mask=fragment_valid_mask,
                        use_cosine_schedule=True
                    )
                    
                    # 应用掩码
                    masked_fragments = apply_masks(fragment_token_ids, fragment_masks)
                    
                    # 前向传播（传递残基类型以支持理化特征）
                    try:
                        residue_types = batch.get('residue_types', None)
                        node_embeddings = self.encoder(backbone_coords, mask=None, residue_types=residue_types)
                        if torch.isnan(node_embeddings).any() or torch.isinf(node_embeddings).any():
                            if self.rank == 0:
                                print(f"  ⚠️  警告: encoder 输出包含 NaN/Inf，跳过此批次")
                            continue
                        
                        try:
                            frag_logits, tors_logits, offset_logits = self.decoder(
                                node_embeddings=node_embeddings,
                                target_fragments=masked_fragments,
                                sequence_lengths=sequence_lengths
                            )
                        except Exception as e:
                            if isinstance(e, torch.cuda.OutOfMemoryError):
                                raise
                            if self.rank == 0:
                                print(f"  ⚠️  警告: decoder 前向传播失败: {e}，跳过此批次")
                                if self.debug_mode:
                                    import traceback
                                    traceback.print_exc()
                            continue
                        
                        if torch.isnan(frag_logits).any() or torch.isinf(frag_logits).any():
                            if self.rank == 0:
                                print(f"  ⚠️  警告: fragment_logits 包含 NaN/Inf，跳过此批次")
                                if self.debug_mode:
                                    print(f"      NaN 数量: {torch.isnan(frag_logits).sum().item()}, Inf 数量: {torch.isinf(frag_logits).sum().item()}")
                                    print(f"      Fragment logits 统计: min={frag_logits.min().item():.4f}, max={frag_logits.max().item():.4f}, mean={frag_logits.mean().item():.4f}")
                                    print(f"      Node embeddings 统计: min={node_embeddings.min().item():.4f}, max={node_embeddings.max().item():.4f}, mean={node_embeddings.mean().item():.4f}")
                                    print(f"      Masked fragments 范围: min={masked_fragments.min().item()}, max={masked_fragments.max().item()}")
                            continue
                        if torch.isnan(tors_logits).any() or torch.isinf(tors_logits).any():
                            if self.rank == 0:
                                print(f"  ⚠️  警告: torsion_logits 包含 NaN/Inf，跳过此批次")
                            continue
                    except Exception as e:
                        if isinstance(e, torch.cuda.OutOfMemoryError):
                            raise
                        if self.rank == 0:
                            print(f"  ⚠️  警告: 前向传播失败: {e}，跳过此批次")
                        continue
                    
                    # 计算损失（传入 offset_logits 和 torsion_raw 以计算真实的回归损失）
                    losses = self.compute_loss(
                        frag_logits, tors_logits,
                        fragment_token_ids, torsion_bins,
                        fragment_mask=fragment_masks,
                        torsion_valid_mask=torsion_valid_mask,
                        offset_logits=offset_logits,
                        torsion_raw=torsion_raw
                    )
                    
                    # 检查损失是否为 NaN 或 Inf
                    if torch.isnan(losses['total_loss']) or torch.isinf(losses['total_loss']):
                        if self.rank == 0:
                            print(f"  ⚠️  警告: 批次 {num_batches} 的损失为 NaN/Inf，跳过")
                            if self.debug_mode:
                                print(f"      Fragment loss: {losses['fragment_loss']}")
                                print(f"      Torsion loss: {losses['torsion_loss']}")
                                print(f"      Fragment logits shape: {frag_logits.shape}, contains NaN: {torch.isnan(frag_logits).any()}")
                                print(f"      Torsion logits shape: {tors_logits.shape}, contains NaN: {torch.isnan(tors_logits).any()}")
                                print(f"      Fragment targets shape: {fragment_token_ids.shape}, min: {fragment_token_ids.min()}, max: {fragment_token_ids.max()}")
                                print(f"      Torsion targets shape: {torsion_bins.shape}, min: {torsion_bins.min()}, max: {torsion_bins.max()}")
                        continue
                    
                    total_loss += losses['total_loss'].item()
                    total_fragment_loss += losses['fragment_loss'].item()
                    total_torsion_loss += losses['torsion_loss'].item()
                    total_structure_loss += losses.get('structure_loss', torch.tensor(0.0)).item()
                    total_offset_loss += losses.get('offset_loss', torch.tensor(0.0)).item()
                    num_batches += 1
                except torch.cuda.OutOfMemoryError as e:
                    if self.rank == 0:
                        print(f"  ⚠️  验证 OOM: {e}，已清理显存并跳过该 Batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
        
        # DDP 模式：聚合所有进程的验证损失
        # 关键：所有 rank 必须无条件参与 all_reduce，避免 collective 不对齐导致死锁
        if self.ddp_enabled:
            stats_tensor = torch.tensor(
                [
                    total_loss,
                    total_fragment_loss,
                    total_torsion_loss,
                    total_structure_loss,
                    total_offset_loss,
                    float(num_batches)
                ],
                device=self.device,
                dtype=torch.float32
            )
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

            total_loss = stats_tensor[0].item()
            total_fragment_loss = stats_tensor[1].item()
            total_torsion_loss = stats_tensor[2].item()
            total_structure_loss = stats_tensor[3].item()
            total_offset_loss = stats_tensor[4].item()
            num_batches = int(stats_tensor[5].item())
        
        # 如果没有批次，返回空字典
        if num_batches == 0:
            return {}
        
        # 检查平均值是否为 NaN
        avg_loss = total_loss / num_batches
        avg_frag_loss = total_fragment_loss / num_batches
        avg_tors_loss = total_torsion_loss / num_batches
        
        # 如果平均值是 NaN，返回空字典（表示验证失败）
        if (avg_loss != avg_loss) or (avg_frag_loss != avg_frag_loss) or (avg_tors_loss != avg_tors_loss):
            if self.rank == 0:
                print(f"  ⚠️  警告: 验证损失为 NaN，可能是数据问题")
            return {}
        
        avg_struct_loss = total_structure_loss / num_batches if num_batches > 0 else 0.0
        avg_offset_loss = total_offset_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'fragment_loss': avg_frag_loss,
            'torsion_loss': avg_tors_loss,
            'structure_loss': avg_struct_loss,
            'offset_loss': avg_offset_loss
        }
    
    def train(
        self,
        num_epochs: int,
        save_dir: Optional[str] = None,
        save_every: int = 10,
        visualize: bool = True,
        plot_every: int = 5,
        resume_from: Optional[str] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0
    ):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            save_dir: 保存目录
            save_every: 每 N 个 epoch 保存一次
            visualize: 是否启用可视化
            plot_every: 每 N 个 epoch 绘制一次图表
            early_stopping_patience: 早停耐心值，如果验证损失连续N轮不下降则停止训练（None表示禁用）
            early_stopping_min_delta: 早停最小改进阈值，只有改进超过此值才认为是有效改进
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 如果指定了恢复训练，加载checkpoint
        # 关键：所有 rank 都必须参与 checkpoint 加载或同步，不能只在 rank 0 执行
        start_epoch = 1
        if resume_from is not None:
            # 所有 rank 都尝试加载 checkpoint（load_checkpoint 内部会处理 rank 0 的打印）
            if self.rank == 0:
                print(f"\n从检查点恢复训练: {resume_from}")
            start_epoch = self.load_checkpoint(resume_from)
            if self.rank == 0:
                print(f"✅ 已加载检查点，从 epoch {start_epoch} 继续训练")
                print(f"   训练历史: {len(self.train_losses)} 个epoch")
                if self.val_loader is not None:
                    print(f"   验证历史: {len(self.val_losses)} 个epoch")
        
        # DDP 模式：强制同步 start_epoch 到所有进程
        # 关键：所有 rank 都必须参与 broadcast，不能只在 rank 0 执行
        if self.ddp_enabled:
            # 确保 tensor 在正确的设备上，所有 rank 都创建 tensor
            epoch_tensor = torch.tensor([start_epoch], dtype=torch.long, device=self.device)
            # 所有 rank 都必须调用 broadcast，src=0 表示从 rank 0 广播到所有 rank
            dist.broadcast(epoch_tensor, src=0)
            start_epoch = int(epoch_tensor.item())
        
        # 早停相关变量
        patience_counter = 0
        best_val_loss_for_patience = float('inf')
        
        if self.rank == 0:
            print(f"开始训练，设备: {self.device}")
            if self.ddp_enabled:
                print(f"DDP 模式: 启用 (world_size={self.world_size})")
            print(f"扩散模型: 启用 (时间步数: {self.num_diffusion_steps})")
            print(f"掩码策略: {self.masking_strategy}, 掩码比例: 动态 (Cosine Schedule)")
            print(f"学习率调度: LinearWarmup ({self.warmup_epochs} epochs) + CosineAnnealing")
            print(f"最大学习率: {self.max_lr:.2e}")
            if early_stopping_patience is not None:
                print(f"早停: 启用 (patience={early_stopping_patience}, min_delta={early_stopping_min_delta:.6f})")
            if resume_from:
                print(f"恢复训练: 从 epoch {start_epoch} 继续，目标 {num_epochs} epochs")
        
        for epoch in range(start_epoch, num_epochs + 1):
            should_stop = False

            # 强制清理内存和显存，防止内存泄漏
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.current_epoch = epoch
            
            if self.rank == 0:
                print(f"\nEpoch {epoch}/{num_epochs}")
                print("-" * 60)
            
            # 训练（传递 epoch 编号）
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics)
            
            if self.rank == 0:
                print(f"Train Loss: {train_metrics['loss']:.4f} "
                      f"(Fragment: {train_metrics['fragment_loss']:.4f}, "
                      f"Torsion: {train_metrics['torsion_loss']:.4f}, "
                      f"Structure: {train_metrics.get('structure_loss', 0.0):.4f}, "
                      f"Offset: {train_metrics.get('offset_loss', 0.0):.4f})")
            
            # 验证
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.val_losses.append(val_metrics)
                
                # 只有当验证指标不为空时才处理
                if val_metrics:
                    if self.rank == 0:
                        print(f"Val Loss: {val_metrics['loss']:.4f} "
                              f"(Fragment: {val_metrics['fragment_loss']:.4f}, "
                              f"Torsion: {val_metrics['torsion_loss']:.4f}, "
                              f"Structure: {val_metrics.get('structure_loss', 0.0):.4f}, "
                              f"Offset: {val_metrics.get('offset_loss', 0.0):.4f})")
                    
                    # 更新学习率（使用WarmupCosineAnnealing调度器）
                    self._update_learning_rate(epoch)
                    
                    # 保存最佳模型（只在 rank 0 保存）
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        if save_dir and self.rank == 0:
                            self.save_checkpoint(
                                os.path.join(save_dir, 'best_model.pt'),
                                epoch, val_metrics['loss']
                            )
                            print(f"✅ 保存最佳模型 (val_loss: {val_metrics['loss']:.4f})")
                    
                    # 早停检查（仅由 rank 0 决策）
                    if early_stopping_patience is not None:
                        if self.rank == 0:
                            # 检查是否有显著改进（超过 min_delta）
                            improvement = best_val_loss_for_patience - val_metrics['loss']
                            if improvement > early_stopping_min_delta:
                                # 有显著改进，重置计数器
                                best_val_loss_for_patience = val_metrics['loss']
                                patience_counter = 0
                                print(f"   早停计数器重置 (改进: {improvement:.6f})")
                            else:
                                # 没有显著改进，增加计数器
                                patience_counter += 1
                                print(f"   早停计数器: {patience_counter}/{early_stopping_patience} (改进: {improvement:.6f})")

                                # 如果达到耐心值，触发停止（由 rank 0 决策，稍后广播）
                                if patience_counter >= early_stopping_patience:
                                    print(f"\n{'='*60}")
                                    print(f"⚠️  早停触发: 验证损失连续 {early_stopping_patience} 轮未改进")
                                    print(f"   最佳验证损失: {best_val_loss_for_patience:.4f} (Epoch {epoch - patience_counter})")
                                    print(f"   当前验证损失: {val_metrics['loss']:.4f}")
                                    print(f"{'='*60}\n")
                                    should_stop = True
                else:
                    if self.rank == 0:
                        print("⚠️  验证集为空，跳过验证")
            
            # 定期保存（只在 rank 0 保存）
            if save_dir and epoch % save_every == 0 and self.rank == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'),
                    epoch, train_metrics['loss']
                )
            
            # 定期可视化（只在 rank 0 执行）
            if visualize and VISUALIZATION_AVAILABLE and save_dir and epoch % plot_every == 0 and self.rank == 0:
                try:
                    plot_path = os.path.join(save_dir, f'training_curves_epoch_{epoch}.png')
                    plot_training_curves(
                        self.train_losses,
                        self.val_losses if self.val_loader is not None else None,
                        save_path=plot_path
                    )
                except Exception as e:
                    print(f"Warning: Failed to plot training curves: {e}")
            elif visualize and not VISUALIZATION_AVAILABLE:
                if epoch == 1 and self.rank == 0:
                    print("   ⚠️  可视化已请求但 matplotlib 未安装，跳过绘图")
            
            # DDP 模式：由 rank 0 统一决策是否停止，并广播到所有 rank
            if self.ddp_enabled:
                stop_tensor = torch.tensor(
                    [1 if (self.rank == 0 and should_stop) else 0],
                    dtype=torch.int,
                    device=self.device
                )
                dist.broadcast(stop_tensor, src=0)
                if int(stop_tensor.item()) == 1:
                    break
            elif should_stop:
                break
        
        # 训练结束后生成最终可视化（只在 rank 0 执行）
        if visualize and VISUALIZATION_AVAILABLE and save_dir and self.rank == 0:
            try:
                final_plot_path = os.path.join(save_dir, 'training_curves_final.png')
                plot_training_curves(
                    self.train_losses,
                    self.val_losses if self.val_loader is not None else None,
                    save_path=final_plot_path
                )
                # 同时生成简化的对比图
                simple_plot_path = os.path.join(save_dir, 'loss_comparison.png')
                plot_loss_comparison(
                    self.train_losses,
                    self.val_losses if self.val_loader is not None else None,
                    save_path=simple_plot_path
                )
            except Exception as e:
                print(f"Warning: Failed to plot final training curves: {e}")
        
        if self.rank == 0:
            print("\n" + "=" * 60)
            print("训练完成！")
            if visualize and save_dir:
                print(f"可视化图表已保存到: {save_dir}")
            print("=" * 60)
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        loss: float
    ):
        """保存检查点（只在 rank 0 执行）"""
        # 如果模型被 DDP 包装，需要移除 module. 前缀
        if isinstance(self.encoder, torch.nn.parallel.DistributedDataParallel):
            encoder_state_dict = self.encoder.module.state_dict()
        else:
            encoder_state_dict = self.encoder.state_dict()
        
        if isinstance(self.decoder, torch.nn.parallel.DistributedDataParallel):
            decoder_state_dict = self.decoder.module.state_dict()
        else:
            decoder_state_dict = self.decoder.state_dict()
        
        checkpoint_data = {
            'epoch': epoch,
            'encoder_state_dict': encoder_state_dict,
            'decoder_state_dict': decoder_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'max_lr': self.max_lr
        }
        # 保存cosine scheduler状态（如果存在）
        if hasattr(self, 'cosine_scheduler'):
            checkpoint_data['cosine_scheduler_state_dict'] = self.cosine_scheduler.state_dict()
        torch.save(checkpoint_data, path)
    
    def _update_learning_rate(self, epoch: int):
        """
        更新学习率：LinearWarmup + CosineAnnealing
        
        Args:
            epoch: 当前epoch（从1开始）
        """
        if epoch <= self.warmup_epochs:
            # Warmup阶段：线性增长
            warmup_lr = self.max_lr * (epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            # CosineAnnealing阶段
            # 注意：CosineAnnealingLR的step需要在warmup之后调用
            if epoch == self.warmup_epochs + 1:
                # 第一次进入cosine阶段，重置scheduler
                self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.total_epochs - self.warmup_epochs,
                    eta_min=1e-6
                )
            self.cosine_scheduler.step()
    
    def load_checkpoint(self, path: str) -> int:
        """
        加载检查点并恢复训练状态
        
        Args:
            path: 检查点文件路径
        
        Returns:
            下一个epoch编号（从checkpoint的epoch+1开始）
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"检查点不存在: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        
        # 加载模型权重
        encoder_state_dict = checkpoint['encoder_state_dict']
        decoder_state_dict = checkpoint['decoder_state_dict']
        
        # 处理可能的module.前缀（兼容从DDP模型保存的checkpoint）
        def remove_module_prefix(state_dict):
            """移除state_dict中的module.前缀"""
            if not state_dict:
                return state_dict
            first_key = next(iter(state_dict.keys()))
            if first_key.startswith('module.'):
                return {k.replace('module.', ''): v for k, v in state_dict.items()}
            return state_dict
        
        encoder_state_dict = remove_module_prefix(encoder_state_dict)
        decoder_state_dict = remove_module_prefix(decoder_state_dict)
        
        # DDP场景下应加载到底层module，避免key前缀(module.)不一致
        encoder_model = self.encoder.module if isinstance(self.encoder, torch.nn.parallel.DistributedDataParallel) else self.encoder
        decoder_model = self.decoder.module if isinstance(self.decoder, torch.nn.parallel.DistributedDataParallel) else self.decoder

        # 恢复训练时必须严格匹配，避免静默部分加载导致loss异常升高
        # 这里先用strict=False收集不匹配信息，再人工判定并抛错
        encoder_incompat = encoder_model.load_state_dict(encoder_state_dict, strict=False)
        decoder_incompat = decoder_model.load_state_dict(decoder_state_dict, strict=False)

        missing_keys = list(encoder_incompat.missing_keys) + list(decoder_incompat.missing_keys)
        unexpected_keys = list(encoder_incompat.unexpected_keys) + list(decoder_incompat.unexpected_keys)
        if missing_keys or unexpected_keys:
            msg = [
                "恢复训练失败：检查点与当前模型结构不一致（检测到参数缺失/多余）。",
                f"missing_keys 数量: {len(missing_keys)}",
                f"unexpected_keys 数量: {len(unexpected_keys)}"
            ]
            if self.rank == 0:
                preview_missing = missing_keys[:10]
                preview_unexpected = unexpected_keys[:10]
                if preview_missing:
                    msg.append(f"missing_keys 示例: {preview_missing}")
                if preview_unexpected:
                    msg.append(f"unexpected_keys 示例: {preview_unexpected}")
            raise RuntimeError("\n".join(msg))
        
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            if self.rank == 0:
                print("   ⚠️  检查点不包含 optimizer_state_dict，将使用新优化器状态继续训练")
        
        # 加载学习率调度器状态
        if 'cosine_scheduler_state_dict' in checkpoint and hasattr(self, 'cosine_scheduler'):
            self.cosine_scheduler.load_state_dict(checkpoint['cosine_scheduler_state_dict'])
        elif 'scheduler_state_dict' in checkpoint:
            # 兼容旧的检查点格式
            if hasattr(self, 'scheduler'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练历史（用于继续绘制曲线）
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        # 恢复最佳验证损失
        # 注意：checkpoint['loss']是训练损失，不是验证损失
        # 应该从val_losses历史中找到最佳验证损失
        if 'val_losses' in checkpoint and checkpoint['val_losses']:
            val_losses = checkpoint['val_losses']
            # 提取所有有效的验证损失值
            val_loss_values = []
            for v in val_losses:
                if v:  # 跳过空字典
                    if isinstance(v, dict):
                        val_loss_values.append(v.get('loss', float('inf')))
                    else:
                        val_loss_values.append(v)
            # 找到最佳验证损失
            if val_loss_values:
                self.best_val_loss = min(val_loss_values)
                if self.rank == 0:
                    print(f"   从历史中恢复最佳验证损失: {self.best_val_loss:.4f}")
            else:
                # 如果没有有效的验证损失，使用checkpoint中的loss（可能是训练损失）
                if 'loss' in checkpoint:
                    self.best_val_loss = checkpoint['loss']
                    if self.rank == 0:
                        print(f"   ⚠️  警告: 使用checkpoint中的loss作为best_val_loss: {self.best_val_loss:.4f}")
        elif 'loss' in checkpoint:
            # 如果没有验证损失历史，使用checkpoint中的loss
            self.best_val_loss = checkpoint['loss']
            if self.rank == 0:
                print(f"   ⚠️  警告: 没有验证损失历史，使用checkpoint中的loss: {self.best_val_loss:.4f}")
        
        # 恢复训练参数（如果存在）
        if 'warmup_epochs' in checkpoint:
            self.warmup_epochs = checkpoint['warmup_epochs']
        if 'total_epochs' in checkpoint:
            # 注意：这里不覆盖，因为用户可能想训练更多epochs
            pass
        if 'max_lr' in checkpoint:
            self.max_lr = checkpoint['max_lr']
        
        # 返回下一个epoch编号
        current_epoch = checkpoint.get('epoch', 0)
        self.current_epoch = int(current_epoch)

        if self.rank == 0:
            lr_values = [pg.get('lr', None) for pg in self.optimizer.param_groups]
            lr_values = [v for v in lr_values if v is not None]
            if lr_values:
                print(f"   恢复后优化器学习率: {[f'{v:.3e}' for v in lr_values]}")
            print(f"   检查点epoch: {current_epoch}, 下一epoch: {current_epoch + 1}")

        return current_epoch + 1