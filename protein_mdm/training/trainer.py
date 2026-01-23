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
from tqdm import tqdm

from models.encoder import BackboneEncoder
from models.decoder import FragmentDecoder
from data.vocabulary import get_vocab
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
        mask_ratio: float = 0.15,
        masking_strategy: str = "random",
        ddp_enabled: bool = False,
        rank: int = 0,
        use_discrete_diffusion: bool = True,
        num_diffusion_steps: int = 1000,
        warmup_epochs: int = 20,
        total_epochs: int = 300
    ):
        """
        初始化训练器
        
        Args:
            encoder: BackboneEncoder 实例（可能是 DDP 包装的）
            decoder: FragmentDecoder 实例（可能是 DDP 包装的）
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            device: 设备（默认：自动检测）
            learning_rate: 学习率
            weight_decay: 权重衰减
            mask_ratio: 掩码比例
            masking_strategy: 掩码策略
            ddp_enabled: 是否启用 DDP
            rank: 当前进程的 rank（用于打印）
        """
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mask_ratio = mask_ratio
        self.masking_strategy = masking_strategy
        self.ddp_enabled = ddp_enabled
        self.rank = rank
        self.use_discrete_diffusion = use_discrete_diffusion
        self.num_diffusion_steps = num_diffusion_steps
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_lr = learning_rate
        
        # 检查是否是 DDP 模型
        self.is_ddp_encoder = hasattr(encoder, 'module')
        self.is_ddp_decoder = hasattr(decoder, 'module')
        
        # 优化器（DDP 模型会自动处理参数收集）
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
    
    def compute_loss(
        self,
        fragment_logits: torch.Tensor,
        torsion_logits: torch.Tensor,
        fragment_targets: torch.Tensor,
        torsion_targets: torch.Tensor,
        fragment_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            fragment_logits: 片段预测 logits [batch_size, M, vocab_size]
            torsion_logits: 扭转角预测 logits [batch_size, M, num_bins]
            fragment_targets: 片段目标 [batch_size, M]
            torsion_targets: 扭转角目标 [batch_size, M]
            fragment_mask: 片段掩码 [batch_size, M]
        
        Returns:
            损失字典
        """
        # 片段损失（交叉熵）
        if fragment_mask is not None:
            # 只计算掩码位置的损失
            masked_logits = fragment_logits[fragment_mask]
            masked_targets = fragment_targets[fragment_mask]
            if len(masked_logits) > 0:
                # 检查是否有无效值
                if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                    fragment_loss = torch.tensor(float('nan'), device=fragment_logits.device)
                else:
                    try:
                        fragment_loss = nn.functional.cross_entropy(
                            masked_logits.view(-1, fragment_logits.shape[-1]),
                            masked_targets.view(-1)
                        )
                        # 检查结果是否为 NaN
                        if torch.isnan(fragment_loss) or torch.isinf(fragment_loss):
                            fragment_loss = torch.tensor(0.0, device=fragment_logits.device)
                    except Exception:
                        fragment_loss = torch.tensor(0.0, device=fragment_logits.device)
            else:
                fragment_loss = torch.tensor(0.0, device=fragment_logits.device)
        else:
            # 检查是否有无效值
            if torch.isnan(fragment_logits).any() or torch.isinf(fragment_logits).any():
                fragment_loss = torch.tensor(float('nan'), device=fragment_logits.device)
            else:
                try:
                    fragment_loss = nn.functional.cross_entropy(
                        fragment_logits.reshape(-1, fragment_logits.shape[-1]),
                        fragment_targets.reshape(-1)
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
            
            # 检查是否有无效值
            if torch.isnan(torsion_logits_trimmed).any() or torch.isinf(torsion_logits_trimmed).any():
                torsion_loss = torch.tensor(0.0, device=torsion_logits.device)
            elif (torsion_targets_trimmed < 0).any() or (torsion_targets_trimmed >= num_bins).any():
                # 目标值超出有效范围，过滤掉无效值
                valid_mask = (torsion_targets_trimmed >= 0) & (torsion_targets_trimmed < num_bins)
                if valid_mask.sum() > 0:
                    torsion_logits_valid = torsion_logits_trimmed[valid_mask]
                    torsion_targets_valid = torsion_targets_trimmed[valid_mask]
                    try:
                        torsion_loss = nn.functional.cross_entropy(
                            torsion_logits_valid.reshape(-1, num_bins),
                            torsion_targets_valid.reshape(-1)
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
                    torsion_loss = nn.functional.cross_entropy(
                        torsion_logits_trimmed.reshape(-1, num_bins),  # [batch_size * min_len, num_bins]
                        torsion_targets_trimmed.reshape(-1)            # [batch_size * min_len]
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
        
        # 3. 物理约束Loss：扭转角平滑性约束
        # 计算相邻扭转角之间的差异，惩罚剧烈变化
        structure_loss = torch.tensor(0.0, device=fragment_logits.device)
        if torsion_logits.shape[1] > 1:
            # 获取预测的扭转角（使用argmax或期望值）
            # 为了可微，我们使用softmax后的期望值
            torsion_probs = torch.softmax(torsion_logits, dim=-1)  # [batch_size, M, num_bins]
            num_bins = torsion_logits.shape[-1]
            # 将bin索引转换为角度（弧度）
            bin_centers = torch.linspace(0, 2 * np.pi, num_bins, device=torsion_logits.device)
            # 计算期望角度
            expected_angles = torch.sum(torsion_probs * bin_centers.unsqueeze(0).unsqueeze(0), dim=-1)  # [batch_size, M]
            
            # 计算相邻角度之间的差异
            angle_diff = expected_angles[:, 1:] - expected_angles[:, :-1]  # [batch_size, M-1]
            # 将角度差归一化到 [-π, π] 范围
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            # 平滑性约束：惩罚大的角度变化（L2正则化）
            structure_loss = torch.mean(angle_diff ** 2)
        
        # 总损失（添加物理约束项，权重可调）
        physical_constraint_weight = 0.1  # 物理约束权重
        total_loss = fragment_loss + torsion_loss + physical_constraint_weight * structure_loss
        
        return {
            'total_loss': total_loss,
            'fragment_loss': fragment_loss,
            'torsion_loss': torsion_loss,
            'structure_loss': structure_loss
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Returns:
            训练指标字典
        """
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0.0
        total_fragment_loss = 0.0
        total_torsion_loss = 0.0
        total_structure_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # 移动到设备
            backbone_coords = batch['backbone_coords'].to(self.device)
            fragment_token_ids = batch['fragment_token_ids'].to(self.device)
            torsion_bins = batch['torsion_bins'].to(self.device)
            sequence_lengths = batch['sequence_lengths'].to(self.device)
            
            # 创建掩码（离散扩散模式：采样时间步）
            current_mask_ratio = self.mask_ratio  # 用于监控
            if self.use_discrete_diffusion:
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
                    mask_ratio=self.mask_ratio,  # 作为默认值，会被覆盖
                    timesteps=timesteps,
                    use_cosine_schedule=True
                )
            else:
                # 传统固定mask_ratio模式（向后兼容）
                fragment_masks = create_masks(
                    fragment_token_ids,
                    strategy=self.masking_strategy,
                    mask_ratio=self.mask_ratio
                )
            
            # 应用掩码
            masked_fragments = apply_masks(fragment_token_ids, fragment_masks)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # Encoder（传递残基类型以支持理化特征）
            residue_types = batch.get('residue_types', None)
            node_embeddings = self.encoder(backbone_coords, mask=None, residue_types=residue_types)
            
            # Decoder
            frag_logits, tors_logits = self.decoder(
                node_embeddings=node_embeddings,
                target_fragments=masked_fragments,
                sequence_lengths=sequence_lengths
            )
            
            # 计算损失
            losses = self.compute_loss(
                frag_logits, tors_logits,
                fragment_token_ids, torsion_bins,
                fragment_mask=fragment_masks
            )
            
            # 检查损失是否为 NaN 或 Inf
            if torch.isnan(losses['total_loss']) or torch.isinf(losses['total_loss']):
                if self.rank == 0:
                    print(f"  ⚠️  警告: 训练批次 {num_batches} 的损失为 NaN/Inf，跳过")
                self.optimizer.zero_grad()  # 清除梯度
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
            num_batches += 1
            
            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'frag': f"{losses['fragment_loss'].item():.4f}",
                'tors': f"{losses['torsion_loss'].item():.4f}",
                'struct': f"{losses.get('structure_loss', torch.tensor(0.0)).item():.4f}",
                'mask_ratio': f"{current_mask_ratio:.3f}",
                'lr': f"{current_lr:.2e}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'fragment_loss': total_fragment_loss / num_batches,
            'torsion_loss': total_torsion_loss / num_batches,
            'structure_loss': total_structure_loss / num_batches
        }
    
    def validate(self) -> Dict[str, float]:
        """
        验证模型
        
        Returns:
            验证指标字典
        """
        if self.val_loader is None:
            return {}
        
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0.0
        total_fragment_loss = 0.0
        total_torsion_loss = 0.0
        total_structure_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                backbone_coords = batch['backbone_coords'].to(self.device)
                fragment_token_ids = batch['fragment_token_ids'].to(self.device)
                torsion_bins = batch['torsion_bins'].to(self.device)
                sequence_lengths = batch['sequence_lengths'].to(self.device)
                
                # 检查输入数据
                if torch.isnan(backbone_coords).any() or torch.isinf(backbone_coords).any():
                    if self.rank == 0:
                        print(f"  ⚠️  警告: backbone_coords 包含 NaN/Inf，跳过此批次")
                    continue
                
                # 创建掩码（验证时使用固定mask_ratio，不采样时间步）
                fragment_masks = create_masks(
                    fragment_token_ids,
                    strategy=self.masking_strategy,
                    mask_ratio=self.mask_ratio
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
                        frag_logits, tors_logits = self.decoder(
                            node_embeddings=node_embeddings,
                            target_fragments=masked_fragments,
                            sequence_lengths=sequence_lengths
                        )
                    except Exception as e:
                        if self.rank == 0:
                            print(f"  ⚠️  警告: decoder 前向传播失败: {e}，跳过此批次")
                            import traceback
                            traceback.print_exc()
                        continue
                    
                    if torch.isnan(frag_logits).any() or torch.isinf(frag_logits).any():
                        if self.rank == 0:
                            print(f"  ⚠️  警告: fragment_logits 包含 NaN/Inf，跳过此批次")
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
                    if self.rank == 0:
                        print(f"  ⚠️  警告: 前向传播失败: {e}，跳过此批次")
                    continue
                
                # 计算损失
                losses = self.compute_loss(
                    frag_logits, tors_logits,
                    fragment_token_ids, torsion_bins,
                    fragment_mask=fragment_masks
                )
                
                # 检查损失是否为 NaN 或 Inf
                if torch.isnan(losses['total_loss']) or torch.isinf(losses['total_loss']):
                    if self.rank == 0:
                        print(f"  ⚠️  警告: 批次 {num_batches} 的损失为 NaN/Inf，跳过")
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
                num_batches += 1
        
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
        
        return {
            'loss': avg_loss,
            'fragment_loss': avg_frag_loss,
            'torsion_loss': avg_tors_loss,
            'structure_loss': avg_struct_loss
        }
    
    def train(
        self,
        num_epochs: int,
        save_dir: Optional[str] = None,
        save_every: int = 10,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
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
            train_sampler: 训练采样器（DDP 时使用 DistributedSampler）
            visualize: 是否启用可视化
            plot_every: 每 N 个 epoch 绘制一次图表
            early_stopping_patience: 早停耐心值，如果验证损失连续N轮不下降则停止训练（None表示禁用）
            early_stopping_min_delta: 早停最小改进阈值，只有改进超过此值才认为是有效改进
        """
        if save_dir and self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        
        # 如果指定了恢复训练，加载checkpoint
        start_epoch = 1
        if resume_from is not None:
            if self.rank == 0:
                print(f"\n从检查点恢复训练: {resume_from}")
            start_epoch = self.load_checkpoint(resume_from)
            if self.rank == 0:
                print(f"✅ 已加载检查点，从 epoch {start_epoch} 继续训练")
                print(f"   训练历史: {len(self.train_losses)} 个epoch")
                if self.val_loader is not None:
                    print(f"   验证历史: {len(self.val_losses)} 个epoch")
        
        # 早停相关变量
        patience_counter = 0
        best_val_loss_for_patience = float('inf')
        
        if self.rank == 0:
            print(f"开始训练，设备: {self.device}")
            if self.use_discrete_diffusion:
                print(f"扩散模型: 启用 (时间步数: {self.num_diffusion_steps})")
                print(f"掩码策略: {self.masking_strategy}, 掩码比例: 动态 (Cosine Schedule)")
            else:
                print(f"掩码策略: {self.masking_strategy}, 掩码比例: {self.mask_ratio} (固定)")
            print(f"学习率调度: LinearWarmup ({self.warmup_epochs} epochs) + CosineAnnealing")
            print(f"最大学习率: {self.max_lr:.2e}")
            if early_stopping_patience is not None:
                print(f"早停: 启用 (patience={early_stopping_patience}, min_delta={early_stopping_min_delta:.6f})")
            if resume_from:
                print(f"恢复训练: 从 epoch {start_epoch} 继续，目标 {num_epochs} epochs")
        
        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            # DDP 模式下，每个 epoch 需要设置 sampler 的 epoch
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            if self.rank == 0:
                print(f"\nEpoch {epoch}/{num_epochs}")
                print("-" * 60)
            
            # 训练
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics)
            
            if self.rank == 0:
                print(f"Train Loss: {train_metrics['loss']:.4f} "
                      f"(Fragment: {train_metrics['fragment_loss']:.4f}, "
                      f"Torsion: {train_metrics['torsion_loss']:.4f}, "
                      f"Structure: {train_metrics.get('structure_loss', 0.0):.4f})")
            
            # 验证
            if self.val_loader is not None:
                val_metrics = self.validate()
                # 确保所有rank都完成了验证
                if self.ddp_enabled:
                    import torch.distributed as dist
                    dist.barrier()
                self.val_losses.append(val_metrics)
                
                # 只有当验证指标不为空时才处理
                if val_metrics:
                    if self.rank == 0:
                        print(f"Val Loss: {val_metrics['loss']:.4f} "
                              f"(Fragment: {val_metrics['fragment_loss']:.4f}, "
                              f"Torsion: {val_metrics['torsion_loss']:.4f}, "
                              f"Structure: {val_metrics.get('structure_loss', 0.0):.4f})")
                    
                    # 更新学习率（使用WarmupCosineAnnealing调度器）
                    self._update_learning_rate(epoch)
                    
                    # 保存最佳模型（只在 rank 0 保存）
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        if save_dir and self.rank == 0:
                            # 在保存前添加barrier，确保所有rank都完成了验证
                            if self.ddp_enabled:
                                import torch.distributed as dist
                                dist.barrier()  # 等待所有rank完成验证
                            self.save_checkpoint(
                                os.path.join(save_dir, 'best_model.pt'),
                                epoch, val_metrics['loss']
                            )
                            print(f"✅ 保存最佳模型 (val_loss: {val_metrics['loss']:.4f})")
                            # 保存后再次barrier，确保保存完成
                            if self.ddp_enabled:
                                dist.barrier()
                    
                    # 早停检查
                    if early_stopping_patience is not None:
                        # 检查是否有显著改进（超过min_delta）
                        improvement = best_val_loss_for_patience - val_metrics['loss']
                        if improvement > early_stopping_min_delta:
                            # 有显著改进，重置计数器
                            best_val_loss_for_patience = val_metrics['loss']
                            patience_counter = 0
                            if self.rank == 0:
                                print(f"   早停计数器重置 (改进: {improvement:.6f})")
                        else:
                            # 没有显著改进，增加计数器
                            patience_counter += 1
                            if self.rank == 0:
                                print(f"   早停计数器: {patience_counter}/{early_stopping_patience} (改进: {improvement:.6f})")
                            
                            # 如果达到耐心值，停止训练
                            if patience_counter >= early_stopping_patience:
                                if self.rank == 0:
                                    print(f"\n{'='*60}")
                                    print(f"⚠️  早停触发: 验证损失连续 {early_stopping_patience} 轮未改进")
                                    print(f"   最佳验证损失: {best_val_loss_for_patience:.4f} (Epoch {epoch - patience_counter})")
                                    print(f"   当前验证损失: {val_metrics['loss']:.4f}")
                                    print(f"{'='*60}\n")
                                break
                else:
                    if self.rank == 0:
                        print("⚠️  验证集为空，跳过验证")
            
            # 定期保存（只在 rank 0 保存）
            if save_dir and epoch % save_every == 0:
                # 添加barrier确保所有rank同步
                if self.ddp_enabled:
                    import torch.distributed as dist
                    dist.barrier()
                if self.rank == 0:
                    self.save_checkpoint(
                        os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'),
                        epoch, train_metrics['loss']
                    )
                if self.ddp_enabled:
                    dist.barrier()
            
            # 定期可视化（只在 rank 0 保存）
            if visualize and VISUALIZATION_AVAILABLE and save_dir and epoch % plot_every == 0:
                # 添加barrier确保所有rank同步
                if self.ddp_enabled:
                    import torch.distributed as dist
                    dist.barrier()
                if self.rank == 0:
                    try:
                        plot_path = os.path.join(save_dir, f'training_curves_epoch_{epoch}.png')
                        plot_training_curves(
                            self.train_losses,
                            self.val_losses if self.val_loader is not None else None,
                            save_path=plot_path
                        )
                    except Exception as e:
                        print(f"Warning: Failed to plot training curves: {e}")
                if self.ddp_enabled:
                    dist.barrier()
            elif visualize and not VISUALIZATION_AVAILABLE:
                if self.ddp_enabled:
                    import torch.distributed as dist
                    dist.barrier()
                if self.rank == 0:
                    if epoch == 1:
                        print("   ⚠️  可视化已请求但 matplotlib 未安装，跳过绘图")
                if self.ddp_enabled:
                    dist.barrier()
        
        # 训练结束后生成最终可视化（只在 rank 0 保存）
        if visualize and VISUALIZATION_AVAILABLE and save_dir:
            # 添加barrier确保所有rank同步
            if self.ddp_enabled:
                import torch.distributed as dist
                dist.barrier()
                if self.rank == 0:
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
                if self.ddp_enabled:
                    dist.barrier()
        
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
        """保存检查点"""
        # 如果是 DDP 模型，需要从 module 中获取 state_dict
        encoder_state_dict = self.encoder.module.state_dict() if self.is_ddp_encoder else self.encoder.state_dict()
        decoder_state_dict = self.decoder.module.state_dict() if self.is_ddp_decoder else self.decoder.state_dict()
        
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
        checkpoint = torch.load(path, map_location=self.device)
        
        # 加载模型权重
        encoder_state_dict = checkpoint['encoder_state_dict']
        decoder_state_dict = checkpoint['decoder_state_dict']
        
        # 处理DDP模型的state_dict键名匹配
        # 保存时已经去掉了module.前缀，所以需要根据当前模型状态添加或保持
        def adjust_state_dict_keys(state_dict, is_ddp):
            """调整state_dict的键名以匹配当前模型结构"""
            if not state_dict:
                return state_dict
            
            # 检查第一个键是否有module.前缀
            first_key = next(iter(state_dict.keys()))
            has_module_prefix = first_key.startswith('module.')
            
            if is_ddp and not has_module_prefix:
                # 当前是DDP，但checkpoint没有module.前缀，需要添加
                return {'module.' + k: v for k, v in state_dict.items()}
            elif not is_ddp and has_module_prefix:
                # 当前不是DDP，但checkpoint有module.前缀，需要去掉
                return {k.replace('module.', ''): v for k, v in state_dict.items()}
            else:
                # 匹配，直接返回
                return state_dict
        
        encoder_state_dict = adjust_state_dict_keys(encoder_state_dict, self.is_ddp_encoder)
        decoder_state_dict = adjust_state_dict_keys(decoder_state_dict, self.is_ddp_decoder)
        
        # 使用strict=False允许部分加载（如果架构有变化）
        self.encoder.load_state_dict(encoder_state_dict, strict=False)
        self.decoder.load_state_dict(decoder_state_dict, strict=False)
        
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
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
        return current_epoch + 1
