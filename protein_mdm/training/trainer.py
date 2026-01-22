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
from tqdm import tqdm

from models.encoder import BackboneEncoder
from models.decoder import FragmentDecoder
from data.vocabulary import get_vocab
from training.masking import create_masks, apply_masks


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
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        mask_ratio: float = 0.15,
        masking_strategy: str = "random"
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
            mask_ratio: 掩码比例
            masking_strategy: 掩码策略
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
        
        # 移动到设备
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
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
        
        # 总损失
        total_loss = fragment_loss + torsion_loss
        
        return {
            'total_loss': total_loss,
            'fragment_loss': fragment_loss,
            'torsion_loss': torsion_loss
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
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # 移动到设备
            backbone_coords = batch['backbone_coords'].to(self.device)
            fragment_token_ids = batch['fragment_token_ids'].to(self.device)
            torsion_bins = batch['torsion_bins'].to(self.device)
            sequence_lengths = batch['sequence_lengths'].to(self.device)
            
            # 创建掩码
            fragment_masks = create_masks(
                fragment_token_ids,
                strategy=self.masking_strategy,
                mask_ratio=self.mask_ratio
            )
            
            # 应用掩码
            masked_fragments = apply_masks(fragment_token_ids, fragment_masks)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # Encoder
            node_embeddings = self.encoder(backbone_coords, mask=None)
            
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
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'frag': f"{losses['fragment_loss'].item():.4f}",
                'tors': f"{losses['torsion_loss'].item():.4f}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'fragment_loss': total_fragment_loss / num_batches,
            'torsion_loss': total_torsion_loss / num_batches
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
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                backbone_coords = batch['backbone_coords'].to(self.device)
                fragment_token_ids = batch['fragment_token_ids'].to(self.device)
                torsion_bins = batch['torsion_bins'].to(self.device)
                sequence_lengths = batch['sequence_lengths'].to(self.device)
                
                # 检查输入数据
                if torch.isnan(backbone_coords).any() or torch.isinf(backbone_coords).any():
                    print(f"  ⚠️  警告: backbone_coords 包含 NaN/Inf，跳过此批次")
                    continue
                
                # 创建掩码
                fragment_masks = create_masks(
                    fragment_token_ids,
                    strategy=self.masking_strategy,
                    mask_ratio=self.mask_ratio
                )
                
                # 应用掩码
                masked_fragments = apply_masks(fragment_token_ids, fragment_masks)
                
                # 前向传播
                try:
                    node_embeddings = self.encoder(backbone_coords, mask=None)
                    if torch.isnan(node_embeddings).any() or torch.isinf(node_embeddings).any():
                        print(f"  ⚠️  警告: encoder 输出包含 NaN/Inf，跳过此批次")
                        continue
                    
                    try:
                        frag_logits, tors_logits = self.decoder(
                            node_embeddings=node_embeddings,
                            target_fragments=masked_fragments,
                            sequence_lengths=sequence_lengths
                        )
                    except Exception as e:
                        print(f"  ⚠️  警告: decoder 前向传播失败: {e}，跳过此批次")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    if torch.isnan(frag_logits).any() or torch.isinf(frag_logits).any():
                        print(f"  ⚠️  警告: fragment_logits 包含 NaN/Inf，跳过此批次")
                        print(f"      NaN 数量: {torch.isnan(frag_logits).sum().item()}, Inf 数量: {torch.isinf(frag_logits).sum().item()}")
                        print(f"      Fragment logits 统计: min={frag_logits.min().item():.4f}, max={frag_logits.max().item():.4f}, mean={frag_logits.mean().item():.4f}")
                        print(f"      Node embeddings 统计: min={node_embeddings.min().item():.4f}, max={node_embeddings.max().item():.4f}, mean={node_embeddings.mean().item():.4f}")
                        print(f"      Masked fragments 范围: min={masked_fragments.min().item()}, max={masked_fragments.max().item()}")
                        continue
                    if torch.isnan(tors_logits).any() or torch.isinf(tors_logits).any():
                        print(f"  ⚠️  警告: torsion_logits 包含 NaN/Inf，跳过此批次")
                        continue
                except Exception as e:
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
            print(f"  ⚠️  警告: 验证损失为 NaN，可能是数据问题")
            return {}
        
        return {
            'loss': avg_loss,
            'fragment_loss': avg_frag_loss,
            'torsion_loss': avg_tors_loss
        }
    
    def train(
        self,
        num_epochs: int,
        save_dir: Optional[str] = None,
        save_every: int = 10
    ):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            save_dir: 保存目录
            save_every: 每 N 个 epoch 保存一次
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        print(f"开始训练，设备: {self.device}")
        print(f"掩码策略: {self.masking_strategy}, 掩码比例: {self.mask_ratio}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)
            
            # 训练
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics)
            
            print(f"Train Loss: {train_metrics['loss']:.4f} "
                  f"(Fragment: {train_metrics['fragment_loss']:.4f}, "
                  f"Torsion: {train_metrics['torsion_loss']:.4f})")
            
            # 验证
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.val_losses.append(val_metrics)
                
                # 只有当验证指标不为空时才处理
                if val_metrics:
                    print(f"Val Loss: {val_metrics['loss']:.4f} "
                          f"(Fragment: {val_metrics['fragment_loss']:.4f}, "
                          f"Torsion: {val_metrics['torsion_loss']:.4f})")
                    
                    # 更新学习率
                    self.scheduler.step(val_metrics['loss'])
                    
                    # 保存最佳模型
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        if save_dir:
                            self.save_checkpoint(
                                os.path.join(save_dir, 'best_model.pt'),
                                epoch, val_metrics['loss']
                            )
                            print(f"✅ 保存最佳模型 (val_loss: {val_metrics['loss']:.4f})")
                else:
                    print("⚠️  验证集为空，跳过验证")
            
            # 定期保存
            if save_dir and epoch % save_every == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'),
                    epoch, train_metrics['loss']
                )
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        loss: float
    ):
        """保存检查点"""
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']
