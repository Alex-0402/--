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

from ..models.encoder import BackboneEncoder
from ..models.decoder import FragmentDecoder
from ..data.vocabulary import get_vocab
from .masking import create_masks, apply_masks


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
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
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
                fragment_loss = nn.functional.cross_entropy(
                    masked_logits.view(-1, fragment_logits.shape[-1]),
                    masked_targets.view(-1)
                )
            else:
                fragment_loss = torch.tensor(0.0, device=fragment_logits.device)
        else:
            fragment_loss = nn.functional.cross_entropy(
                fragment_logits.reshape(-1, fragment_logits.shape[-1]),
                fragment_targets.reshape(-1)
            )
        
        # 扭转角损失（交叉熵）
        torsion_loss = nn.functional.cross_entropy(
            torsion_logits.reshape(-1, torsion_logits.shape[-1]),
            torsion_targets.reshape(-1)
        )
        
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
                
                # 创建掩码
                fragment_masks = create_masks(
                    fragment_token_ids,
                    strategy=self.masking_strategy,
                    mask_ratio=self.mask_ratio
                )
                
                # 应用掩码
                masked_fragments = apply_masks(fragment_token_ids, fragment_masks)
                
                # 前向传播
                node_embeddings = self.encoder(backbone_coords, mask=None)
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
                
                total_loss += losses['total_loss'].item()
                total_fragment_loss += losses['fragment_loss'].item()
                total_torsion_loss += losses['torsion_loss'].item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'fragment_loss': total_fragment_loss / num_batches,
            'torsion_loss': total_torsion_loss / num_batches
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
