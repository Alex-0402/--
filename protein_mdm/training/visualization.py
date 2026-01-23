"""
训练可视化模块

用于绘制训练过程中的损失曲线和其他指标。
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import os


def plot_training_curves(
    train_losses: List[Dict[str, float]],
    val_losses: Optional[List[Dict[str, float]]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = False
):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表，每个元素是包含 'loss', 'fragment_loss', 'torsion_loss' 的字典
        val_losses: 验证损失列表（可选），格式同 train_losses
        save_path: 保存路径（可选）
        show_plot: 是否显示图表（默认 False，因为可能没有显示环境）
    """
    # 提取数据
    epochs = range(1, len(train_losses) + 1)
    
    train_total = [l.get('loss', 0) for l in train_losses]
    train_frag = [l.get('fragment_loss', 0) for l in train_losses]
    train_tors = [l.get('torsion_loss', 0) for l in train_losses]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # 1. 总损失对比
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_total, 'b-', label='Train Loss', linewidth=2)
    if val_losses:
        val_total = [l.get('loss', 0) for l in val_losses if l]
        val_epochs = range(1, len(val_losses) + 1)
        # 只绘制有效的验证损失
        valid_val = [(e, v) for e, v in zip(val_epochs, val_total) if v > 0]
        if valid_val:
            val_epochs_clean, val_total_clean = zip(*valid_val)
            ax1.plot(val_epochs_clean, val_total_clean, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Total Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 片段损失
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_frag, 'g-', label='Train Fragment Loss', linewidth=2)
    if val_losses:
        val_frag = [l.get('fragment_loss', 0) for l in val_losses if l]
        val_epochs = range(1, len(val_losses) + 1)
        valid_val = [(e, v) for e, v in zip(val_epochs, val_frag) if v > 0]
        if valid_val:
            val_epochs_clean, val_frag_clean = zip(*valid_val)
            ax2.plot(val_epochs_clean, val_frag_clean, 'orange', label='Val Fragment Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Fragment Loss', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. 扭转角损失
    ax3 = axes[1, 0]
    ax3.plot(epochs, train_tors, 'm-', label='Train Torsion Loss', linewidth=2)
    if val_losses:
        val_tors = [l.get('torsion_loss', 0) for l in val_losses if l]
        val_epochs = range(1, len(val_losses) + 1)
        valid_val = [(e, v) for e, v in zip(val_epochs, val_tors) if v > 0]
        if valid_val:
            val_epochs_clean, val_tors_clean = zip(*valid_val)
            ax3.plot(val_epochs_clean, val_tors_clean, 'coral', label='Val Torsion Loss', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Torsion Loss', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 所有损失在一个图上
    ax4 = axes[1, 1]
    ax4.plot(epochs, train_total, 'b-', label='Train Total', linewidth=2, alpha=0.7)
    ax4.plot(epochs, train_frag, 'g-', label='Train Fragment', linewidth=1.5, alpha=0.7)
    ax4.plot(epochs, train_tors, 'm-', label='Train Torsion', linewidth=1.5, alpha=0.7)
    if val_losses:
        val_total = [l.get('loss', 0) for l in val_losses if l]
        val_epochs = range(1, len(val_losses) + 1)
        valid_val = [(e, v) for e, v in zip(val_epochs, val_total) if v > 0]
        if valid_val:
            val_epochs_clean, val_total_clean = zip(*valid_val)
            ax4.plot(val_epochs_clean, val_total_clean, 'r--', label='Val Total', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('All Losses', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_loss_comparison(
    train_losses: List[Dict[str, float]],
    val_losses: Optional[List[Dict[str, float]]] = None,
    save_path: Optional[str] = None
):
    """
    绘制简化的损失对比图（单图）
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表（可选）
        save_path: 保存路径（可选）
    """
    epochs = range(1, len(train_losses) + 1)
    
    train_total = [l.get('loss', 0) for l in train_losses]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_total, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    
    if val_losses:
        val_total = [l.get('loss', 0) for l in val_losses if l]
        val_epochs = range(1, len(val_losses) + 1)
        valid_val = [(e, v) for e, v in zip(val_epochs, val_total) if v > 0]
        if valid_val:
            val_epochs_clean, val_total_clean = zip(*valid_val)
            plt.plot(val_epochs_clean, val_total_clean, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss comparison plot saved to: {save_path}")
    
    plt.close()
