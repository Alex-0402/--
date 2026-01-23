"""
分析验证损失平台期问题

专门分析最近50轮验证损失不下降的原因
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def analyze_val_loss_plateau(checkpoint_path: str, recent_epochs: int = 50):
    """
    分析验证损失平台期问题
    
    Args:
        checkpoint_path: checkpoint文件路径
        recent_epochs: 分析最近多少轮
    """
    print("="*80)
    print("验证损失平台期分析")
    print("="*80)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: 检查点文件不存在: {checkpoint_path}")
        return
    
    try:
        print(f"\n加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        epoch = checkpoint.get('epoch', 0)
        print(f"当前Epoch: {epoch}")
        
        # 获取训练历史
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        
        if not val_losses:
            print("❌ 没有验证损失数据")
            return
        
        # 提取损失值
        if isinstance(val_losses[0], dict):
            total_val = [l.get('loss', 0) for l in val_losses if l]
            frag_val = [l.get('fragment_loss', 0) for l in val_losses if l]
            tors_val = [l.get('torsion_loss', 0) for l in val_losses if l]
        else:
            total_val = [l for l in val_losses if l]
            frag_val = []
            tors_val = []
        
        if isinstance(train_losses[0], dict):
            total_train = [l.get('loss', 0) for l in train_losses]
            frag_train = [l.get('fragment_loss', 0) for l in train_losses]
            tors_train = [l.get('torsion_loss', 0) for l in train_losses]
        else:
            total_train = train_losses
            frag_train = []
            tors_train = []
        
        # 分析最近N轮
        recent_start = max(0, len(total_val) - recent_epochs)
        recent_val = total_val[recent_start:]
        recent_train = total_train[recent_start:] if len(total_train) > recent_start else []
        recent_epochs_actual = len(recent_val)
        
        print(f"\n分析最近 {recent_epochs_actual} 轮的验证损失 (Epoch {recent_start+1} - {len(total_val)})")
        print("-"*80)
        
        # 1. 验证损失趋势分析
        if len(recent_val) >= 2:
            # 计算线性趋势
            epochs_idx = np.arange(len(recent_val))
            trend_coef = np.polyfit(epochs_idx, recent_val, 1)[0]
            trend_intercept = np.polyfit(epochs_idx, recent_val, 1)[1]
            
            print(f"\n1. 验证损失趋势分析:")
            print(f"   最近 {recent_epochs_actual} 轮的平均损失: {np.mean(recent_val):.4f}")
            print(f"   损失标准差: {np.std(recent_val):.4f}")
            print(f"   损失范围: [{min(recent_val):.4f}, {max(recent_val):.4f}]")
            print(f"   线性趋势斜率: {trend_coef:.6f} (负值表示下降)")
            
            if abs(trend_coef) < 0.0001:
                print(f"   ⚠️  验证损失基本停滞 (斜率接近0)")
            elif trend_coef > 0:
                print(f"   ⚠️  验证损失在上升 (可能过拟合)")
            else:
                print(f"   ✅ 验证损失在下降，但速度很慢")
            
            # 计算最近50轮的损失变化
            if len(recent_val) >= 50:
                first_50 = recent_val[0]
                last_50 = recent_val[-1]
                change_50 = last_50 - first_50
                change_pct = (change_50 / first_50) * 100
                print(f"\n   最近50轮损失变化: {change_50:+.4f} ({change_pct:+.2f}%)")
                if abs(change_pct) < 1.0:
                    print(f"   ⚠️  损失变化小于1%，基本停滞")
        
        # 2. 训练损失 vs 验证损失对比
        if len(recent_train) > 0 and len(recent_val) > 0:
            min_len = min(len(recent_train), len(recent_val))
            train_recent = recent_train[:min_len]
            val_recent = recent_val[:min_len]
            
            train_mean = np.mean(train_recent)
            val_mean = np.mean(val_recent)
            gap = val_mean - train_mean
            
            print(f"\n2. 训练/验证损失对比:")
            print(f"   训练损失均值: {train_mean:.4f}")
            print(f"   验证损失均值: {val_mean:.4f}")
            print(f"   泛化差距: {gap:.4f}")
            
            if gap > 0.5:
                print(f"   ⚠️  验证损失明显高于训练损失，可能存在过拟合")
            elif gap < 0:
                print(f"   ⚠️  验证损失低于训练损失，可能验证集太小或数据分布问题")
            else:
                print(f"   ✅ 训练和验证损失接近，模型泛化良好")
        
        # 3. 学习率分析
        warmup_epochs = checkpoint.get('warmup_epochs', 20)
        total_epochs = checkpoint.get('total_epochs', 300)
        max_lr = checkpoint.get('max_lr', 5e-4)
        
        print(f"\n3. 学习率调度分析:")
        print(f"   最大学习率: {max_lr:.2e}")
        print(f"   Warmup轮数: {warmup_epochs}")
        print(f"   总训练轮数: {total_epochs}")
        
        # 计算当前epoch的学习率
        current_epoch = epoch
        if current_epoch <= warmup_epochs:
            current_lr = max_lr * (current_epoch / warmup_epochs)
            print(f"   当前学习率: {current_lr:.2e} (Warmup阶段)")
        else:
            # Cosine annealing
            cosine_epochs = current_epoch - warmup_epochs
            cosine_max = total_epochs - warmup_epochs
            current_lr = 1e-6 + (max_lr - 1e-6) * (1 + np.cos(np.pi * cosine_epochs / cosine_max)) / 2
            print(f"   当前学习率: {current_lr:.2e} (Cosine Annealing阶段)")
            print(f"   学习率衰减进度: {cosine_epochs}/{cosine_max} ({cosine_epochs/cosine_max*100:.1f}%)")
            
            if current_lr < max_lr * 0.1:
                print(f"   ⚠️  学习率已经衰减到最大值的10%以下，可能太小")
                print(f"   建议: 考虑使用学习率重启或增加学习率")
        
        # 4. 损失组件分析
        if frag_val and tors_val:
            print(f"\n4. 损失组件分析 (最近{recent_epochs_actual}轮):")
            recent_frag = frag_val[recent_start:]
            recent_tors = tors_val[recent_start:]
            
            frag_mean = np.mean(recent_frag)
            tors_mean = np.mean(recent_tors)
            frag_std = np.std(recent_frag)
            tors_std = np.std(recent_tors)
            
            print(f"   片段损失: 均值={frag_mean:.4f}, 标准差={frag_std:.4f}")
            print(f"   扭转角损失: 均值={tors_mean:.4f}, 标准差={tors_std:.4f}")
            
            # 计算趋势
            if len(recent_frag) >= 2:
                frag_trend = np.polyfit(np.arange(len(recent_frag)), recent_frag, 1)[0]
                tors_trend = np.polyfit(np.arange(len(recent_tors)), recent_tors, 1)[0]
                print(f"   片段损失趋势: {frag_trend:.6f}")
                print(f"   扭转角损失趋势: {tors_trend:.6f}")
                
                if abs(frag_trend) < 0.0001 and abs(tors_trend) < 0.0001:
                    print(f"   ⚠️  两个损失组件都停滞，模型可能已达到性能上限")
        
        # 5. 最佳模型检查
        if len(total_val) > 0:
            best_val_loss = min(total_val)
            best_epoch = total_val.index(best_val_loss) + 1
            current_val_loss = total_val[-1]
            
            print(f"\n5. 最佳模型信息:")
            print(f"   最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch})")
            print(f"   当前验证损失: {current_val_loss:.4f}")
            print(f"   距离最佳: {current_val_loss - best_val_loss:.4f}")
            
            epochs_since_best = len(total_val) - best_epoch
            print(f"   距离最佳模型: {epochs_since_best} 轮")
            
            if epochs_since_best >= 50:
                print(f"   ⚠️  已经 {epochs_since_best} 轮没有改进，建议早停或调整策略")
        
        # 6. 生成详细图表
        print(f"\n6. 生成分析图表...")
        plot_path = os.path.join(os.path.dirname(checkpoint_path), 'val_loss_plateau_analysis.png')
        plot_analysis(total_train, total_val, frag_train, frag_val, tors_train, tors_val,
                     recent_start, warmup_epochs, total_epochs, max_lr, plot_path)
        print(f"   图表已保存: {plot_path}")
        
        # 7. 建议
        print(f"\n" + "="*80)
        print("诊断建议")
        print("="*80)
        
        suggestions = []
        
        # 检查学习率
        if current_epoch > warmup_epochs:
            cosine_epochs = current_epoch - warmup_epochs
            cosine_max = total_epochs - warmup_epochs
            if cosine_epochs / cosine_max > 0.7:
                suggestions.append("学习率可能已经衰减过多，考虑:")
                suggestions.append("  - 使用学习率重启 (CosineAnnealingWarmRestarts)")
                suggestions.append("  - 增加最大学习率")
                suggestions.append("  - 使用更长的训练周期")
        
        # 检查过拟合
        if len(recent_train) > 0 and len(recent_val) > 0:
            if np.mean(recent_val) - np.mean(recent_train) > 0.5:
                suggestions.append("可能存在过拟合，建议:")
                suggestions.append("  - 增加正则化 (weight_decay, dropout)")
                suggestions.append("  - 使用数据增强")
                suggestions.append("  - 早停 (Early Stopping)")
        
        # 检查损失停滞
        if len(recent_val) >= 50:
            first_50 = recent_val[0]
            last_50 = recent_val[-1]
            if abs(last_50 - first_50) / first_50 < 0.01:
                suggestions.append("验证损失停滞，建议:")
                suggestions.append("  - 检查数据质量和分布")
                suggestions.append("  - 尝试不同的优化器 (AdamW -> SGD)")
                suggestions.append("  - 调整模型架构或容量")
                suggestions.append("  - 考虑使用学习率调度器重启")
        
        # 检查最佳模型
        if len(total_val) > 0:
            best_epoch = total_val.index(min(total_val)) + 1
            if len(total_val) - best_epoch >= 50:
                suggestions.append("已长时间未改进，建议:")
                suggestions.append("  - 使用早停机制")
                suggestions.append("  - 加载最佳模型继续微调")
                suggestions.append("  - 尝试不同的超参数组合")
        
        if suggestions:
            for s in suggestions:
                print(s)
        else:
            print("模型训练正常，继续观察")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def plot_analysis(train_total, val_total, train_frag, val_frag, train_tors, val_tors,
                  recent_start, warmup_epochs, total_epochs, max_lr, save_path):
    """绘制详细分析图表"""
    fig = plt.figure(figsize=(16, 12))
    
    epochs_all = range(1, len(train_total) + 1)
    epochs_val = range(1, len(val_total) + 1)
    
    # 1. 总损失对比（全图）
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(epochs_all, train_total, 'b-', label='Train Loss', linewidth=1.5, alpha=0.7)
    ax1.plot(epochs_val, val_total, 'r-', label='Val Loss', linewidth=1.5, alpha=0.7)
    if recent_start > 0:
        ax1.axvline(x=recent_start+1, color='gray', linestyle='--', alpha=0.5, label='Recent Analysis Start')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss (Full History)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 最近50轮详细视图
    ax2 = plt.subplot(3, 2, 2)
    recent_epochs = range(recent_start+1, len(val_total)+1)
    recent_val = val_total[recent_start:]
    recent_train = train_total[recent_start:] if len(train_total) > recent_start else []
    if recent_train:
        recent_train_epochs = range(recent_start+1, len(train_total)+1)
        ax2.plot(recent_train_epochs, recent_train, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax2.plot(recent_epochs, recent_val, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
    
    # 添加趋势线
    if len(recent_val) >= 2:
        z = np.polyfit(range(len(recent_val)), recent_val, 1)
        p = np.poly1d(z)
        ax2.plot(recent_epochs, p(range(len(recent_val))), 'r--', alpha=0.5, label=f'Trend (slope={z[0]:.6f})')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Recent {len(recent_val)} Epochs (Detailed)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 学习率曲线
    ax3 = plt.subplot(3, 2, 3)
    lr_epochs = range(1, total_epochs + 1)
    lrs = []
    for e in lr_epochs:
        if e <= warmup_epochs:
            lr = max_lr * (e / warmup_epochs)
        else:
            cosine_epochs = e - warmup_epochs
            cosine_max = total_epochs - warmup_epochs
            lr = 1e-6 + (max_lr - 1e-6) * (1 + np.cos(np.pi * cosine_epochs / cosine_max)) / 2
        lrs.append(lr)
    ax3.plot(lr_epochs, lrs, 'g-', linewidth=2)
    if len(val_total) > 0:
        ax3.axvline(x=len(val_total), color='r', linestyle='--', alpha=0.5, label='Current Epoch')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 损失组件对比
    ax4 = plt.subplot(3, 2, 4)
    if train_frag and val_frag:
        ax4.plot(epochs_all[:len(train_frag)], train_frag, 'g-', label='Train Fragment', linewidth=1.5, alpha=0.7)
        ax4.plot(epochs_val[:len(val_frag)], val_frag, 'orange', label='Val Fragment', linewidth=1.5, alpha=0.7)
    if train_tors and val_tors:
        ax4.plot(epochs_all[:len(train_tors)], train_tors, 'm-', label='Train Torsion', linewidth=1.5, alpha=0.7)
        ax4.plot(epochs_val[:len(val_tors)], val_tors, 'coral', label='Val Torsion', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Loss Components')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 验证损失移动平均
    ax5 = plt.subplot(3, 2, 5)
    if len(val_total) >= 10:
        window = min(10, len(val_total) // 5)
        moving_avg = np.convolve(val_total, np.ones(window)/window, mode='valid')
        moving_epochs = range(window, len(val_total)+1)
        ax5.plot(epochs_val, val_total, 'r-', alpha=0.3, label='Val Loss (raw)')
        ax5.plot(moving_epochs, moving_avg, 'r-', linewidth=2, label=f'Val Loss (MA{window})')
    else:
        ax5.plot(epochs_val, val_total, 'r-', linewidth=2, label='Val Loss')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.set_title('Validation Loss (Moving Average)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 损失变化率
    ax6 = plt.subplot(3, 2, 6)
    if len(val_total) >= 2:
        val_diff = np.diff(val_total)
        diff_epochs = range(2, len(val_total)+1)
        ax6.plot(diff_epochs, val_diff, 'b-', linewidth=1.5, marker='o', markersize=2)
        ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Loss Change')
        ax6.set_title('Validation Loss Change per Epoch')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="分析验证损失平台期问题")
    parser.add_argument("--checkpoint", type=str, 
                       default="checkpoints/best_model.pt",
                       help="检查点文件路径")
    parser.add_argument("--recent", type=int, default=50,
                       help="分析最近多少轮 (默认50)")
    
    args = parser.parse_args()
    
    # 如果没有best_model.pt，尝试找最新的checkpoint
    if not os.path.exists(args.checkpoint):
        checkpoints_dir = os.path.dirname(args.checkpoint) or "checkpoints"
        if os.path.exists(checkpoints_dir):
            pt_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
            if pt_files:
                # 尝试找最新的checkpoint
                checkpoint_files = [f for f in pt_files if 'epoch' in f]
                if checkpoint_files:
                    # 按epoch编号排序
                    def get_epoch_num(f):
                        try:
                            return int(f.split('epoch_')[1].split('.')[0])
                        except:
                            return 0
                    checkpoint_files.sort(key=get_epoch_num, reverse=True)
                    args.checkpoint = os.path.join(checkpoints_dir, checkpoint_files[0])
                    print(f"使用最新的checkpoint: {args.checkpoint}")
    
    analyze_val_loss_plateau(args.checkpoint, args.recent)


if __name__ == "__main__":
    main()
