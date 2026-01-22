"""
训练诊断脚本 - 分析训练历史和模型性能

使用方法:
    python analyze_training.py --checkpoint checkpoints/checkpoint_epoch_50.pt
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import numpy as np

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data.vocabulary import get_vocab


def analyze_checkpoint(checkpoint_path: str):
    """分析检查点文件"""
    print("="*70)
    print("训练历史分析")
    print("="*70)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: 检查点文件不存在: {checkpoint_path}")
        return
    
    try:
        print(f"\n加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Epoch: {epoch}")
        
        # 分析训练损失
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        
        print(f"\n训练损失历史: {len(train_losses)} 个记录")
        print(f"验证损失历史: {len(val_losses)} 个记录")
        
        if train_losses:
            # 提取损失值
            if isinstance(train_losses[0], dict):
                total_train = [l.get('loss', 0) for l in train_losses]
                frag_train = [l.get('fragment_loss', 0) for l in train_losses]
                tors_train = [l.get('torsion_loss', 0) for l in train_losses]
            else:
                total_train = train_losses
                frag_train = []
                tors_train = []
            
            print("\n训练损失统计:")
            print(f"  初始损失: {total_train[0]:.4f}")
            print(f"  最终损失: {total_train[-1]:.4f}")
            print(f"  损失变化: {total_train[0] - total_train[-1]:.4f}")
            print(f"  最低损失: {min(total_train):.4f} (Epoch {total_train.index(min(total_train)) + 1})")
            
            if len(total_train) > 1:
                # 计算趋势
                recent = total_train[-5:] if len(total_train) >= 5 else total_train
                trend = np.polyfit(range(len(recent)), recent, 1)[0]
                if trend < -0.01:
                    print(f"  ✅ 损失正在下降 (最近趋势: {trend:.4f}/epoch)")
                elif trend > 0.01:
                    print(f"  ⚠️  损失正在上升 (最近趋势: {trend:.4f}/epoch)")
                else:
                    print(f"  ⚠️  损失基本不变 (最近趋势: {trend:.4f}/epoch)")
            
            if frag_train:
                print(f"\n片段损失:")
                print(f"  初始: {frag_train[0]:.4f}, 最终: {frag_train[-1]:.4f}")
            if tors_train:
                print(f"\n扭转角损失:")
                print(f"  初始: {tors_train[0]:.4f}, 最终: {tors_train[-1]:.4f}")
        
        if val_losses:
            if isinstance(val_losses[0], dict):
                total_val = [l.get('loss', 0) for l in val_losses if l]
                frag_val = [l.get('fragment_loss', 0) for l in val_losses if l]
                tors_val = [l.get('torsion_loss', 0) for l in val_losses if l]
            else:
                total_val = [l for l in val_losses if l]
                frag_val = []
                tors_val = []
            
            if total_val:
                print("\n验证损失统计:")
                print(f"  初始损失: {total_val[0]:.4f}")
                print(f"  最终损失: {total_val[-1]:.4f}")
                print(f"  最低损失: {min(total_val):.4f}")
        
        # 理论基线
        vocab = get_vocab()
        vocab_size = vocab.get_vocab_size()
        num_torsion_bins = 72
        
        random_frag_loss = np.log(vocab_size)
        random_tors_loss = np.log(num_torsion_bins)
        random_total = random_frag_loss + random_tors_loss
        
        print("\n" + "="*70)
        print("性能基准对比")
        print("="*70)
        print(f"随机猜测基线:")
        print(f"  片段损失: {random_frag_loss:.4f} (log({vocab_size}))")
        print(f"  扭转角损失: {random_tors_loss:.4f} (log({num_torsion_bins}))")
        print(f"  总损失: {random_total:.4f}")
        
        if train_losses and isinstance(train_losses[0], dict):
            final_loss = train_losses[-1].get('loss', 0)
            final_frag = train_losses[-1].get('fragment_loss', 0)
            final_tors = train_losses[-1].get('torsion_loss', 0)
            
            print(f"\n当前模型性能:")
            print(f"  片段损失: {final_frag:.4f} (vs 基线 {random_frag_loss:.4f})")
            print(f"  扭转角损失: {final_tors:.4f} (vs 基线 {random_tors_loss:.4f})")
            print(f"  总损失: {final_loss:.4f} (vs 基线 {random_total:.4f})")
            
            # 评估
            frag_improvement = (random_frag_loss - final_frag) / random_frag_loss * 100
            tors_improvement = (random_tors_loss - final_tors) / random_tors_loss * 100
            total_improvement = (random_total - final_loss) / random_total * 100
            
            print(f"\n相对基线的改进:")
            print(f"  片段: {frag_improvement:+.2f}%")
            print(f"  扭转角: {tors_improvement:+.2f}%")
            print(f"  总体: {total_improvement:+.2f}%")
            
            if abs(frag_improvement) < 5 and abs(tors_improvement) < 5:
                print("\n⚠️  警告: 模型性能接近随机猜测，需要改进！")
        
    except Exception as e:
        print(f"❌ 错误: 无法加载检查点: {e}")
        import traceback
        traceback.print_exc()


def check_dataset_size():
    """检查数据集规模"""
    print("\n" + "="*70)
    print("数据集规模检查")
    print("="*70)
    
    cache_dir = "data/cache"
    train_file = os.path.join(cache_dir, "train.txt")
    val_file = os.path.join(cache_dir, "val.txt")
    test_file = os.path.join(cache_dir, "test.txt")
    
    train_size = 0
    val_size = 0
    test_size = 0
    
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            train_size = len([l for l in f if l.strip()])
    
    if os.path.exists(val_file):
        with open(val_file, 'r') as f:
            val_size = len([l for l in f if l.strip()])
    
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            test_size = len([l for l in f if l.strip()])
    
    total = train_size + val_size + test_size
    
    print(f"训练集: {train_size} 个样本")
    print(f"验证集: {val_size} 个样本")
    print(f"测试集: {test_size} 个样本")
    print(f"总计: {total} 个样本")
    
    if total < 50:
        print("\n⚠️  警告: 数据集规模过小！")
        print("   建议至少需要 100+ 个样本才能有效训练")
        print("   当前数据量可能导致模型无法学习到有效模式")
    elif total < 100:
        print("\n⚠️  注意: 数据集规模较小")
        print("   建议增加数据量以获得更好的性能")
    else:
        print("\n✅ 数据集规模合理")


def print_recommendations():
    """打印改进建议"""
    print("\n" + "="*70)
    print("改进建议")
    print("="*70)
    
    print("\n1. 增加数据集规模")
    print("   - 当前数据量可能不足以训练有效的模型")
    print("   - 建议至少准备 100-200 个蛋白质结构")
    print("   - 可以下载更多 CATH 数据库的蛋白质结构")
    
    print("\n2. 检查训练过程")
    print("   - 确认训练损失是否在下降")
    print("   - 如果损失不下降，可能需要调整学习率")
    print("   - 尝试更小的学习率 (如 5e-5) 或使用学习率调度")
    
    print("\n3. 调整超参数")
    print("   - 尝试不同的批次大小 (2, 4, 8)")
    print("   - 调整掩码比例 (0.1, 0.15, 0.2)")
    print("   - 增加训练轮数 (100+ epochs)")
    
    print("\n4. 模型架构检查")
    print("   - 确认 Encoder 和 Decoder 的维度匹配")
    print("   - 检查梯度流是否正常")
    print("   - 考虑增加模型容量（更多层或更大隐藏维度）")
    
    print("\n5. 数据质量检查")
    print("   - 验证数据预处理是否正确")
    print("   - 检查片段序列和扭转角的编码是否正确")
    print("   - 确保没有数据泄漏或标签错误")


def main():
    parser = argparse.ArgumentParser(
        description="分析训练历史和模型性能"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint_epoch_50.pt",
        help="检查点文件路径"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="分析所有检查点文件"
    )
    
    args = parser.parse_args()
    
    if args.all:
        # 分析所有检查点
        checkpoints_dir = "checkpoints"
        if os.path.exists(checkpoints_dir):
            pt_files = sorted([
                f for f in os.listdir(checkpoints_dir)
                if f.endswith('.pt') and 'epoch' in f
            ])
            for pt_file in pt_files:
                analyze_checkpoint(os.path.join(checkpoints_dir, pt_file))
                print("\n")
    else:
        analyze_checkpoint(args.checkpoint)
    
    check_dataset_size()
    print_recommendations()
    
    print("\n" + "="*70)
    print("分析完成")
    print("="*70)


if __name__ == "__main__":
    main()
