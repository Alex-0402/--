import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 整理跑分数据 (Mean & Std)
# ==========================================
# 准确率相关指标 (frag_acc, res_exact)
acc_metrics = ['Fragment Acc\n(Higher is better)', 'Residue Exact Matching\n(Higher is better)']
random_acc_means = [0.6203, 0.4155]
random_acc_stds  = [0.0404, 0.0530]

adaptive_acc_means = [0.6925, 0.4603]
adaptive_acc_stds  = [0.0356, 0.0588]

# Clash 碰撞指标 (很小的数值，单独画图)
clash_metrics = ['Clash Score\n(Lower is better)']
random_clash_means = [0.0040]
random_clash_stds  = [0.0022]

adaptive_clash_means = [0.0039]
adaptive_clash_stds  = [0.0021]

# ==========================================
# 2. 设置绘图全局样式 (学术科技风)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['font.size'] = 12

# 颜色配置 (Random 用低调的蓝灰色，Adaptive 用亮眼的珊瑚红/橙色，突出提升)
COLOR_RANDOM = '#6C8EBF'
COLOR_ADAPTIVE = '#E66C5C'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})

# ==========================================
# 3. 绘制准确率柱状图 (左图)
# ==========================================
x = np.arange(len(acc_metrics))
width = 0.35

rects1 = ax1.bar(x - width/2, random_acc_means, width, yerr=random_acc_stds, 
                 label='Random Inference', color=COLOR_RANDOM, capsize=5, alpha=0.9, edgecolor='white')
rects2 = ax1.bar(x + width/2, adaptive_acc_means, width, yerr=adaptive_acc_stds, 
                 label='Adaptive Inference (Ours)', color=COLOR_ADAPTIVE, capsize=5, alpha=0.9, edgecolor='white')

# 添加数值标签
def autolabel(rects, ax, is_acc=True):
    for rect in rects:
        height = rect.get_height()
        # 格式化数字：准确率用百分比，clash用小数
        label_str = f'{height*100:.1f}%' if is_acc else f'{height:.4f}'
        ax.annotate(label_str,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3 if not is_acc else 15),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

autolabel(rects1, ax1, is_acc=True)
autolabel(rects2, ax1, is_acc=True)

ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy Metrics Comparison', pad=20, fontweight='bold', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(acc_metrics)
ax1.set_ylim(0, 0.85)  # 留出空间写字
ax1.legend(loc='upper left')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# ==========================================
# 4. 绘制 Clash 柱状图 (右图)
# ==========================================
x2 = np.arange(len(clash_metrics))

rects3 = ax2.bar(x2 - width/2, random_clash_means, width, yerr=random_clash_stds, 
                 color=COLOR_RANDOM, capsize=5, alpha=0.9, edgecolor='white')
rects4 = ax2.bar(x2 + width/2, adaptive_clash_means, width, yerr=adaptive_clash_stds, 
                 color=COLOR_ADAPTIVE, capsize=5, alpha=0.9, edgecolor='white')

autolabel(rects3, ax2, is_acc=False)
autolabel(rects4, ax2, is_acc=False)

ax2.set_ylabel('Score')
ax2.set_title('Structural Quality (Clash)', pad=20, fontweight='bold', fontsize=14)
ax2.set_xticks(x2)
ax2.set_xticklabels(clash_metrics)
ax2.set_ylim(0, 0.008) # 针对极小的值调整 Y 轴
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# ==========================================
# 5. 调整布局并保存
# ==========================================
plt.tight_layout()
os.makedirs('visualization', exist_ok=True)
output_path = 'visualization/benchmark_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
print(f"图表已成功保存至: {output_path}")
