# 待办事项与核心算法改进计划 (TODO & Core Algorithm Improvements)

## 核心改进：引入动态空间几何偏置 (Dynamic Spatial Geometric Bias)

### 1. 理论推导 (Theoretical Derivation)
当前的 MDM 架构在使用 Transformer 预测离散化学片段 (Fragment Tokens) 时，缺乏对三维欧式空间的显式感知。为了解决这个问题，需要将三维空间约束注入到一维的 Token 序列注意力中。

**推导过程：**
1. **虚拟 $C_\beta$ 坐标与方向**：利用已知的骨架原子 $N_i, CA_i, C_i$，通过固定的局部参考系变换，为每个残基计算出虚拟的侧链起始指向向量 $\vec{v}_i$ 和虚拟 $C_\beta$ 坐标 $x_i^{v\beta}$。
2. **空间距离矩阵**：计算成对距离 $d_{ij} = \|x_i^{v\beta} - x_j^{v\beta}\|_2$，并通过径向基函数 (RBF) 或分箱 (Bucketing) 将其映射为高维偏置张量。
3. **空间相对方向**：计算基团指向的余弦相似度 $cos(\theta_{ij}) = \frac{\vec{v}_i \cdot \vec{v}_j}{\|\vec{v}_i\| \|\vec{v}_j\|}$。
4. **动态注意力偏置融合**：在 Transformer 解码器的 Self-Attention 机制中，叠加基于掩码状态 $m_j \in \{\text{Masked}, \text{Unmasked}\}$ 的偏置项：
   $$ \text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q_i K_j^T}{\sqrt{d}} + \mathbf{Bias}(d_{ij}, \theta_{ij}, m_j) \right) V_j $$

### 2. 为什么可行？(Feasibility & Core Synergies)
- **极低的计算开销**：主链坐标是固定的，因此 $N \times N$ 的距离矩阵和方向矩阵只需在数据加载阶段 (`dataset.py`) 计算一次。Transformer 每层仅需查表并加上 Bias 张量，不会增加模型的时间复杂度，完全兼容现有的多GPU训练。
- **与自适应推理 (Adaptive Inference) 的完美闭环**（最核心亮点）：
  当模型通过 Top1-Top2 Margin 确定了一个空间的“锚点”片段（例如将某个位置从 `[MASK]` 坍缩为确定的 `PHENYL` 大基团）后，对应的指示变量 $m_j$ 发生翻转。
  在下一步推理中，这个已确定的片段会通过 $\mathbf{Bias}$ 向周围的 Token 施加强烈的“体积排斥惩罚 (Volume Penalty)”。周围原本处于犹豫状态（低Margin）的 Token，其预测大基团的概率会被瞬间压制，只能选择小基团，从而导致它们的 Top1-Top2 Margin 瞬间暴涨。
  这就实现了**“物理空间占用 -> 概率分布倾斜 -> 驱动下一步自适应采样”**的连锁反应，完美在神经网络中模拟了“几何数独”的排除法。
- **严谨的文献支撑**：该几何偏置思路在 AlphaFold2 (Evoformer 的 Pair Representation) 和 ProteinMPNN 中均已被验证，是赋予纯序列模型三维感知能力的 SOTA 范式。嫁接到“片段掩码扩散”赛道属于首创。

### 3. 具体实施步骤 (Action Items)
- [x] **数据处理层 (`data/dataset.py` & `geometry.py`)**：
  - 新增计算虚拟 $C_\beta$ 和 $\vec{v}_i$ 的逻辑逻辑。
  - 构建每个 PDB 样本的 $d_{ij}$ 和 $\theta_{ij}$ 成对距离矩阵。
- [x] **模型架构层 (`models/decoder.py`)**：
  - 在 Transformer Decoder 的 Attention 模块中，引入可学习的 Bias Embedding 层 (类似 Relative Position Bias)。
  - 修改 Attention 的 forward 函数，支持传入 `mask_status` 并使其与空间距离 Bias 结合，累加到 `scores (Q*K^T)` 上。
- [ ] **验证与消融实验 (Ablation Studies)**：
  - 对比：加入动态空间偏置 前 vs 后 的模型收敛速度与验证集表现。
  - 核心评估指标：空间位阻冲突率 (Clash Rate) 和侧链恢复的 RMSD。
  - 证明加入该机制后，Top1-Top2 驱动的自适应路径规划依然优于固定的从左到右 (Autoregressive) 顺序列生成。

## 核心改进二：解决二面角离散化的“周期边界断裂” (Circular Continuity in Dihedral Angles)

### 1. 理论缺陷与推导 (Theoretical Problem & Derivation)
**当前痛点：**
目前的架构将二面角（Dihedral Angles）划分为 72 个 bins（每 5 度一个），并使用标准交叉熵（Cross-Entropy, CE）进行多分类。
在 CE 损失的逻辑里，所有的类之间都是正交且等距的。但在物理世界中，角度是一个拓扑环面（Torus）——这意味着 Bin 0（$0^\circ \sim 5^\circ$）和 Bin 71（$355^\circ \sim 360^\circ$）是紧紧贴在一起的。如果真实的侧链角度是 $2^\circ$（归属于 Bin 0），但模型预测成了 $358^\circ$（归属于 Bin 71），在物理上两者几乎没有任何差别，但传统的交叉熵会给定一个极其巨大的惩罚（损失值狂飙）。这在训练平面上制造了一道“人造的能量高墙”，导致那些处于边界附近的柔性侧链极难收敛。

**推导解法（两种路线）：**
- **路线 A (平滑标签/推土机距离)**：在计算损失时应用基于高斯的“环形标签平滑 (Circular Label Smoothing)”，使得相邻的 Bin 也拥有部分目标概率分布。
- **路线 B (三角函数对齐，主流 SOTA 标准)**：在网络输出端，不输出 72 维分类 logits，而是输出一个二维的正余弦向量 $\vec{v} = (\sin \hat{\chi}, \cos \chi)$。
  损失函数改为“真实的 $(\sin \chi, \cos \chi)$ 向量与预测向量之间的均方误差或余弦极值损失”：
  $$ \mathcal{L}_{angle} = 1 - (\sin \chi \sin \hat{\chi} + \cos \chi \cos \hat{\chi}) $$
  这样既彻底消除了 360 度的数学断裂，又能够平滑地衡量物理空间误差。

### 2. 为什么可行？(Feasibility & Impact)
- **极高的投入产出比**：这是修改成本最低、但对指标（RMSD）提升最立竿见影的改动。因为它属于“纠正了原先不符合物理规律的损失函数设定”。
- **完美兼容片段网络**：您的 Transformer 解码器在输出端本就需要预测 Fragment Token 和 Dihedral Bins。引入该改进，只需要将预测分类的角度 Head（全连接层）和损失函数计算（Loss）替换掉，对主体架构没有任何侵入性，还能显著加速模型后期的收敛速度。

### 3. 具体实施步骤 (Action Items)
- [x] **模型架构层 (`models/decoder.py`)**：
  - (已采用路线A，即环形标签平滑，零侵入，维持原维度无需修改)。
- [x] **数据处理层 (`data/geometry.py`)**：
  - (已采用路线A，维持原有的离散 Bins 支持)。
- [x] **损失函数层 (`train.py` 或独立的 `loss.py`)**：
  - 弃用传统的针对角度的 `nn.CrossEntropyLoss`。
  - 已完成基于 `_circular_cross_entropy` 的环形平滑交叉熵损失计算。
- [ ] **指标回测**：
  - 对比原本的 72-bin CE 与新损失函数在长侧链（如精氨酸 R、赖氨酸 K 等拥有高度柔性、多旋转异构体的氨基酸）的恢复精度，预计能看到显著跃升。

## 核心改进三：引入物理能量引导的即时采样 (Energy-Guided Generation)

### 1. 理论缺陷与推导 (Theoretical Problem & Derivation)
**当前痛点：**
目前的 MDM 是纯“数据驱动”的统计模型。在推理阶段，模型如果看到类似“周围全都是亲水氨基酸”的口袋，可能会根据统计学给出极大置信度去塞入一个庞大的残基（如色氨酸）。但在具体的物理由于主链的微小形变，该空间可能并不足以容纳大基团，从而引发严重的原子碰撞（Steric Clashes）。纯靠统计学算出的 Top-1 置信度会导致极其致命的物理错误。

**推导解法（分类器/能量引导扩散）：**
采用前沿的 Energy-Based Guidance。在推理步中，不改变神经网络的参数，而是在计算外围置信度前，加入极轻量的**范德华力排斥势（Lennard-Jones Repulsion）**作为外力引导。
对于某 Mask 位置的原始输出概率 $P_{raw}(x)$，选取 Top-k 候选片段，快速计算其放置在该处的预期物理体积与周围已知原子 (Unmasked) 的重叠代价 $\Delta E_{vdw}$。
然后进行概率重重写（Reweighting）：
$$ P_{guided}(x) \propto P_{raw}(x) \cdot \exp(-\beta \cdot \Delta E_{vdw}) $$

### 2. 为什么可行与核心共振？(Feasibility & Core Synergies)
- **为“置信度”戴上物理枷锁**：与课题的“自适应推理”产生奇妙的化学反应。原本由于统计偏好导致置信度极高（高 Margin）的错判位置，在经过带有斥力的 $\Delta E$ 惩罚后，其大基团的概率会暴跌。这会导致该位置的 Top1-Top2 Margin 骤降，促使自适应推算法“聪明地退缩”，将其优先级推后，保留给未来更多的上下文。直接实现了生物学上的 100% Clash-Free（无重叠）护城河。

### 3. 具体实施步骤 (Action Items)
- [ ] **推理逻辑层 (`inference.py`)**：
  - 在获取模型 logit 输出，计算 Top1-Top2 margin 之前进行拦截。
  - 针对 Top-3 的高概率片段，计算其极其简化的 VDW 碰撞代价（可仅计算相对 $C_\beta$ 或预估原子球心的距离）。
  - 应用 $\exp(-\beta \Delta E)$ 重制 Softmax 概率分布，再送入自适应序列规划池中。


## 核心改进四：基于化学拓扑树的掩码策略 (Kinematic-Tree Aware Masking)

### 1. 理论缺陷与推导 (Theoretical Problem & Derivation)
**当前痛点：**
标准的 MDM 是“均匀随机掩码（Uniform Random Masking）”，所有片段 Token 被盖住的概率均等。但在真实世界的侧链中，原子呈现层级依赖（如 $C_\gamma$ 取决于 $C_\beta$ 的位置）。如果在训练中给了模型外围的末端（如胍基），却要求它猜靠近根部的片段，这违背了生物分子的运动学树（Kinematic Tree）逻辑，导致模型学到“假条件概率”。

**推导解法（非对称拓扑掩码）：**
打破均匀分布，引入**树形层级先验 (Hierarchical Prior)** 前向加噪。
给片段划定拓扑层级（如 $L_0$ 靠主干，$L_1$ 靠外）。修改训练时的 Mask 策略：如果在构造扰动数据时决定 Mask 掉 $L_1$ 节点，那么强制其外围所有子节点 ($L_{>1}$) 必须同时被 Mask；若保留外围节点，则必须保留其父节点。

### 2. 为什么可行与核心共振？(Feasibility & Core Synergies)
- **动态生成路径的极致物理美感**：一旦模型适应了这种树形掩码的训练分布，在进行 Top1-Top2 自适应推理生成时，模型会**本能地对所有靠近主干的根部基团具有极高的置信度**。
- 这将使得网络表现出不可思议的“生长效应”——先在所有残基位置不可辩驳地长出 $L_0$ 级片段，获取这些“内层锚点”后，网络再去依据限制空间去博弈最外层的柔性基团（外层由于不确定性高，会自动落入自适应排序的队尾）。直接达到了课题所期望的“规避复杂子问题”。

### 3. 具体实施步骤 (Action Items)
- [x] **词汇表与结构层 (`data/vocabulary.py`)**：
  - 为建立的各个 Fragment 添加层级 (Level) 属性，记录其在侧链运动学树中的深度。
- [x] **数据处理层 (`data/dataset.py` 内部掩码构建逻辑)**：
  - 重写 `Collate` 或数据提取时加入的 Mask 生成函数。
  - 实现基于拓扑依赖的 Mask 传播法则（Mask 父节点必 Mask 子节点）。
- [ ] **可视化回测 (`inference.py` & 自适应轨迹记录)**：
  - 记录不同阶段 Unmask 掉的 Token 列表。验证生成轨迹是否如理论预期般呈现“从内向外、从刚性向柔性”的自适应拓展。
