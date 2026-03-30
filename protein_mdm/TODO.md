
## 核心修复：架构与数据流的致命隐患修复 (Critical Bug Fixes)

### 🚨 隐患 1：残基-片段的跨重映射断裂（最致命的架构缺陷）
- **问题**：`[MASK]` 预测由于只加了一维绝对位置编码，在 Cross-Attention 时不知道自己从属在主链的哪个氨基酸上。就像让模型“盲猜”主链对应的位置。
- **解决方案**：在 `models/decoder.py` 中，显式地将对应主链 `memory` 节点的隐藏特征通过 `fragment_residue_idx` 映射并加和到 `tgt_emb` 初始特征中。让每个片段天然具有“我是骨架第N条侧链”的身份记忆。
- **状态**：[x] 已修复

### 🚨 隐患 2：Decoder 的 Padding Mask 缺失（注意力污染）
- **问题**：处理变长片段序列时，由 `<PAD>` （即 0）填充的部分完全没有被 Mask 掉（`tgt_key_padding_mask = None` 占位）。有效 Token 被海量空占位的填充特征污染了所有的注意力。
- **解决方案**：引入基于变长 `target_fragments` 的真实长度掩码。将 PAD 部分传递给 Transformer 原生的 `tgt_key_padding_mask`，彻底屏蔽无意义填充的影响。
- **状态**：[x] 已修复

### 🚨 隐患 3：数据集预处理阉割了复杂侧链 ($\chi_2-\chi_4$)
- **问题**：`data/dataset.py` 中目前写死了强制策略：每个氨基酸仅仅监督第一段 Token (`chi1`)，后面的直接拉平为 `0.0` 且置为 `valid_mask=False`。“全原子构象预测”直接退化成了“一维刚棍预测”。
- **解决方案**：开放后续所有的有效片段监督，将各个基团对应的相对角度恢复至 `torsion_raw` 和 `torsion_valid_mask` 中，彻底解封长侧链的表达能力。
- **状态**：[x] 已修复
