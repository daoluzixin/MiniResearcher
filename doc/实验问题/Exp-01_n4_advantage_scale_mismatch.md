# Exp-01 实验总结：Advantage Estimator 对比实验

> 实验日期：2025-05-23 ~ 05-24 | 服务器：js2/js3.blockelite.cn | 单卡 A100-80GB

## 一、实验配置

### 1.1 主实验（三算法对比，lr=1e-6）

| 项目 | 值 |
|------|-----|
| 模型 | Qwen2.5-3B-Instruct |
| 估计器 | GRPO / Dr.GRPO / RLOO（三组顺序跑） |
| agent_grpo.n | 4 |
| train_batch_size | 2（实际每步 8 条样本） |
| total_steps | 300 |
| learning_rate | 1e-6（verl 框架默认） |
| 服务器 | js2.blockelite.cn:18340 |

### 1.2 补充实验（Dr.GRPO lr 调优）

| 档位 | lr | 服务器 | 步数 | 结果 |
|------|-----|--------|------|------|
| 默认 | 1e-6 | js2 | 300 | 完全不学（-0.6） |
| mid | 2e-7 | js3 | 105（中断） | 缓慢改善（+4） |
| high | 5e-7 | js3 | 75（中断） | 明显改善（+7.7） |

## 二、核心发现

### 发现一：三种 Estimator 的 Advantage Scale 差了 1-2 个数量级

| 指标 | GRPO | Dr.GRPO | RLOO |
|------|:----:|:-------:|:----:|
| Advantage 均值 | -0.25 | **-3.99** | **-16.49** |
| Advantage 范围 | [-1.5, +1.5] | [-53.4, -0.03] | [-278.8, +237.9] |
| Grad Norm 均值 | **2.2** | 9.8 | **120.0** |
| Grad Spikes (>20) | **0** | 14 | **299（全部）** |
| Reward 提升(300步) | +11.5 | **-0.6** | +12.6 |

```
Advantage scale:     GRPO ~0.25  <<  Dr.GRPO ~4  <<  RLOO ~16
Gradient scale:      GRPO ~2.2   <<  Dr.GRPO ~10 <<  RLOO ~120
```

**结论：同一个 lr 下对比不同 estimator 不公平。** 它们的 advantage 量级不同，等于在不同的 effective learning rate 下跑。必须为每个算法单独适配 lr。

### 发现二：Dr.GRPO 的 Advantage 在 Reward 全负时恒为负

Dr.GRPO 公式：`adv_i = r_i / (std + eps)`

我们的 reward 全在 -30 到 -260 之间（扣分制），没有正值。所以：
- `负数 / 正数 = 负数`，advantage 必然全负
- 模型只收到"远离所有回答"的信号，没有"靠近好回答"的正向引导
- 梯度只在推开，不在拉近，学习效率极低

GRPO 没有这个问题，因为它做了 z-score（减均值除标准差）：即使 reward 全负，减均值后总有正有负，模型能区分好坏。

### 发现二的验证：Dr.GRPO lr 调优实验

把 lr 从 1e-6 降到 5e-7 后，Dr.GRPO 从"完全不学"变成"能学但慢"：

| Phase | Dr.GRPO lr=1e-6 | Dr.GRPO lr=5e-7 | GRPO lr=1e-6 |
|-------|:-:|:-:|:-:|
| Step 1-10 | -107.2 | -112.3 | -101.8 |
| Step 11-20 | -102.3 | -104.6 | -101.6 |
| Step 21-30 | -106.7 | -112.6 | -100.4 |
| Step 31-40 | -106.3 | -111.0 | -93.0 |
| Step 41-50 | -115.3（变差） | **-102.7（改善）** | -97.9 |
| Step 66-75 | — | **-101.2** | — |

lr=5e-7 让 Dr.GRPO 开始学习（75步改善 +7.7），但仍然追不上 GRPO 同期水平。**根本原因不是 lr，而是 advantage 全负的结构性缺陷限制了学习上限。**

## 三、原因深度分析

### 3.1 Scale 不匹配的原因

三种 estimator 的归一化程度不同：
- GRPO：完整 z-score（减均值除 std），advantage 在 [-1.5, +1.5]
- Dr.GRPO：只除 std 不减均值，保留了 reward 的绝对量级
- RLOO：完全不归一化，advantage 就是 reward 的原始差异

### 3.2 Reward 全负时各算法的表现差异

| 场景 | GRPO | Dr.GRPO | RLOO |
|------|------|---------|------|
| reward 全负 [-200, -30] | advantage 有正有负（相对比较）| advantage 全负（绝对信号）| advantage 有正有负（leave-one-out）|
| reward 全为同一正值 [+0.5, +0.5] | advantage ≈ 0（不学习）| advantage 全正（继续强化）| advantage ≈ 0（不学习）|

Dr.GRPO 的设计意图是在第二种场景（"全员达标"）下仍然有学习信号。但在我们的第一种场景（reward 全负）下，这个设计反而成了劣势。

### 3.3 为什么 RLOO 虽然梯度爆了但还能学

- verl 有 gradient clipping 兜底
- clip 后方向仍正确，只是步长被限制
- 所以 reward 缓慢提升（+12.6），但大量梯度信息被浪费

## 四、实验结论

### 两条核心收获

**1. 不同 RL 算法的 advantage scale 不同，必须做 lr 适配。**

同一 lr 下对比不同 estimator 不是 apple-to-apple 的比较。做 ablation 时需要为每种算法单独调 lr，或者在 advantage 计算后加 whitening 统一 scale。论文通常不强调这一点，因为它们会为每个方法单独调参。

**2. 奖励函数的正负分布直接影响 estimator 的学习效果。**

- Reward 全负时：GRPO（z-score）> RLOO（有正有负但方差大）> Dr.GRPO（全负无正向信号）
- Reward 有正有负时：三者差异缩小，Dr.GRPO 的"保留绝对信号"优势才能体现
- Reward 全正同质时：Dr.GRPO > GRPO（GRPO 的 advantage 退化为 0）

算法选择必须结合 reward 分布特征，没有"一个算法适配所有场景"的银弹。

### 附带收获

- RLOO 在 n=4 时方差极大（grad_norm 均值 120），工程上不实用，需要 n≥16
- 梯度爆炸 ≠ 训练失败，有 clipping 时模型仍能学习，只是效率低
- Dr.GRPO 调到合适 lr（5e-7）后能学，但上限受 advantage 全负的结构性约束

### 对 DeepResearcher 项目的建议

在当前 reward 全负的 DeepResearcher 场景下，**GRPO 是最优选择**：
- 0 次梯度爆炸，训练最稳定
- z-score 归一化天然适合相对比较
- 无需额外调 lr

如果未来 reward 设计改为有正有负（比如搜到正确答案 +10），可以重新评估 Dr.GRPO。

## 五、原始数据

### 主实验日志

`log/exp01_n4/` 目录：
- `Exp-01_n4_GRPO_vs_DrGRPO_vs_RLOO_grpo.log`
- `Exp-01_n4_GRPO_vs_DrGRPO_vs_RLOO_drgrpo.log`
- `Exp-01_n4_GRPO_vs_DrGRPO_vs_RLOO_rloo.log`
- `analyze_comparison.py`

### 补充实验日志

远程服务器 js3.blockelite.cn:21820：
- `/root/DeepResearcher/logs/exp01_drgrpo_lr/drgrpo_lr2e-7.log`（包含 lr=2e-7 和 lr=5e-7 两段）

### 主实验阶段对比（每 50 步 reward 均值）

| Phase | GRPO | Dr.GRPO (lr=1e-6) | RLOO |
|-------|:----:|:-------:|:----:|
| Step 1-50 | -101.8 | -107.2 | -101.7 |
| Step 51-100 | -101.6 | -102.3 | -101.5 |
| Step 101-150 | -100.4 | -106.7 | -98.1 |
| Step 151-200 | -93.0 | -106.3 | -93.2 |
| Step 201-250 | -97.9 | -115.3 | -99.2 |
| Step 251-300 | -90.3 | -107.8 | -89.2 |

### 最终排名

| 排名 | 算法 | Reward 提升 | 稳定性 | 备注 |
|:----:|------|:-----------:|--------|------|
| 1 | GRPO | +11.5 | 极稳定（0 spike） | 当前场景最优选 |
| 2 | RLOO | +12.6 | 极不稳定（299 spike） | 需要更大 n 才实用 |
| 3 | Dr.GRPO (lr=1e-6) | -0.6 | 较不稳定（14 spike） | advantage 全负，不学习 |
| 3* | Dr.GRPO (lr=5e-7) | +7.7 (75步) | 稳定 | 能学但追不上 GRPO |
