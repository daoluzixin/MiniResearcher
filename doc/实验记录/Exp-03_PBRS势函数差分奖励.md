# Exp-3 — PBRS 势函数差分奖励

> 双卡 A100-40G 环境，在 token-level 注入过程奖励，解决多轮搜索前期行为无梯度信号的问题。

## 一、背景与目标

原始框架仅在最终 `<answer>` 标签计算 F1 作为 outcome reward。在 5~10 轮长轨迹中，前 9 轮搜索行为完全无梯度信号——无论搜索得好与坏，reward 都是 0，直到最后一步才能区分。

本实验设计 PBRS（Potential-Based Reward Shaping）势函数差分奖励：对每轮 `<tool_call>` 的 query 质量（与 GT 的 token overlap）计算信息增益势差，注入 token-level reward tensor。目标：策略梯度方差降低约 60%。

## 二、环境准备

同 Exp-1。模型可使用 Exp-2 训练后的 checkpoint（已有一定搜索能力），或从头训练。

## 三、实验原理

### 3.1 势函数设计（Ng et al. 1999）

附加奖励形式：`F(s, s') = γ·Φ(s') - Φ(s)`

关键性质：如果势函数设计满足上述形式，最优策略不变（policy invariance theorem）。

直觉：势差在无穷 horizon 上 telescoping 求和为 `γ^T·Φ(s_T) - Φ(s_0)`，即只是 value function 平移了常数，不改变策略梯度方向。

### 3.2 势函数 Φ(s_t) 的具体定义

s_t = 第 t 轮搜索后的状态（已获取的信息集合）

```
Φ(s_t) = token_overlap(accumulated_info_t, GT_answer) / |GT_tokens|
```

即：到当前轮次为止，收集到的信息与 ground truth 的覆盖率。

每轮势差：
- 好搜索（覆盖新信息）→ 正奖励
- 无用搜索（没覆盖新信息）→ 零奖励
- 重复搜索（与之前重叠）→ 势差为零，不奖不罚

### 3.3 实现方式

在 `reward_manager` 中维护 per-sample 的 `accumulated_tokens` 集合：

```python
def compute_pbrs_reward(sample, gt_tokens, gamma=0.9):
    """
    对每轮 tool_call 的势函数差分 reward
    返回 token-level reward tensor（对应该轮生成 token 位置）
    """
    accumulated = set()
    turn_rewards = []

    for turn_idx, tool_result in enumerate(sample.tool_results):
        new_tokens = extract_new_tokens(tool_result.content)
        accumulated.update(new_tokens)

        # 当前势
        phi_t = len(accumulated & gt_tokens) / len(gt_tokens)
        # 前一步势
        phi_prev = 0.0 if turn_idx == 0 else turn_phi_history[turn_idx - 1]
        # 势差 = 该轮的信息增益
        potential_diff = gamma * phi_t - phi_prev

        turn_rewards.append(potential_diff)
        turn_phi_history.append(phi_t)

    # 注入到对应 token 位置
    return inject_turn_rewards(sample.response_tokens, turn_rewards)
```

## 四、操作步骤

### 4.1 消融实验设计

| 实验 | 方案 | 核心假设 |
|------|------|----------|
| 3a | 无 PBRS（baseline） | 只有 outcome F1 reward |
| 3b | PBRS with γ=0.9 | 完整势函数差分 |
| 3c | PBRS with γ=0.5 | 更激进的短期信号 |
| 3d | PBRS with γ=1.0 | 不衰减，全部势差等权累加 |

### 4.2 启动命令

```bash
# 实验 3a: 无 PBRS baseline
nohup torchrun --nproc_per_node=2 --master_port=29700 \
    grpo_agent_train.py --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp3a_no_pbrs \
    --use_pbrs 0 \
    --epochs 10 --batch_size 1 --learning_rate 1e-6 \
    --num_generations 8 --beta 0.1 \
    --gradient_checkpointing 1 --dtype bfloat16 \
    --use_wandb > logs/exp3/3a_baseline_$(date +%m%d_%H%M).log 2>&1 &

# 实验 3b: PBRS with γ=0.9（推荐）
nohup torchrun --nproc_per_node=2 --master_port=29701 \
    grpo_agent_train.py --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp3b_pbrs_gamma09 \
    --use_pbrs 1 --pbrs_gamma 0.9 \
    --epochs 10 --batch_size 1 --learning_rate 1e-6 \
    --num_generations 8 --beta 0.1 \
    --gradient_checkpointing 1 --dtype bfloat16 \
    --use_wandb > logs/exp3/3b_pbrs09_$(date +%m%d_%H%M).log 2>&1 &

# 实验 3c: PBRS with γ=0.5
nohup torchrun --nproc_per_node=2 --master_port=29702 \
    grpo_agent_train.py --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp3c_pbrs_gamma05 \
    --use_pbrs 1 --pbrs_gamma 0.5 \
    ... > logs/exp3/3c_pbrs05_$(date +%m%d_%H%M).log 2>&1 &

# 实验 3d: PBRS with γ=1.0
nohup torchrun --nproc_per_node=2 --master_port=29703 \
    grpo_agent_train.py --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp3d_pbrs_gamma10 \
    --use_pbrs 1 --pbrs_gamma 1.0 \
    ... > logs/exp3/3d_pbrs10_$(date +%m%d_%H%M).log 2>&1 &
```

### 4.3 核心超参说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_pbrs` | 0 | 开启 PBRS |
| `--pbrs_gamma` | 0.9 | 势函数折扣系数 |
| `--pbrs_overlap_metric` | token_overlap | 信息度量方式 |
| `--pbrs_normalize` | 1 | 是否归一化势差到 [0,1] |

### 4.4 监控指标

| 指标 | 健康范围 | 异常信号 |
|------|----------|----------|
| gradient_variance | 逐步下降 | 上升 = PBRS 信号方向不对 |
| early_turn_reward_mean | > 0（有效的搜索有正信号） | 持续 = 0 = 前期搜索行为无信号 |
| info_gain_per_turn | 各轮逐步递增 | 递减 = 模型学会"占便宜" |

**梯度方差测量方法**：在训练日志中记录同一个 group 内 N 个采样的梯度范数标准差。

## 五、显存分析

PBRS 实现本身不增加额外显存占用（只是 reward 计算逻辑变化）。但需要维护 per-sample 的 `accumulated_tokens` 集合，对于长序列数据会有少量 CPU 开销。

## 六、常见问题

### Q1: 为什么用 token overlap 而不用 ROUGE/BLEU？

多轮 Agent 场景中，每次搜索结果可能包含大量无意义字符（导航栏、页眉等）。token overlap 直接衡量「新信息」的覆盖率，更鲁棒。BLEU/ROUGE 侧重文本相似度，容易被无意义重复 token 误导。

### Q2: GT answer 为空时（无 gt）怎么办？

对于开放式问题（gt=[]），PBRS 势函数退化为 0（无法计算覆盖度）。在这些样本上只使用 outcome reward，不注入过程奖励。

### Q3: 势函数会不会让模型学会「凑字数」而非真正有用的搜索？

会的——如果模型发现反复搜索同一页面能增加覆盖 token 但不提升真正有用信息，势差会给出误导信号。此时通过 query diversity 监控（Exp-5）可以发现并定位这个问题。

## 七、实验记录

### 7.0 Exp-3a 实际训练配置（精简版 baseline）

> 日期：2025-05-xx
> 服务器：AutoDL 双卡 A100-40G
> SSH：`ssh -p 50141 root@connect.westd.seetacloud.com`

**调整动机**：全量数据 + 大 batch 的训练周期太长（~7.5 min/step），先用精简配置验证流程跑通。

| 参数 | 原计划 | 3a 实际 | 说明 |
|------|--------|---------|------|
| data | web_search_agent.jsonl (全量) | web_search_agent_100.parquet (100条) | 快速迭代 |
| train_batch_size | 较大 | 12 | 适配显存 |
| n (采样数/prompt) | 8 | 2 | 减少推理开销 |
| max_turns | 5 | 3 | 减少搜索等待 |
| total_training_steps | 较多 | 100 | 快速看趋势 |
| save_freq | 50 | 10 | 更频繁保存 |
| search_engine | searxng | searxng | 不变 |
| searxng_url | - | http://localhost:8890 | 通过隧道连本地代理 |

预估训练时间：优化后 ~2-3 min/step，100 步约 3-5 小时。

---

### 7.0.1 搜索代理与网络方案

**本地搜索代理**（`/tmp/search_proxy_server.py`，端口 8890）：
- 模拟 SearXNG API（POST `/search`，返回 `{"results": [...]}`）
- 主引擎百度 + 补充 Bing，英文/中文查询均已验证通过

**SSH 反向隧道（未完成）**：
- 目标：GPU 服务器 `localhost:8890` → 本地 Mac 8890
- 现象：`remote port forwarding failed`，杀掉服务器端 8890 占用后仍失败
- 可能原因：AutoDL 平台 SSH 代理层限制端口转发 / TCP TIME_WAIT 未释放
- 备选方案：
  1. 服务器上用 `socat` / `ssh -W` 管道转发
  2. 在服务器端直接跑简易 HTTP 转发脚本
  3. 部署搜索代理到有公网 IP 的中间机器

---

### 7.1 结果对比表（实验后填写）

| 实验 | γ | 梯度方差降低幅度 | early_turn_reward 均值 | F1 reward（终态） |
|------|---|----------------|---------------------|----------------|
| 3a baseline | - | 0%（baseline） | ≈0 | |
| 3b PBRS | 0.9 | | | |
| 3c PBRS | 0.5 | | | |
| 3d PBRS | 1.0 | | | |

### 7.2 结论

- **最佳 γ 值**：________（预期 0.9）
- **梯度方差实际降低了多少**：________（预期约 60%）
- **PBRS 是否帮助了前期轮次的策略更新**：________
