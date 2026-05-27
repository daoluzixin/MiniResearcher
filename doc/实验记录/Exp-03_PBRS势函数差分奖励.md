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

### 7.1 Exp-3a 实际运行记录

#### 7.1.1 运行概况

- **日期**：2025-05-26
- **服务器**：AutoDL 单卡 A100-80G（后期因端口变化切换实例）
- **SSH**：`ssh -p 50141 root@connect.westd.seetacloud.com`（密码 `Fo/AL868s0DE`）
- **框架**：verl 0.2.0.dev + Ray + vLLM，DrGRPO 算法
- **搜索环境**：本地离线缓存 proxy（端口 8890），Jaccard fuzzy 匹配

#### 7.1.2 实际训练配置

```bash
/root/miniconda3/bin/python verl/trainer/main_ppo.py \
    actor_rollout_ref.model.path="/root/models/Qwen/Qwen2___5-3B-Instruct" \
    actor_rollout_ref.model.use_lora=true \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    data.train_files="/root/DeepResearcher/data/train.parquet" \
    data.train_batch_size=12 \
    data.max_response_length=8192 \
    algorithm.adv_estimator=drgrpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.total_training_steps=100 \
    trainer.save_freq=20 \
    trainer.n_gpus_per_node=1 \
    max_turns=3 \
    agent_grpo.n=2 \
    search_engine="rag" \
    reward_model.use_pbrs=false \
    trainer.resume_mode=auto
```

核心参数：batch=12, n=2, max_turns=3, total_steps=100, save_freq=20, 单卡。

#### 7.1.3 训练过程时间线

| 阶段 | Step 范围 | 说明 |
|------|-----------|------|
| Run 1-6 | 1-20 | 早期调试，修复各种启动问题 |
| Run 7 前半 | 21-37 | 从 step 20 checkpoint 续训，缓存为 540 条高质量本地数据 |
| 缓存热更新 | step 38 左右 | 另一个会话补充缓存到 13,589 条，重启 proxy |
| Run 7 后半 | 38-73 | 继续训练，step 73 后崩溃 |

总训练时间约 1.5 小时（step 21-73，每步约 60-80s）。

#### 7.1.4 遇到的问题及解决

**问题 1：GPU 残留进程导致 OOM**

杀掉 trainer 后，`ray::WorkerDict` 仍占 14GB VRAM，新训练启动时 OOM。

解决：`ray stop --force` + `kill` 残留 PID + 等待显存释放。

**问题 2：PyTorch 2.6 checkpoint 加载失败**

```
UnpicklingError: Weights only load failed... numpy.core.multiarray.scalar
```

PyTorch 2.6 默认 `weights_only=True`，但 verl checkpoint 的 extra_state 包含 numpy 数组。

解决：在 `verl/utils/checkpoint/fsdp_checkpoint_manager.py` 中三处 `torch.load` 添加 `weights_only=False`。

**问题 3：搜索缓存质量极差**

初始缓存 12,636 条（49MB），经分析 75% 是英文 query 对应中文无关搜索结果（垃圾数据）。模型产生 2296 个不同 query，精确命中率 0%，fuzzy 匹配（≥0.2）仅 28%。

解决：上传本地高质量缓存（540 条 → 后补充至 2231 条），重启 proxy。但训练中大部分时间使用的是低质量缓存。

**问题 4：序列超长导致 AssertionError 崩溃（Step 73）**

```
AssertionError: max_token_len must be greater than the sequence length. 
Got max_token_len=8192 and max_seq_len=9753
```

多轮 rollout 拼接后某个样本达到 9753 token，超过 `ppo_max_token_len_per_gpu=8192`。

解决方案（未执行，留待后续实验）：将 `ppo_max_token_len_per_gpu` 提升至 16384。

#### 7.1.5 核心指标趋势（Step 21-73）

| 指标 | 早期 (21-30) | 中期 (31-50) | 后期 (51-73) | 趋势判断 |
|------|-------------|-------------|-------------|---------|
| reward (mean) | -0.24 ~ -0.92 | -0.07 ~ -0.96 | -0.53 ~ -0.88 | 无改善，高方差震荡 |
| entropy | 0.56 ~ 1.52 | 0.64 ~ 1.83 | 0.68 ~ 1.59 | 无坍缩，维持多样性 |
| KL loss | 0.001 ~ 0.011 | 0.001 ~ 0.012 | 0.001 ~ 0.019 | 极小，策略未偏离 |
| search_depth | 1.0 ~ 1.6 | 1.0 ~ 1.6 | 1.0 ~ 1.6 | 无上升，未学会深搜 |
| info_gain | 0 ~ 0.06 | 0 ~ 0.03 | 0 ~ 0.06 | 接近零，搜索无效 |
| grad_norm | 2608 ~ 18587 | 5025 ~ 23048 | 5436 ~ 18762 | 波动大，不稳定 |

#### 7.1.6 分析与结论

**核心发现**：纯 F1 reward 在低质量搜索环境下完全无法驱动学习。

具体表现：

1. **Reward 没有学习曲线**——前期偶有好 batch（-0.07），但中后期稳定卡在 -0.65 ~ -0.88，模型收敛到"稳定的差"。
2. **Search depth 始终 1.0-1.6**——模型没学会多轮搜索，因为搜索不提供正向反馈（info_gain ≈ 0）。
3. **无过拟合**——100 条数据跑 ~6 epoch 但 entropy 没坍缩、KL 极小，说明模型连"拟合"都做不到。
4. **根本原因是环境而非算法**——2296 个 model query 在缓存中精确命中 0%，fuzzy 匹配返回大量无关内容，模型从搜索中获取不到有用信息。

**作为 baseline 的价值**：

- 证明了"纯 outcome reward + 噪声搜索环境 = 模型无法学习有效搜索策略"
- 为 Exp-3b（PBRS）提供了对照基线
- 明确了搜索缓存质量是实验成功的前置条件

#### 7.1.7 留待改进

1. 缓存已补充至 2231 条高质量数据（针对模型实际 query 搜索），后续实验使用
2. `ppo_max_token_len_per_gpu` 需提升至 16384 避免超长序列崩溃
3. 考虑扩展训练数据到 200 条以增加样本多样性

---

### 7.2 结果对比表

| 实验 | γ | Reward 均值 | Search Depth | Info Gain | 状态 |
|------|---|------------|-------------|-----------|------|
| 3a baseline | - | -0.72 | 1.25 | ≈0 | 完成（73/100步） |
| 3b PBRS | 0.9 | | | | 待运行 |
| 3c PBRS | 0.5 | | | | 待运行 |
| 3d PBRS | 1.0 | | | | 待运行 |

### 7.3 结论

- **最佳 γ 值**：________（待 3b/3c/3d 完成后对比）
- **PBRS 是否改善了搜索行为**：________
- **3a Baseline 结论**：纯 F1 reward 在噪声搜索环境下无法驱动多轮搜索学习，模型理性选择"少搜或不搜"，reward 无改善趋势
