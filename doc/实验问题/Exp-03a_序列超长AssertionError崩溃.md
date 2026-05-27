# Exp-03a：序列超长导致 AssertionError 崩溃（Step 73）

**日期**: 2025-05-26
**实验**: Exp-3a baseline（无 PBRS）
**服务器**: AutoDL 单卡 A100-80G

---

## 现象

训练在 Step 73 崩溃，报错：

```
AssertionError: max_token_len must be greater than the sequence length. 
Got max_token_len=8192 and max_seq_len=9753
```

## 根因

### 参数含义

- `ppo_max_token_len_per_gpu=8192`：actor update 阶段 dynamic batching 的最大 token 长度限制
- 该参数用于 `verl/trainer/config/ppo_trainer.yaml` 中控制 micro-batch 划分

### 为什么会超长

多轮 rollout（max_turns=3）中，某个样本的完整拼接长度计算：

```
total_tokens = prompt_tokens + response_tokens (turn 1) + search_result_tokens (turn 1) 
             + response_tokens (turn 2) + search_result_tokens (turn 2)
             + response_tokens (turn 3)
```

当搜索结果较长时，3 轮拼接后总长度可达 9000-10000 tokens，超过 8192 限制。

### 触发条件

- 模型实际做了 3 轮搜索（而非 1 轮就给出答案）
- 每轮搜索结果内容较长（fuzzy 匹配返回的无关结果可能很长）
- prompt 本身也较长

## 解决方案

将 `ppo_max_token_len_per_gpu` 从 8192 提升至 **16384**：

```bash
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
```

### 显存影响评估

- 16384 tokens 的单条序列 backward 约需额外 2-4GB 显存
- A100-80G 上有充足余量（训练中峰值 ~50-60GB）
- 如果使用 A100-40G，需要同时降低 batch_size 或开启 gradient checkpointing

### 替代方案

如果不想增大显存开销，可以在 rollout 阶段截断：

```python
# 在 rollout 后、actor update 前增加截断逻辑
max_allowed_len = 8192
if seq_len > max_allowed_len:
    # 方案 A: 截断末尾
    sequence = sequence[:max_allowed_len]
    # 方案 B: 丢弃该样本
    continue
```

但截断会损失多轮交互的完整性，**推荐直接提升限制**。

## 预防措施

```bash
# 在训练配置中，ppo_max_token_len_per_gpu 应为：
# max_response_length + 最大 prompt 长度 + 安全余量
# 例如：8192 (max_response) + 1024 (prompt) + 1024 (buffer) = 10240
# 保守设置 16384 即可覆盖绝大多数情况
```

## 适用场景

当以下条件同时满足时容易触发此问题：
1. 多轮搜索 Agent（max_turns > 1）
2. 搜索结果未做长度截断
3. `ppo_max_token_len_per_gpu` 设置为默认值 8192
4. `max_response_length` 也设为 8192

## 教训

- `ppo_max_token_len_per_gpu` 是一个硬约束（assert），不是软限制
- 多轮 Agent 的实际序列长度 = prompt + 所有轮次的 response + 所有轮次的 search result，远超单轮场景
- 建议设置为 `max_response_length × 2` 作为保底
