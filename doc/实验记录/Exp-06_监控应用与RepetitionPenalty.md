# Exp-6 — 监控体系应用：diversity 告警 + Repetition Penalty

> 基于 Exp-5 监控体系，定位并修复训练过程中的 query diversity 骤降问题。

## 一、背景与目标

根据 Exp-5 的监控体系，在 Exp-2/Exp-3 训练过程中观察到一个典型问题：step 800 附近 query diversity 骤降至 15%（模型锁定单一搜索模板，反复用同一个 query 搜索）。

本实验设计 query-level repetition penalty（区别于 n-gram penalty），在 reward 层面惩罚重复搜索，验证 diversity 恢复和 F1 提升效果。

**目标**：
1. diversity 从骤降点恢复至 45% 以上
2. F1 reward 额外提升 4.2 个点

## 二、环境准备

使用 Exp-5 中监控体系记录的告警历史作为输入：
- diversity 告警触发步：step ≈ 800
- 告警时 diversity 值：15%
- 需要使用 Exp-2 或 Exp-3 训练日志中的告警时刻 checkpoint

## 三、实验原理

### 3.1 Repetition Penalty vs n-gram penalty

| 方法 | 作用层级 | 机制 | 副作用 |
|------|---------|------|--------|
| n-gram penalty | token logits | 对已出现的 n-gram 施加惩罚 | 影响正常术语重复（如科学概念多次出现） |
| **Query-level repetition penalty（ours）** | reward 层面 | 第 t 轮 query 与前 t-1 轮任一 query 的 BLEU > 0.7 时，该轮 reward 额外扣 -0.2 | 仅影响多轮搜索中的重复搜索行为，不影响单轮内 token 生成 |

### 3.2 实现方式

```python
def compute_repetition_penalty(sample_queries, penalty=-0.2, threshold=0.7):
    """
    在 reward 计算后，对重复搜索施加额外惩罚
    第 t 轮的 query 与前 t-1 轮任意一个的 BLEU > threshold → 该轮 reward += penalty
    """
    def bleu(seq1, seq2):
        # 简化 BLEU
        ...

    penalties = []
    for t, query in enumerate(sample_queries):
        if t == 0:
            penalties.append(0.0)
            continue
        max_bleu_with_history = max(bleu(query, sample_queries[i]) for i in range(t))
        if max_bleu_with_history > threshold:
            penalties.append(penalty)
        else:
            penalties.append(0.0)
    return penalties  # 与 turn-level rewards 对齐
```

将 penalties 加到 turn-level rewards 中，参与策略梯度计算。模型通过 RL 梯度学到「重复搜索不划算」。

## 四、操作步骤

### 4.1 消融实验设计

| 实验 | 方案 | 说明 |
|------|------|------|
| 6a | 无 repetition penalty | baseline，验证 diversity 告警出现 |
| 6b | repetition penalty -0.1 | 轻量惩罚 |
| 6c | repetition penalty -0.2（ours） | 推荐强度 |
| 6d | repetition penalty -0.5 | 强惩罚，验证是否会过度惩罚 |

### 4.2 定位告警步并从 checkpoint 启动

```bash
# 从 Exp-5 的告警历史中找到 diversity 骤降的 step
python -c "
import json
with open('./checkpoints/exp2_*/ckpt_history.json') as f:
    data = json.load(f)
    # 找 diversity < 0.15 的 first step
"
```

```bash
# 实验 6a: 无干预 baseline（从告警步后继续训练）
nohup torchrun --nproc_per_node=2 --master_port=29900 \
    grpo_agent_train.py --mode train \
    --from_resume 1 --resume_mode 800 \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp6a_no_penalty \
    --use_repetition_penalty 0 \
    --use_curriculum 1 --use_early_stop_penalty 1 \
    ... > logs/exp6/6a_baseline_$(date +%m%d_%H%M).log 2>&1 &

# 实验 6b: repetition penalty -0.1
nohup torchrun --nproc_per_node=2 --master_port=29901 \
    grpo_agent_train.py --mode train \
    --from_resume 1 --resume_mode 800 \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp6b_penalty_01 \
    --use_repetition_penalty 1 --repetition_penalty 0.1 --repetition_threshold 0.7 \
    ... > logs/exp6/6b_penalty01_$(date +%m%d_%H%M).log 2>&1 &

# 实验 6c: repetition penalty -0.2（ours）
nohup torchrun --nproc_per_node=2 --master_port=29902 \
    grpo_agent_train.py --mode train \
    --from_resume 1 --resume_mode 800 \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp6c_penalty_02 \
    --use_repetition_penalty 1 --repetition_penalty 0.2 --repetition_threshold 0.7 \
    ... > logs/exp6/6c_penalty02_$(date +%m%d_%H%M).log 2>&1 &

# 实验 6d: repetition penalty -0.5
nohup torchrun --nproc_per_node=2 --master_port=29903 \
    grpo_agent_train.py --mode train \
    --from_resume 1 --resume_mode 800 \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp6d_penalty_05 \
    --use_repetition_penalty 1 --repetition_penalty 0.5 --repetition_threshold 0.7 \
    ... > logs/exp6/6d_penalty05_$(date +%m%d_%H%M).log 2>&1 &
```

### 4.3 核心超参说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_repetition_penalty` | 0 | 开启 repetition penalty |
| `--repetition_penalty` | 0.2 | 惩罚强度（负值） |
| `--repetition_threshold` | 0.7 | BLEU 阈值，超过视为重复 |
| `--from_resume` | 0 | 从 checkpoint 恢复 |
| `--resume_mode` | latest | 恢复模式（latest/best/step_number） |

### 4.4 监控指标

| 指标 | 健康范围 | 预期变化 |
|------|----------|---------|
| query_diversity | > 40% | penalty 施加后 diversity 回升 |
| penalty_occurrence_rate | < 30% | 过高 = 阈值设得太低 |
| F1 reward（vs 6a） | 额外提升 4.2+ | 验证 diversity 恢复 → 真实能力提升 |

## 五、实验记录

### 5.1 结果对比表（实验后填写）

| 实验 | penalty | diversity 恢复至 | F1 reward 提升（vs 6a） | penalty_occurrence_rate |
|------|---------|----------------|----------------------|----------------------|
| 6a baseline | 0 | | 0（baseline） | |
| 6b | -0.1 | | | |
| 6c ours | -0.2 | | | |
| 6d | -0.5 | | | |

### 5.2 结论

- **diversity 从 15% 恢复到多少**：________
- **F1 reward 额外提升多少个点**：________（预期 ≥ 4.2）
- **penalty 是否会误伤正常的同义词替换搜索**：________
- **threshold 0.7 是否合适**：________
