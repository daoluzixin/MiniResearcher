# Exp-2 — Curriculum 热身 + Mode Collapse 消融实验

> 双卡 A100-40G 环境 + DuckDuckGo 搜索，解决 3B 模型冷启动时倾向于第 1 轮就终止搜索的问题。
> 数据集：100 条（从 80k 中采样），num_generations=4，epochs=2（约 200 steps）。

## 一、背景与目标

3B 模型冷启动时倾向于第 1 轮就输出 `<answer>` 终止搜索以获取格式正确的基础分。这导致训练初期几乎所有轨迹都是短轨迹（深度 = 1），模型永远学不会多轮搜索。

本实验验证 Curriculum 热身（max_turns 递增）+ early-stop 惩罚（未满最低轮次终止扣分）+ entropy bonus（防止后期探索不足）三招联用的效果。目标：平均轨迹长度从 1.2 轮提升至 4.8 轮。

## 二、环境准备

与 Exp-1 基本相同，核心区别：

| 项目 | Exp-1 | Exp-2 |
|------|-------|-------|
| 数据集 | 80,000 条 | 100 条（快速验证） |
| num_generations | 2 | 4 |
| search_engine | rag | duckduckgo |
| 总 steps | 3000 | 200 |

模型使用 Qwen2.5-3B-Instruct。数据集已保存至 `data/web_search_agent_100.parquet`。

## 三、实验原理

### 3.1 三个机制的协同设计

**Curriculum 热身**：`max_turns` 从 1→3→10 递增，让模型先学会基本工具调用（深度=1），再逐步扩展到多轮搜索。

切换依据：最近 100 个 batch 的平均 F1 reward > 0.3 时升级到下一阶段。如果升级后 reward 持续下降超过 200 步则回退。

**early-stop 惩罚**：若 `actual_turns < min_turns` 且模型输出了 `<answer>`，则额外扣 -0.5 reward。

这是「强制策略」——模型必须先学会多轮搜索才能获得正 reward，不允许用短轨迹骗基础分。

**entropy bonus**：`H_coeff(step) = H_init × cos(π × step / (2 × total_steps))`，前期 H_init=0.01 鼓励探索，防止模型过早锁定单一搜索模板。后期衰减至 0 让策略收敛。

### 3.2 代码实现

在 `grpo_agent_train.py` 中：

```python
# Curriculum: max_turns 递增
max_turns = curriculum_schedule(step, stages=[1, 3, 10], thresholds=[0.0, 0.3, 0.5])

# early-stop 惩罚
if actual_turns < min_turns and has_answer_tag:
    final_reward -= 0.5

# entropy bonus
entropy = compute_policy_entropy(logits)
loss = policy_loss - H_init * math.cos(math.pi * step / (2 * total_steps)) * entropy
```

## 四、操作步骤

### 4.1 消融实验设计

本实验包含 6 个子实验，验证每个机制的独立贡献：

| 实验 | Curriculum | early-stop 惩罚 | entropy bonus | 预期 |
|------|-----------|----------------|---------------|------|
| 2a | ❌ | ❌ | ❌ | baseline，预期深度≈1.2 |
| 2b | ✅ | ❌ | ❌ | 仅 Curriculum |
| 2c | ❌ | ✅ | ❌ | 仅 early-stop |
| 2d | ❌ | ❌ | ✅ | 仅 entropy bonus |
| 2e | ✅ | ✅ | ❌ | Curriculum + early-stop |
| 2f（ours） | ✅ | ✅ | ✅ | 全部机制（最终方案） |

### 4.2 启动命令

```bash
# 实验 2a: 无干预 baseline
nohup torchrun --nproc_per_node=2 --master_port=29600 \
    grpo_agent_train.py --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp2a_baseline \
    --epochs 10 --batch_size 1 --learning_rate 1e-6 \
    --num_generations 8 --beta 0.1 \
    --gradient_checkpointing 1 --dtype bfloat16 \
    --use_wandb > logs/exp2/2a_baseline_$(date +%m%d_%H%M).log 2>&1 &

# 实验 2b: 仅 Curriculum
nohup torchrun --nproc_per_node=2 --master_port=29601 \
    grpo_agent_train.py --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp2b_curriculum \
    --use_curriculum 1 --curriculum_stages 1,3,10 --curriculum_thresholds 0.0,0.3,0.5 \
    ... > logs/exp2/2b_curriculum_$(date +%m%d_%H%M).log 2>&1 &

# 实验 2c: 仅 early-stop penalty
nohup torchrun --nproc_per_node=2 --master_port=29602 \
    grpo_agent_train.py --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp2c_earlystop \
    --use_early_stop_penalty 1 --early_stop_penalty 0.5 --min_turns 2 \
    ... > logs/exp2/2c_earlystop_$(date +%m%d_%H%M).log 2>&1 &

# 实验 2d: 仅 entropy bonus
nohup torchrun --nproc_per_node=2 --master_port=29603 \
    grpo_agent_train.py --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp2d_entropy \
    --use_entropy_bonus 1 --entropy_bonus_init 0.01 --entropy_decay cosine \
    ... > logs/exp2/2d_entropy_$(date +%m%d_%H%M).log 2>&1 &

# 实验 2e: Curriculum + early-stop
nohup torchrun --nproc_per_node=2 --master_port=29604 \
    grpo_agent_train.py --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp2e \
    --use_curriculum 1 --curriculum_stages 1,3,10 --curriculum_thresholds 0.0,0.3,0.5 \
    --use_early_stop_penalty 1 --early_stop_penalty 0.5 --min_turns 2 \
    ... > logs/exp2/2e_curriculum_earlystop_$(date +%m%d_%H%M).log 2>&1 &

# 实验 2f: 全部机制（ours）
nohup torchrun --nproc_per_node=2 --master_port=29605 \
    grpo_agent_train.py --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp2f_ours \
    --use_curriculum 1 --curriculum_stages 1,3,10 --curriculum_thresholds 0.0,0.3,0.5 \
    --use_early_stop_penalty 1 --early_stop_penalty 0.5 --min_turns 2 \
    --use_entropy_bonus 1 --entropy_bonus_init 0.01 --entropy_decay cosine \
    ... > logs/exp2/2f_ours_$(date +%m%d_%H%M).log 2>&1 &
```

### 4.3 核心超参说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_curriculum` | 0 | 开启 Curriculum |
| `--curriculum_stages` | "1,3,10" | max_turns 递增阶段 |
| `--curriculum_thresholds` | "0.0,0.3,0.5" | 切换阈值（F1 reward moving average） |
| `--use_early_stop_penalty` | 0 | 开启 early-stop 惩罚 |
| `--early_stop_penalty` | 0.5 | 惩罚强度 |
| `--min_turns` | 2 | 最低允许轮次 |
| `--use_entropy_bonus` | 0 | 开启 entropy bonus |
| `--entropy_bonus_init` | 0.01 | 初始系数 |
| `--entropy_decay` | cosine | 衰减曲线 |

### 4.4 监控指标

新增以下指标用于本实验：

| 指标 | 健康范围 | 异常信号 |
|------|----------|----------|
| search_depth | 逐步上升至 4~6 | 持续 < 2 = 仍然 short-circuit |
| depth < min_turns rate | < 10% | > 30% = early-stop penalty 强度不够 |
| entropy | 前期高、后期低 | 后期 entropy 不下降 = 模型未收敛 |
| Curriculum stage | 逐步升级 | stage 不升级 = 阈值设置过高 |

## 五、显存分析

3B 模型 + num_generations=4 约需 24GB/卡（相比 n=8 的 42GB 更轻量）。A100-40G 无需 gradient_checkpointing 也能运行，但开启也无妨。

## 六、实验记录

### 6.1 结果对比表（实验后填写）

| 实验 | 平均轨迹深度（终态） | F1 reward（终态） | 达到深度>3的 step | 备注 |
|------|---------------------|-----------------|------------------|------|
| 2a baseline | | | | |
| 2b 仅 Curriculum | | | | |
| 2c 仅 early-stop | | | | |
| 2d 仅 entropy | | | | |
| 2e Curriculum+early-stop | | | | |
| 2f 全部（ours） | | | | |

### 6.2 结论

- **三招各自的贡献量**：________
- **组合效果是否显著优于单招**：________
- **是否触发过 stage 回退**：________
- **entropy bonus 衰减是否符合预期（cosine 曲线）**：________
