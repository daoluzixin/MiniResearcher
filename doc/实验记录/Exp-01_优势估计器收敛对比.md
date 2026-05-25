# Exp-1 — 四种优势估计器收敛对比（Dr.GRPO 变种 included）

> 单卡 A100-80GB 环境，基于 `grpo_agent_train.py` 的优势估计器消融实验。

## 一、背景与目标

原始 GRPO 对 group 内 N 个采样的 reward 做 `(r_i - mean) / std` 归一化。当 group 内所有样本 reward 几乎相同时（多轮 Agent 场景下高频出现——比如所有轨迹都格式错误得 -1），std → 0 导致 advantage 爆炸或归零，梯度信号完全消失。

本实验在同一代码框架内实现并对比 GRPO / RLOO / REINFORCE++ / ReMax 四种优势估计器，同时测试 Dr.GRPO（去均值、仅除标准差）变体，评估各估计器在多轮 Agent 场景下的收敛特性。

**实验假设**：Dr.GRPO 在 group 内 reward 同质性高的场景下能保留更多梯度信号，收敛更快。

## 二、环境准备

### 2.1 硬件要求

| 项目 | 要求 |
|------|------|
| GPU | 2 × NVIDIA A100-40GB |
| CPU 内存 | ≥ 64GB |
| 磁盘 | ≥ 50GB（checkpoint ~5GB/份 × 5） |
| CUDA | ≥ 12.1 |
| NCCL | 随 PyTorch 自带即可 |

### 2.2 软件依赖

```bash
# 核心依赖（与主训练脚本一致）
pip install torch>=2.1.0 transformers>=4.37.0 accelerate modelscope
pip install flash-attn --no-build-isolation   # Flash Attention 2
pip install swanlab                             # 训练可视化
pip install numpy tokenizers
```

### 2.3 下载模型

```bash
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-3B-Instruct', cache_dir='./models')
print(f'模型已下载到: {model_dir}')
"
```

默认下载路径：`~/.cache/modelscope/hub/Qwen/Qwen2.5-3B-Instruct`。

## 三、实验原理

### 3.1 五种优势估计器数学定义

| 估计器 | 公式 | 关键特性 |
|--------|------|----------|
| GRPO | `adv_i = (r_i - μ) / σ` | 经典归一化，reward 同质时 advantage → 0 |
| Dr.GRPO（ours） | `adv_i = r_i / (σ + ε)` | 去均值、仅除标准差，保留绝对 reward 信号 |
| RLOO | `adv_i = r_i - (Σ r_j - r_i) / (N-1)` | leave-one-out baseline，无偏但方差大 |
| REINFORCE++ | token-level discounted return + whitening | 适合有中间奖励，纯 outcome reward 时退化为 GRPO |
| ReMax | `adv_i = r_i - r(gpu_i)` | greedy baseline，需额外一次推理开销 |

### 3.2 Dr.GRPO 的核心洞察

当所有样本 reward 接近时（多轮 Agent 高频场景）：

- GRPO：`μ ≈ r_all`，`σ → 0`，所以 `adv_i → 0`，梯度消失
- Dr.GRPO：`σ` 仍趋于 0，但 `r_i` 的绝对值保留（负 reward → 负 advantage → 梯度推开当前策略）

### 3.3 代码实现位置

优势估计器实现在 `grpo_agent_train.py` 的 `compute_advantages()` 函数，通过 `--advantage_estimator` 参数切换：

```python
def compute_advantages(rewards, estimator='grpo', eps=1e-8):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    if estimator == 'grpo':
        mean, std = rewards.mean(), rewards.std() + eps
        return (rewards - mean) / std
    elif estimator == 'dr_grpo':
        std = rewards.std() + eps
        return rewards / std
    elif estimator == 'rloo':
        # leave-one-out
        ...
    elif estimator == 'reinforce_plusplus':
        ...
    elif estimator == 'remax':
        ...
```

## 四、操作步骤

### 4.1 准备训练数据

同主训练数据，JSONL 格式：

```json
{"query": "搜索最新的强化学习综述论文", "gt": ["reinforcement learning survey"], "needs_tool": true}
{"query": "帮我查一下今天北京的天气", "gt": ["晴", "28°C"], "needs_tool": true}
{"query": "解释量子纠缠的概念", "gt": [], "needs_tool": false}
```

### 4.2 启动五种消融实验

建议每个 estimator 单独跑一个目录，方便对比：

```bash
mkdir -p logs/exp1

# 实验 1a: GRPO baseline
nohup torchrun --nproc_per_node=2 --master_port=29500 \
    grpo_agent_train.py \
    --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp1_grpo \
    --advantage_estimator grpo \
    --num_generations 8 \
    --epochs 10 \
    --batch_size 1 \
    --learning_rate 1e-6 \
    --beta 0.1 \
    --gradient_checkpointing 1 \
    --dtype bfloat16 \
    --save_interval 20 \
    --use_wandb > logs/exp1/grpo_$(date +%m%d_%H%M).log 2>&1 &

# 实验 1b: Dr.GRPO（ours）
nohup torchrun --nproc_per_node=2 --master_port=29501 \
    grpo_agent_train.py \
    --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp1_drgrpo \
    --advantage_estimator dr_grpo \
    --num_generations 8 \
    --epochs 10 \
    --batch_size 1 \
    --learning_rate 1e-6 \
    --beta 0.1 \
    --gradient_checkpointing 1 \
    --dtype bfloat16 \
    --save_interval 20 \
    --use_wandb > logs/exp1/drgrpo_$(date +%m%d_%H%M).log 2>&1 &

# 实验 1c: RLOO
nohup torchrun --nproc_per_node=2 --master_port=29502 \
    grpo_agent_train.py --mode train ... \
    --advantage_estimator rloo \
    --save_dir ./checkpoints/exp1_rloo \
    > logs/exp1/rloo_$(date +%m%d_%H%M).log 2>&1 &

# 实验 1d: REINFORCE++
nohup torchrun --nproc_per_node=2 --master_port=29503 \
    grpo_agent_train.py --mode train ... \
    --advantage_estimator reinforce_plusplus \
    --save_dir ./checkpoints/exp1_rpp \
    > logs/exp1/rpp_$(date +%m%d_%H%M).log 2>&1 &

# 实验 1e: ReMax
nohup torchrun --nproc_per_node=2 --master_port=29504 \
    grpo_agent_train.py --mode train ... \
    --advantage_estimator remax \
    --save_dir ./checkpoints/exp1_remax \
    > logs/exp1/remax_$(date +%m%d_%H%M).log 2>&1 &
```

### 4.3 核心超参说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--advantage_estimator` | grpo/dr_grpo/rloo/reinforce_plusplus/remax | 切换估计器 |
| `--num_generations` | 8 | GRPO group 内候选数，越大优势估计越稳定 |
| `--learning_rate` | 1e-6 | 保守学习率 |
| `--beta` | 0.1 | KL 散度惩罚系数 |
| `--batch_size` | 1 | per-GPU，3B 模型用 gradient_checkpointing |

### 4.4 监控指标

日志输出格式：

```
[Epoch 1/10] Step 10/500 | loss=0.3412 | reward=0.823 | kl=0.0041 | advantage_mean=0.215 | advantage_std=0.892
```

关键指标（所有 estimator 共用）：

| 指标 | 健康范围 | 异常信号 |
|------|----------|----------|
| reward | 逐步上升 | 持续 < 0 或剧烈震荡 |
| advantage_mean | ≈ 0（GRPO）/ < 0（全员差时 Dr.GRPO） | advantage 持续 NaN |
| advantage_std | 0.5~1.5 | 接近 0 说明 group 内 reward 同质性极高 |
| kl | < 0.1 | > 0.5 说明偏离 ref 太远 |

**收敛步数定义**：reward 第一次超过最终收敛值 90% 的 step。

### 4.5 断点续训

```bash
torchrun --nproc_per_node=2 --master_port=29500 \
    grpo_agent_train.py --mode train \
    --from_resume 1 --resume_mode latest \
    ...（其他参数不变）
```

## 五、显存分析

### 5.1 预估显存占用（单卡 per-GPU）

3B 模型 bf16，n=8 采样：

| 组件 | 大小 | 说明 |
|------|------|------|
| 模型参数 (bf16) | ~6.0 GB | 3B × 2 bytes |
| Ref model (bf16) | ~6.0 GB | 冻结，eval 模式 |
| Optimizer states | ~12.0 GB | AdamW 2×fp32 |
| 梯度 (bf16) | ~6.0 GB | 与参数等大 |
| 激活值 (checkpointing) | ~8.0 GB | 3B 较大 |
| Rollout KV Cache (n=8) | ~4.0 GB | 8 个采样 |
| **总计** | **~42 GB** | 超过 40GB，必须使用 FSDP 或减少 n |

**预计 OOM：A100-40G 双卡 DDP 跑 3B + n=8 可能超出**，需要：
1. 启用 `--gradient_checkpointing 1`（已设置）
2. 减小 `--num_generations` 到 4（作为备选）
3. 考虑后续 Exp-4 的 LoRA 方案

### 5.2 降低显存的备选方案

如果 OOM，优先减少 n（num_generations），因为 n 直接影响 KV Cache：

```bash
--num_generations 4   # 默认 8，降到 4 显存约减半
```

## 六、常见问题

### Q1: advantage_std 接近 0 但 reward 也在上升

说明 group 内 reward 分布正在从同质走向异质，是正常现象。如果 std 持续为 0 且 reward 不动，切换到 Dr.GRPO。

### Q2: ReMax 训练时间明显更长

ReMax 需要额外一次 greedy decode 作为 baseline，rollout 时间翻倍。确认 ReMax 的 `num_generations` 至少为 4（N 太小 baseline 波动剧烈）。

### Q3: 不同 estimator 的 reward 曲线量级不同

这是正常的——GRPO 的 advantage 被归一化到 N(0,1)，Dr.GRPO 和 RLOO 的 advantage 量级不同。比较时看收敛速度和最终 F1 reward，不要直接比较 raw advantage 数值。

### Q4: RLOO 在 n=2 时不稳定

RLOO 要求 n ≥ 3 才能做 leave-one-out。代码中做了保护，如果 n < 3 自动降级到 GRPO。

## 七、实验记录

### 7.1 收敛步数对比表（实验后填写）

| estimator | 收敛步数（F1 达 90%） | 最终 F1 reward | advantage_std 均值 | 备注 |
|------------|----------------------|---------------|-------------------|------|
| GRPO | | | | |
| Dr.GRPO（ours） | | | | |
| RLOO | | | | |
| REINFORCE++ | | | | |
| ReMax | | | | |

### 7.2 结论

实验完成后填写：

- **哪个估计器收敛最快**：________
- **Dr.GRPO vs GRPO 差距（Dr.GRPO 收敛步数减少 X%）**：________
- **结论是否支持假设**：________

### 7.3 Checkpoint 目录结构

```
checkpoints/exp1_{estimator}/
├── ckpt_history.json
├── best/
├── step_XX/
└── ...
```

## 八、实战排障记录（js3.blockelite.cn 部署）

### 8.1 服务器连接

| 项目 | 值 |
|------|----|
| IP | js3.blockelite.cn |
| **SSH 端口** | **21816**（注意不是默认的 22，也不是 21812，那是 VNC 转发端口） |
| 用户 | root |
| GPU | NVIDIA A100-SXM4-80GB |
| Python | `/home/vipuser/miniconda3/bin/python` |

> ⚠️ 一开始用 21812 端口一直超时，换成 21816 才连上。21812 是 VNC 端口，不是 SSH。

### 8.2 模型缓存路径

模型通过 ModelScope 下载，路径为 `~/.cache/modelscope/hub/Qwen/Qwen2.5-3B-Instruct`。

verl/vLLM 使用 HuggingFace 的 `transformers` 加载模型，默认读 `~/.cache/huggingface/`。设置 `HF_HOME` 环境变量让 HF 复用 ModelScope 缓存：

```bash
export HF_HOME="/root/.cache/modelscope/hub"
```

Qwen2.5-3B-Instruct 完整缓存约 2.4 GB。验证缓存完整性的方法：看 `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/blobs/` 下是否至少有 safetensors 文件。缓存不完整会导致反复从 HuggingFace 下载，极慢（外网）。

### 8.3 V1~V4 失败原因总结

| 版本 | 症状 | 根因 | 修复方案 |
|------|------|------|----------|
| V1 | Ray OOM Killer，进程被 SIGKILL | batch_size=64 过大 | → V2，batch_size 64→16→8 |
| V2 | SIGTERM 退出 | SSH 超时后服务器主动杀进程 | nohup 后台运行，避免 SSH 断开 |
| V3 | 训练卡在 step 0 不动 | `val_before_train=true` 触发初始验证 OOM | → V4，添加 `+trainer.val_before_train=false` |
| V3 | `train_files` 缺失报错 | Hydra 报错 | 显式指定 `data.train_files=` 和 `data.val_files=` |
| V3/V4 | num_workers 占用大量 RAM | dataloader 的 num_workers=8 每个约 1GB，共占 ~8GB | 添加 `+data.num_workers=0` |
| **V4** | **训练卡在 step 0 不动，GPU 利用率 0%** | **`do_search: true` 导致 scrl 工具调用模式，WorkerDict 在文件轮询中无限等待** | **→ V5，设置 `do_search=false`** |

### 8.4 `do_search=true` 的陷阱（最关键的坑）

**问题现象**：训练看起来启动了，GPU 显存 18543 MB（vLLM 已加载），但 GPU 利用率始终 0%，WorkerDict 进程和主进程都在睡眠态。

**根因分析**：

项目 `verl/trainer/config/ppo_trainer.yaml` 中配置了 `do_search: true`。当此值为 true 时，训练循环调用 `scrl/llm_agent/generation.py` 的 `run_llm_loop()` 方法而非简单的 `generate_sequences()`。在 `run_llm_loop()` 中，每轮生成后调用 `execute_predictions()`，该方法通过**文件信号轮询机制**等待外部 handler 进程：

```python
# scrl/llm_agent/generation.py execute_predictions() 方法
with open(self.config.data_writing_file, 'w', encoding='utf-8') as f:
    json.dump(query_contents, f)
with open(self.config.signal_writing_file, 'w', encoding='utf-8') as f:
    json.dump({'signal': self.config.QUERY_SIGNAL}, f)  # 写入 QUERY_SIGNAL

# 轮询等待 RESPONSE_SIGNAL
while not response_finish:
    with open(self.config.signal_writing_file, 'r', encoding='utf-8') as f:
        if signal_contents['signal'] == self.config.RESPONSE_SIGNAL:
            response_finish = True
    else:
        time.sleep(10)  # ← 每 10 秒轮询一次，永不超时
```

`sclr/handler/handler.py` 是一个独立的外部进程，监听 signal 文件，执行 web search 并写回 `RESPONSE_SIGNAL`。但 V4 脚本只启动了 verl 训练主进程，handler 进程从未启动，导致 WorkerDict **永远等不到 RESPONSE_SIGNAL，卡死在轮询循环中**。

**修复方案**：两个选项

1. **简单可靠**：在命令行添加 `do_search=false`，改用简单模式（直接调用 `generate_sequences`），无需 handler 进程。
2. **完整 scrl 模式**：保留 `do_search=true`，同时启动 handler 进程并确保 `signal/` 目录存在。

本实验使用方案 1：

```bash
do_search=false
```

### 8.5 Hydra 语法注意事项

- `data.num_workers` 等不在默认配置中的参数，命令行覆盖时必须加 `+` 前缀：`+data.num_workers=0`
- 所有组件的 `ulysses_sequence_parallel_size` 默认是 8，单卡训练必须覆盖为 1

### 8.6 完整可用的实验脚本（V5）

```bash
#!/bin/bash
set -euo pipefail

export PATH="/home/vipuser/miniconda3/bin:$PATH"
export PYTHONPATH="/root/DeepResearcher:${PYTHONPATH:-}"
export HF_HOME="/root/.cache/modelscope/hub"
export SWANLAB_API_KEY="c5Q1XvxvCfQ55j8Knzkae"
export PET_NODE_RANK=0
export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1

cd /root/DeepResearcher

ESTIMATORS=("grpo" "drgrpo" "rloo" "reinforce_plusplus" "remax")

for ESTIMATOR in "${ESTIMATORS[@]}"; do
    /home/vipuser/miniconda3/bin/python verl/trainer/main_ppo.py \
        actor_rollout_ref.model.path="Qwen/Qwen2.5-3B-Instruct" \
        data.train_files="/root/DeepResearcher/data/train.parquet" \
        data.val_files="/root/DeepResearcher/data/train.parquet" \
        data.train_batch_size=8 \
        +data.num_workers=0 \
        actor_rollout_ref.actor.ppo_micro_batch_size=8 \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
        critic.ppo_micro_batch_size=8 \
        critic.ulysses_sequence_parallel_size=1 \
        algorithm.adv_estimator="$ESTIMATOR" \
        trainer.total_training_steps=300 \
        trainer.save_freq=50 \
        trainer.test_freq=999999 \
        +trainer.val_before_train=false \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=1 \
        trainer.logger="['console','swanlab']" \
        do_search=false \
        2>&1 | tee "logs/exp01_${ESTIMATOR}.log"
    echo "Estimator $ESTIMATOR completed at $(date)"
done
```

### 8.7 Ray 内存管理

`RAY_memory_monitor_refresh_ms=0` 禁用了 Ray 的内存监控 OOM killer，防止 Ray 在临界内存时主动杀掉 worker 进程。这个设置在内存充裕时是安全的。

### 8.8 进程清理与状态检查

每次重新启动前，确保旧进程和 Ray 残留完全清理：

```bash
# 杀掉所有相关进程
pkill -9 -f "main_ppo"
pkill -9 ray

# 确认 GPU 空闲（应该只有 ~14 MB 占用）
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
# 输出应该是 14 或很小的数字

# 确认无残留 Ray 进程
ps aux | grep -E "main_ppo|ray::" | grep -v grep | wc -l
# 输出应该是 0
```

### 8.9 SSH 连接稳定性

Ray 训练进程会长时间占用内存和 CPU，可能导致 SSH 连接不稳定。如果 SSH 超时：
1. 确认进程是否还在运行（GPU 占用是否 > 14MB）
2. 用长 timeout 重连：`ssh -o ConnectTimeout=60`
3. 所有 nohup 运行的实验，日志都通过 `2>&1 | tee` 实时写入文件，断连不影响日志
