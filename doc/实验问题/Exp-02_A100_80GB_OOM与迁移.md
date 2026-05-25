# Exp-02 A100-80GB 迁移与 OOM 修复记录

**日期**: 2026-05-25
**实验**: Exp-02 baseline (25条数据集, SearXNG搜索, 1×A100-80GB, 50 steps)
**状态**: ✅ 训练稳定运行中（Step 3+），OOM 已修复

---

## 核心教训：单卡 40GB 是不可逾越的瓶颈

### 背景

在 2×A800-40GB（js3 机器）上反复尝试了一整天，遇到的所有 OOM 问题本质上都是同一个根因：**FSDP NO_SHARD 模式下，单卡显存是硬瓶颈，激活值不跨卡共享**。

### 为什么 2×40GB 不等于 80GB

- verl 框架在单节点多 GPU 时使用 FSDP，但 world_size=2 时 FSDP 会尝试 FULL_SHARD
- 对于 3B 模型 + 长序列 backward，即使 FULL_SHARD 也无法将单卡激活值需求降到 40GB 以内
- `update_actor` 的 `loss.backward()` 需要的峰值显存 = 模型参数 + 优化器状态 + 前向激活值 + 反向梯度，在长序列（4096 tokens）下轻松超过 40GB
- 多卡并行只能分摊参数/优化器状态，**不能分摊单条序列的激活值**

### 在 40GB 上的各种无效尝试

| 尝试 | 结果 |
|------|------|
| 截断 response 到 3328 tokens | update_actor 仍 OOM |
| 截断 response 到 1024 tokens | 勉强能跑但严重损害多轮训练效果 |
| optimizer_offload=True | 仍不够，且大幅降低训练速度 |
| 降低 gpu_memory_utilization | vLLM 推理没问题，但 update_actor 阶段无关 |

### 正确方案

**换 A100-80GB 单卡**。一步到位，不需要任何 workaround。

---

## 问题一：ppo_max_token_len_per_gpu=16384 导致 update_actor OOM

**位置**: `verl/workers/actor/dp_actor.py` 第 318 行 `loss.backward()`

**现象**:
- Step 1 侥幸通过（序列较短），Step 2 在 `update_actor` 阶段 CUDA OOM
- 错误：`Tried to allocate 7.75 GiB. GPU 0 has a total capacity of 79.14 GiB of which 4.49 GiB is free`
- PyTorch 已分配 73.35 GiB

**根因**:
- `ppo_max_token_len_per_gpu=16384` 控制 dynamic_bsz 模式下每个 micro-batch 的最大 token 数
- 16384 tokens 允许 ~3-4 条长序列（每条 ~4000 tokens）放入同一个 micro-batch
- backward 时 3-4 条 × 4000 tokens 的激活值 + 模型参数(6GB) + 优化器状态(12GB) + 梯度(6GB) 超过 80GB

**修复**: 将 `ppo_max_token_len_per_gpu` 从 16384 降至 **4096**

```bash
# scripts/run_exp_02_a100.sh
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
```

**效果**:
- 每个 micro-batch 最多处理 ~1 条长序列的 backward
- update_actor 耗时 3.5-5.5 秒，GPU 峰值 ~70GB，余量 ~10GB
- 梯度累积自动处理（16 条序列分成多个 micro-batch，累积梯度后一次 optimizer.step）

---

## 问题二：清理 signal 目录导致 trainer 启动失败

**现象**: 清理旧 rollout 文件时误删了 signal 目录，trainer 启动后立即报 `FileNotFoundError: signal.json`

**根因**: `rm -rf outputs/verl_examples/gsm8k/signal/` 把整个目录删了，trainer 代码假设目录已存在

**修复**: 启动前确保目录存在
```bash
mkdir -p /root/DeepResearcher/outputs/verl_examples/gsm8k/signal
```

---

## 问题三：Handler 进程丢失导致 trainer 卡死

**现象**: trainer signal=1（等待 handler 执行搜索），但 handler 进程不存在，trainer 无限等待

**根因**: 
- 之前的 handler 进程（PID 8891/8893）是旧 trainer 运行时启动的
- 重启 trainer 后，旧 handler 可能因 signal 目录被删而异常退出
- 新 trainer 启动后没有自动拉起 handler

**修复**: 使用 `scripts/run_handler.py` 重新启动 handler（不是 `scripts/start_handler.sh`，后者的 PYTHONPATH 不包含 handler 目录）

```bash
export PATH=/home/vipuser/miniconda3/bin:$PATH
cd /root/DeepResearcher
nohup python3 scripts/run_handler.py > logs/handler/handler.log 2>&1 &
```

**注意**: `run_handler.py` 内含 `sys.path.insert(0, '/root/DeepResearcher/scrl/handler')`，而 `start_handler.sh` 的内联 Python 没有这行，会报 `ModuleNotFoundError: No module named 'web_search_agent'`。

---

## 问题四：start_handler.sh 的 PYTHONPATH 缺失

**现象**: 用 `start_handler.sh` 启动 handler 报 `ModuleNotFoundError: No module named 'web_search_agent'`

**根因**: 
- `web_search_agent` 模块位于 `/root/DeepResearcher/scrl/handler/web_search_agent/`
- `start_handler.sh` 设置 `PYTHONPATH="/root/DeepResearcher:${PYTHONPATH:-}"`，不包含 handler 子目录
- `run_handler.py` 用 `sys.path.insert(0, '/root/DeepResearcher/scrl/handler')` 解决

**修复**: 统一用 `run_handler.py` 启动，或修复 `start_handler.sh` 的 PYTHONPATH：
```bash
export PYTHONPATH="/root/DeepResearcher:/root/DeepResearcher/scrl/handler:${PYTHONPATH:-}"
```

---

## 最终运行参数（A100-80GB）

```bash
# 关键参数
trainer.n_gpus_per_node=1
trainer.nnodes=1
data.train_batch_size=4
actor_rollout_ref.rollout.n=4           # 每条 prompt 采样 4 次 → 16 条轨迹/step
actor_rollout_ref.actor.ppo_mini_batch_size=16
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096  # 关键：防止 backward OOM
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
actor_rollout_ref.rollout.max_model_len=4096
max_turns=6
trainer.total_training_steps=50

# 环境变量
VLLM_ATTENTION_BACKEND=XFORMERS
HF_HUB_OFFLINE=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PET_NODE_RANK=0
PET_WORLD_SIZE=1
PET_RANK=0
```

---

## 运行指标（Step 1-3）

| Step | score/mean | update_actor 耗时 | step 总耗时 | GPU 峰值 |
|------|-----------|----------|---------|---------|
| 1 | 0.025 | 5.5s | 153s | ~70 GB |
| 2 | 0.000 | 3.7s | 95s | ~70 GB |
| 3 | 0.000 | 3.8s | 153s | ~70 GB |

- 平均每步 ~130 秒，50 步预计 ~2 小时完成
- 无截断、无 offload、全长序列训练

---

## 机器信息

- **SSH**: `ssh root@js1.blockelite.cn -p 32912`
- **GPU**: 1×A100-80GB (81920 MiB)
- **训练日志**: `/root/DeepResearcher/train_baseline.log`
- **Handler 日志**: `/root/DeepResearcher/logs/handler/handler.log`
- **训练脚本**: `/root/DeepResearcher/scripts/run_exp_02_a100.sh`
- **Handler 脚本**: `/root/DeepResearcher/scripts/run_handler.py`

---

## 与旧机器（2×A800-40GB）的对比

| 维度 | 2×A800-40GB (js3) | 1×A100-80GB (js1) |
|------|-------------------|-------------------|
| 单卡显存 | 40GB | 80GB |
| update_actor | OOM（需截断到1024 tokens） | 正常（全长 ~830 tokens，峰值70GB） |
| 每步耗时 | N/A（无法稳定运行） | ~130 秒 |
| 训练效果 | 截断后只能学基本 tool_call | 全长多轮交互完整学习 |
| 总耗时 | 浪费一整天未成功 | 约 2 小时完成 50 步 |
