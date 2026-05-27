# MiniResearcher

基于 [DeepResearcher](https://arxiv.org/abs/2504.03160) (EMNLP 2025) 开源框架的复现与改进实验。在单卡 A100-80G 上以 Qwen2.5-3B + LoRA 实现多轮 Web Search Agent 的 GRPO 训练全链路，探索奖励工程（PBRS）、训练稳定性（Curriculum）、行为诊断等方向的改进方案。

> **局限性说明**：本项目受限于 3B 模型容量和 ~100 步训练预算，多数实验为短程验证（proof-of-concept），未完成原始论文 7B + 数千步的完整训练规模。实验结果展示的是方法有效性的方向性验证，而非 SOTA 指标。

## 与原始论文的关系

[DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher) 由上海交大 GAIR 团队提出，是首个在真实网络环境中通过端到端 RL 训练深度研究 Agent 的完整框架（Qwen2.5-7B 全参、纯终局 F1 稀疏奖励、数千步训练），证明了 RL 能涌现多步搜索、交叉验证、自我反思等认知行为。

本项目在此基础上探索"怎么训得稳、训得小"：

| 维度 | 原始论文 | 本项目 |
|------|----------|--------|
| 模型规模 | 7B 全参 | **3B + LoRA rank=64**（单卡 A100-80G） |
| 奖励信号 | 纯终局 F1（稀疏） | **PBRS 势函数差分过程奖励**（密集信号） |
| 训练稳定性 | 未讨论 | **Curriculum + entropy bonus + NaN 防护** |
| 诊断体系 | 无 | **search_depth / diversity / info_gain 监控** |
| 优势估计 | 标准 GRPO | 实现 **Dr.GRPO** 并对比多种 baseline |

## 实验结果（基于真实训练日志）

### Exp-03：PBRS vs Baseline（核心实验）

| 指标 | Baseline (03a, 78步崩溃) | PBRS (03b, 128步) |
|------|--------------------------|-------------------|
| score/mean | −0.486 → −0.786 (恶化) | −0.518 → −0.632 (波动) |
| search_depth | 1.04 → 1.00 (退化) | 1.77 → 2.00 (维持) |
| grad_norm | 5,500 ~ 11,500 (爆炸) | 0.019 ~ 0.035 (极稳定) |
| 结局 | 序列溢出 AssertionError 崩溃 | 正常训练至 step 129 |

**关键发现**：
- PBRS + 数值稳定性修复（fp32 upcast、NaN skip、Adam eps=1e-4）将 grad_norm 从万级爆炸压制到 0.02~0.04
- Baseline 在无过程奖励时 search_depth 退化到 1.0（模型学会跳过搜索），PBRS 维持在 2.0
- 两组 score/mean 均未显著提升（受限于搜索后端质量和 3B 模型容量）

### Exp-02：Curriculum 验证

| 指标 | Baseline (02a, 59步) | Ours (02f, 8步后OOM) |
|------|---------------------|---------------------|
| search_depth | 无记录（模型不搜索） | 0.0 → **2.875** |
| score/mean | −1.0 → −0.65 | −0.625 → −0.306 |
| grad_norm | ~12 (局部稳定) | 10.2 → 2.2 (递减) |

**关键发现**：entropy_coeff=0.01 + early_stop_penalty=−0.5 在 8 步内教会模型使用搜索工具（depth 从 0 到 2.9），但 step 9 因序列超长 OOM 崩溃。

### Exp-01：Dr.GRPO 验证（失败）

300 步训练中 score/mean 全程锁死 −1.000。原因：实验设置为纯文本问答（非 Agent 模式），模型坍缩到固定失败模式。Dr.GRPO 的理论优势在此设置下未能验证。

### 硬件实际使用情况

| 实验 | GPU | 配置 |
|------|-----|------|
| Exp-01 | AutoDL ~A100-80G | 全参, n=4 |
| Exp-02a | 2×A800-40GB | 全参, n=2 |
| Exp-02f | 1×A100-80G (峰值 62GB) | LoRA(64), n=4 |
| Exp-03a/03b | 1×A100-80G | LoRA(64), n=2/4 |

## 改进方向总览

| 方向 | 方案 | 实验状态 |
|------|------|----------|
| 优势估计 | Dr.GRPO + 4 种 baseline 对比 | Exp-01: 代码实现完成，实验设置有误导致失败 |
| 奖励工程 | PBRS 势函数差分（γ=0.9, token overlap 势函数） | Exp-03: ✅ 验证有效（grad_norm 稳定 + depth 维持） |
| 训练稳定性 | Curriculum + early-stop + entropy bonus | Exp-02: ✅ 8步内学会工具调用 |
| 数值稳定性 | fp32 upcast + NaN skip + clamp + Adam eps | Exp-03b: ✅ 解决 LoRA 3B 训练中的梯度爆炸 |
| 行为诊断 | depth/diversity/info_gain 三维监控 | 代码实现完成，集成到训练循环 |
| 参数效率 | LoRA + Frozen-Ref + FSDP CPUOffload | ✅ 单卡 A100-80G 成功训练 |

## 项目结构

```
MiniResearcher/
├── verl/                          # 训练框架核心
│   ├── trainer/ppo/
│   │   ├── core_algos.py          # GRPO/Dr.GRPO/RLOO/REINFORCE++/ReMax
│   │   ├── curriculum_scheduler.py # 课程调度器（max_turns 分阶段递进）
│   │   └── ray_trainer.py         # Ray 分布式训练入口
│   ├── utils/
│   │   ├── behavior_monitor.py    # 三维策略行为监控
│   │   └── fsdp_utils.py          # LoRA FSDP lambda wrap policy
│   └── workers/
│       ├── reward_manager/naive.py # PBRS + 重复惩罚 + 早停惩罚
│       └── rollout/               # vLLM rollout 引擎
├── scrl/                          # 搜索 Agent 基础设施
│   ├── handler/
│   │   ├── handler.py             # 多线程搜索执行器（缓存 + 负载均衡）
│   │   ├── server_handler.py      # 分布式 server handler
│   │   └── web_search_agent/      # 搜索 + 网页浏览工具
│   └── llm_agent/
│       └── generation.py          # 多轮 Agent rollout 生成
├── scripts/
│   ├── experiments/               # 7 个实验的启动脚本
│   ├── search_proxy.py            # 本地搜索代理（百度/Bing）
│   └── build_search_cache.py      # 搜索缓存预构建
├── doc/
│   ├── 实验记录/                   # Exp-01 ~ Exp-07 详细记录
│   └── 实验问题/                   # 踩坑与修复记录
├── logs/                          # 训练日志（含 exp03 baseline/PBRS 完整 log）
├── logs_from_server/              # 服务器拉取的日志 + 对比分析脚本
├── data/                          # 训练/评估数据（Parquet 格式）
└── train_grpo.sh                  # 一键训练入口
```

## 快速开始

### 环境安装

```bash
conda create -n miniresearcher python=3.10
conda activate miniresearcher
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e .
pip3 install -r requirements.txt
```

### 启动训练

```bash
# 1. 启动 Ray
export PET_NODE_RANK=0
ray start --head

# 2. 启动搜索后端
python scrl/handler/server_handler.py   # 远程搜索节点
python scrl/handler/handler.py          # 本地 handler 代理

# 3. 训练
bash train_grpo.sh
```

### 评估

```bash
bash evaluate.sh
python evaluate/cacluate_metrics.py {experiment_name}
```

## 技术栈

- **训练框架**: verl (基于 Ray 的分布式 RLHF/GRPO)
- **推理引擎**: vLLM ≤0.6.3
- **基座模型**: Qwen2.5-3B-Instruct
- **分布式**: FSDP + CPUOffload
- **搜索后端**: SearXNG 自部署 / Serper API
- **微调**: PEFT (LoRA rank=64, alpha=128)
- **监控**: SwanLab

## 核心算法简述

**PBRS 过程奖励**: 势函数 Φ(s_t) = 已收集信息与 GT 的 token 覆盖率。每轮搜索后 shaping reward = γ·Φ(s_{t+1}) - Φ(s_t)，好搜索得正奖励，重复搜索零奖励，理论保证最优策略不变 (Ng et al. 1999)。实测效果：grad_norm 从万级爆炸降至 0.02~0.04。

**Curriculum + Early-Stop**: max_turns 递增 + 未满最低轮次终止扣 -0.5 + entropy bonus (coeff=0.01)。实测效果：8 步内 search_depth 从 0 提升到 2.9。

**Dr.GRPO**: 标准 GRPO 在 group 内 reward 几乎相同时 (std→0) 梯度消失。Dr.GRPO 只除以 std 不减均值，保留绝对 reward 方向性。代码实现完成但实验验证未成功。

**数值稳定性**: 3B LoRA 训练中发现 advantage 出现 -1,000,000 极端值导致 grad_norm 爆炸。通过 fp32 upcast + torch.clamp(-1e4, 1e4) + nan_to_num + Adam eps=1e-4 组合修复。

## 踩坑记录

项目过程中遇到并解决的主要问题（详见 `doc/实验问题/`）：

- **LoRA 参数 NaN 溢出**：advantage 极端值 → grad 爆炸 → 参数 NaN。修复方案见上。
- **序列超长 AssertionError**：多轮搜索返回内容过长突破 max_token_len，需动态调整或截断。
- **PyTorch 2.6 checkpoint 加载失败**：`weights_only=True` 默认行为变更，需显式传 `weights_only=False`。
- **GPU 残留进程 OOM**：训练崩溃后 vLLM worker 未释放显存，需手动 kill。
- **搜索缓存质量问题**：SearXNG 空结果率高影响 reward 信号质量。

## 致谢

本项目基于以下开源工作：

- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher) — 原始框架与训练数据
- [veRL](https://github.com/volcengine/verl) — 分布式 RL 训练基础设施
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) — 搜索 Agent RL 训练范式

## 引用

```bibtex
@misc{zheng2025deepresearcher,
    title={DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments},
    author={Yuxiang Zheng and Dayuan Fu and Xiangkun Hu and Xiaojie Cai and Lyumanshan Ye and Pengrui Lu and Pengfei Liu},
    year={2025},
    eprint={2504.03160},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```
