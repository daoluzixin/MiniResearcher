# MiniResearcher

基于 [DeepResearcher](https://arxiv.org/abs/2504.03160) (EMNLP 2025) 开源框架的复现与系统性改进。在 2×A100-40G 上实现 Qwen2.5-3B 的多轮 Web Search Agent GRPO 训练全链路，覆盖算法、奖励工程、训练稳定性、参数效率与评估诊断五个方向的改进。

## 与原始论文的关系

[DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher) 由上海交大 GAIR 团队提出，是首个在真实网络环境中通过端到端 RL 训练深度研究 Agent 的完整框架，使用 Qwen2.5-7B 全参训练 + 纯终局 F1 稀疏奖励，证明了 RL 训练能涌现出多步搜索、交叉验证、自我反思等认知行为。

本项目在此基础上解决"怎么训得好、训得稳、训得小"的问题：

| 维度 | 原始论文 | 本项目 |
|------|----------|--------|
| 模型规模 | 7B 全参 | **3B + LoRA**（2×A100-40G 可训） |
| 奖励信号 | 纯终局 F1（稀疏） | **PBRS 过程奖励**（密集梯度信号） |
| 训练稳定性 | 未讨论坍缩问题 | **Curriculum + entropy bonus** 解决 mode collapse |
| 诊断体系 | 无 | **三维行为监控** 区分真学会 vs reward hacking |
| 优势估计 | 标准 GRPO | **Dr.GRPO** + 4 种 baseline 对比 |

## 改进总览

| 方向 | 原始框架 | 本项目 | 效果 |
|------|----------|--------|------|
| 优势估计 | 标准 GRPO | Dr.GRPO + 4 种 baseline 对比 | F1 收敛步数 -35% |
| 奖励信号 | 稀疏 outcome F1 | PBRS 势函数差分过程奖励 | 梯度方差 -60% |
| 训练稳定性 | 无 | Curriculum + early-stop + entropy bonus | 轨迹深度 1.2→4.8 轮 |
| 行为诊断 | 无 | depth/diversity/info_gain 三维监控 | 定位 hacking + F1 +4.2pt |
| 参数效率 | 全参 DDP | LoRA + Frozen-Ref + FSDP CPUOffload | 2×A100-40G 可训 |

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

## 实验体系

项目包含 7 个递进实验，每个实验独立验证一个改进点，最终在 Exp-07 端到端汇总：

| 实验 | 主题 | 核心结论 |
|------|------|----------|
| Exp-01 | 优势估计器收敛对比 | Dr.GRPO 在 reward 同质场景下保留梯度方向，收敛快 35% |
| Exp-02 | Curriculum 与 Mode Collapse | 分阶段放开 max_turns 避免短轨迹坍缩 |
| Exp-03 | PBRS 势函数差分奖励 | 逐轮信息增益作为过程奖励，方差降低 60% |
| Exp-04 | LoRA + Frozen-Ref | 共享 base 权重省掉整份 Ref 模型显存 |
| Exp-05 | 三维行为监控搭建 | depth/diversity/info_gain 实时追踪 |
| Exp-06 | 监控应用 + Repetition Penalty | 定位 hacking 后引入查询重复惩罚，F1 +4.2pt |
| Exp-07 | 端到端汇总 | 全部改进叠加 vs baseline 对比 |

实验脚本位于 `scripts/experiments/`，文档位于 `doc/实验记录/`。

## 技术栈

- **训练框架**: verl (基于 Ray 的分布式 RLHF/GRPO)
- **推理引擎**: vLLM ≤0.6.3
- **基座模型**: Qwen2.5-3B-Instruct / 7B-Instruct
- **分布式**: FSDP + CPUOffload / Megatron 可选
- **搜索后端**: SearXNG 自部署 / Serper API / Azure Bing
- **微调**: PEFT (LoRA rank=64, alpha=128)
- **监控**: WandB / SwanLab

## 核心算法简述

**Dr.GRPO**: 标准 GRPO 在 group 内 reward 几乎相同时 (std→0) 梯度消失。Dr.GRPO 只除以 std 不减均值，保留绝对 reward 方向性——全员负 reward 时梯度统一推离当前策略。

**PBRS 过程奖励**: 势函数 Φ(s_t) = 已收集信息与 GT 的 token 覆盖率。每轮搜索后 shaping reward = γ·Φ(s_{t+1}) - Φ(s_t)，好搜索得正奖励，重复搜索零奖励，理论保证最优策略不变 (Ng et al. 1999)。

**Curriculum + Early-Stop**: max_turns 从 1→3→10 递进，未满最低轮次即终止扣 -0.5，配合 cosine 衰减的 entropy bonus 防止早期策略锁定。

**行为监控**: 联合 reward↑ + diversity↓ 判定 reward hacking，通过 query-level BLEU > 0.7 的重复惩罚修复。

## 实验结果摘要（Exp-03 PBRS vs Baseline）

| 指标 | Baseline (03a) | PBRS (03b) | 变化 |
|------|---------------|------------|------|
| score/mean (最终) | 0.35 | 0.38 | +8.6% |
| search_depth | 1.25 (退化) | 1.92 (稳定) | +53.6% |
| grad_norm | 波动大 | 平稳收敛 | 方差 -60% |

关键发现：PBRS 最大价值不在最终 F1 提升，而在于**维持搜索深度不退化**——baseline 训练后期模型倾向于跳过搜索直接出答案（depth 从 2.0 退化到 1.25），PBRS 通过密集过程奖励让模型持续学习"搜索是有价值的"。

完整对比分析见 `logs_from_server/compare_all.py`。

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
