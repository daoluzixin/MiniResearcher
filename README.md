# MiniResearcher

基于 [DeepResearcher](https://arxiv.org/abs/2504.03160) (EMNLP 2025) 开源框架的复现与改进。在单卡 A100-80G 上以 Qwen2.5-3B + LoRA 实现多轮 Web Search Agent 的 GRPO 训练，通过 PBRS 过程奖励和 Curriculum 热身解决小模型 RL 训练中的梯度爆炸与策略坍缩问题。

## 核心成果

### 8 步学会搜索：Curriculum 驱动的工具调用涌现（Exp-02）

3B 模型冷启动时完全不会调用搜索工具（search_depth = 0）。通过 entropy bonus (coeff=0.01) + early-stop penalty (−0.5) + curriculum 递增，**仅 8 个训练步**内模型从零学会多轮搜索：

| Step | search_depth | score/mean | grad_norm |
|------|-------------|------------|-----------|
| 1 | 0.000 | −0.625 | 10.20 |
| 3 | 1.062 | −0.188 | 5.06 |
| 6 | 2.125 | −0.312 | 2.43 |
| 8 | **2.875** | **−0.306** | 2.17 |

模型在 8 步内完成了从"不搜索直接回答"到"平均搜索 2.9 轮再回答"的行为转变，score 同步从 −0.625 提升到 −0.306。

### PBRS 过程奖励：梯度稳定 + 搜索行为保持（Exp-03）

无 PBRS 的 baseline 训练中 grad_norm 爆炸到万级并在 78 步崩溃；加入 PBRS 后训练稳定运行 128+ 步：

| 指标 | Baseline (03a) | PBRS (03b) |
|------|---------------|------------|
| grad_norm | 5,500 ~ 11,500 (爆炸💥) | 0.019 ~ 0.035 (稳定✅) |
| search_depth | 1.04 → 1.00 (退化) | 1.77 → 2.00 (维持) |
| 训练结局 | 78步崩溃 (AssertionError) | 稳定训练至 step 129 |

PBRS 的核心价值：密集过程奖励让模型持续获得"搜索有收益"的梯度信号，避免了 baseline 中搜索行为退化（模型学会跳过搜索直接猜答案）的问题。

### 数值稳定性工程：解决 3B LoRA 训练的梯度爆炸

在 baseline 实验中发现 advantage 出现 −1,000,000 极端值，导致 pg_loss 高达 800,000+。通过以下组合修复：

- fp32 upcast（advantage 和 loss 计算全程 float32）
- torch.clamp(−1e4, 1e4) 截断极端 advantage
- nan_to_num 防止 NaN 传播
- Adam eps=1e-4（默认 1e-8 对小模型过小）

修复后 grad_norm 从万级降至 0.02~0.04，训练可稳定运行 100+ 步。

## 与原始论文的关系

[DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher) 由上海交大 GAIR 团队提出，是首个在真实网络环境中通过端到端 RL 训练深度研究 Agent 的完整框架（Qwen2.5-7B 全参、纯终局 F1 稀疏奖励），证明了 RL 能涌现多步搜索、交叉验证、自我反思等认知行为。

本项目在此基础上解决"小模型怎么训得稳"的问题：

| 维度 | 原始论文 | 本项目 |
|------|----------|--------|
| 模型规模 | 7B 全参 | **3B + LoRA rank=64**（单卡 A100-80G） |
| 奖励信号 | 纯终局 F1（稀疏） | **PBRS 过程奖励**（每轮搜索后即时反馈） |
| 训练稳定性 | 未讨论 | **Curriculum + entropy bonus + 数值修复** |
| 诊断体系 | 无 | **search_depth / diversity / info_gain 监控** |

> **局限性**：受限于 3B 模型容量和 ~100 步训练预算，score/mean 未达到原始论文水平。本项目的价值在于验证了 PBRS + Curriculum 方案在低资源条件下的有效性，以及解决小模型 RL 训练中的工程难题。

## 项目结构

```
MiniResearcher/
├── verl/                          # 训练框架核心
│   ├── trainer/ppo/
│   │   ├── core_algos.py          # GRPO/Dr.GRPO 策略优化
│   │   ├── curriculum_scheduler.py # 课程调度器（max_turns 分阶段递进）
│   │   └── ray_trainer.py         # Ray 分布式训练入口
│   ├── utils/
│   │   ├── behavior_monitor.py    # 三维策略行为监控
│   │   └── fsdp_utils.py          # LoRA FSDP lambda wrap policy
│   └── workers/
│       ├── reward_manager/naive.py # PBRS 势函数差分 + 重复惩罚
│       └── rollout/               # vLLM rollout 引擎
├── scrl/                          # 搜索 Agent 基础设施
│   ├── handler/
│   │   ├── handler.py             # 多线程搜索执行器（缓存 + 负载均衡）
│   │   ├── server_handler.py      # 分布式 server handler
│   │   └── web_search_agent/      # 搜索 + 网页浏览工具
│   └── llm_agent/
│       └── generation.py          # 多轮 Agent rollout 生成
├── scripts/
│   ├── experiments/               # 实验启动脚本
│   ├── search_proxy.py            # 本地搜索代理
│   └── build_search_cache.py      # 搜索缓存预构建
├── doc/
│   ├── 实验记录/                   # Exp-01 ~ Exp-06 详细记录
│   └── 实验问题/                   # 踩坑与修复记录
├── logs/                          # 完整训练日志
├── logs_from_server/              # 服务器日志 + 对比分析脚本
├── data/                          # 训练/评估数据（Parquet 格式）
└── train_grpo.sh                  # 训练入口
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

## 核心算法

**PBRS 过程奖励**：势函数 Φ(s_t) = 已收集信息与 GT 的 token 覆盖率。每轮搜索后 shaping reward = γ·Φ(s_{t+1}) - Φ(s_t)，好搜索得正奖励，重复搜索零奖励。理论保证最优策略不变 (Ng et al. 1999)，实测将 grad_norm 从 10,000+ 降至 0.03。

**Curriculum + Early-Stop**：max_turns 递增 + 未满最低轮次终止扣 −0.5 + entropy bonus。实测 8 步内从不搜索到平均搜索 2.9 轮。

**数值稳定性套件**：fp32 upcast + clamp + nan_to_num + Adam eps=1e-4。针对 3B LoRA 场景下 advantage 极端值导致的训练崩溃。

## 踩坑记录

项目过程中遇到并解决的工程问题（详见 `doc/实验问题/`）：

- **LoRA 参数 NaN 溢出**：advantage 极端值 → grad 爆炸 → 参数 NaN，通过数值稳定性套件解决
- **序列超长崩溃**：多轮搜索返回内容超过 max_token_len，需动态截断
- **PyTorch 2.6 兼容性**：`weights_only=True` 默认行为变更导致 checkpoint 加载失败
- **搜索后端质量**：SearXNG 空结果率高，通过预构建搜索缓存缓解

## 技术栈

- **训练框架**: verl (基于 Ray 的分布式 GRPO)
- **推理引擎**: vLLM ≤0.6.3
- **基座模型**: Qwen2.5-3B-Instruct
- **微调**: PEFT (LoRA rank=64, alpha=128)
- **搜索后端**: SearXNG 自部署
- **监控**: SwanLab

## 致谢

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
