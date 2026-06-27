# MiniResearcher

`MiniResearcher` is my low-resource reproduction and extension of [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher) for multi-turn web-search agent RL.

The central question behind this repo is:

> Under a constrained setup such as `Qwen2.5-3B + LoRA + single-card training`, can a search agent learn stable multi-turn search behavior, and what engineering changes are necessary to make that happen?

This public version keeps the code, selected experiment reports, and representative logs that support that answer, while removing private credentials and machine-specific environment traces.

## Reproduction scope

### What comes from upstream work

- DeepResearcher-style web-search agent training setup
- veRL-based GRPO training pipeline
- long-horizon search trajectories instead of toy single-turn RL tasks

### What I changed in this repo

- **PBRS process reward**
  - Add dense shaping around each tool call instead of relying only on sparse terminal reward.
- **Curriculum for cold start**
  - Use `max_turns` progression, entropy bonus, and early-stop penalty so 3B models learn to search instead of directly answering in round one.
- **Estimator comparison**
  - Compare `GRPO`, `Dr.GRPO`, and `RLOO` under the same framework to understand gradient scale and stability behavior.
- **Numerical stability fixes**
  - Add fp32 upcast, clipping, `nan_to_num`, and optimizer-side safeguards to prevent LoRA NaN collapse.
- **Behavior-level monitoring**
  - Track `search_depth`, `query_diversity`, and `information_gain` to distinguish genuine policy improvement from reward hacking.

## Main findings

### 1. Curriculum makes search behavior emerge quickly

With cold start, the model initially refuses to search (`search_depth = 0`). After adding curriculum, entropy bonus, and early-stop penalty, the model begins to perform multi-turn search within just a few optimization steps.

| Step | search_depth | score/mean | grad_norm |
|------|--------------|------------|-----------|
| 1 | 0.000 | -0.625 | 10.20 |
| 3 | 1.062 | -0.188 | 5.06 |
| 6 | 2.125 | -0.312 | 2.43 |
| 8 | 2.875 | -0.306 | 2.17 |

### 2. PBRS stabilizes low-resource search RL

The baseline setup shows gradient explosion and behavior collapse. PBRS keeps search behavior alive and makes long-horizon training numerically controllable.

| Setting | grad_norm | search behavior | training result |
|---------|-----------|-----------------|-----------------|
| Baseline | 5,500 ~ 11,500 | degrades over time | crashes around step 78 |
| PBRS | 0.019 ~ 0.035 | maintained | stable beyond 120+ steps |

### 3. The hard part is numerical stability, not API glue

The most important engineering work in this repo is around 3B LoRA RL stability:

- extreme `advantage` values
- exploding `pg_loss`
- NaN LoRA weights
- unstable long-sequence rollout gradients

The most useful fixes are:

- fp32 upcast on sensitive loss paths
- clipping for extreme values
- `nan_to_num`
- optimizer epsilon adjustment

## What this repo does and does not claim

- It does **not** claim to beat or match the original 7B full-parameter DeepResearcher result.
- It **does** claim that:
  - low-resource search-agent RL can be trained more stably,
  - PBRS + curriculum is effective in that regime,
  - and behavior-level monitoring is necessary to interpret reward curves correctly.

## Fast navigation

If you only want the highest-signal parts, start here:

- `README.md`
  - project overview and main claims
- `doc/实验记录/`
  - selected experiment writeups
- `logs/selected/`
  - representative raw logs referenced by the README
- `verl/trainer/ppo/core_algos.py`
  - GRPO / Dr.GRPO-related logic
- `verl/trainer/ppo/curriculum_scheduler.py`
  - curriculum scheduling
- `verl/utils/behavior_monitor.py`
  - behavior diagnostics
- `verl/workers/reward_manager/naive.py`
  - PBRS and reward shaping
- `train_grpo.sh`
  - minimal training entry

## Repository layout

```text
MiniResearcher/
├── verl/
│   ├── trainer/ppo/
│   │   ├── core_algos.py
│   │   ├── curriculum_scheduler.py
│   │   └── ray_trainer.py
│   ├── utils/
│   │   ├── behavior_monitor.py
│   │   └── fsdp_utils.py
│   └── workers/
│       ├── reward_manager/naive.py
│       └── rollout/
├── scrl/
│   └── handler/
│       └── web_search_agent/
├── scripts/
│   ├── experiments/
│   ├── build_search_cache.py
│   └── search_proxy.py
├── doc/
│   ├── 实验记录/
│   └── 实验问题/
├── logs/
│   └── selected/
├── data/
└── train_grpo.sh
```

## Public-release cleanup

This repository is published in a reviewable public form:

- browser cookies are not committed
- secrets are loaded from environment variables
- machine-specific `/root/...` and private host assumptions are removed from the main entry points
- only representative logs are kept in versioned artifacts

If you want to rerun experiments locally, pass paths such as `MODEL_PATH`, `PROJECT_ROOT`, and `HF_HOME` via shell variables.

## Minimal training flow

```bash
conda create -n miniresearcher python=3.10
conda activate miniresearcher
pip install -e .
pip install -r requirements.txt

export MODEL_PATH=/path/to/Qwen2.5-3B-Instruct
export PROJECT_ROOT=$(pwd)
export SWANLAB_API_KEY=your_key   # optional

bash train_grpo.sh
```

## Acknowledgements

- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)
- [veRL](https://github.com/volcengine/verl)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1)

## Citation

If this repo helps your work, please cite the original DeepResearcher paper:

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
