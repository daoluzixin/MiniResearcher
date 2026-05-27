# Exp-03b PBRS 训练 LoRA 参数 NaN 溢出

**日期**: 2026-05-27
**实验**: Exp-03b PBRS n4 v2 (DrGRPO + PBRS + LoRA, Qwen2.5-3B-Instruct, 1×4090-24GB, 140 steps)
**状态**: ❌ 未解决 — logits 层面 NaN 是根本原因，当前所有下游补丁均无效

---

## 当前核心问题（交接给新会话排查）

### 一句话总结

**Forward 阶段 logits 中存在 ~100 个 NaN（每步稳定出现），经 backward 传播后污染全部梯度（29M/29M 元素 NaN），导致模型完全无法学习。**

### 关键数据

```
[DEBUG-NaN] logits_rmpad_fp32 BEFORE clamp: nan_count=90~114, inf_count=93~112, 
            shape=torch.Size([11857, 151936]), finite_min=-42.5, finite_max=46.75
[WARN] NaN/Inf in grads (29491200 elements), zeroed. New grad_norm=0.000000
```

- logits 形状 `[~11857, 151936]`（vocab_size=151936），每个 micro-batch 有 90~114 个 NaN、93~112 个 Inf
- 占比极小（~100 / 1.8B = 0.000006%），但 backward 后扩散到**全部 2900 万梯度元素**
- 置零后 grad_norm=0，模型参数完全不更新

### 对比 Exp-03a（无 PBRS baseline，同模型同框架）

| 指标 | Exp-03a | Exp-03b |
|------|---------|---------|
| logits NaN | **无**（从未出现） | 每步 ~100 个 |
| grad_norm | 2608 ~ 18586（正常） | 0.000（NaN 后置零）或 nan |
| 搜索引擎 | `search_engine="rag"`（纯本地） | `search_engine="searxng"`（proxy） |
| 训练代码 | 原版 verl | 加了 fp32 upcast + clamp + NaN 保护 |
| 正常训练 | ✅ 跑了 73 步无问题 | ❌ 从第 1 步就有 logits NaN |

### 为什么 logits 会有 NaN？（待排查）

**最可能的原因：是 dp_actor.py 中新加的 forward 防御代码引入了问题，而非模型本身的问题。**

证据：
1. 3a 用相同模型（Qwen2.5-3B-Instruct + LoRA rank=64），从头训练，logits 从未有 NaN
2. 3b 的 NaN 在 "BEFORE clamp" 就已存在，说明不是 clamp 引入的
3. 但 3b 代码中有 `torch.cuda.amp.autocast(enabled=False)` 包裹 + fp32 upcast 等操作，可能改变了计算路径
4. 3b 用 `search_engine=searxng` 而 3a 用 `search_engine=rag`，数据输入格式可能不同

**需要排查方向：**
1. **对比 3a 和 3b 的 `compute_log_prob` / forward 代码路径差异** — 3a 原版 vs 3b 修改版
2. **确认 logits NaN 是否来自模型 forward 本身** — 在 `model.forward()` 刚输出时就检查，去掉所有 upcast/clamp 后是否还有 NaN
3. **检查输入数据** — searxng 返回的特殊 token 是否导致 embedding lookup 出问题
4. **禁用所有 NaN 保护代码跑一步** — 看是否是保护代码本身引入的

---

## 已尝试的修复（均无效）

### 修复历程

| 方案 | 做法 | 结果 |
|------|------|------|
| ① Rollback + clamp | optimizer step 后检测 NaN 参数并回滚 | 从 step 20 恢复时 work，但参数被冻结 |
| ② Adam eps=1e-4 | AdamW 构造时 eps=1e-4 | 对从头训无效（问题不在 optimizer） |
| ③ 从头训（不恢复 optimizer state） | resume_mode=disable | NaN 依然从 step 1 开始出现 |
| ④ NaN 梯度置零 | grad_norm=nan 时将 NaN grad 置零后继续 | 全部 29M 梯度都是 NaN，置零后 grad_norm=0，等效 skip |
| ⑤ 跳过整步 | grad_norm=nan 时 skip optimizer step | 模型永远不更新 |

### 结论

**所有下游补丁都治标不治本。根因是 forward 输出的 logits 本身有 NaN，必须从 forward 层面解决。**

---

## 训练日志关键 step 对比（v6 从头训）

| Step | search_depth | score/mean | grad_norm | entropy_loss | 备注 |
|------|-------------|-----------|-----------|--------------|------|
| 1 | 0.0 | -1.0 | 0.000 | 0.032 | 无搜索，grad 被置零 |
| 2 | **1.042** | **-0.397** | 0.000 | 0.672 | 模型会搜索！但 grad 仍是 NaN 置零 |
| 3 | 0.0 | -1.0 | 0.000 | 0.035 | |
| 4 | 0.0 | -1.0 | 0.000 | -2.7e17 | **entropy 爆炸**，seqlen=448K |
| 5-6 | 0.0 | -1.0 | 0.000 | 0.031 | 参数从未更新 |

Step 2 证明模型本身会 tool_call（Qwen 的 function calling 能力），但因为梯度全 NaN 导致永远学不动。

---

## 代码变更位置

**当前修改过的文件（服务器上 `/root/DeepResearcher/`）：**

1. **`verl/workers/actor/dp_actor.py`** — `_optimizer_step()` 和 `compute_log_prob()`
   - `compute_log_prob` 中加了 fp32 upcast + clamp + autocast(False)
   - `_optimizer_step` 中加了 NaN grad 置零 + post-step rollback + clamp
   
2. **`verl/workers/fsdp_workers.py`** line 310 — AdamW 加了 `eps=1e-4`

3. **`verl/utils/torch_functional.py`** — `entropy_from_logits` 中 `torch.where(isfinite)` 修复

**建议新会话：先 `git diff` 看完整改动，然后把 `dp_actor.py` 中所有 NaN 保护代码还原到 3a 的原版状态，跑一步看 logits 是否仍有 NaN。**

---

## 环境信息

| 项 | 值 |
|----|------|
| SSH | `ssh -p 41671 root@connect.westd.seetacloud.com` |
| 训练脚本 | `/root/DeepResearcher/run_exp03b_fresh_v6.sh` |
| 日志 | `/root/DeepResearcher/logs/exp03b_pbrs/nohup_fresh_v6.log` |
| 模型 | Qwen2.5-3B-Instruct (path: `/root/autodl-tmp/models/Qwen/Qwen2.5-3B-Instruct`) |
| 框架 | verl 0.2.0.dev + Ray + vLLM 0.21.0 + FSDP |
| GPU | 1×A100-80G (实例为 4090-24G 已换) |
| LoRA | rank=64, alpha=16, targets=q/k/v/o_proj |
| 搜索 | searxng proxy (port 8890) + handler |
| 3a 对照 | `search_engine="rag"`, 同模型同 LoRA，无 logits NaN |

---

## 排查建议优先级

1. **最高优先：对比 3a/3b 的 `compute_log_prob` 代码** — `git stash` 或 `git diff HEAD` 看 `dp_actor.py` 完整改动
2. **高优先：在 model.forward() 输出点直接检查** — 在 autocast/upcast 之前就 print NaN count
3. **中优先：检查 3a 和 3b 的输入差异** — `search_engine=rag` vs `searxng` 生成的 input_ids 是否有非法 token
4. **低优先：去掉所有保护代码，看原版 verl 跑 3b 配置是否正常** — 如果原版也 NaN 则说明是 searxng 数据问题
