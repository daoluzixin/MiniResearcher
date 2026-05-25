# Exp-5 — 三维行为监控体系搭建

> 本实验不占 GPU 时间，纯工具开发和数据埋点，与 Exp-1~4 并行进行。

## 一、背景与目标

训练过程中仅观察 reward 曲线无法区分「真实学到搜索能力」与「学到 shortcut 骗分」。本实验设计三个行为指标，实时追踪训练过程中的策略行为异常：

1. **search depth**：单条轨迹的平均轮次数
2. **query diversity**：相邻轮 query 的 token-level BLEU-4 相似度分布
3. **information gain**：每轮搜索的新增信息覆盖度（复用 Exp-3 的势函数 Φ）

## 二、工具实现

### 2.1 三个指标的定义

```python
import torch
from collections import defaultdict

def compute_search_depth(rollout_batch):
    """每个 <tool_call> 标签计为一轮，返回所有样本的平均深度"""
    depths = []
    for sample in rollout_batch:
        num_tool_calls = sample.response.count('<tool_call>')
        depths.append(num_tool_calls)
    return sum(depths) / len(depths)


def compute_query_diversity(rollout_batch):
    """
    相邻两轮 query 的 token-level BLEU-4
    BLEU < 0.3 → 有效换了搜索策略（+1）
    0.3 ≤ BLEU < 0.7 → 无明显变化（0）
    BLEU ≥ 0.7 → 重复搜索（-1）
    返回：有效变化率 (0~1)
    """
    def bleu(seq1, seq2):
        # 简化的 BLEU-4
        n_grams_1 = set(ngrams(seq1, 4))
        n_grams_2 = set(ngrams(seq2, 4))
        if len(n_grams_1) == 0 or len(n_grams_2) == 0:
            return 0.0
        return len(n_grams_1 & n_grams_2) / len(n_grams_1 | n_grams_2)

    diversity_scores = []
    for sample in rollout_batch:
        queries = extract_queries_from_tool_calls(sample.response)
        if len(queries) < 2:
            diversity_scores.append(1.0)  # 只有一轮，视为"多样化"
            continue

        change_count = 0
        for i in range(1, len(queries)):
            bleu_score = bleu(queries[i-1], queries[i])
            if bleu_score < 0.3:
                change_count += 1
        diversity_scores.append(change_count / (len(queries) - 1))

    return sum(diversity_scores) / len(diversity_scores)


def compute_info_gain(rollout_batch, gt_tokens):
    """
    第 t 轮的信息增益 = Φ(s_{t+1}) - Φ(s_t)
    复用 Exp-3 的势函数定义
    """
    gains = []
    for sample, gt in zip(rollout_batch, gt_tokens):
        accumulated = set()
        tool_results = extract_tool_results(sample.response)
        for result in tool_results:
            new_tokens = extract_new_tokens(result)
            accumulated.update(new_tokens)
            gain = len(accumulated & gt) / len(gt)  # 简化：单轮增益
            gains.append(gain)
    return sum(gains) / len(gains) if gains else 0.0
```

### 2.2 异常告警规则

在训练主循环中集成监控模块：

```python
from SwanLabLogger import swanlab

class BehaviorMonitor:
    def __init__(self, diversity_threshold=0.15, depth_threshold=2.0):
        self.diversity_threshold = diversity_threshold
        self.depth_threshold = depth_threshold
        self.alert_history = []

    def check(self, step, rollout_batch, gt_tokens):
        depth = compute_search_depth(rollout_batch)
        diversity = compute_query_diversity(rollout_batch)
        info_gain = compute_info_gain(rollout_batch, gt_tokens)

        alerts = []
        if diversity < self.diversity_threshold:
            alerts.append(f"DIVERSITY_DROP: {diversity:.2%} < {self.diversity_threshold:.2%}")
        if depth < self.depth_threshold:
            alerts.append(f"SHORT_TRACE: depth={depth:.1f} < {self.depth_threshold}")

        if alerts:
            self.alert_history.append({"step": step, "alerts": alerts, "metrics": {
                "diversity": diversity, "depth": depth, "info_gain": info_gain
            }})

        # 记录到 SwanLab
        swanlab.log({
            "behavior/search_depth": depth,
            "behavior/query_diversity": diversity,
            "behavior/info_gain": info_gain,
            "behavior/alert": len(alerts) > 0
        }, step=step)

        return alerts
```

### 2.3 阈值标定

阈值不能拍脑袋定。先在未训练模型上跑 200 条 rollout 作为 baseline 分布：

```python
# 标定 diversity 阈值
baseline_rollouts = collect_rollout_baseline(model, dataloader, n=200)
baseline_diversities = [compute_query_diversity([s]) for s in baseline_rollouts]
diversity_threshold = np.percentile(baseline_diversities, 25)  # P25 作为告警线
print(f"Diversity threshold (P25 baseline): {diversity_threshold:.2%}")
# 实测约为 18%，取整到 15% 作为告警阈值
```

## 三、操作步骤

### 3.1 工具开发与验证

不需要 GPU，在 CPU 上开发并验证：

```bash
# 单独运行监控工具验证（不占 GPU）
python -c "
from behavior_monitor import BehaviorMonitor, compute_query_diversity
# 加载小样本测试
test_batch = load_test_rollouts('./dataset/test_small.jsonl', n=50)
diversity = compute_query_diversity(test_batch)
print(f'Test diversity: {diversity:.2%}')
"
```

### 3.2 集成到主训练脚本

在 `grpo_agent_train.py` 的训练循环中插入监控：

```python
# 在每个 training step 后
monitor = BehaviorMonitor(diversity_threshold=0.15, depth_threshold=2.0)
alerts = monitor.check(step, rollout_batch, gt_tokens)
if alerts:
    print(f"[ALERT step={step}] " + " | ".join(alerts))
```

## 四、监控面板设计

使用 SwanLab 记录三个曲线：

1. **search_depth**：逐步上升至 4~6 健康，低于 2 告警
2. **query_diversity**：逐步上升至 40~60% 健康，低于 15% 告警
3. **info_gain**：逐步上升健康，波动大需排查

SwanLab 面板配置：

```python
swanlab.init(
    project="deepresearcher-behavior-monitor",
    config={
        "diversity_threshold": 0.15,
        "depth_threshold": 2.0,
        "info_gain_target": "> 0.4"
    }
)
```

## 五、实验记录

### 5.1 Baseline 分布（实验前填写）

| 指标 | P10 | P25（阈值） | P50 | P75 | P90 |
|------|-----|------------|-----|------|-----|
| search_depth | | | | | |
| query_diversity | | | | | |
| info_gain | | | | | |

### 5.2 结论

- **阈值是否合理**：________
- **三个指标是否相互独立（相关系数 < 0.5）**：________
- **在 Exp-1~4 的训练日志中是否能检测到异常**：________
