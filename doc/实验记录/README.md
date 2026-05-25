# 实验总览

> 本目录包含 DeepResearcher 项目（Qwen2.5-3B 多轮 Web Search Agent GRPO 训练）的 7 个实验文档。

## 执行顺序

| 顺序 | 实验 | 关键依赖 | 预计并行 |
|------|------|----------|---------|
| 1 | [Exp-01_优势估计器收敛对比](Exp-01_优势估计器收敛对比.md) | 无 | ✅ 可与 2/3/4 并行 |
| 2 | [Exp-02_Curriculum热启与ModeCollapse消融](Exp-02_Curriculum热启与ModeCollapse消融.md) | 无 | ✅ 可与 1/3/4 并行 |
| 3 | [Exp-03_PBRS势函数差分奖励](Exp-03_PBRS势函数差分奖励.md) | 无 | ✅ 可与 1/2/4 并行 |
| 4 | [Exp-04_ActorLoRA与FrozenRef权重共享](Exp-04_ActorLoRA与FrozenRef权重共享.md) | 无 | ✅ 可与 1/2/3 并行 |
| 5 | [Exp-05_三维行为监控体系搭建](Exp-05_三维行为监控体系搭建.md) | 无，工具开发 | ✅ 穿插在 1~4 训练期间 |
| 6 | [Exp-06_监控应用与RepetitionPenalty](Exp-06_监控应用与RepetitionPenalty.md) | 依赖 2 + 5 | 串行，在 5 完成后 |
| 7 | [Exp-07_端到端汇总实验](Exp-07_端到端汇总实验.md) | 依赖 1~6 全部结论 | 最后 |

## 推荐并行方案

**Phase 1（Week 1~2）**：Exp-1/2/3/4 并行跑，同时 Exp-5 工具开发
**Phase 2（Week 3）**：Exp-6（基于 Phase 1 告警历史）
**Phase 3（Week 4）**：Exp-7 端到端汇总

## 实验与简历条目对应关系

| 简历条目 | 相关实验 |
|----------|---------|
| 条目1：四种优势估计器 | Exp-01 |
| 条目2：PBRS 过程奖励 | Exp-03 |
| 条目3：LoRA + Frozen-Ref | Exp-04 |
| 条目4：Curriculum + Mode Collapse | Exp-02 |
| 条目5：三维行为监控 | Exp-05 + Exp-06 |

## 关键结果记录表（实验完成后填写）

| 实验 | 核心指标 | 结论 |
|------|---------|------|
| Exp-01 | Dr.GRPO 收敛步数减少 35% | ✅ / ❌ |
| Exp-02 | 轨迹深度 0 → 2.9（8 steps 内学会工具调用） | ✅ 验证通过 |
| Exp-03 | 梯度方差降低 60% | ✅ / ❌ |
| Exp-04 | 显存 42GB → 34GB/卡 | ✅ / ❌ |
| Exp-05 | diversity 告警触发 | ✅ / ❌ |
| Exp-06 | diversity 恢复至 45%，F1 +4.2 | ✅ / ❌ |
| Exp-07 | vs baseline 最终提升 | ✅ / ❌ |
