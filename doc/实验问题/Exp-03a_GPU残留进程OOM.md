# Exp-03a：GPU 残留进程导致 OOM

**日期**: 2025-05-26
**实验**: Exp-3a baseline（无 PBRS）
**服务器**: AutoDL 单卡 A100-80G

---

## 现象

杀掉 trainer 进程后重新启动训练，新的 trainer 立即报 CUDA OOM。`nvidia-smi` 显示 `ray::WorkerDict` 进程仍占用约 14GB VRAM。

## 根因

verl 框架通过 Ray 管理 actor/ref/rollout worker。`kill` 或 `Ctrl+C` 终止主进程后，Ray 的子 worker 进程不会自动释放 GPU 显存，它们以 orphan 状态继续占用。

下次启动时，新进程尝试在剩余显存上初始化 vLLM rollout + actor 参数，可用显存不足导致 OOM。

## 解决方案

```bash
# 1. 强制停止所有 Ray 进程
ray stop --force

# 2. 检查并 kill 残留的 Python/Ray 进程
ps aux | grep -E "ray|python" | grep -v grep
kill -9 <残留PID>

# 3. 等待 GPU 显存释放（通常需要 5-10 秒）
watch -n 2 nvidia-smi

# 4. 确认显存已清空后再启动新训练
nvidia-smi  # 确认 GPU Memory Used < 100 MiB
```

## 预防措施

在训练脚本开头加入清理逻辑：

```bash
# 每次启动前确保环境干净
ray stop --force 2>/dev/null || true
sleep 5
```

## 教训

- 不要直接 `kill -9` trainer 后立即重启，必须等 Ray worker 全部退出
- AutoDL 平台重启实例不一定清理 GPU 进程，需手动检查
