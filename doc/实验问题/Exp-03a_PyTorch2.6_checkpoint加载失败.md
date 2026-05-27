# Exp-03a：PyTorch 2.6 Checkpoint 加载失败

**日期**: 2025-05-26
**实验**: Exp-3a baseline（无 PBRS）
**服务器**: AutoDL 单卡 A100-80G

---

## 现象

从 checkpoint 续训时（`resume_mode: auto`），报错：

```
_pickle.UnpicklingError: Weights only load failed. This file can be loaded by setting 
`weights_only=False` or adding numpy.core.multiarray.scalar to the allowlist.
```

## 根因

PyTorch 2.6 将 `torch.load()` 的默认参数从 `weights_only=False` 改为 `weights_only=True`。

verl 框架的 checkpoint 文件中包含 `extra_state`（记录 training step、optimizer state 等），其中使用了 `numpy.core.multiarray.scalar` 类型。新默认值拒绝反序列化这些非 Tensor 对象。

## 解决方案

在 `verl/utils/checkpoint/fsdp_checkpoint_manager.py` 中找到所有 `torch.load` 调用（共 3 处），添加 `weights_only=False`：

```python
# 修改前
state_dict = torch.load(path)

# 修改后
state_dict = torch.load(path, weights_only=False)
```

### 具体位置

文件：`/root/DeepResearcher/verl/utils/checkpoint/fsdp_checkpoint_manager.py`

需要修改的 3 处 `torch.load` 调用分别在：
1. 加载 actor model state_dict
2. 加载 optimizer state_dict
3. 加载 extra_state（training step 等元数据）

## 验证方式

修改后重新启动训练，观察日志中出现 `Resuming from step XX` 确认 checkpoint 加载成功。

## 注意事项

- `weights_only=False` 存在安全风险（允许任意 pickle 反序列化），但在受信任的本地 checkpoint 场景下可以接受
- 长期方案：升级 verl 框架到兼容 PyTorch 2.6 的版本，或在保存 checkpoint 时避免使用 numpy scalar
- 如果使用 PyTorch < 2.6，此问题不会出现
