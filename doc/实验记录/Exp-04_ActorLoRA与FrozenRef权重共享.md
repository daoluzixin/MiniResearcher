# Exp-4 — Actor-LoRA + Frozen-Ref 权重共享架构

> 双卡 A100-40G 环境，将 PPO 训练路径适配 LoRA，实现 0.3% 参数训练 + 显存从 OOM 降至 68GB。

## 一、背景与目标

全参训练 Qwen2.5-3B 在 RL 场景下显存压力极大：模型参数 ~6GB + ref ~6GB + optimizer states ~12GB + 梯度 ~6GB + 激活值 ~8GB + rollout KV cache ~8GB = 远超 40GB，单卡无法放下。

本实验将 Actor 加载 base model 后注入 rank-64 LoRA adapter，仅训练 0.3% 参数。Ref 模型直接复用同一 base weights 作为 KL anchor，配合 FSDP 仅分片可训练参数，实现双卡总显存 68GB。

## 二、环境准备

### 2.1 额外依赖

```bash
pip install peft        # LoRA 实现
pip install bitsandbytes  # 可选：int8/int4 量化支持
```

其他依赖同 Exp-1。

### 2.2 检查模型下载

```bash
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-3B-Instruct', cache_dir='./models')
print(f'模型已下载到: {model_dir}')
"
```

## 三、实验原理

### 3.1 LoRA + Frozen-Ref 的数学基础

Actor = base + LoRA(ΔW)，即 `π_actor(·|s)` 由 `W_base + BA` 参数化。

Ref = base，即 `π_ref(·|s)` 由 `W_base` 参数化。

KL(π_actor || π_ref) 自然度量的就是 LoRA adapter 引入的策略偏移，与全参微调中单独加载一份完整 Ref 模型等价，但省掉了一整份模型的显存（~6GB）。

### 3.2 FSDP lambda wrap policy

FSDP 分片策略：只对 LoRA 的 A/B 矩阵做 all-gather/reduce-scatter，冻结的 base 参数通过 CPUOffload 卸载到 CPU。

```python
def lambda_policy_fn(module):
    # 只有叶子模块 + 有可训练 weight 的模块需要 FSDP 包裹
    if (len(list(module.named_children())) == 0
        and getattr(module, 'weight', None) is not None
        and module.weight.requires_grad):
        return True
    return False
```

冻结 base 参数通信量从全参的 ~6GB 降至 LoRA 的 ~50MB。

### 3.3 LoRA 配置

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,                    # rank
    lora_alpha=128,          # scaling factor = r * 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 加载 base model 后注入 LoRA
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# 预期输出: trainable params: 47,085,888 || all params: 3,024,067,840 || trainable%: 0.31
```

## 四、操作步骤

### 4.1 显存对比实验设计

| 实验 | 方案 | 显存占用 | 预期 |
|------|------|---------|------|
| 4a | 全参 DDP（baseline） | ~90GB/卡 | OOM |
| 4b | 全参 DDP + gradient_checkpointing | ~50GB/卡 | 不 OOM 但紧张 |
| 4c | LoRA + FSDP（rank=16） | ~35GB/双卡 | 可用，参数效率最高 |
| 4d | LoRA + FSDP（rank=64，ours） | ~68GB/双卡 | 可用，精度与效率平衡 |

### 4.2 启动命令

```bash
# 实验 4a: 全参 baseline（预期 OOM，仅测试显存边界）
nohup torchrun --nproc_per_node=2 --master_port=29800 \
    grpo_agent_train.py --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp4a_fullparam \
    --use_lora 0 \
    --num_generations 8 --gradient_checkpointing 0 \
    --epochs 10 --batch_size 1 --learning_rate 1e-6 \
    --beta 0.1 --dtype bfloat16 \
    --use_wandb > logs/exp4/4a_fullparam_$(date +%m%d_%H%M).log 2>&1 &

# 实验 4b: 全参 + gradient_checkpointing
nohup torchrun --nproc_per_node=2 --master_port=29801 \
    grpo_agent_train.py --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp4b_fullparam_gc \
    --use_lora 0 \
    --gradient_checkpointing 1 \
    ... > logs/exp4/4b_fullparam_gc_$(date +%m%d_%H%M).log 2>&1 &

# 实验 4c: LoRA rank=16
nohup torchrun --nproc_per_node=2 --master_port=29802 \
    grpo_agent_train.py --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp4c_lora_r16 \
    --use_lora 1 --lora_rank 16 --lora_alpha 32 \
    --lora_target q_proj,k_proj,v_proj,o_proj \
    ... > logs/exp4/4c_lora_r16_$(date +%m%d_%H%M).log 2>&1 &

# 实验 4d: LoRA rank=64（ours，推荐）
nohup torchrun --nproc_per_node=2 --master_port=29803 \
    grpo_agent_train.py --mode train \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./dataset/web_search_agent.jsonl \
    --save_dir ./checkpoints/exp4d_lora_r64 \
    --use_lora 1 --lora_rank 64 --lora_alpha 128 \
    --lora_target q_proj,k_proj,v_proj,o_proj \
    --use_fsdp 1 \
    ... > logs/exp4/4d_lora_r64_$(date +%m%d_%H%M).log 2>&1 &
```

### 4.3 核心超参说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_lora` | 0 | 开启 LoRA |
| `--lora_rank` | 64 | LoRA rank，越大精度越高但显存越多 |
| `--lora_alpha` | 128 | 缩放因子，通常为 rank × 2 |
| `--lora_target` | q,k,v,o_proj | LoRA 应用到的模块 |
| `--use_fsdp` | 0 | 开启 FSDP（LoRA 模式下推荐开启） |
| `--learning_rate` | 5e-5 | LoRA 场景学习率比全参大（通常 1e-5 ~ 5e-5） |

### 4.4 显存监控方法

在训练脚本中通过 `torch.cuda.max_memory_allocated()` 记录峰值显存：

```python
import torch
torch.cuda.reset_peak_memory_stats()
# ... 训练步骤 ...
peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak GPU memory: {peak_mem:.2f} GB")
```

### 4.5 merge-and-sync

LoRA adapter 训练完成后需要与 base weights 合并才能用于推理：

```python
# 在 demo 模式或保存 checkpoint 时调用
model = model.merge_and_unload()
model.save_pretrained("./checkpoints/exp4d_lora_r64/merged")
```

## 五、显存分析

### 5.1 各方案显存对比

| 组件 | 全参 | LoRA r=64 |
|------|------|-----------|
| 模型参数 bf16 | 6.0 GB | 6.0 GB |
| Ref model | 6.0 GB | 0（复用 base） |
| LoRA params | - | ~0.05 GB |
| Optimizer states | 12.0 GB | ~0.1 GB（只优化 LoRA） |
| 梯度 | 6.0 GB | ~0.05 GB |
| 激活值 | 8.0 GB | 8.0 GB |
| Rollout KV | 8.0 GB | 8.0 GB |
| **总计/卡（双卡各一半）** | **46 GB** | **~34 GB** |

## 六、常见问题

### Q1: LoRA rank 怎么选？

- rank=16：参数效率最高（trainable < 0.1%），但表示空间有限，复杂任务可能欠拟合
- rank=64：平衡点，效果接近全参，trainable 约 0.3%
- rank=128：效果最好但显存接近全参，仅作为上限参考

### Q2: LoRA 学习率怎么调？

LoRA adapter 的有效学习率通常比全参大一个数量级：全参用 1e-6，LoRA 用 5e-5 ~ 1e-4。如果 reward 不上升，先检查学习率是否过小。

### Q3: KL 散度是否还能正确度量？

是的。KL(π_actor || π_ref) = KL(π(base + ΔW) || π(base))，即度量的就是 LoRA adapter 引入的策略偏移。虽然 Ref 模型不是单独加载的，但从数学上完全等价。

## 七、实验记录

### 7.1 显存对比表（实验后填写）

| 实验 | 方案 | 峰值显存/卡 | 是否 OOM | 备注 |
|------|------|-----------|---------|------|
| 4a | 全参 | | | |
| 4b | 全参+GC | | | |
| 4c | LoRA r=16 | | | |
| 4d | LoRA r=64 | | | |

### 7.2 结论

- **哪个方案在 40GB 限制内可用**：________
- **LoRA r=64 的最终 F1 reward vs 全参差距**：________
- **merge 后的推理效果是否正常**：________
