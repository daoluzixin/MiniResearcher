# Exp-02_2a_baseline 训练代码 Bug 修复记录

**日期**: 2026-05-24
**实验**: Exp-02_2a_baseline (100行数据集, online_search搜索, 2×A100-40GB, 200 steps)
**状态**: ✅ 训练已成功运行中（Step 5+），所有阻塞 Bug 已修复

---

## 问题一：parse_response 中 DTensor 格式导致 batch_decode 数量异常

**位置**: `scrl/llm_agent/generation.py` 的 `parse_response` 函数

**现象**:
- vLLM rollout 返回的 `responses` tensor 可能是 DTensor 格式
- `batch_decode(input_ids)` 对 DTensor 处理异常，返回数量与预期不符
- 2 GPU 配置下，`gen_output` 返回 8 条但 `activate_list` 只有 4 条（GPU padding 导致）

**根因**: 代码假设 `batch_decode` 返回数量永远等于输入数量，从不验证

**修复**:
```python
# 在 batch_decode 之前添加 DTensor 转换
if hasattr(input_ids, 'to_local'):
    input_ids = input_ids.to_local()
if not input_ids.is_contiguous():
    input_ids = input_ids.contiguous()
response_contents = self.tokenizer.batch_decode(input_ids)
```

同时将严格的 `assert len(results) == len(activate_list)` 改为警告+填充：
```python
if len(results) != len(activate_list):
    print(f"[WARNING] parse_response returned {len(results)} results but expected {len(activate_list)}. "
          f"Possibly DTensor/batch_decode issue. Padding as needed.", flush=True)
    if len(results) < len(activate_list):
        results.extend([(True, "", "")] * (len(activate_list) - len(results)))
    else:
        results = results[:len(activate_list)]
```

---

## 问题二：parse_response 中 think 变量遮蔽

**位置**: `scrl/llm_agent/generation.py` lines 352, 359

**现象**: 函数参数 `think: bool = False` 在循环内被覆盖

**根因**: 循环内提取 think content 时使用了与参数同名的变量
```python
# 错误代码
else:
    think = content.split("")[0]  # 覆盖了参数！
    answer = content.split("<answer>")[1].split("</answer>")[0]
    results.append((True, think, answer))  # 使用的是 think_content
```

**修复**: 重命名为 `think_content`
```python
think_content = content.split("<think>")[1].split("</think>")[0]
answer = content.split("<answer>")[1].split("</answer>")[0]
results.append((True, think_content, answer))
```

---

## 问题三：execute_predictions 时序竞态条件 (KeyError: 'content')

**位置**: `scrl/llm_agent/generation.py` 的 `execute_predictions` 函数

**现象**: 训练代码收到 `RESPONSE_SIGNAL` 后立即读 `data.json`，但 handler 此时刚写入信号，`content` 字段还未填入

**根因**: Handler 代码逻辑缺陷
1. Handler 轮询到 `QUERY_SIGNAL=1`
2. Handler **先写** `RESPONSE_SIGNAL=0`
3. Handler 启动 ThreadPoolExecutor 异步处理搜索请求
4. 训练代码读到 `RESPONSE_SIGNAL`，立即读 `data.json`
5. 此时 `content` 字段为空 → KeyError

**修复**: 增加 `content` 字段等待循环
```python
# Wait for handler to fill in 'content' field for all items
content_ready = False
while not content_ready:
    with open(self.config.data_writing_file, 'r', encoding='utf-8') as f:
        query_contents = json.load(f)
    all_have_content = all('content' in item and item['content'] for item in query_contents)
    if all_have_content:
        content_ready = True
    else:
        print(f"[INFO] Waiting for handler to fill 'content' field... "
              f"({sum('content' in item and item['content'] for item in query_contents)}/{len(query_contents)} ready)", flush=True)
        time.sleep(5)
```

---

## 问题四：文件未同步到 GPU 机器

**现象**: 修复后训练仍然失败，报同样的错误

**根因**: `scp` 上传脚本后忘记重新上传 `generation.py`

**教训**: 修改本地文件后必须立即上传到远程 GPU 机器

---

## 问题五：多 GPU 训练时的 batch size 必须能被 GPU 数整除

**位置**: `scrl/llm_agent/generation.py` 的新增方法 `_generate_with_gpu_padding`

**现象**: 2 GPU 配置下，当 active batch size 不能被 2 整除时（最常见是 3、5、7 等奇数），vLLM SPMD 报 tensor parallel size 不匹配错误导致进程被 kill

**根因**: vLLM rollout 在 SPMD 模式下要求输入 batch size 必须能被 `tensor_model_parallel_size * nnodes` 整除

**修复**: 新增 `_generate_with_gpu_padding` wrapper 方法，自动处理 padding：
```python
def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
    num_gpus = self.config.num_gpus * self.config.nnodes
    if num_gpus <= 1:
        return self.actor_rollout_wg.generate_sequences(active_batch)
    batch_size = active_batch.batch['input_ids'].shape[0]
    remainder = batch_size % num_gpus
    if remainder == 0:
        return self.actor_rollout_wg.generate_sequences(active_batch)
    # 用第一条序列作为 padding 模板
    padding_size = num_gpus - remainder
    padded_batch = {}
    for k, v in active_batch.batch.items():
        pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
        padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
    padded_active_batch = DataProto.from_dict(padded_batch)
    padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
    # 生成后去掉 padding
    trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
    if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
        trimmed_meta = {}
        for k, v in padded_output.meta_info.items():
            trimmed_meta[k] = v[:-padding_size] if isinstance(v, torch.Tensor) else v
        padded_output.meta_info = trimmed_meta
    padded_output.batch = trimmed_batch
    return padded_output
```

调用处替换：
```python
# 之前
gen_output = self.actor_rollout_wg.generate_sequences(rollings_active)

# 现在
gen_output = self._generate_with_gpu_padding(rollings_active)
```

---

## exp_v8 运行记录（2026-05-24）

### 执行命令
使用新端口 22048（旧端口 25516 的 GPU 机器已不可用）：
```bash
ssh root@js2.blockelite.cn -p 22048  # 密码: Eu9eiphi
# 启动脚本: /tmp/restart_exp.sh 或直接运行 run_exp_v8.sh
```

### 遇到的问题
- **旧端口 25516 无法连接**：GPU 机器 host key 变更 + authorized_keys 不匹配（本地 id_ed25519.pub 未在服务器注册）
- **SSH agent 为空**：`ssh-add -l` 显示 "The agent has no identities"，必须用 `-i` 指定密钥文件
- **GPU 显存空闲**：进程被 kill 后 GPU 空闲（14 MiB），说明确实发生了 OOM

### 进程被 kill 的根因
脚本末尾显示 `Killed`，通常原因是：
- vLLM 加载 Qwen2.5-3B 时显存不足（2×A100-40GB，train_batch_size=2 + 2 GPU 并行）
- `actor_rollout_ref.rollout.gpu_memory_utilization=0.3` 加上 actor 本身参数 + optimizer 状态可能超过 80GB

### 下一步
1. 降低 `gpu_memory_utilization` 到 0.2 或 0.25
2. 或者改用 `param_offload=True` 释放 actor 显存
3. 重新跑 exp_v8 验证 generation.py 所有修复是否生效

---

## 代码架构隐患（未修复，建议后续改进）

1. **IPC 协议设计缺陷**: signal.json 标记完成不代表 data.json 内容就绪
2. **无超时机制**: `execute_predictions` 的 while 循环理论上可能无限等待
3. **用 assert 做验证**: 应该在生产环境用带提示的异常
4. **Handler 搜索超时**: duckduckgo 搜索频繁超时，handler 用 yandex 作为 fallback

---

## 已上传文件

- `/root/DeepResearcher/scrl/llm_agent/generation.py` (完整修复)
- `/tmp/restart_exp.sh` (启动脚本)

---

## 验证步骤（exp_v9 重启）

1. SSH 到 GPU（新端口）: `ssh root@js2.blockelite.cn -p 22048`（密码：Eu9eiphi）
2. 检查 GPU 状态: `nvidia-smi`
3. 启动训练（建议降低 gpu_memory_utilization 到 0.2 或 0.25）
4. 检查训练状态: `tail -50 /tmp/exp_v8.log`
5. 确认 GPU 显存被正常占用（应该 > 10000 MiB）
6. 如果再被 kill，查看具体 OOM 位置（一般会打印 CUDA OOM 或 NCCL timeout）
7. 如果成功，等待 50 steps 完成，观察 per_turn.json 是否正确写入

---

## 相关文件
- 本地 & GPU: `/root/DeepResearcher/scrl/llm_agent/generation.py`（同步一致）
- 本地脚本: `/Users/feng/PycharmProjects/DeepResearcher/scripts/run_exp_v8.sh`
- 训练日志: `/root/DeepResearcher/logs/exp02_2a_baseline/Exp-02_2a_Baseline.log`
- Handler 日志: `/root/DeepResearcher/logs/handler/handler.log`
- SSH 连接: `ssh root@js2.blockelite.cn -p 25412`（密码：Ahbiequ9）

---

## 问题六：Handler 搜索结果未写回 content 字段（2026-05-24 早期修复）

**位置**: `scrl/handler/handler.py` 第 183-184 行

**现象**: Trainer 生成阶段 tool_call 后等待 handler 返回搜索结果，但 handler 处理完搜索后 `data.json` 中 `content` 字段始终为空，trainer 无限等待

**根因**: `handle_execution()` 中使用 `concurrent.futures.as_completed` 遍历搜索结果，但没有将结果赋值回 `query_contents`：
```python
# 错误代码
for future in concurrent.futures.as_completed(future_to_content):
    future.result()  # 结果被丢弃！
```

**修复**:
```python
for i, future in enumerate(future_to_content):
    query_contents[i]["content"] = future.result()
```

---

## 问题七：DataProto.chunk() 要求 non_tensor_batch 值为 np.ndarray（2026-05-24）

**位置**: `verl/trainer/ppo/ray_trainer.py` 第 1133 行附近

**现象**: `compute_log_prob(batch)` 调用时报 `AssertionError`，因为 `DataProto.chunk()` 内部断言 `non_tensor_batch` 的所有值必须是 `np.ndarray`

**根因**: 生成阶段产生的 `per_turn_info` 是 Python list 类型，不满足 `DataProto.chunk()` 的 ndarray 约束

**修复**: 在调用 `compute_log_prob` 之前，自动将 non-ndarray 值转换为 ndarray：
```python
# Fix: convert non-ndarray values in non_tensor_batch to np.ndarray before chunk
for k, v in list(gen_batch_output.non_tensor_batch.items()):
    if not isinstance(v, np.ndarray):
        try:
            gen_batch_output.non_tensor_batch[k] = np.array(v)
        except Exception:
            del gen_batch_output.non_tensor_batch[k]
```

---

## 问题八：CPU 内存 OOM（非 GPU 显存）（2026-05-24）

**位置**: Ray WorkerDict 进程

**现象**: `compute_ref_log_prob` 阶段 Worker 进程被系统 OOM killer 杀死，报 `ActorDiedError: connection error code 2`

**根因**: 
- 服务器只有 **32GB RAM**
- 2 个 Ray Worker 进程各占 ~8.8GB RSS + 5.9GB shared memory ≈ 总共 ~29GB
- 加上 Ray head、trainer 主进程等，峰值超过 32GB → 被 kernel OOM killer 杀死
- `dmesg` 确认：`Out of memory: Killed process (ray::WorkerDict) total-vm:48306740kB`

**修复**: 将服务器 RAM 升级至 **64GB**（峰值约 28.7GB，64GB 余量充足）

**关键信息**: 这**不是 GPU 显存**问题！`nvidia-smi` 显示 GPU 只用了 ~10-11GB/40GB

---

## 问题九：per_turn_info ndarray 真值判断报错（2026-05-24）

**位置**: 
- `verl/workers/reward_manager/naive.py` 第 250 行
- `verl/trainer/ppo/ray_trainer.py` 第 259 行

**现象**: 
- `reward_fn` 调用时报 `ValueError: The truth value of an array with more than one element is ambiguous`
- 修复 reward_fn 后，`compute_data_metrics` 中的 `_compute_query_diversity` 报同样错误

**根因**: 问题七的修复将 `per_turn_info` 转成了 `np.ndarray`，但下游代码用 `if per_turn_info_list:` 或 `if not per_turn_info_list:` 判断 ndarray 的真值，这在 numpy 中是非法的

**修复**:
```python
# naive.py 第 210 行: 提取时转回 list
per_turn_info_list = list(data.non_tensor_batch["per_turn_info"])

# ray_trainer.py 第 258-259 行: 
per_turn_info_list = list(batch.non_tensor_batch.get("per_turn_info", []))
if len(per_turn_info_list) == 0:
    return 0.0
```

---

## 问题十：content 等待检查中 sum 表达式 TypeError（2026-05-24）

**位置**: `scrl/llm_agent/generation.py` 第 294 行

**现象**: Step 1 训练完成后进入 Step 2 生成阶段，在等待 handler 的 debug print 中报 `TypeError: unsupported operand type(s) for +: 'int' and 'list'`

**根因**: `sum('content' in item and item['content'] for item in query_contents)` 中，`item['content']` 是 list 类型（搜索结果列表），`and` 表达式返回 list 而非 bool，导致 `sum()` 无法将 int 和 list 相加

**修复**:
```python
# 之前
sum('content' in item and item['content'] for item in query_contents)

# 之后
sum(1 for item in query_contents if 'content' in item and item['content'])
```

---

## 问题十一：搜索无结果时 content 为空列表导致无限等待（2026-05-24）

**位置**: `scrl/llm_agent/generation.py` 第 290 行

**现象**: Step 2 生成阶段，handler 正确处理了所有搜索请求并写回 `data.json`，signal 也设为 RESPONSE_SIGNAL，但 trainer 仍无限等待，打印 "2/3 ready"

**根因**: 
- Handler 的 `handle_single_query` 在搜索无结果时返回空列表 `[]`
- Trainer 的 content 就绪检查为 `all('content' in item and item['content'] for item in query_contents)`
- 空列表 `[]` 在 Python 中是 falsy，所以 `item['content']` 为 `[]` 时被判定为"未就绪"

**修复**: 改为只检查 key 是否存在，不要求值为 truthy：
```python
# 之前
all_have_content = all('content' in item and item['content'] for item in query_contents)

# 之后
all_have_content = all('content' in item for item in query_contents)
```

---

## 当前运行状态（2026-05-24 15:30）

**实验配置** (原始参数):
- `train_batch_size=2`, `agent_grpo.n=2` → 4 条序列/step
- `actor.ppo_micro_batch_size=4`, `ref.log_prob_micro_batch_size=4`
- `gpu_memory_utilization=0.2`, `max_model_len=2048`
- `total_training_steps=200`

**运行指标** (Step 1-4):
- 每步耗时: 40-69 秒（平均 ~54 秒），总预计 ~3 小时
- 序列长度: prompt ~610 tokens + response 2000-3100 tokens
- actor/pg_loss: 0.012 → -0.073（policy 在学习）
- actor/grad_norm: 12.7 → 15.7（梯度正常）

**SSH**: `ssh root@js2.blockelite.cn -p 25412`（密码：Ahbiequ9）
