#!/bin/bash
# 启动 Exp-02 2a (baseline) 和 2f (ours) 实验
# 带完整调试日志输出，解决之前 hang/静默崩溃问题
set -x

export PATH="/home/vipuser/miniconda3/bin:$PATH"
export PYTHONPATH="/root/DeepResearcher:${PYTHONPATH:-}"
export SWANLAB_API_KEY="c5Q1XvxvCfQ55j8Knzkae"
export PET_NODE_RANK=0
export PET_WORLD_SIZE=1
export PET_RANK=0
export RAY_memory_monitor_refresh_ms=0
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export HF_HOME="/root/.cache/modelscope/hub"
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

cd /root/DeepResearcher

# 清理旧进程和 Ray
pkill -f "main_ppo" 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 3

# 清理旧 checkpoint 和 signal 文件
rm -rf /root/DeepResearcher/outputs/verl_examples/gsm8k/signal/*
mkdir -p /root/DeepResearcher/outputs/verl_examples/gsm8k/signal
mkdir -p /root/DeepResearcher/logs/exp02_debug

PYTHON_BIN="/home/vipuser/miniconda3/bin/python"
MODEL_PATH="/root/models/Qwen/Qwen2___5-3B-Instruct"
TIMESTAMP=$(date +%m%d_%H%M%S)

echo "========================================"
echo "Launching Exp-02 2a (baseline) at $(date)"
echo "========================================"

# --- Exp 2a: Baseline (无 curriculum / early-stop / entropy) ---
nohup "$PYTHON_BIN" verl/trainer/main_ppo.py \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    data.train_files="/root/DeepResearcher/data/web_search_agent_100.parquet" \
    data.val_files="/root/DeepResearcher/data/web_search_agent_100.parquet" \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    +data.num_workers=0 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    '+actor_rollout_ref.actor.fsdp_config.model_dtype=bf16' \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    critic.model.path="$MODEL_PATH" \
    critic.ppo_micro_batch_size=8 \
    critic.ulysses_sequence_parallel_size=1 \
    algorithm.adv_estimator=grpo \
    agent_grpo.n=2 \
    trainer.total_training_steps=200 \
    trainer.save_freq=50 \
    trainer.test_freq=999999 \
    +trainer.val_before_train=false \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.logger="['console','swanlab']" \
    trainer.project_name=verl_examples \
    trainer.experiment_name=Exp02_2a_Baseline \
    data.signal_writing_file=/root/DeepResearcher/outputs/verl_examples/gsm8k/signal/signal.json \
    data.data_writing_file=/root/DeepResearcher/outputs/verl_examples/gsm8k/signal/data.json \
    data.query_signal=1 \
    data.response_signal=0 \
    do_search=true \
    search_engine=online_search \
    max_turns=6 \
    > /root/DeepResearcher/logs/exp02_debug/2a_baseline_${TIMESTAMP}.log 2>&1 &

PID_2A=$!
echo "Exp-02 2a started with PID=$PID_2A"
echo "Log: /root/DeepResearcher/logs/exp02_debug/2a_baseline_${TIMESTAMP}.log"

# 等 2a 的 Ray 初始化完成再启动 2f，避免端口冲突
echo "Waiting for 2a to finish before starting 2f (sequential)..."
wait $PID_2A
EXIT_2A=$?
echo "Exp-02 2a finished with exit code $EXIT_2A at $(date)"

# 清理 signal 文件以免 2f 读到旧数据
rm -rf /root/DeepResearcher/outputs/verl_examples/gsm8k/signal/*
ray stop --force 2>/dev/null || true
sleep 5

echo "========================================"
echo "Launching Exp-02 2f (ours) at $(date)"
echo "========================================"

# --- Exp 2f: 全部机制 (curriculum + early-stop + entropy) ---
nohup "$PYTHON_BIN" verl/trainer/main_ppo.py \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    data.train_files="/root/DeepResearcher/data/web_search_agent_100.parquet" \
    data.val_files="/root/DeepResearcher/data/web_search_agent_100.parquet" \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    +data.num_workers=0 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    '+actor_rollout_ref.actor.fsdp_config.model_dtype=bf16' \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    critic.model.path="$MODEL_PATH" \
    critic.ppo_micro_batch_size=8 \
    critic.ulysses_sequence_parallel_size=1 \
    algorithm.adv_estimator=grpo \
    algorithm.entropy_bonus_coef=0.01 \
    agent_grpo.n=2 \
    trainer.total_training_steps=200 \
    trainer.save_freq=50 \
    trainer.test_freq=999999 \
    +trainer.val_before_train=false \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.logger="['console','swanlab']" \
    trainer.project_name=verl_examples \
    trainer.experiment_name=Exp02_2f_Ours \
    data.signal_writing_file=/root/DeepResearcher/outputs/verl_examples/gsm8k/signal/signal.json \
    data.data_writing_file=/root/DeepResearcher/outputs/verl_examples/gsm8k/signal/data.json \
    data.query_signal=1 \
    data.response_signal=0 \
    do_search=true \
    search_engine=online_search \
    max_turns=6 \
    reward_model.use_curriculum=true \
    reward_model.curriculum_start_turn=2 \
    reward_model.curriculum_end_turn=5 \
    reward_model.early_stop_penalty_coef=0.5 \
    reward_model.curriculum_bonus_coef=0.1 \
    > /root/DeepResearcher/logs/exp02_debug/2f_ours_${TIMESTAMP}.log 2>&1 &

PID_2F=$!
echo "Exp-02 2f started with PID=$PID_2F"
echo "Log: /root/DeepResearcher/logs/exp02_debug/2f_ours_${TIMESTAMP}.log"

wait $PID_2F
EXIT_2F=$?
echo "Exp-02 2f finished with exit code $EXIT_2F at $(date)"

echo "========================================"
echo "All experiments done."
echo "  2a exit=$EXIT_2A"
echo "  2f exit=$EXIT_2F"
echo "========================================"
