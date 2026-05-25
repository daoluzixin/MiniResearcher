#!/bin/bash
set -euo pipefail

export PATH="/home/vipuser/miniconda3/bin:$PATH"
export PYTHONPATH="/root/DeepResearcher:${PYTHONPATH:-}"
export SWANLAB_API_KEY="c5Q1XvxvCfQ55j8Knzkae"
export PET_NODE_RANK=0
export RAY_memory_monitor_refresh_ms=0
export RAY_DEDUP_LOGS=0
export RAY_enable_recursive_ray_get=True
export HYDRA_FULL_ERROR=1
export HF_HOME="/root/.cache/modelscope/hub"
export TORCHDYNAMO_DISABLE=1

cd /root/DeepResearcher

PYTHON_BIN="/home/vipuser/miniconda3/bin/python"
MODEL_PATH="/root/models/Qwen/Qwen2___5-3B-Instruct"
LOG_DIR="/root/DeepResearcher/logs/exp02_2a_baseline"
EXP_NAME="Exp-02_2a_Baseline"

mkdir -p "$LOG_DIR"
mkdir -p /root/DeepResearcher/outputs/verl_examples/gsm8k/signal

echo "========================================"
echo "Starting $EXP_NAME at $(date)"
echo "Mechanisms: curriculum=OFF, early_stop_penalty=OFF, entropy_bonus=OFF"
echo "========================================"

LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

"$PYTHON_BIN" verl/trainer/main_ppo.py \
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
    data.signal_writing_file=/root/DeepResearcher/outputs/verl_examples/gsm8k/signal/signal.json \
    data.data_writing_file=/root/DeepResearcher/outputs/verl_examples/gsm8k/signal/data.json \
    data.query_signal=1 \
    data.response_signal=0 \
    do_search=true \
    search_engine=online_search \
    2>&1 | tee -a "$LOG_FILE"

echo "========================================"
echo "Completed $EXP_NAME at $(date), exit code: $?"
echo "========================================"
