#!/bin/bash
set -euo pipefail

# Exp-01 v6: 三种优势估计器对比 (n=4)
# GRPO vs Dr.GRPO vs RLOO
# Model: Qwen2.5-3B-Instruct, 300 steps each, single A100-80GB

export PATH="/home/vipuser/miniconda3/bin:$PATH"
export PYTHONPATH="/root/DeepResearcher:${PYTHONPATH:-}"
export SWANLAB_API_KEY="c5Q1XvxvCfQ55j8Knzkae"
export PET_NODE_RANK=0
export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1
export HF_HOME="/root/.cache/modelscope/hub"
export TORCHDYNAMO_DISABLE=1

cd /root/DeepResearcher

PYTHON_BIN="/home/vipuser/miniconda3/bin/python"
MODEL_PATH="/root/models/Qwen/Qwen2___5-3B-Instruct"
LOG_DIR="/root/DeepResearcher/logs/exp01_n4"
EXP_NAME="Exp-01_n4_GRPO_vs_DrGRPO_vs_RLOO"

mkdir -p "$LOG_DIR"
mkdir -p /root/DeepResearcher/outputs/verl_examples/gsm8k/signal

# 只跑三个核心 estimator
ESTIMATORS=("grpo" "drgrpo" "rloo")

for ESTIMATOR in "${ESTIMATORS[@]}"; do
    ESTIMATOR_LOG="${LOG_DIR}/${EXP_NAME}_${ESTIMATOR}.log"
    echo "========================================" | tee -a "$ESTIMATOR_LOG"
    echo "Starting estimator: $ESTIMATOR (n=4) at $(date)" | tee -a "$ESTIMATOR_LOG"
    echo "========================================" | tee -a "$ESTIMATOR_LOG"

    "$PYTHON_BIN" verl/trainer/main_ppo.py \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        data.train_files="/root/DeepResearcher/data/train.parquet" \
        data.val_files="/root/DeepResearcher/data/train.parquet" \
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
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
        actor_rollout_ref.rollout.max_model_len=2048 \
        actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
        critic.model.path="/root/models/Qwen/Qwen2___5-3B-Instruct" \
        critic.ppo_micro_batch_size=8 \
        critic.ulysses_sequence_parallel_size=1 \
        algorithm.adv_estimator="$ESTIMATOR" \
        trainer.total_training_steps=300 \
        trainer.save_freq=50 \
        trainer.test_freq=999999 \
        +trainer.val_before_train=false \
        trainer.nnodes=1 trainer.n_gpus_per_node=1 \
        trainer.logger="['console','swanlab']" \
        do_search=false \
        2>&1 | tee -a "$ESTIMATOR_LOG"

    echo "Completed estimator: $ESTIMATOR at $(date), exit code: $?" | tee -a "$ESTIMATOR_LOG"
    echo "" | tee -a "$ESTIMATOR_LOG"

    sleep 5
done

echo "========================================" | tee -a "${LOG_DIR}/${EXP_NAME}_summary.log"
echo "ALL 3 ESTIMATORS COMPLETED at $(date)" | tee -a "${LOG_DIR}/${EXP_NAME}_summary.log"
echo "========================================" | tee -a "${LOG_DIR}/${EXP_NAME}_summary.log"
