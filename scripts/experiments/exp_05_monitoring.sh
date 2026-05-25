#!/bin/bash
set -euo pipefail

# =============================================================================
# Experiment 05: Multi-Dimensional Policy Behavior Monitoring
# =============================================================================

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
DATA_DIR="/root/DeepResearcher/data/train.parquet"
SEARCH_ENGINE="${SEARCH_ENGINE:-rag}"
NUM_GPUS="${NUM_GPUS:-2}"
EXP_NAME="${EXP_NAME:-exp05_monitoring}"
REP_PENALTY_COEF="${REP_PENALTY_COEF:-0.05}"
TOTAL_STEPS="${TOTAL_STEPS:-3000}"
LOG_DIR="/root/DeepResearcher/logs/${EXP_NAME}"

mkdir -p "$LOG_DIR"
mkdir -p /root/DeepResearcher/outputs/verl_examples/gsm8k/signal

echo "============================================================"
echo "  Experiment 05: Multi-Dimensional Policy Behavior Monitoring"
echo "  Query repetition penalty coef: $REP_PENALTY_COEF"
echo "============================================================"

LOG_FILE="${LOG_DIR}/${EXP_NAME}_penalty${REP_PENALTY_COEF}.log"

"$PYTHON_BIN" verl/trainer/main_ppo.py \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=true \
    '+actor_rollout_ref.actor.fsdp_config.model_dtype=bf16' \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.model.path="$MODEL_PATH" \
    critic.ulysses_sequence_parallel_size=1 \
    data.train_files="$DATA_DIR" \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.train_batch_size=512 \
    +data.num_workers=0 \
    algorithm.adv_estimator=drgrpo \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.entropy_bonus_coef=0.01 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.experiment_name="${EXP_NAME}_penalty${REP_PENALTY_COEF}" \
    trainer.logger="['console','swanlab']" \
    trainer.save_freq=500 \
    trainer.test_freq=100 \
    +trainer.val_before_train=false \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    max_turns=5 \
    do_search=true \
    agent_grpo.n=2 \
    search_engine="$SEARCH_ENGINE" \
    reward_model.reward_manager=naive \
    reward_model.use_pbrs=true \
    reward_model.pbrs_gamma=0.5 \
    reward_model.use_curriculum=true \
    reward_model.curriculum_start_turn=2 \
    reward_model.curriculum_end_turn=5 \
    reward_model.early_stop_penalty_coef=0.1 \
    reward_model.curriculum_bonus_coef=0.1 \
    reward_model.use_query_monitoring=true \
    reward_model.query_repetition_penalty_coef="$REP_PENALTY_COEF" 2>&1 | tee -a "$LOG_FILE"

echo "Done. Log saved to $LOG_FILE"
