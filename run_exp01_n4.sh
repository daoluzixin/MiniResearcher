#!/bin/bash
set -euo pipefail

# Exp-01: estimator comparison for n=4

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-3B-Instruct}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/selected/exp01_n4}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PET_NODE_RANK="${PET_NODE_RANK:-0}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export HYDRA_FULL_ERROR=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/modelscope/hub}"
export TORCHDYNAMO_DISABLE=1

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/outputs/verl_examples/gsm8k/signal"

EXP_NAME="Exp-01_n4_GRPO_vs_DrGRPO_vs_RLOO"
ESTIMATORS=("grpo" "drgrpo" "rloo")

for ESTIMATOR in "${ESTIMATORS[@]}"; do
    ESTIMATOR_LOG="${LOG_DIR}/${EXP_NAME}_${ESTIMATOR}.log"

    "${PYTHON_BIN}" verl/trainer/main_ppo.py \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        data.train_files="${PROJECT_ROOT}/data/train.parquet" \
        data.val_files="${PROJECT_ROOT}/data/train.parquet" \
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
        critic.model.path="${MODEL_PATH}" \
        critic.ppo_micro_batch_size=8 \
        critic.ulysses_sequence_parallel_size=1 \
        algorithm.adv_estimator="${ESTIMATOR}" \
        trainer.total_training_steps=300 \
        trainer.save_freq=50 \
        trainer.test_freq=999999 \
        +trainer.val_before_train=false \
        trainer.nnodes=1 trainer.n_gpus_per_node=1 \
        trainer.logger="['console','swanlab']" \
        do_search=false \
        2>&1 | tee -a "${ESTIMATOR_LOG}"
done
