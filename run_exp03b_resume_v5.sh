#!/bin/bash
set -euo pipefail

# Exp-03b: PBRS resume run from the last clean checkpoint.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-3B-Instruct}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/selected/exp03b_pbrs}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/ckpts/verl_examples/exp03b_pbrs_n4_v2}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PET_NODE_RANK="${PET_NODE_RANK:-0}"
export PET_WORLD_SIZE="${PET_WORLD_SIZE:-1}"
export PET_RANK="${PET_RANK:-0}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export HYDRA_FULL_ERROR=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/modelscope/hub}"
export TORCHDYNAMO_DISABLE=1

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"
rm -rf "${CKPT_DIR}/global_step_40"
LOG_FILE="${LOG_DIR}/trainer_resume_v5.log"

python verl/trainer/main_ppo.py \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.use_lora=true \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.lora_dropout=0.0 \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    critic.model.path="${MODEL_PATH}" \
    critic.ulysses_sequence_parallel_size=1 \
    data.train_files="${PROJECT_ROOT}/data/train.parquet" \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
    data.train_batch_size=12 \
    +data.num_workers=0 \
    algorithm.adv_estimator=drgrpo \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.entropy_bonus_coef=0.0 \
    trainer.total_training_steps=140 \
    trainer.experiment_name=exp03b_pbrs_n4_v2 \
    trainer.logger=[console] \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    +trainer.val_before_train=false \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    trainer.resume_mode=auto \
    max_turns=3 \
    do_search=true \
    agent_grpo.n=4 \
    search_engine=searxng \
    reward_model.reward_manager=naive \
    reward_model.use_pbrs=true \
    reward_model.pbrs_gamma=0.9 \
    reward_model.use_curriculum=false \
    reward_model.use_query_monitoring=false \
    2>&1 | tee -a "${LOG_FILE}"
