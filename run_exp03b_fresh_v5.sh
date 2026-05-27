#!/bin/bash
set -euo pipefail

# Exp-03b PBRS v5 - FRESH START (from scratch)
# All stability fixes from v4 + root cause NaN fixes:
#   1. entropy_coeff=0.01 (10x stronger entropy regularization)
#   2. advantage clipping: [-5, 5] (in core_algos.py)
#   3. kl_loss_coef=0.005 (5x stronger KL penalty)
#   4. lr_lambda fix (warmup=0 returns 1.0)
#   5. fp32 upcast + clamp(-1e4, 1e4) + nan_to_num for logits (dp_actor.py)
#   6. autocast(enabled=False) for entropy/log_prob computation
#   7. entropy_from_logits: torch.where(isfinite) to handle -inf correctly
#   8. NaN backward skip + grad NaN skip
#   9. OOM mitigation: ppo_max_token_len_per_gpu=12288
# Reason for fresh start: step 40 checkpoint had NaN in LoRA weights
# (layers.23.self_attn.q_proj lora_A/B), corrupted during step 20-40 training.

export PATH="/root/miniconda3/bin:$PATH"
export PYTHONPATH="/root/DeepResearcher:${PYTHONPATH:-}"
export PET_NODE_RANK=0
export PET_WORLD_SIZE=1
export PET_RANK=0
export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1
export HF_HOME="/root/.cache/modelscope/hub"
export TORCHDYNAMO_DISABLE=1
export LD_LIBRARY_PATH="/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

cd /root/DeepResearcher

LOG_DIR="/root/DeepResearcher/logs/exp03b_pbrs"
LOG_FILE="${LOG_DIR}/trainer_fresh_v5.log"

mkdir -p "$LOG_DIR"

echo "========================================" | tee "$LOG_FILE"
echo "Starting Exp-03b PBRS FRESH v5 at $(date)" | tee -a "$LOG_FILE"
echo "Training from SCRATCH with all NaN fixes:" | tee -a "$LOG_FILE"
echo "  - fp32 upcast + clamp + nan_to_num for logits" | tee -a "$LOG_FILE"
echo "  - autocast(enabled=False) for entropy/log_prob" | tee -a "$LOG_FILE"
echo "  - entropy_from_logits: torch.where(isfinite) fix" | tee -a "$LOG_FILE"
echo "  - NaN skip (backward + optimizer step)" | tee -a "$LOG_FILE"
echo "  - entropy_coeff=0.01, kl_loss_coef=0.005" | tee -a "$LOG_FILE"
echo "  - ppo_max_token_len_per_gpu=12288" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

python verl/trainer/main_ppo.py \
    actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen/Qwen2.5-3B-Instruct \
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
    critic.model.path=/root/autodl-tmp/models/Qwen/Qwen2.5-3B-Instruct \
    critic.ulysses_sequence_parallel_size=1 \
    data.train_files=/root/DeepResearcher/data/train.parquet \
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
    trainer.experiment_name=exp03b_pbrs_fresh_v5 \
    trainer.logger=[console] \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    +trainer.val_before_train=false \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    trainer.resume_mode=disable \
    max_turns=3 \
    do_search=true \
    agent_grpo.n=4 \
    search_engine=searxng \
    reward_model.reward_manager=naive \
    reward_model.use_pbrs=true \
    reward_model.pbrs_gamma=0.9 \
    reward_model.use_curriculum=false \
    reward_model.use_query_monitoring=false \
    2>&1 | tee -a "$LOG_FILE"
