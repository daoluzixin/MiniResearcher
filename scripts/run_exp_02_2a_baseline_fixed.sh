#!/bin/bash
set -x
cd /root/DeepResearcher

export PATH=/home/vipuser/miniconda3/bin:$PATH
export VLLM_ATTENTION_BACKEND=XFORMERS
export PET_NODE_RANK=0
export PET_WORLD_SIZE=1
export PET_RANK=0
# Force offline mode - use local model only (already cached at ~/models/Qwen/Qwen2.5-3B-Instruct)
export HF_HUB_OFFLINE=1
# Use the LOCAL model path to avoid any network download
LOCAL_MODEL_PATH=/root/models/Qwen/Qwen2.5-3B-Instruct

python3 -m verl.trainer.main_ppo \
  data.train_files=/root/DeepResearcher/data/web_search_agent_100.parquet \
  data.val_files=/root/DeepResearcher/data/web_search_agent_100.parquet \
  data.train_batch_size=2 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.signal_writing_file=/root/DeepResearcher/outputs/verl_examples/gsm8k/signal/signal.json \
  data.data_writing_file=/root/DeepResearcher/outputs/verl_examples/gsm8k/signal/data.json \
  data.query_signal=1 \
  data.response_signal=0 \
  actor_rollout_ref.model.path=${LOCAL_MODEL_PATH} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.prompt_length=512 \
  actor_rollout_ref.rollout.response_length=512 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.kl_ctrl.type=fixed \
  algorithm.adv_estimator=grpo \
  algorithm.entropy_bonus_coef=0.0 \
  algorithm.kl_penalty=kl \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  critic.ulysses_sequence_parallel_size=1 \
  reward_model.ulysses_sequence_parallel_size=1 \
  reward_model.enable=False \
  reward_model.use_curriculum=False \
  reward_model.curriculum_bonus_coef=0.0 \
  reward_model.early_stop_penalty_coef=0.0 \
  reward_model.query_repetition_penalty_coef=0.0 \
  do_search=True \
  max_turns=6 \
  agent_grpo.n=2 \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.logger='[console]' \
  trainer.project_name=verl_examples \
  trainer.experiment_name=gsm8k \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_training_steps=25 \
  trainer.save_freq=50 \
  "$@"
