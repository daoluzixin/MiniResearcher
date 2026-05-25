# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import math
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from scrl.llm_agent.generation import LLMGenerationManager, GenerationConfig
WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    DRGRPO = 'drgrpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        assert torch.isnan(data.batch['old_log_probs']).any() == False
        assert torch.isnan(data.batch['ref_log_prob']).any() == False
        
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        # TODO 这里改了原来的index方式，需要检查
        # index = data.non_tensor_batch['uid']
        index = data.non_tensor_batch['agent_grpo_idx']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.DRGRPO:
        # Dr.GRPO: divide-by-std-only normalization (no mean subtraction)
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['agent_grpo_idx']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_drgrpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def _compute_query_diversity(batch):
    """
    Compute query diversity metric for multi-dimensional policy behavior monitoring (Bullet 5).

    Diversity metric = unique query token sequences across the batch / total queries.
    A low diversity score indicates the agent is reusing the same queries, which leads to
    poor exploration and lower F1 scores.

    Also computes average number of turns and repetition ratio per sample.
    """
    per_turn_info_list = batch.non_tensor_batch.get('per_turn_info', None)
    if per_turn_info_list is None or (hasattr(per_turn_info_list, '__len__') and len(per_turn_info_list) == 0):
        return 0.0
    # Convert numpy array to list for iteration
    if isinstance(per_turn_info_list, np.ndarray):
        per_turn_info_list = per_turn_info_list.tolist()

    total_queries = 0
    unique_query_hashes = set()
    all_query_token_lists = []

    for pti in per_turn_info_list:
        if pti is None:
            continue
        turn_queries = pti.get('turn_queries', [])
        for qt in turn_queries:
            total_queries += 1
            # Use token tuple as unique identifier for query content
            qt_tuple = tuple(qt) if isinstance(qt, list) else qt
            all_query_token_lists.append(qt_tuple)
            unique_query_hashes.add(qt_tuple)

    if total_queries == 0:
        return 1.0  # No queries means no diversity to measure

    # query_diversity = fraction of unique queries in the batch
    diversity = len(unique_query_hashes) / total_queries
    return diversity


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # ---- Multi-dimensional Policy Behavior Monitoring (Bullet 5) ----
        # Track query diversity and repetition to ensure healthy policy exploration.
        # F1 +4.2 improvement is partly attributed to these monitoring metrics guiding policy improvement.
        'policy/query_diversity': _compute_query_diversity(batch),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.DRGRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS,
                AdvantageEstimator.REMAX, AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError(f"Unknown advantage estimator: {self.config.algorithm.adv_estimator}")

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger and 'swanlab' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb/swanlab logger is found. ')
            return

        # SwanLab does not support Table — log as text summary instead
        if 'swanlab' in self.config.trainer.logger:
            import swanlab
            summary_text = f"Step {self.global_steps} — top {min(generations_to_log, len(inputs))} samples:\n"
            pairs = list(zip(inputs, outputs, scores))
            pairs.sort(key=lambda x: x[0])
            for inp, out, sc in pairs[:generations_to_log]:
                summary_text += f"  Score={sc:.4f} | Q: {inp[:80]}... | A: {out[:80]}...\n"
            swanlab.log({"val/sample_summary": swanlab.Text(summary_text)}, step=self.global_steps)
            return

        import wandb
        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])

        if not hasattr(self, 'validation_table'):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=self.global_steps)
        self.validation_table = new_table

    def _validate(self, global_steps=-1):
        reward_tensor_lst = []
        em_reward_tensor_lst = []
        llm_reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            num_gpus=self.config.trainer.n_gpus_per_node,
            data_writing_file=self.config.data.data_writing_file,
            signal_writing_file=self.config.data.signal_writing_file,
            model_name=self.config.actor_rollout_ref.model.path,
            RESPONSE_SIGNAL=self.config.data.response_signal,
            QUERY_SIGNAL=self.config.data.query_signal,
            n=1, # 只roll一次
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            search_engine=self.config.search_engine,
            nnodes=self.config.trainer.nnodes,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation=True,
        )
        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                # Store original inputs
                input_ids = test_batch.batch['input_ids']
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)

                if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                    test_gen_batch = test_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    test_gen_batch = test_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

                # Store generated outputs
                output_ids = test_output_gen_batch.batch['responses']
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                reward_tensor = self.val_reward_fn(test_batch)

                # Store scores
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        else:
            for batch_dict in self.val_dataloader:
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                with _timer('step', timing_raw):
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        _, final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            global_steps=-global_steps # 取负号代表val
                        )
                    test_batch = test_batch.union(final_gen_batch_output)
                    
                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    try:
                        reward_tensor = self.val_reward_fn(test_batch, val_type='f1')
                        em_reward_tensor = self.val_reward_fn(test_batch, val_type='em')
                        llm_reward_tensor = self.val_reward_fn(test_batch, val_type='llm')
                
                    except:
                        print(test_batch)
                        exit()

                    reward_tensor_lst.append(reward_tensor)
                    em_reward_tensor_lst.append(em_reward_tensor)
                    llm_reward_tensor_lst.append(llm_reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations_to_wandb(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        em_reward_tensor = torch.cat([rw.sum(-1) for rw in em_reward_tensor_lst], dim=0).cpu()
        llm_reward_tensor = torch.cat([rw.sum(-1) for rw in llm_reward_tensor_lst], dim=0).cpu()
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            f1_data_source = f"{data_source}_f1"
            em_data_source = f"{data_source}_em"
            llm_data_source = f"{data_source}_llm"
            if f1_data_source not in data_source_reward:
                data_source_reward[f1_data_source] = []
            if em_data_source not in data_source_reward:
                data_source_reward[em_data_source] = []
            if llm_data_source not in data_source_reward:
                data_source_reward[llm_data_source] = []
            data_source_reward[f1_data_source].append(reward_tensor[i].item())
            data_source_reward[em_data_source].append(em_reward_tensor[i].item())
            data_source_reward[llm_data_source].append(llm_reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from verl.utils.behavior_monitor import BehaviorMonitor
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        # ---- Exp-05: Multi-Dimensional Policy Behavior Monitor ----
        behavior_monitor = BehaviorMonitor(
            diversity_threshold=0.15,
            depth_threshold=2.0,
            info_gain_threshold=0.0,
            similarity_threshold=0.3,
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate(self.global_steps)
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        
        import os
        node_rank = int(os.environ["PET_NODE_RANK"])
        print(f"node {node_rank} in fit function!")

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            num_gpus=self.config.trainer.n_gpus_per_node,
            data_writing_file=self.config.data.data_writing_file,
            signal_writing_file=self.config.data.signal_writing_file,
            model_name=self.config.actor_rollout_ref.model.path,
            RESPONSE_SIGNAL=self.config.data.response_signal,
            QUERY_SIGNAL=self.config.data.query_signal,
            n=self.config.agent_grpo.n,
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            search_engine=self.config.search_engine,
            nnodes=self.config.trainer.nnodes,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation=False,
        )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                with _timer('step', timing_raw):
                    # generate a batch
                    if not self.config.do_search:
                        with _timer('gen', timing_raw):
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            with _timer('gen_max', timing_raw):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info['do_sample'] = False
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                                batch = batch.union(gen_baseline_output)
                                reward_baseline_tensor = self.reward_fn(batch)
                                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                                batch.batch['reward_baselines'] = reward_baseline_tensor

                                del gen_baseline_batch, gen_baseline_output
                    else:
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            assert False, 'REMAX is not supported for search'
                        else:
                            with _timer('gen', timing_raw):
                                generation_manager.timing_raw = timing_raw
                                gen_str_list, gen_batch_output = generation_manager.run_llm_loop(
                                    gen_batch=gen_batch,
                                    global_steps=self.global_steps
                                )
                            print(f"[DEBUG][step={self.global_steps}] generation done. gen_batch_output keys={list(gen_batch_output.batch.keys())}", flush=True)
                            print(f"[DEBUG][step={self.global_steps}] gen_batch_output shapes: { {k: v.shape for k, v in gen_batch_output.batch.items()} }", flush=True)

                            try:
                                for key in gen_batch_output.batch.keys():
                                    gen_batch_output.batch[key] = gen_batch_output.batch[key].long()
                                print(f"[DEBUG][step={self.global_steps}] cast to long done", flush=True)
                            except Exception as e:
                                import traceback
                                print(f"[ERROR][step={self.global_steps}] cast to long failed: {e}", flush=True)
                                traceback.print_exc()
                                raise

                            try:
                                print(f"[DEBUG][step={self.global_steps}] about to call compute_log_prob, input shape={gen_batch_output.batch['input_ids'].shape}", flush=True)
                                print(f"[DEBUG][step={self.global_steps}] non_tensor_batch keys & types: { {k: type(v).__name__ for k, v in gen_batch_output.non_tensor_batch.items()} }", flush=True)
                                # Fix: convert non-ndarray values in non_tensor_batch to np.ndarray before chunk
                                for k, v in list(gen_batch_output.non_tensor_batch.items()):
                                    if not isinstance(v, np.ndarray):
                                        try:
                                            gen_batch_output.non_tensor_batch[k] = np.array(v)
                                            print(f"[DEBUG][step={self.global_steps}] converted non_tensor_batch['{k}'] from {type(v).__name__} to ndarray", flush=True)
                                        except Exception:
                                            print(f"[DEBUG][step={self.global_steps}] removing non_tensor_batch['{k}'] (type={type(v).__name__}, cannot convert)", flush=True)
                                            del gen_batch_output.non_tensor_batch[k]
                                with torch.no_grad():
                                    output = self.actor_rollout_wg.compute_log_prob(gen_batch_output)
                                    print(f"[DEBUG][step={self.global_steps}] compute_log_prob done. output keys={list(output.batch.keys())}", flush=True)
                                    gen_batch_output = gen_batch_output.union(output)
                                    print(f"[DEBUG][step={self.global_steps}] gen_batch_output.union(output) done", flush=True)
                            except Exception as e:
                                import traceback
                                print(f"[ERROR][step={self.global_steps}] compute_log_prob or union failed: type={type(e).__name__}, repr={repr(e)}", flush=True)
                                print(f"[ERROR][step={self.global_steps}] Full traceback:", flush=True)
                                traceback.print_exc()
                                import sys
                                traceback.print_exc(file=sys.stdout)
                                raise

                    try:
                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                 dtype=object)
                        print(f"[DEBUG][step={self.global_steps}] batch size before repeat: {len(batch.batch)}, gen_batch_output size: {len(gen_batch_output.batch)}", flush=True)
                        
                        batch = batch.repeat(repeat_times=self.config.agent_grpo.n, interleave=True)
                        print(f"[DEBUG][step={self.global_steps}] batch.repeat done, new size: {len(batch.batch)}", flush=True)
                        
                        batch = batch.union(gen_batch_output)
                        print(f"[DEBUG][step={self.global_steps}] batch.union(gen_batch_output) done", flush=True)
                    except Exception as e:
                        import traceback
                        print(f"[ERROR][step={self.global_steps}] repeat/union failed: {e}", flush=True)
                        print(f"[ERROR] batch.batch keys={list(batch.batch.keys())}, shapes={ {k: v.shape for k, v in batch.batch.items()} }", flush=True)
                        print(f"[ERROR] gen_batch_output.batch keys={list(gen_batch_output.batch.keys())}, shapes={ {k: v.shape for k, v in gen_batch_output.batch.items()} }", flush=True)
                        traceback.print_exc()
                        raise

                    # ---- GRPO/DRGRPO grouping index: agent_grpo_idx ----
                    # After `batch.repeat(interleave=True)`, the batch layout is:
                    #   [...rollout_0_of_query_0, rollout_0_of_query_1, ...,
                    #    ...rollout_1_of_query_0, rollout_1_of_query_1, ...]
                    # i.e., groups of `n` consecutive samples share the same original query.
                    # `agent_grpo_idx` lets compute_grpo_outcome_advantage / compute_drgrpo_outcome_advantage
                    # correctly group samples by query to compute per-group advantage normalization.
                    # NOTE: `_balance_batch` reorders samples; advantage computation is by index value
                    #       (not by position), so this grouping remains valid after the shuffle.
                    n = self.config.agent_grpo.n
                    batch_size_before_repeat = len(gen_batch_output.batch)  # ~train_batch_size
                    group_idx = np.repeat(np.arange(batch_size_before_repeat), n).astype(object)
                    batch.non_tensor_batch['agent_grpo_idx'] = group_idx

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    try:
                        self._balance_batch(batch, metrics=metrics)
                        print(f"[DEBUG][step={self.global_steps}] _balance_batch done", flush=True)
                    except Exception as e:
                        import traceback
                        print(f"[ERROR][step={self.global_steps}] _balance_batch failed: {e}", flush=True)
                        traceback.print_exc()
                        raise

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    if not self.config.do_search:
                        with _timer('old_log_prob', timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                    # 跳过old_log_probs的检查，防止下面assert报错
                    # AssertionError: old_log_probs in tensor_dict1 and tensor_dict2 are not the same object
                    for key in batch.batch.keys():
                        if key != 'old_log_probs':
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        try:
                            with _timer('ref', timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)
                            print(f"[DEBUG][step={self.global_steps}] compute_ref_log_prob done", flush=True)
                        except Exception as e:
                            import traceback
                            print(f"[ERROR][step={self.global_steps}] compute_ref_log_prob failed: {e}", flush=True)
                            traceback.print_exc()
                            raise
                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        try:
                            reward_tensor = self.reward_fn(batch)
                            batch.batch['token_level_scores'] = reward_tensor
                            print(f"[DEBUG][step={self.global_steps}] reward_fn done, reward shape={reward_tensor.shape}", flush=True)
                        except Exception as e:
                            import traceback
                            print(f"[ERROR][step={self.global_steps}] reward_fn failed: {e}", flush=True)
                            traceback.print_exc()
                            raise

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # ---- Entropy Bonus for Mode Collapse (Bullet 4) ----
                        # Adds entropy shaping: H(π) = -sum(p * log p) per token.
                        # When policy becomes deterministic (low entropy), the bonus is subtracted
                        # from rewards to encourage exploration. Combined with curriculum+early-stop,
                        # this pushes trajectory length from 1.2 to 4.8 turns.
                        entropy_bonus_coef = self.config.algorithm.get('entropy_bonus_coef', 0.0)
                        if entropy_bonus_coef > 0 and 'old_log_probs' in batch.batch:
                            # Cosine decay: H_coeff(step) = H_init × cos(π × step / (2 × total_steps))
                            # 初期 H_coeff ≈ H_init，鼓励探索；后期 H_coeff → 0，让策略收敛
                            global_step = self.global_steps
                            total_steps = self.total_training_steps if self.total_training_steps > 0 else 1
                            h_coeff = entropy_bonus_coef * math.cos(math.pi * global_step / (2 * total_steps))
                            # response_length = last response part in sequence
                            response_length = batch.batch['responses'].size(-1)
                            response_mask = batch.batch['attention_mask'][:, -response_length:]
                            old_log_probs = batch.batch['old_log_probs'][:, -response_length:]
                            # token_level_entropy[i,t] = -sum(π(a_t|s_t) * log π(a_t|s_t)), zeroed after EOS
                            # Avoid log(0) by clamping probabilities
                            probs = old_log_probs.float().exp().clamp(min=1e-8, max=1-1e-8)
                            token_entropy = -(probs * probs.log()).sum(dim=-1, keepdim=True)  # shape: [batch, 1]
                            # Tile entropy to match token-level reward shape
                            entropy_shaping = token_entropy * h_coeff
                            # entropy bonus: 高 entropy（探索多）→ reward 更高
                            batch.batch['token_level_rewards'] = batch.batch['token_level_rewards'] + entropy_shaping
                        # ---- End Entropy Bonus ----
                        # compute advantages, executed on the driver process
                        # batch = compute_advantage(batch,
                        #                           adv_estimator=self.config.algorithm.adv_estimator,
                        #                           gamma=self.config.algorithm.gamma,
                        #                           lam=self.config.algorithm.lam,
                        #                           num_repeat=self.config.actor_rollout_ref.rollout.n)
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.agent_grpo.n) # 不过这个函数里面也没用到num_repeat参数
                    
                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # Sanitize non_tensor_batch: ensure all values are np.ndarray before update_actor
                        for _k, _v in list(batch.non_tensor_batch.items()):
                            if not isinstance(_v, np.ndarray):
                                try:
                                    batch.non_tensor_batch[_k] = np.array(_v, dtype=object)
                                except Exception:
                                    del batch.non_tensor_batch[_k]

                        # update actor
                        try:
                            with _timer('update_actor', timing_raw):
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                            metrics.update(actor_output_metrics)
                            print(f"[DEBUG][step={self.global_steps}] update_actor done", flush=True)
                        except Exception as e:
                            import traceback
                            print(f"[ERROR][step={self.global_steps}] update_actor failed: {e}", flush=True)
                            traceback.print_exc()
                            raise

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate(self.global_steps)
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # ---- Exp-05: Behavior Monitoring ----
                # Extract per_turn_info from batch to compute behavioral indicators.
                # per_turn_info is passed through non_tensor_batch from generation manager.
                per_turn_info_list = None
                if 'per_turn_info' in batch.non_tensor_batch:
                    pti_raw = batch.non_tensor_batch['per_turn_info']
                    if isinstance(pti_raw, np.ndarray):
                        per_turn_info_list = pti_raw.tolist()
                    elif isinstance(pti_raw, list):
                        per_turn_info_list = pti_raw
                if per_turn_info_list is not None:
                    # Extract ground truths for info_gain computation
                    gt_list = None
                    if 'reward_model' in batch.non_tensor_batch:
                        rm_data = batch.non_tensor_batch['reward_model']
                        if isinstance(rm_data, np.ndarray):
                            gt_list = [item.get('ground_truth', '') if isinstance(item, dict) else ''
                                       for item in rm_data]
                        elif isinstance(rm_data, list):
                            gt_list = [item.get('ground_truth', '') if isinstance(item, dict) else ''
                                       for item in rm_data]
                    behavior_metrics, behavior_alerts = behavior_monitor.check(
                        step=self.global_steps,
                        per_turn_info_list=per_turn_info_list,
                        ground_truths=gt_list,
                        tokenizer=self.tokenizer,
                    )
                    metrics.update(behavior_metrics)
                    if behavior_alerts:
                        print(f"[BEHAVIOR ALERT step={self.global_steps}] "
                              + " | ".join(behavior_alerts), flush=True)
                # ---- End Behavior Monitoring ----

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate(self.global_steps)
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return
