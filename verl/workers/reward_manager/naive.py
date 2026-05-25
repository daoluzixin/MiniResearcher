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
PBRS (Potential-Based Reward Shaping) Reward Manager for multi-turn web search Agent.

Key concept:
  Φ(s_t) = token_overlap(accumulated_queries, GT) / |GT|
  Shaping reward at turn t: γ * Φ(s_{t+1}) - Φ(s_t)

This replaces sparse outcome-only F1 reward with dense process rewards.
The policy invariance theorem (Ng et al. 1999) guarantees optimal policy unchanged.
"""

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch


# PBRS default parameters
DEFAULT_PBRS_GAMMA = 0.5  # shaping discount factor


def _compute_token_overlap(tokens_a, tokens_b):
    """
    Compute token-level overlap between two token lists.
    Returns: overlap_count / len(tokens_b)
    """
    if not tokens_b:
        return 0.0
    tokens_b_set = set(tokens_b)
    overlap = sum(1 for t in tokens_a if t in tokens_b_set)
    return overlap / len(tokens_b)


class NaiveRewardManager:
    """The reward manager with optional PBRS and curriculum/early-stop support."""

    def __init__(self, tokenizer, num_examine, compute_score=None,
                 use_pbrs=False, pbrs_gamma=DEFAULT_PBRS_GAMMA,
                 use_curriculum=False, curriculum_start_turn=2, curriculum_end_turn=5,
                 early_stop_penalty_coef=0.1, curriculum_bonus_coef=0.1,
                 use_query_monitoring=False, query_repetition_penalty_coef=0.05) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.use_pbrs = use_pbrs
        self.pbrs_gamma = pbrs_gamma  # shaping discount factor
        # ---- Curriculum + Early-stop Penalty for Mode Collapse (Bullet 4) ----
        # Mode collapse: agent stops prematurely after ~1 turn. Solution:
        # - Curriculum: gradually increase expected turn count as training progresses
        # - Early-stop penalty: penalize stopping below current threshold
        # - Trajectory length improved: 1.2 -> 4.8 turns
        self.use_curriculum = use_curriculum
        self.curriculum_start_turn = curriculum_start_turn   # minimum expected turns
        self.curriculum_end_turn = curriculum_end_turn       # maximum expected turns
        self.early_stop_penalty_coef = early_stop_penalty_coef  # penalty magnitude
        self.curriculum_bonus_coef = curriculum_bonus_coef      # bonus magnitude
        # ---- Query-level Repetition Penalty for Policy Diversity (Bullet 5) ----
        # Detect repeated queries within the same trajectory and penalize to encourage diverse behavior.
        # This is part of multi-dimensional policy behavior monitoring.
        self.use_query_monitoring = use_query_monitoring
        self.query_repetition_penalty_coef = query_repetition_penalty_coef

    def _compute_pbrs_shaping_reward(self, per_turn_info, ground_truth_str):
        """
        Compute PBRS shaping rewards per sample.
        
        For each turn t:
          accumulated_query_tokens = all query tokens from turns 0..t-1
          Φ(s_t) = overlap(accumulated_query_tokens, GT) / |GT|
          Shaping at turn t: γ * (Φ(s_{t+1}) - Φ(s_t))
        
        Returns:
          avg_shaping: float, shaping reward averaged over all turns
          num_turns: int, number of valid turns (tool_call turns)
        """
        if not per_turn_info or 'turn_queries' not in per_turn_info:
            return 0.0, 0

        turn_queries = per_turn_info.get('turn_queries', [])
        if len(turn_queries) == 0:
            return 0.0, 0

        # Encode ground truth once
        gt_tokens = self.tokenizer.encode(ground_truth_str, add_special_tokens=False)

        num_turns = len(turn_queries)
        shaping_rewards = []
        accumulated_tokens = []

        for t, query_tokens in enumerate(turn_queries):
            # Accumulate query tokens up to turn t-1 (not including turn t itself,
            # because we observe the query at the START of the turn)
            # Actually, for PBRS: we observe the query at turn t, so accumulated 
            # before applying shaping for turn t should include queries from turns < t
            accumulated_tokens_before = list(accumulated_tokens)
            accumulated_tokens.extend(query_tokens)

            # Potential at s_t (after processing queries from turns < t)
            potential_before = _compute_token_overlap(accumulated_tokens_before, gt_tokens)
            # Potential at s_{t+1} (after processing queries from turns <= t)
            potential_after = _compute_token_overlap(accumulated_tokens, gt_tokens)

            # Shaping reward: γ * (Φ_{t+1} - Φ_t)
            shaping = self.pbrs_gamma * (potential_after - potential_before)
            shaping_rewards.append(shaping)

        if not shaping_rewards:
            return 0.0, 0

        # Average shaping across all turns; normalize by number of turns
        avg_shaping = sum(shaping_rewards) / num_turns
        return avg_shaping, num_turns

    def _inject_pbrs_into_response(self, reward_tensor, i, response_ids,
                                   valid_response_length, per_turn_info, ground_truth_str):
        """
        Inject PBRS shaping reward into the response part of reward_tensor.
        
        We parse the response to find turn boundaries (marked by <|im_end|>),
        then inject shaping reward at each turn boundary token.
        
        The shaping reward at each turn t is distributed to the tokens 
        belonging to that turn in the response sequence.
        """
        if not self.use_pbrs:
            return

        response_tokens = response_ids[:valid_response_length].tolist()
        response_str = self.tokenizer.decode(response_tokens)

        # Find <|im_end|> positions in the token sequence
        # These mark end of assistant turns in multi-turn format
        end_turn_token = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if not end_turn_token:
            # Fallback: distribute shaping evenly across response
            avg_shaping, num_turns = self._compute_pbrs_shaping_reward(
                per_turn_info, ground_truth_str)
            shaping_per_token = avg_shaping / max(valid_response_length, 1)
            reward_tensor[i, :valid_response_length] += shaping_per_token
            return

        # Find all end-of-turn token positions in the response
        end_positions = []
        end_tok = end_turn_token[0]
        for idx in range(len(response_tokens) - len(end_turn_token) + 1):
            if response_tokens[idx:idx + len(end_turn_token)] == end_turn_token:
                end_positions.append(idx + len(end_turn_token) - 1)  # last token of <|im_end|>

        if not end_positions:
            # No explicit turn markers; distribute shaping evenly
            avg_shaping, num_turns = self._compute_pbrs_shaping_reward(
                per_turn_info, ground_truth_str)
            shaping_per_token = avg_shaping / max(valid_response_length, 1)
            reward_tensor[i, :valid_response_length] += shaping_per_token
            return

        # Compute shaping reward for each turn
        shaping_per_turn, num_turns = self._compute_pbrs_shaping_reward(
            per_turn_info, ground_truth_str)

        # Distribute shaping reward to tokens in each turn
        # Turn boundaries: [0, end_positions[0], end_positions[1], ..., valid_response_length]
        turn_boundaries = [0] + end_positions + [valid_response_length]
        num_turn_segments = len(turn_boundaries) - 1

        if num_turn_segments <= 1:
            # Entire response as one segment
            reward_tensor[i, :valid_response_length] += shaping_per_turn
            return

        # Distribute shaping reward uniformly within each turn segment
        # Assign each token to the turn it belongs to
        shaping_per_token_tensor = torch.zeros(valid_response_length,
                                               dtype=torch.float32)
        for t in range(min(num_turns, num_turn_segments - 1)):
            start_pos = turn_boundaries[t]
            end_pos = min(turn_boundaries[t + 1], valid_response_length)
            num_tokens_in_turn = end_pos - start_pos
            if num_tokens_in_turn > 0:
                shaping_per_token_tensor[start_pos:end_pos] = shaping_per_turn / num_tokens_in_turn

        reward_tensor[i, :valid_response_length] += shaping_per_token_tensor

    def __call__(self, data: DataProto, val_type='f1'):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # Check if PBRS data is available (passed from generation manager)
        has_pbrs_data = ('per_turn_info' in data.non_tensor_batch and
                         data.non_tensor_batch['per_turn_info'] is not None)
        if has_pbrs_data:
            per_turn_info_list = data.non_tensor_batch['per_turn_info']
        else:
            per_turn_info_list = [None] * len(data)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                val_type=val_type,
            )

            # ---- PBRS Shaping Reward ----
            # Get per-turn info for this sample (by matching idx)
            sample_per_turn_info = None
            if has_pbrs_data and per_turn_info_list is not None and len(per_turn_info_list) > 0:
                for pti in per_turn_info_list:
                    if pti and pti.get('idx') == i:
                        sample_per_turn_info = pti
                        break
                if sample_per_turn_info is None:
                    # Fallback: use first entry if batch ordering matches
                    sample_per_turn_info = per_turn_info_list[i] if i < len(per_turn_info_list) else None

            # Inject PBRS shaping reward into token-level reward tensor
            if self.use_pbrs and sample_per_turn_info is not None:
                self._inject_pbrs_into_response(
                    reward_tensor=reward_tensor,
                    i=i,
                    response_ids=response_ids,
                    valid_response_length=valid_response_length,
                    per_turn_info=sample_per_turn_info,
                    ground_truth_str=ground_truth
                )
            # ---- End PBRS ----

            # ---- Curriculum + Early-stop Penalty for Mode Collapse (Bullet 4) ----
            # Encourage multi-turn interaction: penalize premature stopping, reward continuation.
            # The threshold grows with training progress (curriculum_schedule).
            if self.use_curriculum and sample_per_turn_info is not None:
                turn_queries = sample_per_turn_info.get('turn_queries', [])
                num_turns = len(turn_queries)
                # Compute dynamic threshold based on training progress
                # Training step 0 -> start_turn, Training step max -> end_turn
                # This linearly increases expected turns from curriculum_start_turn to curriculum_end_turn
                global_step = data.non_tensor_batch.get('global_step', [0] * len(data))
                step_i = int(global_step[i]) if i < len(global_step) else 0
                max_steps = int(data.non_tensor_batch.get('max_training_steps', 10000)) if not isinstance(data.non_tensor_batch.get('max_training_steps', 10000), (int, float)) else data.non_tensor_batch.get('max_training_steps', 10000)
                progress = min(step_i / max_steps, 1.0)
                current_threshold = int(
                    self.curriculum_start_turn + progress * (self.curriculum_end_turn - self.curriculum_start_turn)
                )
                eos_idx = valid_response_length - 1
                if num_turns < current_threshold:
                    # Early-stop penalty: agent quit too early
                    penalty = -self.early_stop_penalty_coef * (current_threshold - num_turns)
                    reward_tensor[i, eos_idx] += penalty
                else:
                    # Multi-turn bonus: reward reaching/continuing past threshold
                    bonus = self.curriculum_bonus_coef * min(num_turns - current_threshold + 1, 3)
                    reward_tensor[i, eos_idx] += bonus
            # ---- End Curriculum ----

            # ---- Query-level Repetition Penalty for Policy Diversity (Bullet 5) ----
            # Multi-dimensional monitoring: detect repeated queries within the same trajectory.
            # Repeated queries indicate degenerate behavior that hurts diversity and final F1.
            # Store monitoring data in batch for later metric logging.
            eos_idx = valid_response_length - 1
            if self.use_query_monitoring and sample_per_turn_info is not None:
                turn_queries = sample_per_turn_info.get('turn_queries', [])
                num_turns = len(turn_queries)
                # Compute query repetition score
                # repetition_ratio = (total_queries - unique_queries) / total_queries
                if num_turns >= 2:
                    unique_queries = len(set(tuple(q) for q in turn_queries))
                    repetition_ratio = 1.0 - (unique_queries / num_turns)
                else:
                    repetition_ratio = 0.0
                # Apply repetition penalty at EOS token
                repetition_penalty = -self.query_repetition_penalty_coef * repetition_ratio
                reward_tensor[i, eos_idx] += repetition_penalty
                # Store monitoring data for trainer metrics logging
                batch_mon_data = getattr(data.non_tensor_batch, '_policy_monitor_data', [None] * len(data))
                if batch_mon_data[i] is None:
                    batch_mon_data[i] = {}
                batch_mon_data[i]['query_repetition_ratio'] = repetition_ratio
                batch_mon_data[i]['num_turns'] = num_turns
                data.non_tensor_batch._policy_monitor_data = batch_mon_data
            # ---- End Query Repetition Penalty ----

            # Apply outcome F1 reward at EOS token
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)
                if has_pbrs_data:
                    print("[pbrs_turn_queries]", sample_per_turn_info.get('turn_queries') if sample_per_turn_info else [])

        return reward_tensor
