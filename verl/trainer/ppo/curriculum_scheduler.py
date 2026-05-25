"""
Curriculum Scheduler for max_turns hard scheduling.

Implements: max_turns stages [1, 3, 6] with F1-based promotion/demotion.
- Promote to next stage when F1 moving average > threshold
- Demote (rollback) if F1 drops consistently for rollback_patience steps after promotion
"""
from collections import deque
from typing import List


class CurriculumScheduler:
    """
    Hard curriculum scheduling for max_turns.

    Stages: [1, 3, 6] (or configurable)
    Promotion: F1 moving average (window) > threshold for current stage
    Demotion: F1 drops below (threshold - 0.1) for rollback_patience consecutive steps
    """

    def __init__(self, stages=None, thresholds=None, window_size=50, rollback_patience=10):
        """
        Args:
            stages: List of max_turns values, e.g. [1, 3, 6]
            thresholds: F1 thresholds to promote FROM stage i to stage i+1.
                        Length = len(stages) - 1. E.g. [0.2, 0.35] means:
                        - promote from stage 0 (turns=1) to stage 1 (turns=3) when F1 > 0.2
                        - promote from stage 1 (turns=3) to stage 2 (turns=6) when F1 > 0.35
            window_size: Size of F1 moving average window
            rollback_patience: Steps of F1 decline after promotion before rolling back
        """
        self.stages = stages or [1, 3, 6]
        self.thresholds = thresholds or [0.2, 0.35]
        assert len(self.thresholds) == len(self.stages) - 1, \
            f"thresholds length ({len(self.thresholds)}) must be len(stages)-1 ({len(self.stages)-1})"

        self.window_size = window_size
        self.rollback_patience = rollback_patience

        # State
        self.current_stage_idx = 0
        self.f1_history = deque(maxlen=window_size)
        self.steps_since_promotion = 0
        self.f1_at_promotion = 0.0
        self.decline_counter = 0

    @property
    def current_max_turns(self):
        return self.stages[self.current_stage_idx]

    @property
    def current_min_turns(self):
        """Minimum turns for early-stop penalty at current stage."""
        if self.current_stage_idx == 0:
            return 1  # At stage 0 (max_turns=1), no penalty possible
        # At higher stages, require at least (current_max_turns - 2) turns
        return max(1, self.stages[self.current_stage_idx] - 2)

    def get_moving_average(self):
        if not self.f1_history:
            return 0.0
        return sum(self.f1_history) / len(self.f1_history)

    def step(self, f1_reward):
        """
        Update curriculum state with new F1 reward observation.

        Args:
            f1_reward: Average F1 reward for the current step's batch

        Returns:
            dict with curriculum state info for logging
        """
        self.f1_history.append(f1_reward)
        self.steps_since_promotion += 1

        f1_ma = self.get_moving_average()

        # --- Promotion check ---
        if self.current_stage_idx < len(self.stages) - 1:
            threshold = self.thresholds[self.current_stage_idx]
            if f1_ma > threshold and len(self.f1_history) >= min(5, self.window_size):
                # Promote!
                self.current_stage_idx += 1
                self.f1_at_promotion = f1_ma
                self.steps_since_promotion = 0
                self.decline_counter = 0
                print(f"[Curriculum] PROMOTED to stage {self.current_stage_idx} "
                      f"(max_turns={self.current_max_turns}), "
                      f"F1_MA={f1_ma:.4f} > threshold={threshold:.4f}", flush=True)

        # --- Demotion (rollback) check ---
        if self.current_stage_idx > 0 and self.steps_since_promotion >= 3:
            demotion_threshold = self.thresholds[self.current_stage_idx - 1] - 0.1
            if f1_ma < demotion_threshold:
                self.decline_counter += 1
            else:
                self.decline_counter = 0

            if self.decline_counter >= self.rollback_patience:
                # Demote!
                self.current_stage_idx -= 1
                self.decline_counter = 0
                self.steps_since_promotion = 0
                print(f"[Curriculum] DEMOTED back to stage {self.current_stage_idx} "
                      f"(max_turns={self.current_max_turns}), "
                      f"F1_MA={f1_ma:.4f} < demotion_threshold={demotion_threshold:.4f}",
                      flush=True)

        return {
            'curriculum/stage': self.current_stage_idx,
            'curriculum/max_turns': self.current_max_turns,
            'curriculum/min_turns': self.current_min_turns,
            'curriculum/f1_moving_avg': f1_ma,
            'curriculum/decline_counter': self.decline_counter,
        }
