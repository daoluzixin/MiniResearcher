"""
Multi-Dimensional Policy Behavior Monitor (Exp-05).

Tracks three behavioral indicators during training to distinguish
genuine search capability from reward-hacking shortcuts:

1. search_depth: average number of tool-call turns per trajectory
2. query_diversity: adjacent-turn query dissimilarity (1 - BLEU-4 overlap)
3. info_gain: incremental information coverage per turn (token overlap with GT)

The monitor integrates with the trainer loop via `BehaviorMonitor.check()`,
which returns metrics dict for SwanLab logging and optional alert strings.
"""

from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Indicator 1: Search Depth
# ---------------------------------------------------------------------------

def compute_search_depth(per_turn_info_list: List[Dict]) -> float:
    """
    Average number of search turns per trajectory.

    Args:
        per_turn_info_list: List of per-sample dicts, each with 'turn_queries'.
            turn_queries is List[List[int]] — one token list per search turn.

    Returns:
        Mean search depth across all samples.
    """
    if not per_turn_info_list:
        return 0.0
    depths = []
    for sample in per_turn_info_list:
        if sample is None:
            depths.append(0)
            continue
        turn_queries = sample.get('turn_queries', [])
        depths.append(len(turn_queries))
    return float(np.mean(depths)) if depths else 0.0


# ---------------------------------------------------------------------------
# Indicator 2: Query Diversity (1 - adjacent BLEU-4 overlap)
# ---------------------------------------------------------------------------

def _ngrams(tokens: List[int], n: int) -> set:
    """Extract n-gram set from a token sequence."""
    if len(tokens) < n:
        return set()
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _jaccard_ngram_similarity(seq1: List[int], seq2: List[int], n: int = 4) -> float:
    """
    Compute Jaccard similarity over n-grams between two token sequences.
    Returns value in [0, 1]; higher means more similar (less diverse).
    """
    ng1 = _ngrams(seq1, n)
    ng2 = _ngrams(seq2, n)
    if not ng1 or not ng2:
        return 0.0
    intersection = len(ng1 & ng2)
    union = len(ng1 | ng2)
    return intersection / union if union > 0 else 0.0


def compute_query_diversity(per_turn_info_list: List[Dict],
                            similarity_threshold: float = 0.3) -> float:
    """
    Compute query diversity: fraction of adjacent query pairs that are
    sufficiently different (Jaccard-4 similarity < threshold).

    Args:
        per_turn_info_list: List of per-sample dicts with 'turn_queries'.
        similarity_threshold: Below this similarity, adjacent queries are
            considered "diverse" (default 0.3).

    Returns:
        Mean diversity score across all samples (0~1, higher = more diverse).
    """
    if not per_turn_info_list:
        return 0.0

    diversity_scores = []
    for sample in per_turn_info_list:
        if sample is None:
            diversity_scores.append(1.0)
            continue
        turn_queries = sample.get('turn_queries', [])
        if len(turn_queries) < 2:
            # Only one turn or zero: considered "diverse" (no repetition possible)
            diversity_scores.append(1.0)
            continue

        diverse_count = 0
        for i in range(1, len(turn_queries)):
            sim = _jaccard_ngram_similarity(turn_queries[i - 1], turn_queries[i])
            if sim < similarity_threshold:
                diverse_count += 1
        diversity_scores.append(diverse_count / (len(turn_queries) - 1))

    return float(np.mean(diversity_scores)) if diversity_scores else 0.0


# ---------------------------------------------------------------------------
# Indicator 3: Information Gain (per-turn token coverage of GT)
# ---------------------------------------------------------------------------

def compute_info_gain(per_turn_info_list: List[Dict],
                      ground_truths: Optional[List[str]] = None,
                      tokenizer=None) -> float:
    """
    Compute average per-turn information gain: incremental token coverage
    of the ground truth across search turns.

    info_gain_t = |accumulated_tokens_t ∩ GT_tokens| / |GT_tokens|
    Δ_gain_t = info_gain_t - info_gain_{t-1}
    Final metric = mean of all Δ_gain across all samples and turns.

    Args:
        per_turn_info_list: List of per-sample dicts with 'turn_queries'.
        ground_truths: List of GT strings (one per sample). If None, returns 0.
        tokenizer: Tokenizer with .encode() method. Required if ground_truths provided.

    Returns:
        Mean incremental information gain per turn across all samples.
    """
    if not per_turn_info_list or ground_truths is None or tokenizer is None:
        return 0.0

    all_gains = []
    for sample, gt_str in zip(per_turn_info_list, ground_truths):
        if sample is None or not gt_str:
            continue
        turn_queries = sample.get('turn_queries', [])
        if not turn_queries:
            continue

        gt_tokens = set(tokenizer.encode(str(gt_str), add_special_tokens=False))
        if not gt_tokens:
            continue

        accumulated = set()
        prev_coverage = 0.0
        for query_tokens in turn_queries:
            accumulated.update(query_tokens)
            coverage = len(accumulated & gt_tokens) / len(gt_tokens)
            delta = coverage - prev_coverage
            all_gains.append(delta)
            prev_coverage = coverage

    return float(np.mean(all_gains)) if all_gains else 0.0


# ---------------------------------------------------------------------------
# BehaviorMonitor: Unified check + alert
# ---------------------------------------------------------------------------

class BehaviorMonitor:
    """
    Integrates the three behavioral indicators and produces:
    - A metrics dict for logging (SwanLab / console)
    - Alert strings when indicators cross thresholds

    Usage in training loop:
        monitor = BehaviorMonitor()
        metrics, alerts = monitor.check(step, per_turn_info_list, ground_truths, tokenizer)
        all_metrics.update(metrics)
    """

    def __init__(self,
                 diversity_threshold: float = 0.15,
                 depth_threshold: float = 2.0,
                 info_gain_threshold: float = 0.0,
                 similarity_threshold: float = 0.3):
        """
        Args:
            diversity_threshold: Alert if diversity drops below this.
            depth_threshold: Alert if average depth drops below this.
            info_gain_threshold: Alert if info_gain is below this (0 = disabled).
            similarity_threshold: Jaccard-4 similarity below which adjacent
                queries are considered "diverse".
        """
        self.diversity_threshold = diversity_threshold
        self.depth_threshold = depth_threshold
        self.info_gain_threshold = info_gain_threshold
        self.similarity_threshold = similarity_threshold
        self.alert_history: List[Dict] = []

    def check(self,
              step: int,
              per_turn_info_list: List[Dict],
              ground_truths: Optional[List[str]] = None,
              tokenizer=None) -> Tuple[Dict[str, float], List[str]]:
        """
        Compute all three indicators and return metrics + alerts.

        Args:
            step: Current training step.
            per_turn_info_list: From generation manager (non_tensor_batch['per_turn_info']).
            ground_truths: List of GT answer strings (for info_gain).
            tokenizer: Tokenizer instance (for info_gain).

        Returns:
            metrics: Dict of float values for logging.
            alerts: List of alert message strings (empty if healthy).
        """
        depth = compute_search_depth(per_turn_info_list)
        diversity = compute_query_diversity(
            per_turn_info_list,
            similarity_threshold=self.similarity_threshold
        )
        info_gain = compute_info_gain(per_turn_info_list, ground_truths, tokenizer)

        metrics = {
            'behavior/search_depth': depth,
            'behavior/query_diversity': diversity,
            'behavior/info_gain': info_gain,
        }

        alerts = []
        if diversity < self.diversity_threshold:
            alerts.append(
                f"DIVERSITY_DROP: {diversity:.2%} < {self.diversity_threshold:.2%}"
            )
        if depth < self.depth_threshold:
            alerts.append(
                f"SHORT_TRACE: depth={depth:.1f} < {self.depth_threshold}"
            )
        if self.info_gain_threshold > 0 and info_gain < self.info_gain_threshold:
            alerts.append(
                f"LOW_INFO_GAIN: {info_gain:.4f} < {self.info_gain_threshold}"
            )

        metrics['behavior/alert_count'] = float(len(alerts))

        if alerts:
            self.alert_history.append({
                'step': step,
                'alerts': alerts,
                'metrics': metrics.copy(),
            })

        return metrics, alerts
