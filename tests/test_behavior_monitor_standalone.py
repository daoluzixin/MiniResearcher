"""Standalone test for behavior_monitor (no torch dependency)."""
import sys
import importlib.util

# Load module directly to bypass verl/__init__.py torch dependency
spec = importlib.util.spec_from_file_location(
    'behavior_monitor', 'verl/utils/behavior_monitor.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

compute_search_depth = mod.compute_search_depth
compute_query_diversity = mod.compute_query_diversity
compute_info_gain = mod.compute_info_gain
BehaviorMonitor = mod.BehaviorMonitor


def test_all():
    # Sample 1: 3 turns with diverse queries
    sample1 = {
        'idx': 0,
        'turn_queries': [
            [101, 202, 303, 404, 505, 606],
            [707, 808, 909, 1010, 1111, 1212],
            [1313, 1414, 1515, 1616, 1717, 1818],
        ]
    }

    # Sample 2: 2 turns with repetitive queries (identical)
    sample2 = {
        'idx': 1,
        'turn_queries': [
            [101, 202, 303, 404, 505, 606],
            [101, 202, 303, 404, 505, 606],
        ]
    }

    # Sample 3: only 1 turn
    sample3 = {
        'idx': 2,
        'turn_queries': [[201, 301, 401, 501]],
    }

    per_turn_info_list = [sample1, sample2, sample3]

    # ---- search_depth ----
    depth = compute_search_depth(per_turn_info_list)
    print(f'search_depth: {depth:.2f}  (expected: 2.0)')
    assert abs(depth - 2.0) < 1e-6

    # ---- query_diversity ----
    diversity = compute_query_diversity(per_turn_info_list, similarity_threshold=0.3)
    expected_div = (1.0 + 0.0 + 1.0) / 3
    print(f'query_diversity: {diversity:.4f}  (expected: {expected_div:.4f})')
    assert abs(diversity - expected_div) < 1e-4

    # ---- info_gain without tokenizer ----
    ig = compute_info_gain(per_turn_info_list, ground_truths=None, tokenizer=None)
    assert ig == 0.0
    print(f'info_gain (no GT): {ig:.4f} OK')

    # ---- BehaviorMonitor.check() ----
    monitor = BehaviorMonitor(diversity_threshold=0.15, depth_threshold=2.0)
    metrics, alerts = monitor.check(step=1, per_turn_info_list=per_turn_info_list)
    print(f'\nMetrics: {metrics}')
    print(f'Alerts: {alerts}')
    assert 'behavior/search_depth' in metrics
    assert 'behavior/query_diversity' in metrics
    assert 'behavior/info_gain' in metrics
    assert 'behavior/alert_count' in metrics
    assert len(alerts) == 0  # depth=2.0 (not below), diversity=0.67 (above 0.15)

    # ---- Alert triggering (all repetitive) ----
    low_div = [sample2, sample2, sample2]
    metrics2, alerts2 = monitor.check(step=2, per_turn_info_list=low_div)
    print(f'\nLow-diversity: diversity={metrics2["behavior/query_diversity"]:.4f}')
    print(f'Alerts: {alerts2}')
    assert any('DIVERSITY_DROP' in a for a in alerts2)

    # ---- Alert triggering (short trace) ----
    short_trace = [sample3, sample3]  # depth = 1.0 < 2.0
    metrics3, alerts3 = monitor.check(step=3, per_turn_info_list=short_trace)
    print(f'\nShort-trace: depth={metrics3["behavior/search_depth"]:.2f}')
    print(f'Alerts: {alerts3}')
    assert any('SHORT_TRACE' in a for a in alerts3)

    # ---- Empty input ----
    metrics4, alerts4 = monitor.check(step=4, per_turn_info_list=[])
    assert metrics4['behavior/search_depth'] == 0.0
    assert metrics4['behavior/query_diversity'] == 0.0
    print(f'\nEmpty input OK')

    # ---- info_gain with mock tokenizer ----
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(100, 120))  # GT tokens: [100..119]

    sample_ig = {
        'idx': 0,
        'turn_queries': [
            [100, 101, 102, 200, 201],  # covers 100,101,102 of GT
            [103, 104, 105, 300, 301],  # covers 103,104,105 additionally
        ]
    }
    ig = compute_info_gain([sample_ig], ground_truths=['some ground truth'], tokenizer=MockTokenizer())
    # turn 1: coverage = 3/20 = 0.15, delta = 0.15
    # turn 2: coverage = 6/20 = 0.30, delta = 0.15
    # mean delta = 0.15
    print(f'\ninfo_gain with mock tokenizer: {ig:.4f}  (expected: 0.15)')
    assert abs(ig - 0.15) < 1e-4

    print('\n=== ALL TESTS PASSED ===')


if __name__ == '__main__':
    test_all()
