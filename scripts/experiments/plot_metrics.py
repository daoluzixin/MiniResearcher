#!/usr/bin/env python3
"""
plot_metrics.py — Plot experiment metrics from SwanLab-exported CSV files

Usage:
  1. Export runs from SwanLab web UI (swanlab.cn → your project → Runs → Export CSV)
  2. Place CSV files in outputs/ folder (one per experiment run)
  3. Run this script:

  # Convergence comparison
  python scripts/experiments/plot_metrics.py \
      --runs outputs/exp00_baseline.csv outputs/exp01_drgrpo.csv \
      --metrics critic/score/mean critic/rewards/mean \
      --output outputs/convergence.png

  # Trajectory length comparison
  python scripts/experiments/plot_metrics.py \
      --runs outputs/exp00_baseline.csv outputs/exp04_curriculum.csv \
      --metrics trajectory/mean_turns \
      --output outputs/traj_length.png

  # Diversity dashboard (exp_05 monitoring run)
  python scripts/experiments/plot_metrics.py \
      --runs outputs/exp05_monitoring.csv \
      --metrics policy/query_diversity trajectory/mean_turns critic/score/mean \
      --plot_type diversity \
      --output outputs/monitoring_dashboard.png

  # Variance comparison (baseline vs PBRS)
  python scripts/experiments/plot_metrics.py \
      --runs outputs/exp00_baseline.csv outputs/exp02_pbrs.csv \
      --metrics critic/rewards/mean critic/rewards/std \
      --plot_type variance \
      --output outputs/variance_comparison.png

CSV format expected:
  step,metric,value
  100,critic/score/mean,0.23
  100,critic/rewards/mean,0.45
  100,critic/rewards/std,0.12
  200,critic/score/mean,0.31
  ...

  SwanLab export: Project → Runs → Select run → Export → CSV
  (SwanLab web UI will generate a CSV with step-by-step metric values)
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot experiment metrics from SwanLab-exported CSV files"
    )
    parser.add_argument(
        "--runs", "-r",
        type=str,
        nargs="+",
        required=True,
        help="Path to one or more SwanLab-exported CSV files (one per experiment)"
    )
    parser.add_argument(
        "--metrics", "-m",
        type=str,
        nargs="+",
        default=["critic/score/mean"],
        help="Metric names to plot (column 'metric' in CSV)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/metrics.png",
        help="Output file path"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        default="convergence",
        choices=["convergence", "traj_length", "diversity", "variance"],
        help="Type of plot to generate"
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=3000,
        help="Maximum training step to show"
    )
    parser.add_argument(
        "--run_names",
        type=str,
        nargs="+",
        default=None,
        help="Display names for each run (defaults to filename)"
    )
    return parser.parse_args()


def load_csv(path: str) -> pd.DataFrame:
    """Load a SwanLab-exported CSV into a tidy DataFrame."""
    df = pd.read_csv(path)
    required = {"step", "metric", "value"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(
            f"CSV '{path}' missing required columns {missing}. "
            f"Found: {list(df.columns)}"
        )
    return df


def get_metric(df: pd.DataFrame, metric_name: str, max_step: int) -> tuple:
    """Extract step and value arrays for a specific metric."""
    sub = df[df["metric"] == metric_name]
    sub = sub[sub["step"] <= max_step].sort_values("step")
    return sub["step"].values, sub["value"].values


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth a series with a moving average."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_convergence(
    dfs: list, run_names: list, metrics: list, output: str, max_step: int
):
    """Plot convergence curves for one or more metrics across runs."""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1, n_metrics, figsize=(6 * n_metrics, 5), squeeze=False
    )
    palette = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]

    for col_idx, metric in enumerate(metrics):
        ax = axes[0, col_idx]
        for i, (df, name) in enumerate(zip(dfs, run_names)):
            try:
                steps, values = get_metric(df, metric, max_step)
            except Exception:
                continue
            if len(values) == 0:
                continue
            window = max(3, min(50, len(values) // 10))
            if window > 1:
                values = moving_average(values, window)
                steps = steps[: len(values)]
            color = palette[i % len(palette)]
            label = f"{name}" + (f" / {metric.split('/')[-1]}" if n_metrics > 1 else "")
            ax.plot(steps, values, label=label, color=color, alpha=0.85, linewidth=1.8)
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def plot_trajectory_length(
    dfs: list, run_names: list, output: str, max_step: int
):
    """Plot trajectory length (mean turns) over training steps."""
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    # Try multiple possible metric names
    metric_candidates = [
        "trajectory/mean_turns", "trajectory/turns/mean",
        "policy/num_turns", "critic/score/mean",
    ]

    for i, (df, name) in enumerate(zip(dfs, run_names)):
        for metric in metric_candidates:
            try:
                steps, values = get_metric(df, metric, max_step)
            except Exception:
                continue
            if len(values) > 0:
                break
        else:
            print(f"Warning: No trajectory metric found for '{name}', skipping")
            continue

        window = max(3, min(20, len(values) // 10))
        if window > 1:
            values = moving_average(values, window)
            steps = steps[: len(values)]
        color = palette[i % len(palette)]
        ax.plot(steps, values, label=name, color=color, linewidth=1.8)

    ax.axhline(y=4.8, color="green", linestyle="--", alpha=0.6, label="Target: 4.8 turns")
    ax.axhline(y=1.2, color="orange", linestyle="--", alpha=0.6, label="Baseline: 1.2 turns")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Trajectory Length (turns)", fontsize=12)
    ax.set_title("Trajectory Length Over Training", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def plot_diversity_dashboard(
    dfs: list, run_names: list, output: str, max_step: int
):
    """Plot a 2×2 monitoring dashboard from the full pipeline run."""
    if not dfs:
        return
    df = dfs[0]
    name = run_names[0] if run_names else "Run"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def safe_plot(ax, metric, title, ylabel, ref_lines=None, ref_labels=None):
        try:
            steps, vals = get_metric(df, metric, max_step)
        except Exception as e:
            ax.text(0.5, 0.5, f"Data unavailable:\n{e}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return
        if len(steps) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            return
        window = max(3, min(20, len(vals) // 10))
        vals_smooth = moving_average(vals, window) if window > 1 else vals
        steps_smooth = steps[:len(vals_smooth)]
        ax.plot(steps_smooth, vals_smooth, color="#4CAF50", linewidth=1.8)
        if ref_lines:
            for ref_val, ref_label in zip(ref_lines, ref_labels or []):
                ax.axhline(y=ref_val, linestyle="--", alpha=0.5, label=ref_label)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        if ref_labels:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    safe_plot(
        axes[0, 0], "policy/query_diversity", "Query Diversity (Bullet 5)",
        "Unique Query Fraction",
        ref_lines=[0.45, 0.15], ref_labels=["Target: 45%", "Alert: 15%"]
    )
    safe_plot(
        axes[0, 1], "critic/score/mean", "F1 Score Convergence",
        "F1 Score"
    )
    safe_plot(
        axes[1, 0], "critic/rewards/mean", "Training Reward",
        "Reward"
    )
    safe_plot(
        axes[1, 1], "trajectory/mean_turns", "Trajectory Length",
        "Turns",
        ref_lines=[4.8, 1.2]
    )

    plt.suptitle(f"Multi-Dimensional Monitoring Dashboard — {name}", fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def plot_variance_comparison(
    dfs: list, run_names: list, output: str, max_step: int
):
    """Plot reward mean±std to visualize variance reduction from PBRS."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = ["#2196F3", "#4CAF50"]

    for col_idx, (df, name) in enumerate(zip(dfs, run_names)):
        ax = axes[col_idx]
        try:
            steps_mean, vals_mean = get_metric(df, "critic/rewards/mean", max_step)
            _, vals_std = get_metric(df, "critic/rewards/std", max_step)
        except Exception as e:
            ax.text(0.5, 0.5, f"No data:\n{e}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name)
            continue

        if len(vals_mean) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(name)
            continue

        window = max(3, min(50, len(vals_mean) // 10))
        if window > 1:
            vals_mean = moving_average(vals_mean, window)
            if len(vals_std) > 0:
                vals_std = moving_average(vals_std, window)
            steps = steps_mean[:len(vals_mean)]
        else:
            steps = steps_mean

        ax.plot(steps, vals_mean, label=name, linewidth=1.8, color=palette[col_idx])
        if len(vals_std) > 0:
            std_n = min(len(vals_std), len(vals_mean))
            ax.fill_between(
                steps[:std_n],
                vals_mean[:std_n] - vals_std[:std_n],
                vals_mean[:std_n] + vals_std[:std_n],
                alpha=0.2, color=palette[col_idx]
            )
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Reward Mean ± Std")
        ax.set_title(f"{name}\n(Reward Variability)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Reward Variance Comparison: Baseline vs PBRS (Target: -60% variance)",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load all CSV files
    dfs, run_names = [], args.run_names or []
    for path in args.runs:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
        try:
            dfs.append(load_csv(path))
        except Exception as e:
            print(f"ERROR loading {path}: {e}")
            sys.exit(1)

    # Default run names from filenames
    if not run_names:
        run_names = [Path(p).stem for p in args.runs]

    if args.plot_type == "convergence":
        plot_convergence(dfs, run_names, args.metrics, args.output, args.max_step)

    elif args.plot_type == "traj_length":
        plot_trajectory_length(dfs, run_names, args.output, args.max_step)

    elif args.plot_type == "diversity":
        plot_diversity_dashboard(dfs, run_names, args.output, args.max_step)

    elif args.plot_type == "variance":
        plot_variance_comparison(dfs, run_names, args.output, args.max_step)


if __name__ == "__main__":
    main()
