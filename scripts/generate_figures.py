#!/usr/bin/env python3
"""
CDR Figures — Generate evaluation charts from baseline data.

Produces:
    eval/results/fig_latency.png    (p50/p95 latency per pipeline stage)

This is the script behind `make figures`.

Usage:
    PYTHONPATH=src python scripts/generate_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def generate_latency_chart() -> Path:
    """Generate pipeline stage latency chart (p50/p95)."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("❌  matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    # Baseline latency data (from eval/results/baseline_v0_1.md)
    # Format: stage -> (p50_ms, p95_ms)
    stages = {
        "parse_question": (800, 1500),
        "plan_search": (1200, 2200),
        "retrieve": (3000, 6000),
        "deduplicate": (50, 120),
        "screen": (2500, 5000),
        "parse_docs": (1500, 3500),
        "extract_data": (4000, 8000),
        "assess_rob2": (3500, 7000),
        "synthesize": (5000, 10000),
        "critique": (2000, 4500),
        "verify": (2500, 5500),
        "compose": (1500, 3000),
        "publish": (200, 500),
    }

    labels = list(stages.keys())
    p50 = [stages[s][0] for s in labels]
    p95 = [stages[s][1] for s in labels]

    # Color-code by phase
    phase_colors = {
        "parse_question": "#3b82f6",
        "plan_search": "#3b82f6",
        "retrieve": "#3b82f6",
        "deduplicate": "#3b82f6",
        "screen": "#f59e0b",
        "parse_docs": "#f59e0b",
        "extract_data": "#f59e0b",
        "assess_rob2": "#a855f7",
        "synthesize": "#a855f7",
        "critique": "#ef4444",
        "verify": "#22c55e",
        "compose": "#22c55e",
        "publish": "#22c55e",
    }
    colors_p50 = [phase_colors[s] for s in labels]
    colors_p95 = [phase_colors.get(s, "#94a3b8") for s in labels]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    x = range(len(labels))
    width = 0.35

    bars_p50 = ax.bar(
        [i - width / 2 for i in x],
        [v / 1000 for v in p50],
        width,
        label="p50",
        color=colors_p50,
        alpha=0.85,
        edgecolor="none",
    )
    bars_p95 = ax.bar(
        [i + width / 2 for i in x],
        [v / 1000 for v in p95],
        width,
        label="p95",
        color=colors_p95,
        alpha=0.45,
        edgecolor="none",
    )

    ax.set_xlabel("Pipeline Stage", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Latency (seconds)", color="#94a3b8", fontsize=11)
    ax.set_title(
        "CDR Pipeline Latency — p50 / p95 per Stage",
        color="#e2e8f0",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9, color="#94a3b8")
    ax.tick_params(axis="y", colors="#94a3b8")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # Grid
    ax.grid(axis="y", color="#334155", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    # Legend
    legend = ax.legend(
        loc="upper left",
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#94a3b8",
        fontsize=10,
    )

    # Spine styling
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # Annotations for hotspots
    max_idx = p95.index(max(p95))
    ax.annotate(
        f"{p95[max_idx] / 1000:.1f}s",
        xy=(max_idx + width / 2, p95[max_idx] / 1000),
        xytext=(max_idx + 1.5, p95[max_idx] / 1000 + 0.5),
        arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.2),
        fontsize=9,
        color="#ef4444",
        fontweight="bold",
    )

    plt.tight_layout()

    out_path = ROOT / "eval" / "results" / "fig_latency.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return out_path


def main() -> int:
    print("📊  CDR Figures — Generating evaluation charts")
    print("=" * 50)
    print()

    chart_path = generate_latency_chart()
    print(f"  ✅ {chart_path.relative_to(ROOT)}")
    print(f"     Size: {chart_path.stat().st_size // 1024} KB")
    print()
    print("✅  Figures complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
