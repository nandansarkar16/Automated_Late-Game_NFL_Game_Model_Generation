#!/usr/bin/env python3
"""Build thesis-ready figures from ``history.jsonl`` (per-generation GA log).

Writes PNGs under ``<run-dir>/figures/`` by default:
  - ga_fitness_convergence.png   — best / mean fitness + population min–max band
  - ga_simulation_metrics.png    — win rate, yards/play, plays/game, score diff (gen-best)
  - ga_evaluation_components.png — composite metric components (0–1 scale)

Usage:
  python3 tools/plot_ga_summary.py --run-dir results/m1_local_overnight
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_history(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r.get("generation", 0)))
    return rows


def _plot_fitness(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = [int(r["generation"]) for r in rows]
    best = [float(r["best_fitness"]) for r in rows]
    mean = [float(r["mean_fitness"]) for r in rows]
    lo = [float(r["min_fitness"]) for r in rows]
    hi = [float(r["max_fitness"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.fill_between(g, lo, hi, alpha=0.22, label="Population min–max")
    ax.plot(g, mean, "o-", lw=1.6, ms=5, label="Population mean")
    ax.plot(g, best, "s-", lw=2.2, ms=6, label="Generation best (elitism)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Composite fitness")
    ax.set_title("GA fitness convergence")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_simulation(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = [int(r["generation"]) for r in rows]
    wr = [float(r.get("best_win_rate_mean", 0.0)) for r in rows]
    ypp = [float(r.get("best_avg_yards_per_play", 0.0)) for r in rows]
    ppg = [float(r.get("best_avg_plays_per_game", 0.0)) for r in rows]
    sc = [float(r.get("best_avg_final_score_diff", 0.0)) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    ax0, ax1, ax2, ax3 = axes.flat

    ax0.plot(g, wr, "o-", color="#1f77b4", lw=1.8, ms=5)
    ax0.set_ylabel("Mean win rate (gen-best)")
    ax0.set_title("Simulation: win rate")
    ax0.grid(True, linestyle="--", alpha=0.35)

    ax1.plot(g, ypp, "o-", color="#ff7f0e", lw=1.8, ms=5)
    ax1.set_ylabel("Avg yards / play")
    ax1.set_title("Simulation: yards per play")
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2.plot(g, ppg, "o-", color="#2ca02c", lw=1.8, ms=5)
    ax2.set_ylabel("Avg plays / game")
    ax2.set_title("Simulation: pace (plays per game)")
    ax2.grid(True, linestyle="--", alpha=0.35)

    ax3.plot(g, sc, "o-", color="#d62728", lw=1.8, ms=5)
    ax3.set_ylabel("Avg final score diff (offense)")
    ax3.set_title("Simulation: score differential")
    ax3.grid(True, linestyle="--", alpha=0.35)

    for ax in axes.flat:
        ax.set_xlabel("Generation")
    fig.suptitle("Metrics of the generation-best candidate (per seed average)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_components(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [
        "StrategicDiversity",
        "StateSensitivity",
        "OutcomePlausibility",
        "Robustness",
        "NonDegeneracy",
        "YardsPlausibility",
    ]
    g = [int(r["generation"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    cmap = plt.cm.tab10.colors
    for i, key in enumerate(keys):
        ys = []
        for r in rows:
            m = r.get("best_metrics") or {}
            ys.append(float(m.get(key, 0.0)))
        ax.plot(g, ys, "o-", lw=1.5, ms=4, color=cmap[i % len(cmap)], label=key)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Component score (0–1)")
    ax.set_title("Evaluation components for generation-best model")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument(
        "--history",
        type=Path,
        help="Path to history.jsonl (default: <run-dir>/history.jsonl)",
    )
    args = ap.parse_args()
    run_dir: Path = args.run_dir
    hist_path = args.history or (run_dir / "history.jsonl")
    if not hist_path.exists():
        print(f"[error] missing {hist_path}", file=sys.stderr)
        return 2

    rows = _load_history(hist_path)
    if not rows:
        print("[error] empty history", file=sys.stderr)
        return 2

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    p1 = fig_dir / "ga_fitness_convergence.png"
    p2 = fig_dir / "ga_simulation_metrics.png"
    p3 = fig_dir / "ga_evaluation_components.png"

    _plot_fitness(rows, p1)
    _plot_simulation(rows, p2)
    _plot_components(rows, p3)

    print(f"[ok] {p1}")
    print(f"[ok] {p2}")
    print(f"[ok] {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
