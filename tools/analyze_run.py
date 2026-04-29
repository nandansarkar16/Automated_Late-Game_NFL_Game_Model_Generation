#!/usr/bin/env python3
"""Post-hoc analysis of a GA run.

Builds the data the thesis needs from whatever the run actually produced:

1. Per-generation fitness stats (best / mean / min / max / median / std),
   reconstructed from either `progress.jsonl` (preferred, richer) or
   `heartbeat.jsonl` (fallback for runs where on_generation silently
   dropped progress rows).
2. A fitness convergence plot (optional, matplotlib) written to PNG.
3. An initial-vs-final play_outcomes diff, ready to drop into a table.

Usage
-----
    python3 tools/analyze_run.py --run-dir results/m1_local_overnight \
        --config configs/m1_local_overnight.json \
        [--plot out.png]

All artifacts land inside ``run-dir`` unless explicit paths are passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _gen_stats_from_progress(rows: List[Dict]) -> List[Dict]:
    return [
        {
            "generation": int(r.get("generation", i)),
            "best_fitness": float(r.get("best_fitness", float("nan"))),
            "mean_fitness": float(r.get("mean_fitness", float("nan"))),
            "min_fitness": float(r.get("min_fitness", float("nan"))),
            "max_fitness": float(r.get("max_fitness", float("nan"))),
            "median_fitness": float(r.get("median_fitness", float("nan"))),
            "std_fitness": float(r.get("std_fitness", float("nan"))),
            "source": "progress",
        }
        for i, r in enumerate(rows)
    ]


def _gen_stats_from_heartbeat(rows: List[Dict], population: int) -> List[Dict]:
    """Group candidate_eval / cache_hit heartbeats by generation.

    Elite candidates (elitism carries top-k genes unchanged) do not re-enter
    fitness_func, so their fitness values are not in the heartbeat stream
    for the new generation. We backfill: if a generation has fewer than
    `population` entries, pad the missing ones with the best fitnesses from
    the previous generation. This is the correct reconstruction for
    keep_elitism because PyGAD preserves top-k by fitness exactly.
    """
    by_gen: Dict[int, List[float]] = {}
    best_trace: Dict[int, float] = {}
    for r in rows:
        ev = r.get("event")
        if ev not in {"candidate_eval", "cache_hit"}:
            continue
        g = r.get("generation_in_progress")
        try:
            g = int(g)
        except (TypeError, ValueError):
            continue
        try:
            f = float(r.get("fitness"))
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        by_gen.setdefault(g, []).append(f)
        best_trace[g] = max(best_trace.get(g, float("-inf")), f)

    if not by_gen:
        return []

    max_gen = max(by_gen)
    out: List[Dict] = []
    # Maintain a running sorted list of recent fitnesses we can use to backfill
    # elite slots for a generation that is missing them.
    prev_top: List[float] = []
    for g in range(max_gen + 1):
        vals = list(by_gen.get(g, []))
        if population and len(vals) < population and prev_top:
            # Pad with elites from previous generation (top-k unchanged).
            needed = population - len(vals)
            vals.extend(prev_top[:needed])
        if not vals:
            continue
        vals_sorted = sorted(vals, reverse=True)
        prev_top = vals_sorted
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        median = statistics.median(vals)
        out.append(
            {
                "generation": g,
                "best_fitness": vals_sorted[0],
                "mean_fitness": float(mean),
                "min_fitness": vals_sorted[-1],
                "max_fitness": vals_sorted[0],
                "median_fitness": float(median),
                "std_fitness": float(std),
                "n_real_evals": len(by_gen.get(g, [])),
                "n_with_elites": len(vals),
                "source": "heartbeat",
            }
        )
    return out


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _plot_convergence(rows: List[Dict], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] matplotlib unavailable, skipping plot: {exc}", file=sys.stderr)
        return
    if not rows:
        print("[warn] no per-gen rows, skipping plot", file=sys.stderr)
        return
    gens = [r["generation"] for r in rows]
    best = [r["best_fitness"] for r in rows]
    mean = [r["mean_fitness"] for r in rows]
    mn = [r["min_fitness"] for r in rows]
    mx = [r["max_fitness"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.fill_between(gens, mn, mx, alpha=0.2, label="pop min/max")
    ax.plot(gens, mean, marker="o", linewidth=1.5, label="population mean")
    ax.plot(gens, best, marker="s", linewidth=2.0, label="generation best")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Composite fitness")
    ax.set_title("GA fitness convergence")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _diff_play_outcomes(
    base_params: Dict, final_params: Dict
) -> List[Tuple[str, int, Dict, Dict]]:
    """Per-(combo, outcome_idx) rows showing base -> final tuple deltas.

    Only rows with any field change are returned, sorted by combo key then
    outcome index.
    """
    base = base_params.get("play_outcomes", {})
    final = final_params.get("play_outcomes", {})
    out: List[Tuple[str, int, Dict, Dict]] = []
    for combo in sorted(set(base) | set(final)):
        bo = base.get(combo, [])
        fo = final.get(combo, [])
        n = max(len(bo), len(fo))
        for i in range(n):
            b = bo[i] if i < len(bo) else {}
            f = fo[i] if i < len(fo) else {}
            if _outcome_changed(b, f):
                out.append((combo, i, b, f))
    return out


def _outcome_changed(a: Dict, b: Dict) -> bool:
    if not a or not b:
        return True
    for key in ("yards", "time", "turnover", "prob"):
        if _scalar_differs(a.get(key), b.get(key)):
            return True
    return False


def _scalar_differs(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) > 1e-6
    return a != b


def _fmt_outcome(o: Dict) -> str:
    if not o:
        return "-"
    return (
        f"y={int(round(float(o.get('yards', 0))))}, "
        f"t={int(round(float(o.get('time', 0))))}, "
        f"to={bool(o.get('turnover'))}, "
        f"p={float(o.get('prob', 0.0)):.3f}"
    )


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_best(run_dir: Path) -> Optional[Dict]:
    path = run_dir / "best_candidate.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="Directory with run artifacts")
    parser.add_argument(
        "--config",
        help="Config that produced the run, used for initial params. "
        "Optional if best_candidate.json is absent.",
    )
    parser.add_argument("--plot", help="Path for convergence PNG (default: <run-dir>/convergence.png)")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG generation",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"[error] run_dir not found: {run_dir}", file=sys.stderr)
        return 2

    progress_rows = _read_jsonl(run_dir / "progress.jsonl")
    heartbeat_rows = _read_jsonl(run_dir / "heartbeat.jsonl")

    population = 0
    if args.config:
        try:
            cfg = _load_config(Path(args.config))
            population = int(cfg.get("search", {}).get("population", 0))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not read config: {exc}", file=sys.stderr)

    if progress_rows:
        stats = _gen_stats_from_progress(progress_rows)
        print(f"[ok] built {len(stats)} generations from progress.jsonl")
    else:
        stats = _gen_stats_from_heartbeat(heartbeat_rows, population)
        print(
            f"[ok] built {len(stats)} generations from heartbeat.jsonl "
            f"(progress.jsonl unavailable; elite slots backfilled using previous-gen top-k)"
        )

    gen_stats_csv = run_dir / "generation_stats.csv"
    _write_csv(gen_stats_csv, stats)
    print(f"[ok] wrote {gen_stats_csv}")

    if not args.no_plot:
        plot_path = Path(args.plot) if args.plot else run_dir / "convergence.png"
        _plot_convergence(stats, plot_path)
        if plot_path.exists():
            print(f"[ok] wrote {plot_path}")

    if args.config:
        best = _load_best(run_dir)
        if best is None:
            print(
                "[skip] best_candidate.json not present yet; "
                "rerun after the GA finishes for the params diff table.",
                file=sys.stderr,
            )
        else:
            cfg = _load_config(Path(args.config))
            base_params = cfg.get("search", {}).get("base_params", {})
            final_params = best.get("params", {})
            diff_rows = _diff_play_outcomes(base_params, final_params)
            diff_path = run_dir / "params_diff.csv"
            with diff_path.open("w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["combo", "outcome_idx", "base", "final"])
                for combo, idx, base, final in diff_rows:
                    w.writerow([combo, idx, _fmt_outcome(base), _fmt_outcome(final)])
            print(f"[ok] wrote {diff_path} ({len(diff_rows)} changed outcomes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
