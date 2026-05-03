#!/usr/bin/env python3
"""Pure random search baseline: sample N candidates uniformly from the GA
gene space, evaluate each with the same fitness function, and log results.

Used to answer: does the GA actually outperform random sampling?

Usage
-----
    python3 tools/random_search.py \
        --config configs/m1_random_init.json \
        --n-samples 200 \
        --out results/random_search_baseline \
        --seed 9999
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.evaluator import evaluate_candidate
from search.genetic_search import (
    _build_gene_space,
    _combo_keys,
    _decode_solution_to_params,
    _extract_fixed_probs,
)


def _fmt_hms(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--out", required=True, help="Output directory for results")
    parser.add_argument("--seed", type=int, default=9999)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "random_search_results.jsonl"
    summary_path = out_dir / "random_search_summary.json"

    search_cfg = cfg["search"]
    eval_cfg = copy.deepcopy(cfg["eval"])
    eval_cfg.pop("heartbeat_log_path", None)
    if isinstance(eval_cfg.get("solver"), dict):
        eval_cfg["solver"].pop("heartbeat_log_path", None)

    base_params = copy.deepcopy(search_cfg["base_params"])
    vector_cfg = search_cfg.get("vector_gene", {})
    outcomes_per_combo = int(search_cfg.get("outcomes_per_combo", 5))
    combos = _combo_keys(base_params)
    fixed_probs = _extract_fixed_probs(base_params, combos, outcomes_per_combo)
    outcomes_total = len(combos) * outcomes_per_combo
    gene_space = _build_gene_space(outcomes_total, vector_cfg)

    rng = random.Random(args.seed)
    fitnesses = []
    run_start = time.time()

    print(f"Random search baseline")
    print(f"  config     : {args.config}")
    print(f"  n_samples  : {args.n_samples}")
    print(f"  seed       : {args.seed}")
    print(f"  output     : {args.out}")
    print(f"  ticks      : {base_params['max_time_ticks']}")
    print(f"  seeds      : {eval_cfg['seeds']}")
    print()

    with results_path.open("w", encoding="utf-8") as out_f:
        for i in range(args.n_samples):
            sample = [rng.uniform(float(g["low"]), float(g["high"])) for g in gene_space]
            candidate_params = _decode_solution_to_params(
                sample, base_params, combos, outcomes_per_combo, fixed_probs, vector_cfg
            )

            t0 = time.time()
            result = evaluate_candidate(candidate_params, eval_cfg)
            elapsed = time.time() - t0
            fitness = float(result["fitness"])
            fitnesses.append(fitness)

            row = {
                "sample_idx": i,
                "fitness": fitness,
                "win_rate_mean": float(result.get("win_rate_mean", 0.0)),
                "eval_sec": elapsed,
                "elapsed_total_sec": time.time() - run_start,
                "metrics": result.get("metrics", {}),
            }
            out_f.write(json.dumps(row, sort_keys=True) + "\n")
            out_f.flush()

            # Running stats every 10 samples
            n_done = i + 1
            elapsed_total = time.time() - run_start
            rate = elapsed_total / n_done
            eta = rate * (args.n_samples - n_done)
            print(
                f"  [{n_done:>3}/{args.n_samples}] "
                f"fitness={fitness:.4f}  "
                f"best_so_far={max(fitnesses):.4f}  "
                f"eval={elapsed:.1f}s  "
                f"elapsed={_fmt_hms(elapsed_total)}  "
                f"ETA={_fmt_hms(eta)}",
                flush=True,
            )

    # Final summary
    summary = {
        "n_samples": args.n_samples,
        "seed": args.seed,
        "config": args.config,
        "best_fitness": max(fitnesses),
        "mean_fitness": statistics.mean(fitnesses),
        "median_fitness": statistics.median(fitnesses),
        "std_fitness": statistics.pstdev(fitnesses) if len(fitnesses) > 1 else 0.0,
        "min_fitness": min(fitnesses),
        "pct_above_0_65": sum(1 for f in fitnesses if f > 0.65) / len(fitnesses),
        "pct_above_0_70": sum(1 for f in fitnesses if f > 0.70) / len(fitnesses),
        "total_wall_sec": time.time() - run_start,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print()
    print("=" * 52)
    print("  RANDOM SEARCH RESULTS")
    print("=" * 52)
    print(f"  n_samples   : {args.n_samples}")
    print(f"  best        : {summary['best_fitness']:.4f}")
    print(f"  mean        : {summary['mean_fitness']:.4f}")
    print(f"  median      : {summary['median_fitness']:.4f}")
    print(f"  std         : {summary['std_fitness']:.4f}")
    print(f"  min         : {summary['min_fitness']:.4f}")
    print(f"  % above 0.65: {summary['pct_above_0_65']*100:.1f}%")
    print(f"  % above 0.70: {summary['pct_above_0_70']*100:.1f}%")
    print(f"  total time  : {_fmt_hms(summary['total_wall_sec'])}")


if __name__ == "__main__":
    main()
