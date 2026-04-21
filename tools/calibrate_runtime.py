#!/usr/bin/env python3
# Pre-flight timing helper: runs one evaluate_candidate at a chosen
# max_time_ticks and prints per-seed solve time plus a wall-clock
# projection for the GA config you'd actually launch.
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


def _load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_hms(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def main():
    parser = argparse.ArgumentParser(
        description="Time one candidate evaluation so you can size GA configs honestly."
    )
    parser.add_argument("--config", required=True, help="Base config to borrow base_params + eval from")
    parser.add_argument("--max-ticks", type=int, default=None, help="Override base_params.max_time_ticks for the probe")
    parser.add_argument("--seeds", type=int, default=None, help="Override eval.seeds count (uses first N of configured seeds, or [11,19,23])")
    parser.add_argument("--sim-games", type=int, default=None, help="Override eval.sim_games for the probe")
    parser.add_argument("--policy-probe-states", type=int, default=None, help="Override eval.policy_probe_states for the probe")
    parser.add_argument("--population", type=int, default=None, help="GA population you intend to run (for projection)")
    parser.add_argument("--generations", type=int, default=None, help="GA generations you intend to run (for projection)")
    parser.add_argument("--target-hours", type=float, default=None, help="If set, also suggests (pop, gens) combos under this budget")
    parser.add_argument("--stress", type=int, default=0, help="Stress mode: after the base eval, also run N random gene samples drawn from the GA gene_space so per-candidate variance is observed. 0 disables.")
    parser.add_argument("--stress-seed", type=int, default=4242, help="Random seed for the stress-mode gene sampler")
    parser.add_argument("--stress-timeout", type=float, default=600.0, help="Per-stress-candidate wall-clock limit in seconds; skips remainder if exceeded and marks as outlier")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    base_params = copy.deepcopy(cfg["search"]["base_params"])
    eval_cfg = copy.deepcopy(cfg["eval"])

    # Don't spam a run's real heartbeat with probe data.
    eval_cfg.pop("heartbeat_log_path", None)
    if isinstance(eval_cfg.get("solver"), dict):
        eval_cfg["solver"].pop("heartbeat_log_path", None)

    if args.max_ticks is not None:
        base_params["max_time_ticks"] = int(args.max_ticks)
    if args.sim_games is not None:
        eval_cfg["sim_games"] = int(args.sim_games)
    if args.policy_probe_states is not None:
        eval_cfg["policy_probe_states"] = int(args.policy_probe_states)

    seed_pool = eval_cfg.get("seeds", [11, 19, 23])
    if args.seeds is not None:
        n = max(1, int(args.seeds))
        if len(seed_pool) >= n:
            eval_cfg["seeds"] = seed_pool[:n]
        else:
            default = [11, 19, 23, 29, 31]
            eval_cfg["seeds"] = (seed_pool + default)[:n]

    n_seeds = len(eval_cfg["seeds"])
    max_ticks = int(base_params["max_time_ticks"])

    print("=== Calibration probe ===")
    print(f"  max_time_ticks     : {max_ticks}")
    print(f"  seeds              : {eval_cfg['seeds']}  (n={n_seeds})")
    print(f"  sim_games          : {eval_cfg.get('sim_games')}")
    print(f"  policy_probe_states: {eval_cfg.get('policy_probe_states')}")
    print("Running one evaluate_candidate...", flush=True)

    t0 = time.time()
    result = evaluate_candidate(base_params, eval_cfg)
    elapsed = time.time() - t0

    solver_meta = result.get("solver_meta", [])
    if solver_meta:
        per_seed = [float(m.get("elapsed_sec", 0.0)) for m in solver_meta]
        lp_solves = sum(int(m.get("lp_solves", 0)) for m in solver_meta)
        lp_failed = sum(int(m.get("lp_failed", 0)) for m in solver_meta)
        reachable = [int(m.get("reachable_states", 0)) for m in solver_meta]
        print("--- solver per seed ---")
        for i, m in enumerate(solver_meta):
            print(
                f"  seed {eval_cfg['seeds'][i]:>4}: "
                f"elapsed={float(m.get('elapsed_sec', 0.0)):7.2f}s  "
                f"lp_solves={int(m.get('lp_solves', 0)):>7}  "
                f"lp_failed={int(m.get('lp_failed', 0)):>4}  "
                f"reachable={int(m.get('reachable_states', 0)):>7}"
            )
        solve_mean = sum(per_seed) / len(per_seed)
        print(f"--- aggregates ---")
        print(f"  mean solve sec / seed       : {solve_mean:.2f}s")
        print(f"  total LP solves (all seeds) : {lp_solves}")
        print(f"  total LP failures           : {lp_failed}")
        print(f"  reachable states mean       : {sum(reachable)/len(reachable):.0f}")
    print(f"  total candidate eval time    : {elapsed:.2f}s ({_fmt_hms(elapsed)})")
    print(f"  fitness                      : {result.get('fitness'):.4f}")
    print(f"  win_rate_mean                : {result.get('win_rate_mean'):.4f}")

    stress_samples = []
    if args.stress > 0:
        print()
        print(f"=== Stress mode: {args.stress} random gene samples from GA gene_space ===")
        search_cfg = cfg.get("search", {})
        base_params_for_gene = copy.deepcopy(search_cfg["base_params"])
        base_params_for_gene["max_time_ticks"] = max_ticks
        vector_cfg = search_cfg.get("vector_gene", {})
        outcomes_per_combo = int(search_cfg.get("outcomes_per_combo", 5))
        combos = _combo_keys(base_params_for_gene)
        fixed_probs = _extract_fixed_probs(base_params_for_gene, combos, outcomes_per_combo)
        outcomes_total = len(combos) * outcomes_per_combo
        gene_space = _build_gene_space(outcomes_total, vector_cfg)

        rng = random.Random(int(args.stress_seed))
        for k in range(int(args.stress)):
            sample = []
            for g in gene_space:
                lo = float(g["low"])
                hi = float(g["high"])
                sample.append(rng.uniform(lo, hi))
            candidate_params = _decode_solution_to_params(
                sample,
                base_params_for_gene,
                combos,
                outcomes_per_combo,
                fixed_probs,
                vector_cfg,
            )
            t0 = time.time()
            stress_result = evaluate_candidate(candidate_params, eval_cfg)
            dt = time.time() - t0
            smeta = stress_result.get("solver_meta", [])
            per_seed = [float(m.get("elapsed_sec", 0.0)) for m in smeta] if smeta else [0.0]
            reach = [int(m.get("reachable_states", 0)) for m in smeta] if smeta else [0]
            stress_samples.append({
                "idx": k,
                "total_sec": dt,
                "per_seed": per_seed,
                "reachable": reach,
                "fitness": float(stress_result.get("fitness", 0.0)),
            })
            print(
                f"  sample {k+1:>2}/{args.stress}: "
                f"total={dt:7.2f}s  "
                f"per_seed={'/'.join(f'{x:.0f}' for x in per_seed)}s  "
                f"reachable_max={max(reach):>7}  "
                f"fitness={stress_result.get('fitness', 0.0):.3f}"
                + ("  [TIMEOUT-like, flagged]" if dt > float(args.stress_timeout) else "")
            )

        if stress_samples:
            times = [s["total_sec"] for s in stress_samples]
            all_times = [elapsed] + times
            print()
            print("--- stress distribution (including the base eval) ---")
            print(f"  n                           : {len(all_times)}")
            print(f"  min                         : {min(all_times):.1f}s")
            print(f"  median                      : {statistics.median(all_times):.1f}s")
            print(f"  mean                        : {statistics.mean(all_times):.1f}s")
            print(f"  max                         : {max(all_times):.1f}s")
            if len(all_times) >= 2:
                print(f"  stdev                       : {statistics.pstdev(all_times):.1f}s")

    pop = args.population
    gens = args.generations
    if pop is None or gens is None:
        search_cfg = cfg.get("search", {})
        pop = pop if pop is not None else int(search_cfg.get("population", 10))
        gens = gens if gens is not None else int(search_cfg.get("generations", 15))

    total_evals = pop * gens
    # Use stress mean if available, otherwise fall back to the base eval alone.
    if stress_samples:
        projection_sec = statistics.mean([s["total_sec"] for s in stress_samples] + [elapsed])
        projection_max = max([s["total_sec"] for s in stress_samples] + [elapsed])
        proj_source = "mean(base + stress)"
    else:
        projection_sec = elapsed
        projection_max = elapsed
        proj_source = "base eval only"
    projected = total_evals * projection_sec
    projected_worst = total_evals * projection_max
    print()
    print(f"=== Projection (source: {proj_source}) ===")
    print(f"  pop={pop}, gens={gens}  =>  {total_evals} evals")
    print(f"  projected wall-clock (mean rate): {_fmt_hms(projected)} ({projected/3600:.2f}h)")
    if stress_samples:
        print(f"  projected wall-clock (worst rate): {_fmt_hms(projected_worst)} ({projected_worst/3600:.2f}h)")
    # Real runs have cache hits on elites and repeated children; rough discount:
    discount = 0.75
    print(f"  with ~{int((1-discount)*100)}% cache-hit discount: {_fmt_hms(projected*discount)} ({projected*discount/3600:.2f}h)")

    if args.target_hours is not None:
        budget_sec = float(args.target_hours) * 3600.0
        max_evals = int(budget_sec / max(1e-9, projection_sec))
        effective_evals = int(max_evals / discount)
        print("=== Suggestions under budget ===")
        print(f"  budget {args.target_hours:.1f}h => ~{max_evals} raw evals (~{effective_evals} with cache discount)")
        # Show a few (pop, gens) combos hitting effective_evals
        print("  candidate (population, generations) combos:")
        for p in (8, 10, 12, 16):
            g = max(1, effective_evals // p)
            print(f"    population={p:>3} -> generations={g:>3}  ({p*g} evals)")


if __name__ == "__main__":
    main()
