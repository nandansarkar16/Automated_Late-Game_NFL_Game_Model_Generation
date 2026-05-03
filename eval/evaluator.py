from __future__ import annotations

# This file runs one candidate end-to-end (solve, simulate, score).
import copy
from collections import defaultdict
import json
import random
import time
from typing import Dict, List, Tuple

from agents.dp_lp_solver import solve_equilibrium_policy
from env.quarter_strategy import QuarterStrategy, State
from eval.metrics import (
    composite_score,
    degeneracy_penalty,
    outcome_plausibility,
    robustness,
    state_sensitivity,
    strategic_diversity,
)


# This file runs one candidate end-to-end (solve, simulate, score).

Bucket = Tuple[int, int, int, int]


def _append_jsonl(path: str | None, row: Dict):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def bucket_state(state: State) -> Bucket:
    yardline, down, distance, ticks, score_diff = state
    time_bucket = 0 if ticks <= 4 else 1 if ticks <= 10 else 2
    score_bucket = 0 if score_diff < 0 else 1 if score_diff == 0 else 2
    distance_bucket = 0 if distance <= 3 else 1 if distance <= 8 else 2
    return (time_bucket, score_bucket, down, distance_bucket)


def normalize(counts: List[float]) -> List[float]:
    total = sum(counts)
    if total <= 0:
        return [0.0 for _ in counts]
    return [c / total for c in counts]


def _band_score(value: float, low: float, high: float, slack: float) -> float:
    if low <= value <= high:
        return 1.0
    slack = max(1e-9, float(slack))
    if value < low:
        return max(0.0, 1.0 - (low - value) / slack)
    return max(0.0, 1.0 - (value - high) / slack)


def evaluate_policy_usage(
    model: QuarterStrategy,
    pi_off: Dict[State, Tuple[float, ...]],
    n_states: int,
    seed: int,
) -> Tuple[Dict[Bucket, List[float]], Dict[str, List[float]], Dict[str, float], List[float]]:
    rng = random.Random(seed)
    n_actions = model.offensive_playbook_size()

    bucket_counts = defaultdict(lambda: [0.0] * n_actions)
    regime_counts = {
        "trailing_late": [0.0] * n_actions,
        "tied_mid": [0.0] * n_actions,
        "leading_late": [0.0] * n_actions,
    }
    regime_totals = {k: 0.0 for k in regime_counts.keys()}
    global_counts = [0.0] * n_actions

    for _ in range(n_states):
        s = model.initial_position()
        perturb = model._normalize_state(
            (
                max(1, min(99, s[0] + rng.randint(-15, 15))),
                max(1, min(4, s[1] + rng.randint(-1, 1))),
                max(1, min(20, s[2] + rng.randint(-4, 4))),
                max(1, min(model.max_ticks, s[3] + rng.randint(-5, 5))),
                max(-model.max_score_diff, min(model.max_score_diff, s[4] + rng.randint(-7, 7))),
            )
        )

        probs = pi_off.get(perturb)
        if probs is None:
            # Skip unknown probe states for policy-shape metrics to avoid
            # inflating diversity/sensitivity with synthetic uniform play.
            continue
        for action, p in enumerate(probs):
            bucket_counts[bucket_state(perturb)][action] += p
            global_counts[action] += p

            if perturb[4] < 0 and perturb[3] <= 8:
                regime_counts["trailing_late"][action] += p
                regime_totals["trailing_late"] += p
            elif perturb[4] == 0 and perturb[3] > 8:
                regime_counts["tied_mid"][action] += p
                regime_totals["tied_mid"] += p
            elif perturb[4] > 0 and perturb[3] <= 8:
                regime_counts["leading_late"][action] += p
                regime_totals["leading_late"] += p

    regime_rates = {k: normalize(v) for k, v in regime_counts.items()}
    return bucket_counts, regime_rates, regime_totals, global_counts


def evaluate_candidate(params: Dict, eval_cfg: Dict) -> Dict:
    seeds = eval_cfg.get("seeds", [11, 19, 23])
    sim_games = int(eval_cfg.get("sim_games", 300))
    solve_cfg = eval_cfg.get("solver", {})
    heartbeat_log_path = eval_cfg.get("heartbeat_log_path")
    eval_marker = time.time_ns()
    _append_jsonl(
        heartbeat_log_path,
        {
            "event": "candidate_eval_start",
            "eval_marker": eval_marker,
            "seed_count": len(seeds),
            "sim_games": sim_games,
        },
    )

    seed_scores = []
    win_rates = []
    avg_yards_per_play_vals = []
    avg_plays_per_game_vals = []
    avg_final_score_diff_vals = []
    avg_offense_scoring_plays_vals = []
    solver_meta = []
    usage_metrics = []

    for s in seeds:
        _append_jsonl(
            heartbeat_log_path,
            {
                "event": "seed_start",
                "eval_marker": eval_marker,
                "seed": int(s),
            },
        )
        model = QuarterStrategy(params, seed=s)
        # Inner loop: solve equilibrium policy profile with DP + LP.
        seed_solve_cfg = copy.deepcopy(solve_cfg)
        seed_solve_cfg["heartbeat_log_path"] = heartbeat_log_path
        seed_solve_cfg["heartbeat_seed"] = int(s)
        solved = solve_equilibrium_policy(model, seed_solve_cfg)
        pi_off = solved["pi_off"]
        pi_def = solved["pi_def"]
        _append_jsonl(
            heartbeat_log_path,
            {
                "event": "seed_solved",
                "eval_marker": eval_marker,
                "seed": int(s),
                "solver_meta": solved.get("meta", {}),
            },
        )

        # Then evaluate that policy profile by simulation.
        sim_stats = model.simulate_profile_stats(pi_off, pi_def, sim_games)
        wr = float(sim_stats["win_rate"])
        win_rates.append(wr)
        seed_scores.append(2 * wr - 1)
        avg_yards_per_play_vals.append(float(sim_stats["avg_yards_per_play"]))
        avg_plays_per_game_vals.append(float(sim_stats["avg_plays_per_game"]))
        avg_final_score_diff_vals.append(float(sim_stats["avg_final_score_diff"]))
        avg_offense_scoring_plays_vals.append(float(sim_stats["avg_offense_scoring_plays_per_game"]))

        b_counts, regimes, regime_totals, global_counts = evaluate_policy_usage(
            model,
            pi_off,
            n_states=int(eval_cfg.get("policy_probe_states", 800)),
            seed=s + 1000,
        )
        usage_metrics.append((b_counts, regimes, regime_totals, global_counts))
        solver_meta.append(solved["meta"])
        _append_jsonl(
            heartbeat_log_path,
            {
                "event": "seed_sim_done",
                "eval_marker": eval_marker,
                "seed": int(s),
                "win_rate": float(sim_stats["win_rate"]),
                "avg_yards_per_play": float(sim_stats["avg_yards_per_play"]),
                "avg_plays_per_game": float(sim_stats["avg_plays_per_game"]),
                "avg_final_score_diff": float(sim_stats["avg_final_score_diff"]),
                "avg_offense_scoring_plays_per_game": float(sim_stats["avg_offense_scoring_plays_per_game"]),
            },
        )

    diversity_vals = [strategic_diversity(x[0]) for x in usage_metrics]
    sensitivity_vals = [state_sensitivity(x[1], x[2]) for x in usage_metrics]
    degeneracy_vals = [degeneracy_penalty(x[3]) for x in usage_metrics]

    aggregate_diversity = sum(diversity_vals) / len(diversity_vals)
    aggregate_sensitivity = sum(sensitivity_vals) / len(sensitivity_vals)
    aggregate_degeneracy = sum(degeneracy_vals) / len(degeneracy_vals)
    non_degeneracy = max(0.0, min(1.0, 1.0 - aggregate_degeneracy))
    plausibility = outcome_plausibility(params, eval_cfg.get("plausibility", {}))
    robust = robustness(seed_scores)
    win_rate_mean = sum(win_rates) / len(win_rates)
    avg_yards_per_play = sum(avg_yards_per_play_vals) / len(avg_yards_per_play_vals)
    avg_plays_per_game = sum(avg_plays_per_game_vals) / len(avg_plays_per_game_vals)
    avg_final_score_diff = sum(avg_final_score_diff_vals) / len(avg_final_score_diff_vals)
    avg_offense_scoring_plays = sum(avg_offense_scoring_plays_vals) / len(avg_offense_scoring_plays_vals)

    yards_low, yards_high = eval_cfg.get("yards_per_play_range", [2.0, 9.0])
    realism_slack = float(eval_cfg.get("realism_slack", 8.0))
    yards_score = _band_score(avg_yards_per_play, float(yards_low), float(yards_high), realism_slack)

    weights = eval_cfg.get("weights", [1.0 / 6.0] * 5)
    metrics = composite_score(
        aggregate_diversity,
        aggregate_sensitivity,
        plausibility,
        robust,
        non_degeneracy,
        weights,
    )
    yards_weight = float(eval_cfg.get("yards_score_weight", 1.0 / 6.0))
    metrics["AvgYardsPerPlay"] = avg_yards_per_play
    metrics["AvgPlaysPerGame"] = avg_plays_per_game
    metrics["AvgFinalScoreDiff"] = avg_final_score_diff
    metrics["AvgOffenseScoringPlaysPerGame"] = avg_offense_scoring_plays
    metrics["YardsPlausibility"] = yards_score
    metrics["DegeneracyPenalty"] = aggregate_degeneracy
    metrics["composite_score"] += yards_weight * yards_score

    hard_constraints = {
        "valid_probabilities": _check_probabilities(params),
        "nontrivial_clock": params.get("max_time_ticks", 0) >= 10,
    }
    constraint_penalty_weight = float(eval_cfg.get("constraint_penalty_weight", 0.1))
    constraint_penalty = constraint_penalty_weight * sum(0 if ok else 1 for ok in hard_constraints.values())
    metrics["composite_score"] -= constraint_penalty

    result = {
        "fitness": metrics["composite_score"],
        "metrics": metrics,
        "win_rate_mean": win_rate_mean,
        "win_rate_std_proxy": 1.0 - robust,
        "seed_win_rates": win_rates,
        "avg_yards_per_play_mean": avg_yards_per_play,
        "avg_plays_per_game_mean": avg_plays_per_game,
        "avg_final_score_diff_mean": avg_final_score_diff,
        "avg_offense_scoring_plays_per_game_mean": avg_offense_scoring_plays,
        "solver_meta": solver_meta,
        "constraints": hard_constraints,
        "penalties": {
            "constraint_penalty": constraint_penalty,
            "total_penalty": constraint_penalty,
        },
        "params": copy.deepcopy(params),
    }
    _append_jsonl(
        heartbeat_log_path,
        {
            "event": "candidate_eval_done",
            "eval_marker": eval_marker,
            "fitness": float(result["fitness"]),
            "win_rate_mean": float(result["win_rate_mean"]),
            "avg_yards_per_play_mean": float(result["avg_yards_per_play_mean"]),
            "avg_plays_per_game_mean": float(result["avg_plays_per_game_mean"]),
            "avg_final_score_diff_mean": float(result["avg_final_score_diff_mean"]),
        },
    )
    return result


def _check_probabilities(params: Dict) -> bool:
    for items in params.get("play_outcomes", {}).values():
        s = sum(float(x.get("prob", 0.0)) for x in items)
        if abs(s - 1.0) > 1e-6:
            return False
    return True
