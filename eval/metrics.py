from __future__ import annotations

# Metric helpers used to score each candidate game model.
import math
import statistics
from typing import Dict, Iterable, List, Tuple


def entropy(probs: Iterable[float]) -> float:
    e = 0.0
    for p in probs:
        if p > 1e-12:
            e -= p * math.log(p + 1e-12)
    return e


def normalize_entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts]
    max_e = math.log(len(counts)) if counts else 1.0
    if max_e <= 0:
        return 0.0
    return entropy(probs) / max_e


def strategic_diversity(action_hist: Dict[Tuple, List[int]]) -> float:
    if not action_hist:
        return 0.0
    vals = [normalize_entropy(counts) for counts in action_hist.values()]
    return sum(vals) / len(vals)


def state_sensitivity(
    regime_action_rates: Dict[str, List[float]],
    regime_totals: Dict[str, float] | None = None,
    min_total: float = 1e-9,
) -> float:
    # Reward measurable strategy shifts between regimes, but only when
    # we have actual support in those regimes.
    if regime_totals is None:
        keys = [k for k, rates in regime_action_rates.items() if sum(rates) > min_total]
    else:
        keys = [k for k in regime_action_rates.keys() if float(regime_totals.get(k, 0.0)) > min_total]
    if len(keys) < 2:
        return 0.0
    distances = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            p = regime_action_rates[keys[i]]
            q = regime_action_rates[keys[j]]
            distances.append(sum(abs(a - b) for a, b in zip(p, q)) / 2.0)
    return sum(distances) / len(distances) if distances else 0.0


def _range_score(value: float, low: float, high: float, slack: float = 0.0) -> float:
    if low <= value <= high:
        return 1.0
    if slack <= 1e-12:
        return 0.0
    if value < low:
        return max(0.0, 1.0 - (low - value) / slack)
    return max(0.0, 1.0 - (value - high) / slack)


def outcome_plausibility(candidate_params: Dict, cfg: Dict | None = None) -> float:
    cfg = cfg or {}
    yard_low = float(cfg.get("expected_yards_min", -3.0))
    yard_high = float(cfg.get("expected_yards_max", 12.0))
    yard_slack = float(cfg.get("expected_yards_slack", 8.0))
    time_low = float(cfg.get("expected_time_min", 1.5))
    time_high = float(cfg.get("expected_time_max", 5.5))
    time_slack = float(cfg.get("expected_time_slack", 2.0))
    to_low = float(cfg.get("expected_turnover_min", 0.01))
    to_high = float(cfg.get("expected_turnover_max", 0.35))
    to_slack = float(cfg.get("expected_turnover_slack", 0.2))

    scores = []
    for items in candidate_params["play_outcomes"].values():
        probs = [max(0.0, float(item.get("prob", 0.0))) for item in items]
        total = sum(probs)
        if total <= 1e-12:
            continue
        probs = [p / total for p in probs]
        exp_yards = sum(p * float(item["yards"]) for p, item in zip(probs, items))
        exp_time = sum(p * float(item["time"]) for p, item in zip(probs, items))
        exp_turnover = sum(p for p, item in zip(probs, items) if bool(item.get("turnover", False)))
        combo_score = (
            _range_score(exp_yards, yard_low, yard_high, yard_slack)
            + _range_score(exp_time, time_low, time_high, time_slack)
            + _range_score(exp_turnover, to_low, to_high, to_slack)
        ) / 3.0
        scores.append(combo_score)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def robustness(seed_scores: List[float]) -> float:
    if not seed_scores:
        return 0.0
    if len(seed_scores) == 1:
        return 1.0
    stdev = statistics.pstdev(seed_scores)
    return 1.0 / (1.0 + 3.0 * stdev)


def degeneracy_penalty(global_action_counts: List[int]) -> float:
    total = sum(global_action_counts)
    if total == 0:
        return 1.0
    shares = [c / total for c in global_action_counts]
    dominant = max(shares)
    return max(0.0, (dominant - 0.65) / 0.35)


def composite_score(
    strategic_diversity_value: float,
    state_sensitivity_value: float,
    outcome_plausibility_value: float,
    robustness_value: float,
    non_degeneracy_value: float,
    weights: List[float] | Tuple[float, float, float, float, float],
) -> Dict:
    w1, w2, w3, w4, w5 = weights
    score = (
        w1 * strategic_diversity_value
        + w2 * state_sensitivity_value
        + w3 * outcome_plausibility_value
        + w4 * robustness_value
        + w5 * non_degeneracy_value
    )
    return {
        "composite_score": score,
        "StrategicDiversity": strategic_diversity_value,
        "StateSensitivity": state_sensitivity_value,
        "OutcomePlausibility": outcome_plausibility_value,
        "Robustness": robustness_value,
        "NonDegeneracy": non_degeneracy_value,
    }
