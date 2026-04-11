# GA outer loop: mutate tuple genes, evaluate, and log progress.
import copy
import json
import random
import time
from typing import Dict, List, Sequence, Tuple

import pygad

from eval.evaluator import evaluate_candidate


def _renormalize_probs(params: Dict):
    for items in params["play_outcomes"].values():
        total = sum(max(1e-5, float(x["prob"])) for x in items)
        for x in items:
            x["prob"] = max(1e-5, float(x["prob"])) / total


def _combo_keys(base_params: Dict) -> List[str]:
    return [
        f"{off}|{def_}"
        for off in base_params["offense_plays"]
        for def_ in base_params["defense_plays"]
    ]


def _extract_fixed_probs(base_params: Dict, combos: Sequence[str], outcomes_per_combo: int) -> Dict[str, List[float]]:
    fixed: Dict[str, List[float]] = {}
    for combo in combos:
        items = base_params["play_outcomes"][combo]
        if len(items) != outcomes_per_combo:
            raise ValueError(
                f"{combo} has {len(items)} outcomes but expected {outcomes_per_combo}; "
                "update config play_outcomes to fixed 5 outcomes per matchup"
            )
        probs = [float(x["prob"]) for x in items]
        total = sum(probs)
        if total <= 0:
            raise ValueError(f"invalid fixed probabilities for {combo}")
        fixed[combo] = [p / total for p in probs]
    return fixed


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _decode_solution_to_params(
    solution: Sequence[float],
    base_params: Dict,
    combos: Sequence[str],
    outcomes_per_combo: int,
    fixed_probs: Dict[str, List[float]],
    vector_cfg: Dict,
) -> Dict:
    # Decode one 135-gene vector into structured play outcome tables.
    params = copy.deepcopy(base_params)

    yards_min = float(vector_cfg.get("yards_min", -20))
    yards_max = float(vector_cfg.get("yards_max", 60))
    time_min = float(vector_cfg.get("time_min", 1))
    time_max = float(vector_cfg.get("time_max", 8))
    turnover_min = float(vector_cfg.get("turnover_min", 0.0))
    turnover_max = float(vector_cfg.get("turnover_max", 1.0))
    turnover_threshold = float(vector_cfg.get("turnover_threshold", 0.5))

    idx = 0
    for combo in combos:
        decoded = []
        for outcome_idx in range(outcomes_per_combo):
            yards_gene = solution[idx]
            time_gene = solution[idx + 1]
            turnover_gene = solution[idx + 2]
            idx += 3

            yards = int(round(_clamp(yards_gene, yards_min, yards_max)))
            elapsed = int(round(_clamp(time_gene, time_min, time_max)))
            elapsed = max(1, elapsed)
            turnover_proxy = _clamp(turnover_gene, turnover_min, turnover_max)
            turnover = bool(turnover_proxy >= turnover_threshold)

            decoded.append(
                {
                    "yards": yards,
                    "time": elapsed,
                    "turnover": turnover,
                    "prob": fixed_probs[combo][outcome_idx],
                }
            )
        params["play_outcomes"][combo] = decoded

    return params


def _solution_key(solution: Sequence[float]) -> Tuple[float, ...]:
    return tuple(round(float(x), 6) for x in solution)


def _build_gene_space(outcomes_total: int, vector_cfg: Dict) -> List[Dict[str, float]]:
    yards_min = float(vector_cfg.get("yards_min", -20))
    yards_max = float(vector_cfg.get("yards_max", 60))
    time_min = float(vector_cfg.get("time_min", 1))
    time_max = float(vector_cfg.get("time_max", 8))
    turnover_min = float(vector_cfg.get("turnover_min", 0.0))
    turnover_max = float(vector_cfg.get("turnover_max", 1.0))

    gene_space: List[Dict[str, float]] = []
    for _ in range(outcomes_total):
        gene_space.append({"low": yards_min, "high": yards_max})
        gene_space.append({"low": time_min, "high": time_max})
        gene_space.append({"low": turnover_min, "high": turnover_max})
    return gene_space


def _base_solution_from_params(base_params: Dict, combos: Sequence[str], outcomes_per_combo: int) -> List[float]:
    genes: List[float] = []
    for combo in combos:
        items = base_params["play_outcomes"][combo]
        for i in range(outcomes_per_combo):
            item = items[i]
            genes.append(float(item["yards"]))
            genes.append(float(max(1, int(item["time"]))))
            genes.append(1.0 if bool(item.get("turnover", False)) else 0.0)
    return genes


def _tuple_change_summary(reference_params: Dict, current_params: Dict, combos: Sequence[str], outcomes_per_combo: int) -> Dict:
    combo_changes: Dict[str, Dict[str, int]] = {}
    changed_tuples: List[Dict] = []
    turnover_flips = 0

    for combo in combos:
        ref_items = reference_params["play_outcomes"][combo]
        cur_items = current_params["play_outcomes"][combo]
        changed_count = 0
        yards_delta_sum = 0
        time_delta_sum = 0
        combo_turnover_flips = 0

        for idx in range(outcomes_per_combo):
            ref = ref_items[idx]
            cur = cur_items[idx]
            old_tuple = (int(ref["yards"]), int(ref["time"]), bool(ref["turnover"]))
            new_tuple = (int(cur["yards"]), int(cur["time"]), bool(cur["turnover"]))
            if old_tuple == new_tuple:
                continue

            changed_count += 1
            yards_delta_sum += abs(new_tuple[0] - old_tuple[0])
            time_delta_sum += abs(new_tuple[1] - old_tuple[1])
            if old_tuple[2] != new_tuple[2]:
                combo_turnover_flips += 1
                turnover_flips += 1
            changed_tuples.append(
                {
                    "combo": combo,
                    "outcome_index": idx,
                    "from": {"yards": old_tuple[0], "time": old_tuple[1], "turnover": old_tuple[2]},
                    "to": {"yards": new_tuple[0], "time": new_tuple[1], "turnover": new_tuple[2]},
                }
            )

        combo_changes[combo] = {
            "changed_outcomes": changed_count,
            "yards_abs_delta_sum": yards_delta_sum,
            "time_abs_delta_sum": time_delta_sum,
            "turnover_flips": combo_turnover_flips,
        }

    return {
        "changed_outcomes_total": len(changed_tuples),
        "turnover_flips_total": turnover_flips,
        "combo_changes": combo_changes,
        "changed_tuples": changed_tuples,
    }


def genetic_search(search_cfg: Dict, eval_cfg: Dict):
    rng = random.Random(search_cfg.get("seed", 12345))

    pop_size = int(search_cfg.get("population", 48))
    generations = int(search_cfg.get("generations", 40))
    crossover_rate = float(search_cfg.get("crossover_rate", 0.7))
    mutation_rate = float(search_cfg.get("mutation_rate", 0.2))
    tournament = int(search_cfg.get("tournament_size", 3))
    elites = int(search_cfg.get("elites", 6))

    base_params = copy.deepcopy(search_cfg["base_params"])
    vector_cfg = search_cfg.get("vector_gene", {})

    outcomes_per_combo = int(search_cfg.get("outcomes_per_combo", 5))
    combos = _combo_keys(base_params)
    fixed_probs = _extract_fixed_probs(base_params, combos, outcomes_per_combo)

    outcomes_total = len(combos) * outcomes_per_combo
    num_genes = outcomes_total * 3
    gene_space = _build_gene_space(outcomes_total, vector_cfg)
    base_solution = _base_solution_from_params(base_params, combos, outcomes_per_combo)

    cache: Dict[Tuple[float, ...], Dict] = {}
    history: List[Dict] = []
    last_best_params = copy.deepcopy(base_params)
    log_generations = bool(search_cfg.get("log_generations", True))
    generation_log_path = search_cfg.get("generation_log_path")
    heartbeat_log_path = search_cfg.get("heartbeat_log_path")
    heartbeat_every = max(1, int(search_cfg.get("heartbeat_every", 1)))
    eval_counter = 0
    cache_hit_counter = 0
    cache_miss_counter = 0
    best_seen_fitness = float("-inf")
    run_start = time.time()

    def summarize_solver_meta(entries: List[Dict]) -> Dict:
        if not entries:
            return {}
        elapsed = [float(x.get("elapsed_sec", 0.0)) for x in entries]
        lp_solves = [int(x.get("lp_solves", 0)) for x in entries]
        lp_failed = [int(x.get("lp_failed", 0)) for x in entries]
        reachable = [int(x.get("reachable_states", 0)) for x in entries]
        n = float(len(entries))
        return {
            "seed_count": int(len(entries)),
            "elapsed_sec_mean": sum(elapsed) / n,
            "elapsed_sec_max": max(elapsed),
            "lp_solves_mean": sum(lp_solves) / n,
            "lp_failed_total": sum(lp_failed),
            "reachable_states_mean": sum(reachable) / n,
        }

    def fitness_func(ga_instance, solution, solution_idx):
        nonlocal eval_counter, cache_hit_counter, cache_miss_counter, best_seen_fitness
        # Cache expensive evaluations so repeated solutions are cheap.
        key = _solution_key(solution)
        eval_counter += 1
        generation_idx = int(max(0, ga_instance.generations_completed))
        if key not in cache:
            t0 = time.time()
            params = _decode_solution_to_params(
                solution,
                base_params,
                combos,
                outcomes_per_combo,
                fixed_probs,
                vector_cfg,
            )
            cache[key] = evaluate_candidate(params, eval_cfg)
            eval_sec = time.time() - t0
            cache_miss_counter += 1
            fit = float(cache[key]["fitness"])
            if fit > best_seen_fitness:
                best_seen_fitness = fit
            if heartbeat_log_path and (eval_counter % heartbeat_every == 0):
                row = {
                    "event": "candidate_eval",
                    "eval_index": eval_counter,
                    "generation_in_progress": generation_idx,
                    "solution_idx": int(solution_idx),
                    "fitness": fit,
                    "best_seen_fitness": best_seen_fitness,
                    "cache_hits": cache_hit_counter,
                    "cache_misses": cache_miss_counter,
                    "cache_size": len(cache),
                    "candidate_eval_sec": eval_sec,
                    "elapsed_sec": time.time() - run_start,
                }
                with open(heartbeat_log_path, "a", encoding="utf-8") as hlog:
                    hlog.write(json.dumps(row, sort_keys=True) + "\n")
        else:
            cache_hit_counter += 1
            fit = float(cache[key]["fitness"])
            if fit > best_seen_fitness:
                best_seen_fitness = fit
            if heartbeat_log_path and (eval_counter % heartbeat_every == 0):
                row = {
                    "event": "cache_hit",
                    "eval_index": eval_counter,
                    "generation_in_progress": generation_idx,
                    "solution_idx": int(solution_idx),
                    "fitness": fit,
                    "best_seen_fitness": best_seen_fitness,
                    "cache_hits": cache_hit_counter,
                    "cache_misses": cache_miss_counter,
                    "cache_size": len(cache),
                    "elapsed_sec": time.time() - run_start,
                }
                with open(heartbeat_log_path, "a", encoding="utf-8") as hlog:
                    hlog.write(json.dumps(row, sort_keys=True) + "\n")
        return fit

    def on_generation(ga_instance):
        nonlocal last_best_params
        fits = ga_instance.last_generation_fitness
        if fits is None or len(fits) == 0:
            return
        best_idx = max(range(len(fits)), key=lambda i: fits[i])
        best_solution = ga_instance.population[best_idx]
        best_eval = cache[_solution_key(best_solution)]
        best_params = best_eval["params"]
        delta_from_prev = _tuple_change_summary(last_best_params, best_params, combos, outcomes_per_combo)
        delta_from_base = _tuple_change_summary(base_params, best_params, combos, outcomes_per_combo)
        fit_list = [float(x) for x in fits]
        fit_sorted = sorted(fit_list)
        fit_mean = float(sum(fit_list) / len(fit_list))
        fit_min = float(fit_sorted[0])
        fit_max = float(fit_sorted[-1])
        fit_median = float(fit_sorted[len(fit_sorted) // 2])
        fit_var = sum((x - fit_mean) ** 2 for x in fit_list) / len(fit_list)
        fit_std = fit_var ** 0.5
        history.append(
            {
                "generation": int(ga_instance.generations_completed - 1),
                "best_fitness": float(best_eval["fitness"]),
                "mean_fitness": fit_mean,
                "min_fitness": fit_min,
                "max_fitness": fit_max,
                "median_fitness": fit_median,
                "std_fitness": fit_std,
                "best_win_rate_mean": float(best_eval.get("win_rate_mean", 0.0)),
                "best_seed_win_rates": copy.deepcopy(best_eval.get("seed_win_rates", [])),
                "best_avg_yards_per_play": float(best_eval.get("avg_yards_per_play_mean", 0.0)),
                "best_avg_plays_per_game": float(best_eval.get("avg_plays_per_game_mean", 0.0)),
                "best_avg_final_score_diff": float(best_eval.get("avg_final_score_diff_mean", 0.0)),
                "best_avg_offense_scoring_plays_per_game": float(
                    best_eval.get("avg_offense_scoring_plays_per_game_mean", 0.0)
                ),
                "best_constraints": copy.deepcopy(best_eval.get("constraints", {})),
                "best_penalties": copy.deepcopy(best_eval.get("penalties", {})),
                "best_solver_summary": summarize_solver_meta(best_eval.get("solver_meta", [])),
                "best_metrics": copy.deepcopy(best_eval["metrics"]),
                "best_params": copy.deepcopy(best_params),
                "tuple_delta_from_prev_best": delta_from_prev,
                "tuple_delta_from_base": delta_from_base,
            }
        )
        if generation_log_path:
            with open(generation_log_path, "a", encoding="utf-8") as logf:
                logf.write(json.dumps(history[-1], sort_keys=True) + "\n")
        if log_generations:
            g = int(ga_instance.generations_completed - 1)
            changed = delta_from_prev["changed_outcomes_total"]
            print(
                f"[gen {g}] best={best_eval['fitness']:.6f} mean={fit_mean:.6f} "
                f"wr={best_eval.get('win_rate_mean', 0.0):.4f} "
                f"plays={best_eval.get('avg_plays_per_game_mean', 0.0):.2f} "
                f"changed_tuples_vs_prev={changed}",
                flush=True,
            )
        last_best_params = copy.deepcopy(best_params)

    initial_population = []
    initial_population.append(base_solution)
    noise_sigma = float(vector_cfg.get("init_sigma", 2.0))
    for _ in range(pop_size - 1):
        v = [g + rng.gauss(0.0, noise_sigma) for g in base_solution]
        initial_population.append(v)

    ga = pygad.GA(
        random_seed=int(search_cfg.get("seed", 12345)),
        num_generations=generations,
        sol_per_pop=pop_size,
        num_parents_mating=max(2, pop_size // 2),
        num_genes=num_genes,
        initial_population=initial_population,
        gene_space=gene_space,
        gene_type=float,
        parent_selection_type="tournament",
        K_tournament=tournament,
        keep_elitism=min(elites, pop_size),
        crossover_type="single_point",
        crossover_probability=crossover_rate,
        mutation_type="random",
        mutation_probability=mutation_rate,
        fitness_func=fitness_func,
        on_generation=on_generation,
        suppress_warnings=True,
    )
    ga.run()

    best_solution, best_fitness, _ = ga.best_solution()
    best_eval = cache[_solution_key(best_solution)]

    # Ensure fixed probabilities remain normalized in final emitted params.
    best_params = copy.deepcopy(best_eval["params"])
    _renormalize_probs(best_params)

    return best_params, history
