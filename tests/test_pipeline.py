import tempfile
import unittest

from agents.dp_lp_solver import solve_equilibrium_policy
from cli.run_experiment import run_experiment
from env.quarter_strategy import QuarterStrategy
from eval.evaluator import evaluate_candidate


def small_config():
    base_params = {
        "offense_plays": ["run", "short_pass", "deep_pass"],
        "defense_plays": ["base", "blitz", "prevent"],
        "max_time_ticks": 6,
        "max_score_diff": 14,
        "field_goal": {"enabled": True, "yardline_threshold": 35, "success_prob": 0.75, "miss_score_cost": 1},
        "punt": {
            "enabled": True,
            "yardline_threshold": 55,
            "net_yards": 40,
            "touchback_prob": 0.2,
            "pin_prob": 0.2,
            "pin_yardline": 95,
            "touchback_score_cost": 2,
            "pin_score_cost": 1,
            "standard_score_cost": 2,
        },
        "initial_state_distribution": [{"state": [70, 1, 10, 6, 0], "prob": 1.0}],
        "play_outcomes": {},
    }

    for op in base_params["offense_plays"]:
        for dp in base_params["defense_plays"]:
            base_params["play_outcomes"][f"{op}|{dp}"] = [
                {"yards": 2, "time": 2, "turnover": False, "prob": 0.35},
                {"yards": 7, "time": 2, "turnover": False, "prob": 0.25},
                {"yards": -3, "time": 3, "turnover": False, "prob": 0.15},
                {"yards": 0, "time": 2, "turnover": True, "prob": 0.15},
                {"yards": 12, "time": 3, "turnover": False, "prob": 0.10},
            ]

    return {
        "master_seed": 123,
        "search": {
            "seed": 123,
            "population": 6,
            "elites": 2,
            "tournament_size": 3,
            "crossover_rate": 0.7,
            "mutation_rate": 0.2,
            "generations": 2,
            "outcomes_per_combo": 5,
            "reeval_top_k": 1,
            "reeval_reps": 1,
            "reeval_seed_step": 100,
            "genome_space": {
                "prob_sigma": 0.03,
                "continuous": {"max_time_ticks": {"min": 4, "max": 8, "sigma": 1.0}},
                "discrete": {"turnover_on_downs": {"values": [True, False]}},
            },
            "base_params": base_params,
        },
        "eval": {
            "seeds": [3, 7],
            "sim_games": 40,
            "policy_probe_states": 100,
            "weights": [0.25, 0.25, 0.25, 0.2, 0.05],
            "solver": {"type": "dp_lp", "tol": 1e-9},
        },
    }


class PipelineTests(unittest.TestCase):
    def test_dp_solver_policy_shape(self):
        cfg = small_config()
        model = QuarterStrategy(cfg["search"]["base_params"], seed=5)
        solved = solve_equilibrium_policy(model, cfg["eval"]["solver"])
        self.assertIn("pi_off", solved)
        self.assertIn("pi_def", solved)

    def test_evaluator_uses_solver(self):
        cfg = small_config()
        out = evaluate_candidate(cfg["search"]["base_params"], cfg["eval"])
        self.assertIn("solver_meta", out)
        self.assertGreater(len(out["solver_meta"]), 0)

    def test_end_to_end_smoke(self):
        cfg = small_config()
        with tempfile.TemporaryDirectory() as d:
            out = run_experiment(cfg, d)
            self.assertIn("best_candidate", out)


if __name__ == "__main__":
    unittest.main()
