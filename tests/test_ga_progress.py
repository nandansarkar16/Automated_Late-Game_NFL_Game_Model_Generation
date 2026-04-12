import tempfile
import unittest

from cli.run_experiment import run_experiment


def medium_config():
    base_params = {
        "offense_plays": ["run", "short_pass", "deep_pass"],
        "defense_plays": ["base", "blitz", "prevent"],
        "max_time_ticks": 8,
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
        "initial_state_distribution": [
            {"state": [80, 1, 10, 8, 0], "prob": 0.5},
            {"state": [45, 2, 6, 6, -3], "prob": 0.5},
        ],
        "play_outcomes": {},
    }

    for op in base_params["offense_plays"]:
        for dp in base_params["defense_plays"]:
            base_params["play_outcomes"][f"{op}|{dp}"] = [
                {"yards": -2, "time": 3, "turnover": False, "prob": 0.18},
                {"yards": 3, "time": 2, "turnover": False, "prob": 0.34},
                {"yards": 8, "time": 2, "turnover": False, "prob": 0.24},
                {"yards": 0, "time": 2, "turnover": True, "prob": 0.14},
                {"yards": 6, "time": 3, "turnover": False, "prob": 0.10},
            ]

    return {
        "master_seed": 321,
        "search": {
            "seed": 321,
            "population": 8,
            "elites": 2,
            "tournament_size": 3,
            "crossover_rate": 0.7,
            "mutation_rate": 0.2,
            "generations": 4,
            "outcomes_per_combo": 5,
            "vector_gene": {
                "yards_min": -20,
                "yards_max": 60,
                "time_min": 1,
                "time_max": 8,
                "turnover_min": 0.0,
                "turnover_max": 1.0,
                "turnover_threshold": 0.5,
                "init_sigma": 2.0,
            },
            "base_params": base_params,
        },
        "eval": {
            "seeds": [11, 19],
            "sim_games": 30,
            "policy_probe_states": 80,
            "weights": [0.25, 0.25, 0.25, 0.2, 0.05],
            "solver": {"type": "dp_lp", "tol": 1e-9},
        },
    }


class GAProgressTests(unittest.TestCase):
    def test_ga_preserves_fixed_probs_and_bounds(self):
        cfg = medium_config()
        fixed_probs = {
            combo: [float(x["prob"]) for x in items]
            for combo, items in cfg["search"]["base_params"]["play_outcomes"].items()
        }

        with tempfile.TemporaryDirectory() as out:
            run_experiment(cfg, out)
            import json
            import os

            with open(os.path.join(out, "best_candidate.json"), "r", encoding="utf-8") as f:
                best = json.load(f)
            params = best["params"]

            for combo, items in params["play_outcomes"].items():
                self.assertEqual(len(items), 5)
                probs = [float(x["prob"]) for x in items]
                self.assertAlmostEqual(sum(probs), 1.0, places=7)
                for p_new, p_fixed in zip(probs, fixed_probs[combo]):
                    self.assertAlmostEqual(p_new, p_fixed, places=7)

                for item in items:
                    self.assertTrue(-20 <= int(item["yards"]) <= 60)
                    self.assertTrue(1 <= int(item["time"]) <= 8)
                    self.assertIn(bool(item["turnover"]), [True, False])

    def test_ga_history_shows_search_movement(self):
        cfg = medium_config()
        with tempfile.TemporaryDirectory() as out:
            run_experiment(cfg, out)
            import csv
            import os

            with open(os.path.join(out, "history.csv")) as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), cfg["search"]["generations"])
            best_vals = [float(r["best_fitness"]) for r in rows]
            mean_vals = [float(r["mean_fitness"]) for r in rows]

            self.assertGreaterEqual(max(best_vals), best_vals[0])
            moved = len({round(x, 8) for x in best_vals}) > 1 or len({round(x, 8) for x in mean_vals}) > 1
            self.assertTrue(moved)

    def test_ga_can_resume_from_checkpoint(self):
        cfg = medium_config()
        cfg["search"]["population"] = 4
        cfg["search"]["elites"] = 1
        cfg["search"]["generations"] = 1
        cfg["eval"]["seeds"] = [11]
        cfg["eval"]["sim_games"] = 10
        cfg["eval"]["policy_probe_states"] = 20

        with tempfile.TemporaryDirectory() as out:
            run_experiment(cfg, out)

            cfg["search"]["generations"] = 2
            run_experiment(cfg, out)

            import csv
            import os

            with open(os.path.join(out, "history.csv")) as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 2)
            self.assertEqual([int(r["generation"]) for r in rows], [0, 1])


if __name__ == "__main__":
    unittest.main()
