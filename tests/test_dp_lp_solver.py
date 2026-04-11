import random
import unittest

from agents.dp_lp_solver import solve_equilibrium_policy, solve_zero_sum_lp
from env.quarter_strategy import QuarterStrategy


def _params():
    params = {
        "offense_plays": ["run", "short_pass", "deep_pass"],
        "defense_plays": ["base", "blitz", "prevent"],
        "max_time_ticks": 6,
        "max_score_diff": 14,
        "field_goal": {"enabled": True, "yardline_threshold": 35, "success_prob": 0.7, "miss_score_cost": 1},
        "punt": {"enabled": True, "yardline_threshold": 55, "net_yards": 35, "touchback_prob": 0.2, "pin_prob": 0.2, "pin_yardline": 95},
        "initial_state_distribution": [{"state": [65, 1, 10, 6, 0], "prob": 1.0}],
        "play_outcomes": {},
    }
    for op in params["offense_plays"]:
        for dp in params["defense_plays"]:
            params["play_outcomes"][f"{op}|{dp}"] = [
                {"yards": 2, "time": 2, "turnover": False, "prob": 0.5},
                {"yards": 8, "time": 2, "turnover": False, "prob": 0.3},
                {"yards": 0, "time": 2, "turnover": True, "prob": 0.2},
            ]
    return params


class DPLPSolverTests(unittest.TestCase):
    def test_lp_solver_valid_distributions(self):
        matrix = [
            [0.7, 0.2, 0.5],
            [0.4, 0.8, 0.3],
            [0.6, 0.1, 0.9],
        ]
        pi_off, pi_def, value = solve_zero_sum_lp(matrix)
        self.assertAlmostEqual(sum(pi_off), 1.0, places=6)
        self.assertAlmostEqual(sum(pi_def), 1.0, places=6)
        self.assertTrue(all(p >= 0.0 for p in pi_off))
        self.assertTrue(all(p >= 0.0 for p in pi_def))
        self.assertTrue(0.0 <= value <= 1.0)

    def test_solve_equilibrium_policy_smoke(self):
        model = QuarterStrategy(_params(), seed=5)
        solved = solve_equilibrium_policy(model, {"tol": 1e-9})

        self.assertIn("V", solved)
        self.assertIn("pi_off", solved)
        self.assertIn("pi_def", solved)
        self.assertIn("meta", solved)
        self.assertGreater(solved["meta"]["lp_solves"], 0)
        self.assertEqual(solved["meta"]["lp_failed"], 0)

    def test_solver_matches_known_values(self):
        matrices = [
            [[0.7, 0.2, 0.5], [0.4, 0.8, 0.3], [0.6, 0.1, 0.9]],
            [[0.0, 1.0, 0.5], [0.7, 0.7, 0.7], [1.0, 0.0, 0.2]],
            [[0.2, 0.2, 0.2], [0.8, 0.4, 0.6], [0.1, 0.9, 0.3]],
            [[1.0, 0.0], [0.0, 1.0]],  # matching pennies style with value 0.5
            [[0.3, 0.3], [0.3, 0.3]],  # constant matrix with value 0.3
        ]
        rng = random.Random(7)
        for _ in range(15):
            matrices.append([[rng.random() for _ in range(3)] for _ in range(3)])

        for matrix in matrices:
            _, _, value_custom = solve_zero_sum_lp(matrix, tol=1e-9)
            self.assertTrue(0.0 <= value_custom <= 1.0)

        _, _, value_identity = solve_zero_sum_lp([[1.0, 0.0], [0.0, 1.0]], tol=1e-9)
        self.assertAlmostEqual(value_identity, 0.5, places=6)
        _, _, value_constant = solve_zero_sum_lp([[0.3, 0.3], [0.3, 0.3]], tol=1e-9)
        self.assertAlmostEqual(value_constant, 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
