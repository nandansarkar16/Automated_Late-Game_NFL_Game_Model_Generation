import unittest

from env.quarter_strategy import QuarterStrategy
from search.genetic_search import _renormalize_probs


def _base_params():
    params = {
        "offense_plays": ["run", "short_pass", "deep_pass"],
        "defense_plays": ["base", "blitz", "prevent"],
        "max_time_ticks": 12,
        "max_score_diff": 21,
        "field_goal": {"enabled": True, "yardline_threshold": 35, "success_prob": 0.75, "miss_score_cost": 1},
        "punt": {
            "enabled": True,
            "yardline_threshold": 55,
            "net_yards": 40,
            "touchback_prob": 0.2,
            "pin_prob": 0.3,
            "pin_yardline": 95,
            "touchback_score_cost": 2,
            "pin_score_cost": 1,
            "standard_score_cost": 2,
        },
        "initial_state_distribution": [{"state": [70, 1, 10, 12, 0], "prob": 1.0}],
        "opponent_response": {
            "clock_ticks": [2],
            "clock_probs": [1.0],
            "deep_own_territory": [1.0, 0.0, 0.0],
            "normal": [1.0, 0.0, 0.0],
            "short_field": [1.0, 0.0, 0.0],
            "score_values": [0, 3, 7],
        },
        "play_outcomes": {},
    }
    for op in params["offense_plays"]:
        for dp in params["defense_plays"]:
            params["play_outcomes"][f"{op}|{dp}"] = [
                {"yards": 0, "time": 2, "turnover": False, "prob": 1.0},
            ]
    return params


class QuarterStrategyTests(unittest.TestCase):
    def test_terminal_value(self):
        model = QuarterStrategy(_base_params(), seed=1)
        self.assertEqual(model.terminal_value((70, 1, 10, 0, 3)), 1.0)
        self.assertEqual(model.terminal_value((70, 1, 10, 0, 0)), 0.0)

    def test_payoff_matrix_entry_matches_transition_expectation(self):
        model = QuarterStrategy(_base_params(), seed=2)
        state = (70, 1, 10, 12, 0)

        matrix = model.build_payoff_matrix(state, lambda s: 1.0 if s[4] > 0 else 0.0)
        transitions = model.action_pair_transitions(state, 0, 0)
        expected = sum(prob * (1.0 if nxt[4] > 0 else 0.0) for prob, nxt in transitions)

        self.assertAlmostEqual(matrix[0][0], expected, places=9)

    def test_punt_branch_probabilities(self):
        params = _base_params()
        model = QuarterStrategy(params, seed=3)
        state = (70, 4, 10, 10, 0)
        transitions = model.action_pair_transitions(state, 0, 0)

        self.assertAlmostEqual(sum(p for p, _ in transitions), 1.0, places=9)
        yardlines = sorted({s[0] for _, s in transitions})
        self.assertIn(80, yardlines)
        self.assertIn(95, yardlines)

    def test_touchdown_uses_opponent_response_clock(self):
        params = _base_params()
        params["play_outcomes"]["run|base"] = [
            {"yards": 80, "time": 2, "turnover": False, "prob": 1.0},
        ]
        params["opponent_response"]["normal"] = [1.0, 0.0, 0.0]
        params["opponent_response"]["clock_ticks"] = [4]
        params["opponent_response"]["clock_probs"] = [1.0]
        model = QuarterStrategy(params, seed=4)

        transitions = model.action_pair_transitions((70, 1, 10, 12, 0), 0, 0)
        self.assertEqual(len(transitions), 1)
        prob, nxt = transitions[0]
        self.assertAlmostEqual(prob, 1.0, places=9)
        self.assertEqual(nxt, (75, 1, 10, 6, 7))

    def test_turnover_short_field_response_applies_opponent_points(self):
        params = _base_params()
        params["play_outcomes"]["run|base"] = [
            {"yards": 0, "time": 2, "turnover": True, "prob": 1.0},
        ]
        params["turnover_score_cost"] = 3
        params["opponent_response"]["short_field"] = [0.0, 0.0, 1.0]
        params["opponent_response"]["clock_ticks"] = [2]
        params["opponent_response"]["clock_probs"] = [1.0]
        model = QuarterStrategy(params, seed=5)

        transitions = model.action_pair_transitions((70, 1, 10, 12, 0), 0, 0)
        self.assertEqual(len(transitions), 1)
        prob, nxt = transitions[0]
        self.assertAlmostEqual(prob, 1.0, places=9)
        self.assertEqual(nxt, (75, 1, 10, 8, -10))

    def test_probability_renormalize(self):
        params = _base_params()
        params["play_outcomes"]["run|base"] = [
            {"yards": 0, "time": 2, "turnover": False, "prob": -0.3},
            {"yards": 4, "time": 2, "turnover": False, "prob": 0.1},
            {"yards": 0, "time": 2, "turnover": True, "prob": 0.2},
        ]
        _renormalize_probs(params)
        s = sum(x["prob"] for x in params["play_outcomes"]["run|base"])
        self.assertAlmostEqual(s, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
