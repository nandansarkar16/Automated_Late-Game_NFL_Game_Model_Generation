from __future__ import annotations

# Main game environment used by the solver and simulator.
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

State = Tuple[int, int, int, int, int]
Outcome = Tuple[int, int, bool]


@dataclass(frozen=True)
class OutcomeSupport:
    outcomes: Sequence[Outcome]
    probs: Sequence[float]


class QuarterStrategy:
    """Fourth-quarter abstraction with action-pair transition kernels for DP/LP."""

    def __init__(self, params: Dict, seed: int | None = None):
        self.params = params
        self.rng = random.Random(seed)

        self.offense_plays: List[str] = params.get("offense_plays", ["run", "short_pass", "deep_pass"])
        self.defense_plays: List[str] = params.get("defense_plays", ["base", "blitz", "prevent"])
        self.play_support = self._build_outcome_support(params["play_outcomes"])

        self.max_ticks = int(params.get("max_time_ticks", 30))
        self.max_score_diff = int(params.get("max_score_diff", 21))

        self.field_goal = params.get(
            "field_goal",
            {
                "enabled": True,
                "yardline_threshold": 35,
                "success_prob": 0.78,
                "miss_score_cost": 1,
            },
        )
        self.punt = params.get(
            "punt",
            {
                "enabled": True,
                "yardline_threshold": 55,
                "net_yards": 40,
                "touchback_prob": 0.25,
                "pin_prob": 0.2,
                "pin_yardline": 95,
                "touchback_score_cost": 2,
                "pin_score_cost": 1,
                "standard_score_cost": 2,
            },
        )

        self.turnover_score_cost = int(params.get("turnover_score_cost", 3))
        self.turnover_on_downs_score_cost = int(params.get("turnover_on_downs_score_cost", 2))
        self.opponent_response = params.get(
            "opponent_response",
            {
                "clock_ticks": [2, 4, 6],
                "clock_probs": [0.4, 0.4, 0.2],
                "deep_own_territory": [0.68, 0.18, 0.14],
                "normal": [0.5, 0.25, 0.25],
                "short_field": [0.25, 0.25, 0.5],
                "score_values": [0, 3, 7],
            },
        )

        self.init_distribution = params.get(
            "initial_state_distribution",
            [{"state": [75, 1, 10, self.max_ticks, 0], "prob": 1.0}],
        )

    def _build_outcome_support(self, plays_cfg: Dict) -> Dict[Tuple[int, int], OutcomeSupport]:
        support: Dict[Tuple[int, int], OutcomeSupport] = {}
        for oi, off_name in enumerate(self.offense_plays):
            for di, def_name in enumerate(self.defense_plays):
                key = f"{off_name}|{def_name}"
                items = plays_cfg[key]
                outcomes: List[Outcome] = []
                probs: List[float] = []
                total = 0.0
                for item in items:
                    elapsed = max(1, int(item["time"]))
                    outcomes.append((int(item["yards"]), elapsed, bool(item.get("turnover", False))))
                    p = float(item["prob"])
                    probs.append(p)
                    total += p
                if total <= 0.0:
                    raise ValueError(f"invalid distribution for {key}")
                probs = [p / total for p in probs]
                support[(oi, di)] = OutcomeSupport(tuple(outcomes), tuple(probs))
        return support

    def offensive_playbook_size(self) -> int:
        return len(self.offense_plays)

    def defensive_playbook_size(self) -> int:
        return len(self.defense_plays)

    def initial_position(self) -> State:
        sample = self._sample_weighted(self.init_distribution, key="prob")
        s = sample["state"]
        return self._normalize_state((int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4])))

    def is_terminal(self, state: State) -> bool:
        return state[3] <= 0

    def terminal_value(self, state: State) -> float:
        return 1.0 if state[4] > 0 else 0.0

    def game_over(self, state: State) -> bool:
        return self.is_terminal(state)

    def win(self, state: State) -> bool:
        return self.terminal_value(state) == 1.0

    def action_pair_transitions(self, state: State, off_action: int, def_action: int) -> List[Tuple[float, State]]:
        if self.is_terminal(state):
            return [(1.0, state)]
        if off_action < 0 or off_action >= self.offensive_playbook_size():
            raise ValueError(f"invalid offensive action {off_action}")
        if def_action < 0 or def_action >= self.defensive_playbook_size():
            raise ValueError(f"invalid defensive action {def_action}")

        yardline, down, distance, ticks, score_diff = state
        support = self.play_support[(off_action, def_action)]

        transitions: List[Tuple[float, State]] = []
        for p, outcome in zip(support.probs, support.outcomes):
            yards, elapsed, turnover = outcome
            next_ticks = max(0, ticks - elapsed)

            if turnover:
                transitions.extend(
                    self._post_change_transitions(
                        next_ticks=next_ticks,
                        score_diff=score_diff,
                        offense_points=0,
                        possession_cost=self.turnover_score_cost,
                        regime="short_field",
                        reset_yardline=75,
                        base_prob=p,
                    )
                )
                continue

            next_yardline = yardline - yards
            next_down = down + 1
            next_distance = distance - yards
            next_score_diff = score_diff

            if next_yardline <= 0:
                transitions.extend(
                    self._post_change_transitions(
                        next_ticks=next_ticks,
                        score_diff=next_score_diff,
                        offense_points=7,
                        possession_cost=0,
                        regime="normal",
                        reset_yardline=75,
                        base_prob=p,
                    )
                )
                continue

            if next_distance <= 0:
                next_down = 1
                next_distance = min(10, next_yardline)
                nxt = self._normalize_state((next_yardline, next_down, next_distance, next_ticks, next_score_diff))
                transitions.append((p, nxt))
                continue

            if next_down > 4:
                # On failed 4th down, branch into FG/punt/turnover outcomes.
                transitions.extend(
                    self._failed_fourth_transitions(next_yardline, next_ticks, next_score_diff, base_prob=p)
                )
                continue

            nxt = self._normalize_state((next_yardline, next_down, next_distance, next_ticks, next_score_diff))
            transitions.append((p, nxt))

        return self._merge_transitions(transitions)

    def build_payoff_matrix(self, state: State, value_fn: Callable[[State], float]) -> List[List[float]]:
        matrix: List[List[float]] = []
        for off_action in range(self.offensive_playbook_size()):
            row = []
            for def_action in range(self.defensive_playbook_size()):
                val = 0.0
                for prob, nxt in self.action_pair_transitions(state, off_action, def_action):
                    val += prob * float(value_fn(nxt))
                row.append(val)
            matrix.append(row)
        return matrix

    def simulate_profile(self, pi_off: Dict[State, Sequence[float]], pi_def: Dict[State, Sequence[float]], n: int) -> float:
        stats = self.simulate_profile_stats(pi_off, pi_def, n)
        return float(stats["win_rate"])

    def simulate_profile_stats(self, pi_off: Dict[State, Sequence[float]], pi_def: Dict[State, Sequence[float]], n: int) -> Dict[str, float]:
        wins = 0
        total_plays = 0
        total_expected_yards = 0.0
        total_final_score_diff = 0.0
        total_offense_scoring_plays = 0
        for _ in range(n):
            state = self.initial_position()
            while not self.is_terminal(state):
                off_probs = pi_off.get(state)
                def_probs = pi_def.get(state)
                if off_probs is None:
                    off_probs = [1.0 / self.offensive_playbook_size()] * self.offensive_playbook_size()
                if def_probs is None:
                    def_probs = [1.0 / self.defensive_playbook_size()] * self.defensive_playbook_size()
                off_action = self._sample_index(off_probs)
                def_action = self._sample_index(def_probs)
                support = self.play_support[(off_action, def_action)]
                exp_yards = sum(p * o[0] for p, o in zip(support.probs, support.outcomes))
                transitions = self.action_pair_transitions(state, off_action, def_action)
                prev_state = state
                state = self._sample_state_transition(transitions)
                total_plays += 1
                total_expected_yards += exp_yards
                delta = float(state[4] - prev_state[4])
                if delta > 0:
                    total_offense_scoring_plays += 1
            wins += int(self.terminal_value(state) == 1.0)
            total_final_score_diff += float(state[4])

        games = max(1, n)
        return {
            "win_rate": wins / games,
            "avg_yards_per_play": total_expected_yards / max(1, total_plays),
            "avg_plays_per_game": total_plays / games,
            "avg_final_score_diff": total_final_score_diff / games,
            "avg_offense_scoring_plays_per_game": total_offense_scoring_plays / games,
        }

    def _failed_fourth_transitions(
        self,
        yardline: int,
        next_ticks: int,
        score_diff: int,
        base_prob: float,
    ) -> List[Tuple[float, State]]:
        out: List[Tuple[float, State]] = []

        if self.field_goal.get("enabled", True) and yardline <= int(self.field_goal.get("yardline_threshold", 35)):
            success_prob = min(1.0, max(0.0, float(self.field_goal.get("success_prob", 0.78))))
            miss_cost = int(self.field_goal.get("miss_score_cost", 1))
            out.extend(
                self._post_change_transitions(
                    next_ticks=next_ticks,
                    score_diff=score_diff,
                    offense_points=3,
                    possession_cost=0,
                    regime="normal",
                    reset_yardline=75,
                    base_prob=base_prob * success_prob,
                )
            )
            out.extend(
                self._post_change_transitions(
                    next_ticks=next_ticks,
                    score_diff=score_diff,
                    offense_points=0,
                    possession_cost=miss_cost,
                    regime="normal",
                    reset_yardline=75,
                    base_prob=base_prob * (1.0 - success_prob),
                )
            )
            return out

        if self.punt.get("enabled", True) and yardline >= int(self.punt.get("yardline_threshold", 55)):
            # Punt realism: touchback, pin deep, or standard net punt.
            touchback_prob = max(0.0, float(self.punt.get("touchback_prob", 0.25)))
            pin_prob = max(0.0, float(self.punt.get("pin_prob", 0.2)))
            rem = max(0.0, 1.0 - touchback_prob - pin_prob)
            total = touchback_prob + pin_prob + rem
            if total <= 0:
                touchback_prob, pin_prob, rem = 0.25, 0.2, 0.55
                total = 1.0
            touchback_prob /= total
            pin_prob /= total
            rem /= total

            touchback_cost = int(self.punt.get("touchback_score_cost", 2))
            pin_cost = int(self.punt.get("pin_score_cost", 1))
            standard_cost = int(self.punt.get("standard_score_cost", 2))
            pin_yardline = int(self.punt.get("pin_yardline", 95))
            net_yards = int(self.punt.get("net_yards", 40))

            standard_yardline = max(20, min(95, 100 - max(1, yardline - net_yards)))
            out.extend(
                self._post_change_transitions(
                    next_ticks=next_ticks,
                    score_diff=score_diff,
                    offense_points=0,
                    possession_cost=touchback_cost,
                    regime="normal",
                    reset_yardline=80,
                    base_prob=base_prob * touchback_prob,
                )
            )
            out.extend(
                self._post_change_transitions(
                    next_ticks=next_ticks,
                    score_diff=score_diff,
                    offense_points=0,
                    possession_cost=pin_cost,
                    regime="deep_own_territory",
                    reset_yardline=pin_yardline,
                    base_prob=base_prob * pin_prob,
                )
            )
            out.extend(
                self._post_change_transitions(
                    next_ticks=next_ticks,
                    score_diff=score_diff,
                    offense_points=0,
                    possession_cost=standard_cost,
                    regime="normal",
                    reset_yardline=standard_yardline,
                    base_prob=base_prob * rem,
                )
            )
            return out

        out.extend(
            self._post_change_transitions(
                next_ticks=next_ticks,
                score_diff=score_diff,
                offense_points=0,
                possession_cost=self.turnover_on_downs_score_cost,
                regime="short_field",
                reset_yardline=75,
                base_prob=base_prob,
            )
        )
        return out

    def _post_change_transitions(
        self,
        next_ticks: int,
        score_diff: int,
        offense_points: int,
        possession_cost: int,
        regime: str,
        reset_yardline: int,
        base_prob: float,
    ) -> List[Tuple[float, State]]:
        score_values = [int(v) for v in self.opponent_response.get("score_values", [0, 3, 7])]
        score_probs = self._normalized_probs(self.opponent_response.get(regime, self.opponent_response.get("normal", [0.5, 0.25, 0.25])))
        clock_values = [int(v) for v in self.opponent_response.get("clock_ticks", [2, 4, 6])]
        clock_probs = self._normalized_probs(self.opponent_response.get("clock_probs", [0.4, 0.4, 0.2]))

        transitions: List[Tuple[float, State]] = []
        for score_p, opp_points in zip(score_probs, score_values):
            for clock_p, extra_ticks in zip(clock_probs, clock_values):
                final_ticks = max(0, next_ticks - extra_ticks)
                final_score_diff = score_diff + offense_points - possession_cost - opp_points
                state = self._normalize_state(
                    (reset_yardline, 1, min(10, reset_yardline), final_ticks, final_score_diff)
                )
                transitions.append((base_prob * score_p * clock_p, state))
        return transitions

    def _normalize_state(self, state: State) -> State:
        yardline, down, distance, ticks, score_diff = state
        yardline = max(1, min(99, int(yardline)))
        down = max(1, min(4, int(down)))
        distance = max(1, min(min(20, yardline), int(distance)))
        ticks = max(0, min(self.max_ticks, int(ticks)))
        score_diff = max(-self.max_score_diff, min(self.max_score_diff, int(score_diff)))
        return (yardline, down, distance, ticks, score_diff)

    def _sample_index(self, probs: Sequence[float]) -> int:
        total = sum(float(p) for p in probs)
        if total <= 0.0:
            return 0
        r = self.rng.random() * total
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += float(p)
            if r <= cumulative:
                return i
        return len(probs) - 1

    def _sample_weighted(self, items: Sequence[Dict], key: str):
        weights = [max(0.0, float(x.get(key, 0.0))) for x in items]
        idx = self._sample_index(weights)
        return items[idx]

    def _sample_state_transition(self, transitions: Sequence[Tuple[float, State]]) -> State:
        probs = [max(0.0, float(p)) for p, _ in transitions]
        idx = self._sample_index(probs)
        return transitions[idx][1]

    def _merge_transitions(self, transitions: Sequence[Tuple[float, State]]) -> List[Tuple[float, State]]:
        merged: Dict[State, float] = {}
        for p, state in transitions:
            if p <= 0.0:
                continue
            merged[state] = merged.get(state, 0.0) + p
        total = sum(merged.values())
        if total <= 0.0:
            return [(1.0, self._normalize_state((75, 1, 10, 0, 0)))]
        return [(p / total, s) for s, p in merged.items()]

    def _normalized_probs(self, probs: Sequence[float]) -> List[float]:
        cleaned = [max(0.0, float(p)) for p in probs]
        total = sum(cleaned)
        if total <= 0.0:
            if not cleaned:
                return [1.0]
            return [1.0 / len(cleaned)] * len(cleaned)
        return [p / total for p in cleaned]
