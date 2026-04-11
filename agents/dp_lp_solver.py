from __future__ import annotations

# Solves the DP recursion and per-state zero-sum matrix equilibria.
import time
from itertools import combinations
from typing import Dict, List, Sequence, Tuple
import json

from env.quarter_strategy import QuarterStrategy, State


Matrix = List[List[float]]


def _append_jsonl(path: str | None, row: Dict):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _solve_linear_system(a: List[List[float]], b: List[float], tol: float) -> List[float] | None:
    n = len(a)
    if n == 0 or any(len(row) != n for row in a) or len(b) != n:
        return None

    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) <= tol:
            return None
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) <= tol:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def _embed_strategy(indices: Tuple[int, ...], values: Sequence[float], size: int) -> List[float]:
    out = [0.0] * size
    for idx, val in zip(indices, values):
        out[idx] = float(val)
    return out


def _normalize_probs(probs: List[float], tol: float) -> List[float] | None:
    clipped = [0.0 if abs(x) <= tol else x for x in probs]
    if any(x < -tol for x in clipped):
        return None
    s = sum(max(0.0, x) for x in clipped)
    if s <= tol:
        return None
    return [max(0.0, x) / s for x in clipped]


def _payoff_row(matrix: Matrix, row: int, pi_def: Sequence[float]) -> float:
    return sum(matrix[row][j] * pi_def[j] for j in range(len(pi_def)))


def _payoff_col(matrix: Matrix, col: int, pi_off: Sequence[float]) -> float:
    return sum(pi_off[i] * matrix[i][col] for i in range(len(pi_off)))


def solve_zero_sum_lp(matrix: Matrix, tol: float = 1e-9) -> Tuple[List[float], List[float], float]:
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        raise ValueError("matrix must be non-empty")
    if any(len(row) != n_cols for row in matrix):
        raise ValueError("matrix must be rectangular")

    max_support = min(n_rows, n_cols)
    for k in range(1, max_support + 1):
        for row_support in combinations(range(n_rows), k):
            for col_support in combinations(range(n_cols), k):
                # Solve offense mix over row support and value v:
                # sum_i x_i M_ij = v for j in support, and sum_i x_i = 1.
                off_a: List[List[float]] = []
                off_b: List[float] = []
                for col in col_support:
                    off_a.append([matrix[row][col] for row in row_support] + [-1.0])
                    off_b.append(0.0)
                off_a.append([1.0] * k + [0.0])
                off_b.append(1.0)
                off_sol = _solve_linear_system(off_a, off_b, tol)
                if off_sol is None:
                    continue
                x_support = off_sol[:-1]
                v_off = off_sol[-1]
                if any(x < -tol for x in x_support):
                    continue
                pi_off = _embed_strategy(row_support, x_support, n_rows)
                pi_off = _normalize_probs(pi_off, tol)
                if pi_off is None:
                    continue

                # Solve defense mix over col support and value w:
                # sum_j M_ij y_j = w for i in support, and sum_j y_j = 1.
                def_a: List[List[float]] = []
                def_b: List[float] = []
                for row in row_support:
                    def_a.append([matrix[row][col] for col in col_support] + [-1.0])
                    def_b.append(0.0)
                def_a.append([1.0] * k + [0.0])
                def_b.append(1.0)
                def_sol = _solve_linear_system(def_a, def_b, tol)
                if def_sol is None:
                    continue
                y_support = def_sol[:-1]
                v_def = def_sol[-1]
                if any(y < -tol for y in y_support):
                    continue
                pi_def = _embed_strategy(col_support, y_support, n_cols)
                pi_def = _normalize_probs(pi_def, tol)
                if pi_def is None:
                    continue

                value = 0.5 * (v_off + v_def)

                # Equilibrium checks.
                if any(_payoff_col(matrix, c, pi_off) < value - 1e-7 for c in range(n_cols)):
                    continue
                if any(_payoff_row(matrix, r, pi_def) > value + 1e-7 for r in range(n_rows)):
                    continue

                return pi_off, pi_def, float(value)

    raise RuntimeError("failed to find zero-sum equilibrium via support enumeration")


def _initial_states(model: QuarterStrategy) -> List[State]:
    states = []
    for item in model.init_distribution:
        s = item["state"]
        states.append((int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4])))
    return [model._normalize_state(s) for s in states]


def _reachable_states_by_time(
    model: QuarterStrategy,
    heartbeat_log_path: str | None = None,
    heartbeat_seed=None,
) -> Dict[int, List[State]]:
    # Build the reachable set once so DP only solves states that can occur.
    by_time: Dict[int, set] = {t: set() for t in range(model.max_ticks + 1)}
    for s in _initial_states(model):
        by_time[s[3]].add(s)

    for t in range(model.max_ticks, 0, -1):
        t_start = time.time()
        layer = list(by_time[t])
        for state in layer:
            if model.is_terminal(state):
                continue
            for off_action in range(model.offensive_playbook_size()):
                for def_action in range(model.defensive_playbook_size()):
                    for _, nxt in model.action_pair_transitions(state, off_action, def_action):
                        by_time[nxt[3]].add(nxt)
        _append_jsonl(
            heartbeat_log_path,
            {
                "event": "reachable_layer_done",
                "seed": heartbeat_seed,
                "time_layer": t,
                "input_states": len(layer),
                "cumulative_reachable_states": int(sum(len(v) for v in by_time.values())),
                "layer_elapsed_sec": time.time() - t_start,
            },
        )

    return {k: list(v) for k, v in by_time.items()}


def solve_equilibrium_policy(model: QuarterStrategy, solve_cfg: Dict | None = None) -> Dict:
    solve_cfg = solve_cfg or {}
    start = time.time()
    heartbeat_log_path = solve_cfg.get("heartbeat_log_path")
    heartbeat_every_t = max(1, int(solve_cfg.get("heartbeat_every_t", 1)))
    heartbeat_seed = solve_cfg.get("heartbeat_seed")

    _append_jsonl(
        heartbeat_log_path,
        {
            "event": "solver_reachable_start",
            "seed": heartbeat_seed,
            "max_ticks": model.max_ticks,
        },
    )
    reachable = _reachable_states_by_time(model, heartbeat_log_path=heartbeat_log_path, heartbeat_seed=heartbeat_seed)
    _append_jsonl(
        heartbeat_log_path,
        {
            "event": "solver_reachable_done",
            "seed": heartbeat_seed,
            "reachable_states": int(sum(len(v) for v in reachable.values())),
            "elapsed_sec": time.time() - start,
        },
    )
    V: Dict[State, float] = {}
    pi_off: Dict[State, Tuple[float, ...]] = {}
    pi_def: Dict[State, Tuple[float, ...]] = {}

    lp_solves = 0
    lp_failed = 0

    for state in reachable.get(0, []):
        V[state] = model.terminal_value(state)

    for t in range(1, model.max_ticks + 1):
        # Backward induction over time layers.
        layer_start = time.time()
        layer_state_count = 0
        for state in reachable.get(t, []):
            layer_state_count += 1
            if model.is_terminal(state):
                V[state] = model.terminal_value(state)
                continue

            def value_fn(next_state: State) -> float:
                if next_state in V:
                    return V[next_state]
                if model.is_terminal(next_state):
                    return model.terminal_value(next_state)
                return 0.0

            matrix = model.build_payoff_matrix(state, value_fn)
            try:
                p_off, p_def, value = solve_zero_sum_lp(matrix, tol=float(solve_cfg.get("tol", 1e-9)))
                lp_solves += 1
            except Exception:
                # Keep the pipeline alive if LP fails on a corner case state.
                lp_failed += 1
                n_off = model.offensive_playbook_size()
                n_def = model.defensive_playbook_size()
                p_off = [1.0 / n_off] * n_off
                p_def = [1.0 / n_def] * n_def
                value = sum(sum(row) for row in matrix) / (len(matrix) * len(matrix[0]))

            V[state] = float(value)
            pi_off[state] = tuple(p_off)
            pi_def[state] = tuple(p_def)

        if t % heartbeat_every_t == 0 or t == model.max_ticks:
            _append_jsonl(
                heartbeat_log_path,
                {
                    "event": "solver_time_layer_done",
                    "seed": heartbeat_seed,
                    "time_layer": t,
                    "layer_state_count": layer_state_count,
                    "lp_solves_so_far": lp_solves,
                    "lp_failed_so_far": lp_failed,
                    "layer_elapsed_sec": time.time() - layer_start,
                    "elapsed_sec": time.time() - start,
                },
            )

    meta = {
        "elapsed_sec": time.time() - start,
        "lp_solves": lp_solves,
        "lp_failed": lp_failed,
        "reachable_states": sum(len(v) for v in reachable.values()),
        "states_by_time": {str(k): len(v) for k, v in reachable.items()},
    }

    return {"V": V, "pi_off": pi_off, "pi_def": pi_def, "meta": meta}
