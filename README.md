# Automated Late-Game NFL Game Model Generation

This repo is my working code for a simplified fourth-quarter football game model.
The main workflow is:

1. Define a game model (state transitions, play outcomes, scoring logic).
2. Solve equilibrium policies with dynamic programming + per-state zero-sum matrix solve.
3. Score that game model with simulation-based metrics.
4. Use a genetic algorithm to search for better game model parameters.

## Project layout

- `run_experiment.py`
  - tiny root wrapper so experiments can be launched from repo root.

- `cli/run_experiment.py`
  - loads config, runs GA search, writes `history.jsonl`, `history.csv`, `best_candidate.json`, and `report.md`.

- `env/quarter_strategy.py`
  - core environment.
  - state is `(yardline, down, distance, time_ticks, score_diff)`.
  - handles transition logic for play outcomes, first downs, touchdowns, field goals, and punts.

- `agents/dp_lp_solver.py`
  - inner-loop solver.
  - does backward DP over time and solves each state’s offense/defense matrix game.

- `eval/evaluator.py`
  - evaluates one candidate model end-to-end.
  - runs solver, simulates policy profile, computes final fitness and metrics.

- `eval/metrics.py`
  - metric helper functions (diversity, sensitivity, plausibility, robustness, non-degeneracy).

- `search/genetic_search.py`
  - GA outer loop (`PyGAD`).
  - decodes genes into outcome tuples, evaluates candidates, logs tuple changes and fitness traces.

- `configs/*.json`
  - experiment configs.
  - `m1_smoke.json` is the quick check.
  - `m1_zoo_parity_pilot.json` is the heavy local/zoo-style config.

- `tests/`
  - unit and integration tests for environment, solver, GA behavior, and pipeline.

- `hw5/`
  - old class files kept for reference only.

## Setup

Use Python 3.9+.

Install dependency:

```bash
python3 -m pip install --user pygad
```

## Run commands

Quick smoke run:

```bash
python3 run_experiment.py --config configs/m1_smoke.json --out results/m1_smoke
```

Heavier run:

```bash
python3 run_experiment.py --config configs/m1.json --out results/m1
```

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

## Output files

Each run writes:

- `history.jsonl` (full per-generation records)
- `history.csv` (easy spreadsheet view)
- `best_candidate.json` (best params + metrics)
- `report.md` (auto summary)

For heavy debug runs, heartbeat logs may also be written to `results/.../heartbeat.jsonl`.

## Zoo run helper

For long runs, use the helper script instead of running the experiment command manually.

Start in background:

```bash
python3 zoo_run.py start --config configs/m1_zoo_full.json --out results/zoo_full
```

Check progress + stage + ETA:

```bash
python3 zoo_run.py status --out results/zoo_full
```

Tail heartbeat/progress/log output:

```bash
python3 zoo_run.py tail --out results/zoo_full --tail-lines 40
```

Stop run:

```bash
python3 zoo_run.py stop --out results/zoo_full
```

`configs/m1_zoo_full.json` is the default full config with `max_time_ticks=30`, which is a better runtime/reachability tradeoff than 45 on local hardware.

