# Automated Late-Game NFL Game Model Generation

This repo is my working code for a simplified fourth-quarter football game model.
The main workflow is:

1. Define a game model (state transitions, play outcomes, scoring logic).
2. Solve equilibrium policies with dynamic programming + per-state zero-sum matrix solve.
3. Score that game model with simulation-based metrics.
4. Use a genetic algorithm to search for better game model parameters.

The game is still intentionally simplified. The state is
`(yardline, down, distance, time_ticks, score_diff)`, so it does not carry
explicit possession or drive IDs. To make that abstraction less unrealistic,
the environment now uses a compressed opponent-response jump after scores and
other possession-ending events. That jump removes extra clock and applies a
small opponent scoring distribution based on whether the opponent would be
starting deep, in a normal situation, or on a short field.

## Project layout

- `run_experiment.py`
  - tiny root wrapper so experiments can be launched from repo root.

- `cli/run_experiment.py`
  - loads config, runs GA search, writes `history.jsonl`, `history.csv`, `best_candidate.json`, and `report.md`.

- `env/quarter_strategy.py`
  - core environment.
  - state is `(yardline, down, distance, time_ticks, score_diff)`.
  - handles play outcomes, first downs, touchdowns, field goals, punts, and the compressed opponent-response layer after possession changes.

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

If a long Zoo run stops mid-way, restarting the same output directory now
resumes from the last completed GA generation using a saved checkpoint instead
of starting from generation 0 again.

## Local overnight runs

`configs/m1_local_overnight.json` is a smaller but still legitimate config
targeted at laptop-scale runs (~2-5h wall clock). It uses `max_time_ticks=12`,
tightened `vector_gene` bounds to prevent the GA from wandering into gene
regions that explode the reachable state set, and 2 seeds per eval.

Before committing to a long run, sanity-check wall clock with the calibrator:

```bash
python3 tools/calibrate_runtime.py \
  --config configs/m1_local_overnight.json \
  --seeds 2 --population 10 --generations 12 \
  --target-hours 5 --stress 4
```

`--stress N` samples N random candidates from the GA gene space and reports
the real per-eval time distribution (min/median/mean/max). The base eval alone
is not representative because the GA mutates genes that directly drive solver
cost.

Launch the same way as a Zoo run:

```bash
python3 zoo_run.py start --config configs/m1_local_overnight.json --out results/m1_local_overnight
python3 zoo_run.py status --out results/m1_local_overnight
```

The status output now includes a one-line banner with current generation,
candidate, seed, DP time layer, elapsed, and ETA. Use `--verbose` for the
full JSON dump.
