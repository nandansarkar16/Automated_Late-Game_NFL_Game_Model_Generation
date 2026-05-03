# Automated Game-Model Synthesis for Late-Game NFL Strategy

This repository contains the code for my senior thesis in Applied Mathematics at Yale. The project asks the question: instead of finding the best way to *play* a game, can we automatically find the best *model* of the game itself?

## What this project does

The game studied here is a simplified version of **NFL Strategy**, a two-player simulation of American football. In the late-game two-minute drill, an offense tries to score before time runs out while a defense tries to burn the clock. On each play, both sides simultaneously pick a play call, and the outcome (yards gained, time consumed, possession lost) is sampled from a probability table specific to that matchup. The full collection of these tables is called the **model** of the game.

Most work on NFL Strategy takes the model as given and asks: what is the best way to play? This project inverts that. The model itself is what we search over. The goal is to automatically find model parameters such that the resulting game is strategically interesting---its optimal policies are genuinely mixed, sensitive to game context, and statistically plausible as late-game football.

## How it works

The pipeline has two nested components.

**Inner loop (exact equilibrium solver).** For any fixed model, the game is a finite-horizon two-player zero-sum stochastic game. Because every play consumes clock, the state graph is acyclic and can be solved exactly by backward induction. At each reachable game state, a 3×3 zero-sum matrix game is assembled and solved by support enumeration---a method that checks all possible mixed-strategy support pairs and returns the exact equilibrium. This gives both players' optimal mixed strategies and the game value at every state.

**Outer loop (genetic algorithm).** The model parameters (yards, time consumed, and turnover flag for each play-outcome tuple) are encoded as a real-valued genome with 135 genes. A real-coded genetic algorithm searches over this space, evaluating each candidate by running the inner loop and scoring the resulting equilibrium game on six criteria:

1. **Strategic Diversity** --- does the optimal policy genuinely mix across play calls?
2. **State Sensitivity** --- does the policy shift sensibly between trailing, tied, and leading situations?
3. **Outcome Plausibility** --- are the expected yards, clock consumption, and turnover rates within realistic football ranges?
4. **Robustness** --- is the fitness stable across different random seeds?
5. **Non-Degeneracy** --- does no single play dominate across all situations?
6. **Yards-per-Play Plausibility** --- are simulated drives producing realistic yardage?

These six terms are averaged equally to produce a single composite fitness score in [0, 1].

## Repository layout

```
env/quarter_strategy.py       # Game environment: state transitions, scoring, opponent response
agents/dp_lp_solver.py        # Inner loop: backward DP + per-state zero-sum LP solver
eval/evaluator.py             # Evaluates one candidate model end-to-end
eval/metrics.py               # Individual fitness term implementations
search/genetic_search.py      # Outer loop: genetic algorithm (via PyGAD)
cli/run_experiment.py         # Entry point: loads config, runs GA, writes artifacts
run_experiment.py             # Thin root wrapper to launch from repo root
zoo_run.py                    # Helper for starting/monitoring/stopping long runs
tools/calibrate_runtime.py    # Times a single candidate eval and projects total runtime
tools/analyze_run.py          # Post-run analysis and convergence plots
tools/random_search.py        # Random-search baseline (no evolution)
configs/                      # Experiment configurations (JSON)
results/                      # Output directories (written at runtime)
tests/                        # Unit and integration tests
```

## Setup

Requires Python 3.9+.

Install the one external dependency:

```bash
python3 -m pip install --user pygad
```

## Running an experiment

**Quick smoke test** (completes in ~2 minutes, just checks that the pipeline is wired correctly):

```bash
python3 run_experiment.py --config configs/m1_smoke.json --out results/m1_smoke
```

**Production run** (the config used for the main thesis results, ~5–7 hours on a laptop):

```bash
python3 zoo_run.py start --config configs/m1_local_overnight.json --out results/m1_local_overnight
```

**Check status of a running experiment:**

```bash
python3 zoo_run.py status --out results/m1_local_overnight
```

**Tail live output:**

```bash
python3 zoo_run.py tail --out results/m1_local_overnight --tail-lines 40
```

**Stop a run:**

```bash
python3 zoo_run.py stop --out results/m1_local_overnight
```

If a long run is interrupted, restarting with the same `--out` directory resumes from the last completed generation rather than starting over.

## Estimating runtime before committing

Before launching a long run, use the calibration tool to time a single candidate evaluation and project total wall-clock time:

```bash
python3 tools/calibrate_runtime.py \
  --config configs/m1_local_overnight.json \
  --population 20 --generations 9 \
  --target-hours 5 --stress 4
```

`--stress N` samples N additional random candidates from the gene space to get a realistic distribution of evaluation times (some candidates are faster or slower than the hand-designed baseline).

## Running tests

```bash
python3 -m unittest discover -s tests -v
```

## Output files

Each run writes the following into its output directory:

| File | Contents |
|---|---|
| `history.jsonl` | Full per-generation records (population fitnesses, best candidate) |
| `history.csv` | Same data in tabular form |
| `best_candidate.json` | Best evolved model parameters and all fitness metrics |
| `progress.jsonl` | Per-generation summary (best/mean fitness, used for plotting) |
| `report.md` | Auto-generated text summary |
| `heartbeat.jsonl` | (Long runs) Inner-loop progress logs per time layer |

## Post-run analysis

After a run completes, generate convergence plots and a parameter diff table:

```bash
python3 tools/analyze_run.py \
  --run-dir results/m1_local_overnight \
  --config configs/m1_local_overnight.json
```

This writes `convergence.png` and `params_diff.csv` into the run directory.

## Random search baseline

To evaluate how much the genetic algorithm adds over pure random sampling:

```bash
python3 tools/random_search.py \
  --config configs/m1_local_overnight.json \
  --n-samples 200 \
  --out results/random_search_baseline \
  --seed 9999
```

## Configuration

All experiment parameters live in a JSON config file. The key sections are:

- **`search`** --- population size, generations, elites, gene bounds, initialization strategy
- **`eval`** --- number of seeds, simulated games per seed, policy probe states, plausibility bands
- **`master_seed`** --- top-level random seed for reproducibility
