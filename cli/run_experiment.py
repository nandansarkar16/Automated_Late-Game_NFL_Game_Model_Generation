# CLI entrypoint that loads config, runs GA, and writes artifacts.
import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

from eval.evaluator import evaluate_candidate
from search.genetic_search import genetic_search


def _load_config(path: str):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if path.endswith(".yaml") or path.endswith(".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requested but PyYAML is not installed") from e
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    raise ValueError(f"Unsupported config extension: {path}")


def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def _write_history_csv(path: Path, history):
    fields = [
        "generation",
        "best_fitness",
        "mean_fitness",
        "StrategicDiversity",
        "StateSensitivity",
        "OutcomePlausibility",
        "Robustness",
        "DegeneracyPenalty",
        "YardsPlausibility",
        "AvgYardsPerPlay",
        "AvgPlaysPerGame",
        "AvgFinalScoreDiff",
        "AvgOffenseScoringPlaysPerGame",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            metrics = row.get("best_metrics", {})
            writer.writerow(
                {
                    "generation": row.get("generation"),
                    "best_fitness": row.get("best_fitness"),
                    "mean_fitness": row.get("mean_fitness"),
                    "StrategicDiversity": metrics.get("StrategicDiversity"),
                    "StateSensitivity": metrics.get("StateSensitivity"),
                    "OutcomePlausibility": metrics.get("OutcomePlausibility"),
                    "Robustness": metrics.get("Robustness"),
                    "DegeneracyPenalty": metrics.get("DegeneracyPenalty"),
                    "YardsPlausibility": metrics.get("YardsPlausibility"),
                    "AvgYardsPerPlay": metrics.get("AvgYardsPerPlay"),
                    "AvgPlaysPerGame": metrics.get("AvgPlaysPerGame"),
                    "AvgFinalScoreDiff": metrics.get("AvgFinalScoreDiff"),
                    "AvgOffenseScoringPlaysPerGame": metrics.get("AvgOffenseScoringPlaysPerGame"),
                }
            )


def _write_report(path: Path, cfg, best_eval, history):
    text = []
    text.append("# Milestone 1 Experiment Report")
    text.append("")
    text.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    text.append("")
    text.append("## Objective")
    text.append("Optimize fourth-quarter game-model parameters for composite realism/strategic quality using GA outer-loop and DP+LP equilibrium inner-loop.")
    text.append("")
    text.append("## Configuration")
    text.append("```json")
    text.append(json.dumps(cfg, indent=2, sort_keys=True))
    text.append("```")
    text.append("")
    text.append("## Best Candidate Metrics")
    text.append("```json")
    text.append(json.dumps(best_eval["metrics"], indent=2, sort_keys=True))
    text.append("```")
    text.append("")
    text.append(f"Win-rate mean: {best_eval['win_rate_mean']:.4f}")
    if best_eval.get("solver_meta"):
        metas = best_eval["solver_meta"]
        mean_solve = sum(float(m.get("elapsed_sec", 0.0)) for m in metas) / len(metas)
        lp_solves = sum(int(m.get("lp_solves", 0)) for m in metas)
        lp_failed = sum(int(m.get("lp_failed", 0)) for m in metas)
        text.append(f"Mean DP+LP solve time per seed: {mean_solve:.4f}s")
        text.append(f"LP solves (total): {lp_solves}; LP failures (total): {lp_failed}")
    text.append("")
    text.append("## Evolution Summary")
    text.append(f"Generations: {len(history)}")
    text.append(f"Best fitness achieved: {max(h['best_fitness'] for h in history):.4f}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))


def run_experiment(config: dict, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    master_seed = int(config.get("master_seed", 20260224))
    random.seed(master_seed)

    search_cfg = dict(config["search"])
    eval_cfg = dict(config["eval"])
    heartbeat_path = str(Path(out_dir) / "heartbeat.jsonl")
    search_cfg["checkpoint_base"] = str(Path(out_dir) / "ga_checkpoint")
    search_cfg["generation_log_path"] = str(Path(out_dir) / "progress.jsonl")
    search_cfg["heartbeat_log_path"] = heartbeat_path
    # Plumb the same heartbeat log through eval_cfg so the inner DP/LP solver
    # emits reach_layer / solver_time_layer / seed_* events to the same file.
    # Without this the status banner's dp_layer/reach_layer stay null.
    eval_cfg["heartbeat_log_path"] = heartbeat_path
    solver_cfg = dict(eval_cfg.get("solver", {}))
    solver_cfg["heartbeat_log_path"] = heartbeat_path
    eval_cfg["solver"] = solver_cfg

    best_params, history = genetic_search(search_cfg, eval_cfg)
    best_eval = evaluate_candidate(best_params, eval_cfg)

    history_path = Path(out_dir) / "history.jsonl"
    history_csv_path = Path(out_dir) / "history.csv"
    best_path = Path(out_dir) / "best_candidate.json"
    report_path = Path(out_dir) / "report.md"

    _write_jsonl(history_path, history)
    _write_history_csv(history_csv_path, history)

    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_eval, f, indent=2, sort_keys=True)

    _write_report(report_path, config, best_eval, history)

    return {
        "history_jsonl": str(history_path),
        "history_csv": str(history_csv_path),
        "best_candidate": str(best_path),
        "report": str(report_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run GA + DP/LP fourth-quarter NFL strategy experiments")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config")
    parser.add_argument("--out", required=True, help="Output directory for artifacts")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    outputs = run_experiment(cfg, args.out)
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
