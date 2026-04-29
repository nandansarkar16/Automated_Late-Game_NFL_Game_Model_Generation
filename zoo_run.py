#!/usr/bin/env python3
# Simple helper for launching and monitoring long Zoo-style experiments.
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

DEFAULT_CONFIG = "configs/m1_zoo_full.json"
DEFAULT_OUT = "results/zoo_v2"
META_FILE = "zoo_job_meta.json"
STATUS_FILE = "zoo_status.json"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _detach_child():
    # Ignore terminal hangups so the long-running job survives disconnects.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _tail_jsonl(path: Path, n: int = 50) -> List[Dict]:
    if not path.exists():
        return []
    q = deque(maxlen=n)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                q.append(json.loads(line))
            except Exception:
                continue
    return list(q)


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _estimate_expected_evals(search_cfg: Dict) -> int:
    pop = int(search_cfg.get("population", 0))
    gens = int(search_cfg.get("generations", 0))
    reeval_top_k = int(search_cfg.get("reeval_top_k", 0))
    reeval_reps = int(search_cfg.get("reeval_reps", 0))
    return gens * (pop + reeval_top_k * max(0, reeval_reps))


def _write_json(path: Path, obj: Dict):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def start_run(config_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / META_FILE
    if meta_path.exists():
        meta = _load_json(meta_path)
        pid = int(meta.get("pid", -1))
        if pid > 0 and _is_pid_running(pid):
            print(f"A run is already active (pid={pid}). Use 'python3 zoo_run.py status' or 'stop'.")
            return

    cfg = _load_json(config_path)
    search_cfg = cfg.get("search", {})

    cmd = [
        sys.executable,
        "run_experiment.py",
        "--config",
        str(config_path),
        "--out",
        str(out_dir),
    ]

    log_path = out_dir / "zoo_runner.log"
    err_path = out_dir / "zoo_runner.err.log"

    with log_path.open("a", encoding="utf-8") as logf, err_path.open("a", encoding="utf-8") as errf:
        logf.write(f"[{_now_iso()}] launching background job\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=errf,
            preexec_fn=_detach_child,
            start_new_session=True,
            close_fds=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    meta = {
        "pid": proc.pid,
        "pgid": int(os.getpgid(proc.pid)),
        "started_at": _now_iso(),
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "python": sys.executable,
        "search": {
            "population": int(search_cfg.get("population", 0)),
            "generations": int(search_cfg.get("generations", 0)),
            "expected_eval_calls": _estimate_expected_evals(search_cfg),
        },
        "logs": {
            "stdout": str(log_path),
            "stderr": str(err_path),
            "heartbeat": str(out_dir / "heartbeat.jsonl"),
            "progress": str(out_dir / "progress.jsonl"),
            "status": str(out_dir / STATUS_FILE),
        },
    }
    _write_json(meta_path, meta)

    print(f"Started zoo run in background. pid={proc.pid}")
    print(f"Config: {config_path}")
    print(f"Output dir: {out_dir}")
    print(f"Use: python3 zoo_run.py status")


def _fmt_hms(sec) -> str:
    if sec is None:
        return "?"
    try:
        sec = max(0, int(round(float(sec))))
    except Exception:
        return "?"
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def _find_latest(rows: List[Dict], event: str) -> Dict:
    for row in reversed(rows):
        if row.get("event") == event:
            return row
    return {}


def _find_latest_any(rows: List[Dict], events: set) -> Dict:
    for row in reversed(rows):
        if row.get("event") in events:
            return row
    return {}


def _search_cfg_from_meta(meta: Dict) -> Dict:
    try:
        cfg_path = Path(meta.get("config_path", ""))
        if cfg_path.exists():
            return json.loads(cfg_path.read_text()).get("search", {})
    except Exception:
        pass
    return {}


def _estimate_eta_sec(
    heartbeat_rows: List[Dict],
    total_gens: int,
    population: int,
    elites,
) -> float | None:
    """ETA in seconds, based on recent per-miss cost and remaining misses.

    Uses an exponentially weighted average of the last N `candidate_eval`
    durations (where N = min(half the completed misses, population)) so the
    estimate tracks the GA's current cost regime rather than averaging gen 0
    in forever.
    """
    if not heartbeat_rows or total_gens <= 0 or population <= 0:
        return None
    eval_rows = [
        r for r in heartbeat_rows if r.get("event") == "candidate_eval"
    ]
    if not eval_rows:
        return None
    try:
        elites_int = int(elites) if elites is not None else 1
    except Exception:
        elites_int = 1
    elites_int = max(0, min(elites_int, population))

    # Expected total misses: gen 0 runs all `population` slots fresh; each
    # subsequent gen only re-evaluates population - elites new candidates
    # (elites are carried through unchanged via keep_elitism).
    expected_total = population + max(0, total_gens - 1) * max(1, population - elites_int)
    completed = len(eval_rows)
    remaining = max(0, expected_total - completed)
    if remaining == 0:
        return 0.0

    # Exponentially weighted average of recent eval durations.
    recent_n = max(population, min(completed, 2 * population))
    recent = eval_rows[-recent_n:]
    alpha = 2.0 / (recent_n + 1.0)
    ewma = float(recent[0].get("candidate_eval_sec", 0.0))
    for row in recent[1:]:
        try:
            ewma = alpha * float(row.get("candidate_eval_sec", 0.0)) + (1 - alpha) * ewma
        except Exception:
            continue
    return float(remaining) * ewma


def _build_status(meta: Dict) -> Dict:
    out_dir = Path(meta["out_dir"])
    progress_path = out_dir / "progress.jsonl"
    heartbeat_path = out_dir / "heartbeat.jsonl"

    pid = int(meta.get("pid", -1))
    running = pid > 0 and _is_pid_running(pid)

    progress_rows = _tail_jsonl(progress_path, n=1000)
    # Pull a generous heartbeat window so we can reconstruct the whole current
    # candidate's state, not just whichever event happened last.
    heartbeat_rows = _tail_jsonl(heartbeat_path, n=2000)

    completed_gens = len(progress_rows)
    total_gens = int(meta.get("search", {}).get("generations", 0))
    gen_pct = (completed_gens / total_gens * 100.0) if total_gens > 0 else 0.0

    latest_progress = progress_rows[-1] if progress_rows else {}
    latest_heartbeat = heartbeat_rows[-1] if heartbeat_rows else {}

    ga_eval_event = _find_latest_any(heartbeat_rows, {"candidate_eval", "cache_hit"})
    cand_start = _find_latest(heartbeat_rows, "candidate_eval_start")
    cand_done = _find_latest(heartbeat_rows, "candidate_eval_done")
    seed_event = _find_latest_any(heartbeat_rows, {"seed_start", "seed_solved", "seed_sim_done"})
    dp_event = _find_latest(heartbeat_rows, "solver_time_layer_done")
    reach_event = _find_latest(heartbeat_rows, "reachable_layer_done")

    gen_in_progress = ga_eval_event.get("generation_in_progress")
    if gen_in_progress is None:
        gen_in_progress = completed_gens
    solution_idx = ga_eval_event.get("solution_idx")
    pop = int(meta.get("search", {}).get("population", 0))

    current_seed = seed_event.get("seed")

    # Read config once (same file already parsed in start_run) so we can pull
    # seeds_total and max_time_ticks without re-opening the file twice.
    cfg_snapshot: Dict = {}
    try:
        cfg_path = Path(meta.get("config_path", ""))
        if cfg_path.exists():
            cfg_snapshot = json.loads(cfg_path.read_text())
    except Exception:
        cfg_snapshot = {}

    seeds_total = len(cfg_snapshot.get("eval", {}).get("seeds", []) or []) or None

    dp_layer = dp_event.get("time_layer") if dp_event else None
    reach_layer = reach_event.get("time_layer") if reach_event else None
    max_ticks = None
    if cand_start and cand_start.get("max_time_ticks"):
        try:
            max_ticks = int(cand_start["max_time_ticks"])
        except Exception:
            max_ticks = None
    if max_ticks is None:
        try:
            max_ticks = int(cfg_snapshot.get("search", {}).get("base_params", {}).get("max_time_ticks", 0)) or None
        except Exception:
            max_ticks = None

    # Figure out the sub-candidate phase (which of the DP stages is happening right now).
    stage_parts = []
    if total_gens:
        gen_display = int(gen_in_progress)
        # During an in-progress candidate we want 1-indexed "gen X/total",
        # but once an eval has finished we're already mid-way toward gen X+1
        # until on_generation flips the counter.
        if ga_eval_event and ga_eval_event is not cand_done:
            gen_display += 1
        stage_parts.append(f"gen {gen_display}/{total_gens}")
    if pop and solution_idx is not None:
        try:
            stage_parts.append(f"cand {int(solution_idx) + 1}/{pop}")
        except Exception:
            pass
    if current_seed is not None:
        stage_parts.append(f"seed={current_seed}")

    latest_ev = latest_heartbeat.get("event", "")
    if latest_ev == "reachable_layer_done" and reach_layer is not None:
        if max_ticks:
            stage_parts.append(f"reach_layer {int(reach_layer)}/{max_ticks}")
        else:
            stage_parts.append(f"reach_layer {int(reach_layer)}")
    elif latest_ev == "solver_time_layer_done" and dp_layer is not None:
        if max_ticks:
            stage_parts.append(f"DP_layer {int(dp_layer)}/{max_ticks}")
        else:
            stage_parts.append(f"DP_layer {int(dp_layer)}")
    elif latest_ev in {"seed_solved", "seed_sim_done"}:
        stage_parts.append(latest_ev)
    elif latest_ev in {"candidate_eval", "cache_hit", "candidate_eval_done"}:
        stage_parts.append(latest_ev)

    elapsed_sec = None
    eta_sec = None
    started_at = meta.get("started_at")
    if started_at:
        try:
            start_dt = datetime.fromisoformat(started_at.replace("Z", ""))
            elapsed_sec = max(0.0, (datetime.now(UTC) - start_dt.replace(tzinfo=UTC)).total_seconds())
        except Exception:
            elapsed_sec = None

    # ETA: use per-cache-miss rate weighted toward recent evals instead of
    # sec_per_completed_generation. Rationale:
    #   - Gen 0 is a cold-start outlier (no elitism cache hits yet), so a
    #     naive elapsed/completed_gens rate over-predicts by 10-20%.
    #   - The GA's per-eval cost shifts as the population walks through gene
    #     space; recent evals are a better predictor than early ones.
    #   - Heartbeats carry the real number of completed misses, so we can
    #     compute s/miss × expected_remaining_misses directly.
    eta_sec = _estimate_eta_sec(
        heartbeat_rows=heartbeat_rows,
        total_gens=total_gens,
        population=pop,
        elites=_search_cfg_from_meta(meta).get("elites", 1),
    )

    stage_str = "  ".join(stage_parts) if stage_parts else (latest_ev or "idle")
    if elapsed_sec is not None:
        stage_str += f"  elapsed={_fmt_hms(elapsed_sec)}"
    if eta_sec is not None:
        stage_str += f"  ETA={_fmt_hms(eta_sec)}"

    # Best-fitness trace for quick eyeballing.
    best_so_far = None
    mean_last = None
    if progress_rows:
        try:
            best_so_far = max(float(r.get("best_fitness", float("-inf"))) for r in progress_rows)
            mean_last = float(progress_rows[-1].get("mean_fitness", 0.0))
        except Exception:
            pass

    status = {
        "timestamp": _now_iso(),
        "running": running,
        "pid": pid,
        "generation": {
            "completed": completed_gens,
            "total": total_gens,
            "percent": gen_pct,
        },
        "current_candidate": {
            "generation_in_progress": gen_in_progress,
            "solution_idx": solution_idx,
            "population": pop,
            "current_seed": current_seed,
            "seeds_total": seeds_total,
            "dp_layer": dp_layer,
            "reach_layer": reach_layer,
            "max_time_ticks": max_ticks,
            "cache_hits": ga_eval_event.get("cache_hits") if ga_eval_event else None,
            "cache_misses": ga_eval_event.get("cache_misses") if ga_eval_event else None,
        },
        "best_fitness_so_far": best_so_far,
        "last_gen_mean_fitness": mean_last,
        "latest_stage": stage_str,
        "latest_progress": latest_progress,
        "latest_heartbeat": latest_heartbeat,
        "counts": {
            "progress_rows": len(progress_rows),
            "heartbeat_rows": len(heartbeat_rows),
        },
        "timing": {
            "elapsed_sec": elapsed_sec,
            "elapsed_hms": _fmt_hms(elapsed_sec) if elapsed_sec is not None else None,
            "eta_sec": eta_sec,
            "eta_hms": _fmt_hms(eta_sec) if eta_sec is not None else None,
        },
        "paths": meta.get("logs", {}),
    }
    return status


def status_run(out_dir: Path, verbose: bool = False):
    meta_path = out_dir / META_FILE
    if not meta_path.exists():
        print("No run metadata found. Start with: python3 zoo_run.py start")
        return

    meta = _load_json(meta_path)
    status = _build_status(meta)
    _write_json(out_dir / STATUS_FILE, status)

    run_state = "RUNNING" if status.get("running") else "STOPPED"
    gen = status.get("generation", {})
    best = status.get("best_fitness_so_far")
    mean = status.get("last_gen_mean_fitness")
    summary_bits = [
        f"[{run_state}] pid={status.get('pid')}",
        f"gens_done={gen.get('completed')}/{gen.get('total')}",
        f"{status.get('latest_stage', '')}",
    ]
    if best is not None:
        summary_bits.append(f"best={best:.4f}")
    if mean is not None:
        summary_bits.append(f"last_gen_mean={mean:.4f}")
    print("  ".join(b for b in summary_bits if b))

    if verbose:
        print(json.dumps(status, indent=2, sort_keys=True))


def stop_run(out_dir: Path):
    meta_path = out_dir / META_FILE
    if not meta_path.exists():
        print("No run metadata found.")
        return
    meta = _load_json(meta_path)
    pid = int(meta.get("pid", -1))
    pgid = int(meta.get("pgid", -1))

    stopped = False
    if pgid > 0:
        try:
            os.killpg(pgid, signal.SIGTERM)
            stopped = True
        except Exception:
            pass
    if not stopped and pid > 0:
        try:
            os.kill(pid, signal.SIGTERM)
            stopped = True
        except Exception:
            pass

    print("Stopped." if stopped else "Could not stop process (it may already be done).")


def tail_logs(out_dir: Path, n: int):
    hb = out_dir / "heartbeat.jsonl"
    pg = out_dir / "progress.jsonl"
    runlog = out_dir / "zoo_runner.log"
    print(f"--- heartbeat ({hb})")
    for row in _tail_jsonl(hb, n=n):
        print(json.dumps(row, sort_keys=True))
    print(f"--- progress ({pg})")
    for row in _tail_jsonl(pg, n=n):
        print(json.dumps(row, sort_keys=True))
    print(f"--- runner.log tail ({runlog})")
    if runlog.exists():
        lines = runlog.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines[-n:]:
            print(line)


def main():
    parser = argparse.ArgumentParser(description="Background runner for Zoo-style experiments")
    parser.add_argument("action", nargs="?", default="start", choices=["start", "status", "stop", "tail"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--tail-lines", type=int, default=30)
    parser.add_argument("--verbose", action="store_true", help="Dump full status JSON, not just the one-line banner")
    args = parser.parse_args()

    out_dir = Path(args.out)
    config_path = Path(args.config)

    if args.action == "start":
        start_run(config_path=config_path, out_dir=out_dir)
    elif args.action == "status":
        status_run(out_dir=out_dir, verbose=args.verbose)
    elif args.action == "stop":
        stop_run(out_dir=out_dir)
    elif args.action == "tail":
        tail_logs(out_dir=out_dir, n=args.tail_lines)


if __name__ == "__main__":
    main()
