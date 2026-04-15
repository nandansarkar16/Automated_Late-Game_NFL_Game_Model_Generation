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
DEFAULT_OUT = "results/zoo_full"
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


def _build_status(meta: Dict) -> Dict:
    out_dir = Path(meta["out_dir"])
    progress_path = out_dir / "progress.jsonl"
    heartbeat_path = out_dir / "heartbeat.jsonl"

    pid = int(meta.get("pid", -1))
    running = pid > 0 and _is_pid_running(pid)

    progress_rows = _tail_jsonl(progress_path, n=1000)
    heartbeat_rows = _tail_jsonl(heartbeat_path, n=200)

    completed_gens = len(progress_rows)
    total_gens = int(meta.get("search", {}).get("generations", 0))
    gen_pct = (completed_gens / total_gens * 100.0) if total_gens > 0 else 0.0

    latest_progress = progress_rows[-1] if progress_rows else {}
    latest_heartbeat = heartbeat_rows[-1] if heartbeat_rows else {}

    # Stage text based on latest heartbeat event.
    stage = "idle"
    if latest_heartbeat:
        ev = latest_heartbeat.get("event", "")
        if ev == "reachable_layer_done":
            stage = f"reachable states build (seed={latest_heartbeat.get('seed')}, t={latest_heartbeat.get('time_layer')})"
        elif ev == "solver_time_layer_done":
            stage = f"dp solve layer (seed={latest_heartbeat.get('seed')}, t={latest_heartbeat.get('time_layer')})"
        elif ev == "seed_start":
            stage = f"seed solve started (seed={latest_heartbeat.get('seed')})"
        elif ev == "seed_solved":
            stage = f"seed solved (seed={latest_heartbeat.get('seed')})"
        elif ev == "seed_sim_done":
            stage = f"seed simulation done (seed={latest_heartbeat.get('seed')})"
        elif ev == "candidate_eval":
            stage = "candidate evaluation complete"
        elif ev == "candidate_eval_done":
            stage = "candidate evaluation done"
        else:
            stage = ev

    elapsed_sec = None
    eta_sec = None
    started_at = meta.get("started_at")
    if started_at:
        try:
            start_dt = datetime.fromisoformat(started_at.replace("Z", ""))
            elapsed_sec = max(0.0, (datetime.now(UTC) - start_dt.replace(tzinfo=UTC)).total_seconds())
        except Exception:
            elapsed_sec = None

    if elapsed_sec is not None and completed_gens > 0 and total_gens > completed_gens:
        sec_per_gen = elapsed_sec / completed_gens
        eta_sec = sec_per_gen * (total_gens - completed_gens)

    status = {
        "timestamp": _now_iso(),
        "running": running,
        "pid": pid,
        "generation": {
            "completed": completed_gens,
            "total": total_gens,
            "percent": gen_pct,
        },
        "latest_stage": stage,
        "latest_progress": latest_progress,
        "latest_heartbeat": latest_heartbeat,
        "counts": {
            "progress_rows": len(progress_rows),
            "heartbeat_rows": len(heartbeat_rows),
        },
        "timing": {
            "elapsed_sec": elapsed_sec,
            "eta_sec": eta_sec,
        },
        "paths": meta.get("logs", {}),
    }
    return status


def status_run(out_dir: Path):
    meta_path = out_dir / META_FILE
    if not meta_path.exists():
        print("No run metadata found. Start with: python3 zoo_run.py start")
        return

    meta = _load_json(meta_path)
    status = _build_status(meta)
    _write_json(out_dir / STATUS_FILE, status)

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
    args = parser.parse_args()

    out_dir = Path(args.out)
    config_path = Path(args.config)

    if args.action == "start":
        start_run(config_path=config_path, out_dir=out_dir)
    elif args.action == "status":
        status_run(out_dir=out_dir)
    elif args.action == "stop":
        stop_run(out_dir=out_dir)
    elif args.action == "tail":
        tail_logs(out_dir=out_dir, n=args.tail_lines)


if __name__ == "__main__":
    main()
