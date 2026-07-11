"""
batch_status.py — Check the status of a Gemini Batch job.

Usage:
    python batch_status.py batches/abc123
    python batch_status.py batch/jobs/nbc_vol1.json   # reads job_id from file
    python batch_status.py batches/abc123 --watch      # poll every 60 s
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from batch_common import JOBS_DIR, load_api_key


# Map SDK state values to human-readable strings
_STATE_LABELS = {
    "JOB_STATE_PENDING":   "PENDING   – queued, not yet started",
    "JOB_STATE_RUNNING":   "RUNNING   – processing",
    "JOB_STATE_SUCCEEDED": "SUCCEEDED – ready to download",
    "JOB_STATE_FAILED":    "FAILED    – job-level failure",
    "JOB_STATE_CANCELLED": "CANCELLED",
    "JOB_STATE_CANCELLING":"CANCELLING",
}


def resolve_job_id(arg: str) -> str:
    """Accept either a raw job ID or a path to a jobs/*.json file."""
    p = Path(arg)
    if p.exists() and p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        return data["job_id"]
    # Could also be a bare stem like "nbc_vol1" → look in jobs dir
    candidate = JOBS_DIR / f"{arg}.json"
    if candidate.exists():
        data = json.loads(candidate.read_text(encoding="utf-8"))
        return data["job_id"]
    return arg   # assume it's already a raw job ID


def fetch_and_print(client, job_id: str) -> str:
    """Fetch job and pretty-print status.  Returns the state string."""
    try:
        job = client.batches.get(name=job_id)
    except Exception as e:
        print(f"ERROR fetching job: {e}")
        return "ERROR"

    state_raw = str(job.state.name)  # gives clean "JOB_STATE_SUCCEEDED"
    state_label = _STATE_LABELS.get(state_raw, state_raw)

    print("─" * 56)
    print(f"  Job ID    : {job.name}")
    print(f"  State     : {state_label}")

    # Progress — SDK exposes completed/total when available
    total     = getattr(job, "request_count", None)
    completed = getattr(job, "completed_count", None)
    failed    = getattr(job, "failed_count", None)
    if total is not None:
        pct = (completed or 0) / total * 100 if total else 0
        bar_len = 30
        filled  = int(bar_len * pct / 100)
        bar     = "█" * filled + "░" * (bar_len - filled)
        print(f"  Progress  : [{bar}] {pct:.0f}%  ({completed}/{total})")
        if failed:
            print(f"  Failed    : {failed}")

    created  = getattr(job, "create_time", None)
    updated  = getattr(job, "update_time", None)
    finished = getattr(job, "end_time", None)
    if created:
        print(f"  Created   : {created}")
    if updated:
        print(f"  Updated   : {updated}")
    if finished:
        print(f"  Completed : {finished}")

    # Output URI (available once succeeded)
    dest = getattr(job, "dest", None) or getattr(job, "output_config", None)
    if dest:
        print(f"  Output URI: {dest}")

    print("─" * 56)
    return state_raw


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Check status of a Gemini Batch job."
    )
    ap.add_argument(
        "job",
        help="Job ID (e.g. batches/abc123), jobs JSON file, or job name stem",
    )
    ap.add_argument(
        "--watch", action="store_true",
        help="Keep polling every --interval seconds until job finishes",
    )
    ap.add_argument(
        "--interval", type=int, default=60,
        help="Poll interval in seconds (default: 60)",
    )
    args = ap.parse_args()

    try:
        from google import genai
    except ImportError:
        sys.exit("ERROR: google-genai not installed. pip install google-genai")

    job_id = resolve_job_id(args.job)
    client = genai.Client(api_key=load_api_key())

    if not args.watch:
        fetch_and_print(client, job_id)
        return

    terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}
    while True:
        state = fetch_and_print(client, job_id)
        if state in terminal:
            if state == "JOB_STATE_SUCCEEDED":
                print("\nJob complete — run: python batch_download.py <job_file>")
            break
        print(f"  (next check in {args.interval}s — Ctrl-C to stop)")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()