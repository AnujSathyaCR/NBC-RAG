"""
batch_download.py — Download Gemini Batch results and write per-page Markdown.

Usage:
    python batch_download.py batch/jobs/nbc_vol1.json
    python batch_download.py batch/jobs/nbc_vol1.json --pages-dir output/pages
    python batch_download.py batch/jobs/nbc_vol1.json --rebuild   # overwrite existing .md

What it does:
  1. Loads job metadata to find the output file URI.
  2. Downloads all result lines from the batch output.
  3. Matches each response to its page number via custom_id.
  4. Runs normalize_output() on the text (identical to extract.py).
  5. Writes output/pages/page_NNN.md  (skip if exists, unless --rebuild).
  6. Logs success/failure to batch/results/success.log and failures.log.
  7. Writes batch/results/usage.json with per-page and aggregate token counts + cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from batch_common import (
    ROOT,
    PAGES_DIR,
    RESULTS_DIR,
    load_api_key,
    load_job,
    log_failure,
    log_success,
    normalize_output,
    cost_for_model,
)

# ---------------------------------------------------------------------------
# Known finish-reasons that are not retryable / produce no text
# ---------------------------------------------------------------------------
_EMPTY_FINISH_REASONS = {"MAX_TOKENS", "SAFETY", "RECITATION", "OTHER", "BLOCKLIST",
                          "PROHIBITED_CONTENT", "SPII"}
_RETRYABLE_FINISH_REASONS = {"STOP"}  # normal completion


# ---------------------------------------------------------------------------
# Result parsing helpers
# ---------------------------------------------------------------------------

def parse_page_key(key: str) -> int | None:
    """Extract page number from key like 'page_042'."""
    try:
        return int(key.split("_")[1])
    except (IndexError, ValueError):
        return None


def extract_text_from_response(response: dict) -> tuple[str, str]:
    """Return (text, finish_reason) from a single Gemini response dict."""
    candidates = response.get("candidates") or []
    if not candidates:
        # Check for prompt feedback / blocking at request level
        pf = response.get("promptFeedback") or {}
        reason = pf.get("blockReason") or "NO_CANDIDATES"
        return "", reason

    cand = candidates[0]
    finish_reason = cand.get("finishReason", "UNKNOWN")
    parts = (cand.get("content") or {}).get("parts") or []
    text  = "".join(p.get("text", "") for p in parts)
    return text, finish_reason


def extract_usage(response: dict) -> dict[str, int]:
    """Pull token counts from usageMetadata."""
    meta = response.get("usageMetadata") or {}
    return {
        "prompt_tokens":  meta.get("promptTokenCount",          0),
        "output_tokens":  meta.get("candidatesTokenCount",      0),
        "thought_tokens": meta.get("thoughtsTokenCount",        0),
        "total_tokens":   meta.get("totalTokenCount",           0),
    }


# ---------------------------------------------------------------------------
# Download via SDK
# ---------------------------------------------------------------------------

def download_results(job: dict) -> list[dict]:
    """Fetch all result lines for a completed batch job.

    Returns a list of raw response dicts as returned by the Batch API,
    each still containing the original ``custom_id``.
    """
    try:
        from google import genai
    except ImportError:
        sys.exit("ERROR: google-genai not installed. pip install google-genai")

    client = genai.Client(api_key=load_api_key())
    job_id = job["job_id"]

    print(f"Fetching results for job: {job_id}")

    # Poll until succeeded (in case caller runs download right after submit)
    terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}
    while True:
        batch_job = client.batches.get(name=job_id)
        state = str(batch_job.state.name)  # gives clean "JOB_STATE_SUCCEEDED"
        print(f"  State: {state}")
        if state in terminal:
            break
        print("  Job still running — waiting 30 s…")
        time.sleep(30)

    if state != "JOB_STATE_SUCCEEDED":  # already stripped above
        sys.exit(f"ERROR: Job ended with state={state}. Cannot download results.")

    # Get the output file name from the job's dest field
    dest      = getattr(batch_job, "dest", None)
    file_name = getattr(dest, "file_name", None) if dest else None
    if not file_name:
        sys.exit("ERROR: Could not find output file_name in completed job. "
                 f"dest={dest}")

    print(f"  Output file: {file_name}")
    print("  Downloading result JSONL…")

    # Download the output JSONL via the Files API
    import tempfile, os
    file_content = client.files.download(file=file_name)

    # file_content is bytes; split into lines and parse
    results: list[dict] = []
    raw_bytes = bytes(file_content) if not isinstance(file_content, bytes) else file_content
    for lineno, line in enumerate(raw_bytes.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  [WARN] line {lineno}: JSON parse error — {e}")

    print(f"  Downloaded {len(results)} result records.")
    return results


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_results(
    results: list[dict],
    pages_dir: Path,
    model: str,
    rebuild: bool,
) -> dict:
    """Write .md files and accumulate usage stats.

    Returns a usage summary dict.
    """
    costs = cost_for_model(model)

    agg = {
        "prompt_tokens":  0,
        "output_tokens":  0,
        "thought_tokens": 0,
        "total_tokens":   0,
        "estimated_cost_usd": 0.0,
    }
    per_page: list[dict] = []

    succeeded = 0
    skipped   = 0
    failed    = 0

    for record in results:
        custom_id = record.get("key") or record.get("custom_id", "")
        page_num  = parse_page_key(custom_id)

        if page_num is None:
            print(f"  [WARN] unrecognised custom_id: {custom_id!r} — skipping")
            continue

        # --- Response / error block ---
        response = record.get("response") or {}
        error    = record.get("error")

        if error:
            reason = error.get("message") or str(error)
            print(f"  [FAIL] page {page_num:04d}: API error — {reason[:120]}")
            log_failure(page_num, f"API error: {reason[:200]}")
            failed += 1
            continue

        raw_text, finish_reason = extract_text_from_response(response)
        usage = extract_usage(response)

        # Cost for this page
        page_cost = (
            usage["prompt_tokens"]  / 1000 * costs["input"]
            + usage["output_tokens"] / 1000 * costs["output"]
            + usage["thought_tokens"]/ 1000 * costs["thought"]
        )

        per_page.append({
            "page": page_num,
            "finish_reason": finish_reason,
            **usage,
            "estimated_cost_usd": round(page_cost, 6),
        })

        # Aggregate
        for k in ("prompt_tokens", "output_tokens", "thought_tokens", "total_tokens"):
            agg[k] += usage[k]
        agg["estimated_cost_usd"] += page_cost

        # --- Check for unusable outputs ---
        if finish_reason in _EMPTY_FINISH_REASONS or not raw_text.strip():
            reason_msg = f"finish_reason={finish_reason}, empty={not raw_text.strip()}"
            print(f"  [FAIL] page {page_num:04d}: {reason_msg}")
            log_failure(page_num, reason_msg)
            failed += 1
            continue

        # --- Normalise and write ---
        md_path = pages_dir / f"page_{page_num:03d}.md"
        if md_path.exists() and not rebuild:
            skipped += 1
            continue

        text = normalize_output(raw_text)
        md_path.write_text(text + "\n", encoding="utf-8")
        log_success(page_num, len(text))
        succeeded += 1
        print(f"  [OK]   page {page_num:04d}  finish={finish_reason}  "
              f"{len(text):,} chars  tokens={usage['total_tokens']}")

    print(f"\nResults: {succeeded} written, {skipped} skipped, {failed} failed")

    agg["estimated_cost_usd"] = round(agg["estimated_cost_usd"], 4)
    return {
        "model":        model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals":       agg,
        "per_page":     per_page,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download Gemini Batch results and write Markdown files."
    )
    ap.add_argument("job_file", help="Path to batch/jobs/<name>.json")
    ap.add_argument("--pages-dir", default=None,
                    help="Output directory for .md files (default: output/pages)")
    ap.add_argument("--usage-out", default=None,
                    help="Path for usage JSON (default: batch/results/usage_<name>.json)")
    ap.add_argument("--rebuild", action="store_true",
                    help="Overwrite existing .md files")
    args = ap.parse_args()

    job_file = Path(args.job_file)
    if not job_file.exists():
        sys.exit(f"ERROR: job file not found: {job_file}")

    job = load_job(job_file)
    model = job.get("model", "gemini-2.5-pro")

    pages_dir = Path(args.pages_dir) if args.pages_dir else PAGES_DIR
    if not pages_dir.is_absolute():
        pages_dir = ROOT / pages_dir
    pages_dir.mkdir(parents=True, exist_ok=True)

    usage_out = (
        Path(args.usage_out)
        if args.usage_out
        else RESULTS_DIR / f"usage_{job_file.stem}.json"
    )

    print(f"Job:       {job['job_id']}")
    print(f"Model:     {model}")
    print(f"Pages dir: {pages_dir}")
    print(f"Rebuild:   {args.rebuild}")
    print()

    results = download_results(job)
    usage   = process_results(results, pages_dir, model, args.rebuild)

    usage_out.write_text(json.dumps(usage, indent=2), encoding="utf-8")
    print(f"\nUsage written to {usage_out}")

    t = usage["totals"]
    print(f"\n{'─'*40}")
    print(f"  Prompt tokens : {t['prompt_tokens']:>12,}")
    print(f"  Output tokens : {t['output_tokens']:>12,}")
    print(f"  Thought tokens: {t['thought_tokens']:>12,}")
    print(f"  Total tokens  : {t['total_tokens']:>12,}")
    print(f"  Est. cost     : ${t['estimated_cost_usd']:.4f} USD")
    print(f"{'─'*40}")

    # Surface pages that still need a retry
    failures_log = RESULTS_DIR / "failures.log"
    if failures_log.exists():
        failed_pages = [
            ln.split("\t")[0].replace("page_", "")
            for ln in failures_log.read_text(encoding="utf-8").splitlines()
            if "\tFAIL\t" in ln
        ]
        if failed_pages:
            print(f"\nFailed pages ({len(failed_pages)}): {failed_pages[:20]}")
            print(
                "Re-run failed pages with:\n"
                f"  python batch_build.py --pages {' '.join(failed_pages[:20])} "
                f"--output {job_file.stem}_retry\n"
                f"  python batch_submit.py batch/requests/{job_file.stem}_retry.jsonl"
            )


if __name__ == "__main__":
    main()