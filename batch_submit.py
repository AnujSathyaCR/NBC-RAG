"""
batch_submit.py — Submit a JSONL file as a Gemini Batch job.

TWO MODES:

  --file   (recommended, handles any size)
    Uploads the JSONL via the Files API, then passes the resource name
    (e.g. "files/abc123") to batches.create().  No GCS required.

    python batch_submit.py batch/requests/nbc_vol1.jsonl --file

  --inline  (small batches only, < 20 MB)
    Reads the JSONL and passes requests as a plain Python list.
    Your 70 MB file will NOT work here — use --file instead.

    python batch_submit.py batch/requests/small.jsonl --inline

JSONL FORMAT expected by the Gemini Batch API:
  {"key": "page_001", "request": {"contents": [...], "generation_config": {...}}}

Output:
    batch/jobs/<job_name>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

from batch_common import (
    JOBS_DIR,
    load_api_key,
    save_job,
)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_requests_as_inline(path: Path) -> list[dict]:
    """
    Convert our JSONL (custom_id / request) into the inline format
    the SDK expects: a list of dicts with 'contents' and optional 'config'.

    Note: inline mode loses the custom_id mapping, so pages are identified
    by their position in the list.  Use --file mode to preserve custom_id.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"ERROR: malformed JSON on line {lineno}: {e}")
            req = obj.get("request", obj)   # unwrap if wrapped
            entry: dict = {"contents": req["contents"]}
            if "generation_config" in req:
                entry["config"] = req["generation_config"]
            records.append(entry)
    return records


def rewrite_jsonl_keys(src: Path, dst: Path) -> int:
    """
    The Gemini Batch API JSONL format uses "key" (not "custom_id") and
    "request" as the top-level fields.  Rewrite our internal format if needed.

    Returns number of lines written.
    """
    written = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Rename custom_id → key if present
            if "custom_id" in obj and "key" not in obj:
                obj["key"] = obj.pop("custom_id")
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Path A: Files API upload → resource name
# ---------------------------------------------------------------------------

def submit_via_file(jsonl_path: Path, model: str, display_name: str) -> dict:
    """
    Upload the JSONL via the Files API, then create a batch job using
    the resource name (e.g. "files/abc123") as src.

    This is the correct approach for files > 20 MB.
    """
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError:
        sys.exit("ERROR: google-genai not installed.  pip install google-genai")

    api_key = load_api_key()
    client  = genai.Client(api_key=api_key)

    # Rewrite custom_id → key into a temp file
    tmp_path = jsonl_path.with_suffix(".upload.jsonl")
    try:
        line_count = rewrite_jsonl_keys(jsonl_path, tmp_path)
        size_mb    = tmp_path.stat().st_size / 1_048_576
        print(f"Uploading {jsonl_path.name} ({size_mb:.1f} MB) via Files API…")

        uploaded = client.files.upload(
            file=str(tmp_path),
            config=gtypes.UploadFileConfig(
                display_name=display_name,
                mime_type="application/jsonl",
            ),
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    # The SDK wants the resource name ("files/abc123"), NOT the full https:// URI
    file_name = uploaded.name   # e.g. "files/g63n6myu1sgd"
    print(f"Uploaded file name : {file_name}")
    print(f"Uploaded file URI  : {uploaded.uri}")

    print("Creating batch job…")
    batch_job = client.batches.create(
        model=model,
        src=file_name,          # ← resource name, not the https URI
        config=gtypes.CreateBatchJobConfig(display_name=display_name),
    )

    return _build_metadata(batch_job, model, display_name, jsonl_path,
                           line_count, src=file_name)


# ---------------------------------------------------------------------------
# Path B: inline list (< 20 MB)
# ---------------------------------------------------------------------------

def submit_inline(jsonl_path: Path, model: str, display_name: str) -> dict:
    """Pass requests as a plain Python list — no file upload needed."""
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError:
        sys.exit("ERROR: google-genai not installed.  pip install google-genai")

    size_mb = jsonl_path.stat().st_size / 1_048_576
    if size_mb > 20:
        print(
            f"WARNING: file is {size_mb:.1f} MB — inline mode is limited to ~20 MB.\n"
            "         Use --file mode instead if this fails."
        )

    api_key  = load_api_key()
    client   = genai.Client(api_key=api_key)
    requests = read_requests_as_inline(jsonl_path)

    print(f"Submitting {len(requests)} inline requests…")
    batch_job = client.batches.create(
        model=model,
        src=requests,           # plain list of dicts with 'contents'
        config={"display_name": display_name},
    )

    return _build_metadata(batch_job, model, display_name, jsonl_path,
                           len(requests), src="inline")


# ---------------------------------------------------------------------------
# Shared metadata builder
# ---------------------------------------------------------------------------

def _build_metadata(
    batch_job,
    model: str,
    display_name: str,
    jsonl_path: Path,
    line_count: int,
    src: str,
) -> dict:
    job_id = batch_job.name
    state  = str(batch_job.state)
    print(f"Job created : {job_id}")
    print(f"State       : {state}")
    return {
        "job_id":          job_id,
        "display_name":    display_name,
        "model":           model,
        "jsonl_path":      str(jsonl_path),
        "request_count":   line_count,
        "submission_time": datetime.now(timezone.utc).isoformat(),
        "state":           state,
        "src":             src,
    }


# ---------------------------------------------------------------------------
# One-shot JSONL repair (fixes existing files built before this patch)
# ---------------------------------------------------------------------------

def repair_model_prefix(jsonl_path: Path) -> None:
    """Rewrite model names in an existing JSONL from short to models/ form.

    Safe to run multiple times (idempotent).
    """
    tmp = jsonl_path.with_suffix(".repair.jsonl")
    fixed = 0
    with open(jsonl_path, encoding="utf-8") as fin, \
         open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            req = obj.get("request", {})
            m = req.get("model", "")
            if m and not m.startswith("models/"):
                req["model"] = f"models/{m}"
                fixed += 1
            # Also rename custom_id → key
            if "custom_id" in obj and "key" not in obj:
                obj["key"] = obj.pop("custom_id")
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(jsonl_path)
    print(f"Repaired {fixed} model prefixes in {jsonl_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Submit a JSONL batch request file to Gemini Batch API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("jsonl", help="Path to the JSONL request file")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--job-name", default=None,
                    help="Base name for job metadata file (default: JSONL stem)")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--file", action="store_true",
        help="Upload via Files API then submit (recommended, handles any size)",
    )
    mode.add_argument(
        "--inline", action="store_true",
        help="Pass requests inline as a list (small batches < 20 MB only)",
    )
    mode.add_argument(
        "--repair", action="store_true",
        help="Fix model prefix and key field in an existing JSONL (no submission)",
    )

    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        sys.exit(f"ERROR: JSONL file not found: {jsonl_path}")

    job_name = args.job_name or jsonl_path.stem
    job_file = JOBS_DIR / f"{job_name}.json"

    if args.repair:
        repair_model_prefix(jsonl_path)
        return

    if args.file:
        metadata = submit_via_file(jsonl_path, args.model, job_name)
    else:
        metadata = submit_inline(jsonl_path, args.model, job_name)

    save_job(job_file, metadata)
    print(f"\nJob metadata saved to: {job_file}")
    print(json.dumps(metadata, indent=2))
    print(
        f"\nNext steps:\n"
        f"  python batch_status.py {metadata['job_id']} --watch\n"
        f"  python batch_download.py {job_file}"
    )


if __name__ == "__main__":
    main()