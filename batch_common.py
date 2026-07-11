"""
batch_common.py — Shared utilities for the Gemini batch pipeline.

Provides:
  - build_gemini_request()   Single source of truth for request construction.
  - normalize_output()       Preserved exactly from extract.py.
  - load_prompt()            Load a prompt file.
  - load_api_key()           Load API key from .env.
  - cost_for_model()         Per-token cost lookup.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PAGES_DIR   = ROOT / "output" / "pages_v2"
BATCH_DIR   = ROOT / "batch"
REQUESTS_DIR = BATCH_DIR / "requests"
RESULTS_DIR  = BATCH_DIR / "results"
JOBS_DIR     = BATCH_DIR / "jobs"

for _d in (PAGES_DIR, REQUESTS_DIR, RESULTS_DIR, JOBS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model cost table  (USD per 1 000 tokens, as of 2025-06)
# Gemini Batch API is 50 % of online pricing.
# ---------------------------------------------------------------------------
_COST_PER_1K: dict[str, dict[str, float]] = {
    "gemini-2.5-pro": {
        "input":   0.000625,   # $1.25 / 1M  × 0.5  batch discount
        "output":  0.005,      # $10.00 / 1M × 0.5
        "thought": 0.0015,     # $3.00  / 1M × 0.5  (thinking tokens)
    },
    "gemini-2.5-flash": {
        "input":   0.0000375,
        "output":  0.000375,
        "thought": 0.0000375,
    },
}
_COST_DEFAULT = {"input": 0.0, "output": 0.0, "thought": 0.0}


def cost_for_model(model: str) -> dict[str, float]:
    """Return per-1k-token costs for *model* (falls back to zeros)."""
    key = model.lower().split("/")[-1]   # strip any "models/" prefix
    for k, v in _COST_PER_1K.items():
        if k in key:
            return v
    return _COST_DEFAULT


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit(
            "ERROR: GEMINI_API_KEY not found.\n"
            "  echo 'GEMINI_API_KEY=your_key' > .env"
        )
    return key


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------
def load_prompt(prompt_path: Path | str | None = None) -> str:
    path = Path(prompt_path) if prompt_path else (ROOT / "extract_nbc_pages.md")
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        sys.exit(f"ERROR: prompt file not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Output normaliser  (preserved verbatim from extract.py)
# ---------------------------------------------------------------------------
def normalize_output(raw: str) -> str:
    """Coerce the model's reply into ``---\\nfrontmatter\\n---\\n\\nbody``.

    Models sometimes wrap the whole reply in a ```markdown fence, or wrap only
    the YAML frontmatter in a ```yaml fence.  The latter is corrosive: stripping
    the opening fence leaves bare YAML with no ``---`` delimiters and a stray
    closing ``` between frontmatter and body, which breaks combine.py's
    frontmatter parser.  Handle both.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        nl = text.find("\n")
        text = text[nl + 1:] if nl != -1 else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    if not text.startswith("---") and re.match(r"^[A-Za-z_][\w ]*:", text):
        m = re.search(r"^(?:---|```)\s*$", text, re.MULTILINE)
        if m:
            fm = text[: m.start()].strip()
            body = text[m.end():].strip()
            text = f"---\n{fm}\n---\n\n{body}"
    return text.strip()


# ---------------------------------------------------------------------------
# Core request builder  (single source of truth)
# ---------------------------------------------------------------------------
def build_gemini_request(
    page_num: int,
    pdf_bytes: bytes,
    prompt: str,
    model: str = "gemini-2.5-pro",
    max_output_tokens: int = 65536,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Build a Gemini API request dict for one page.

    The same structure is used for:
      * Interactive (online) calls via the Python SDK.
      * Batch JSONL lines submitted to the Batch API.

    Returns a plain Python dict compatible with both paths.
    """
    import base64

    # Batch API JSONL requires "models/" prefix inside each request
    model_full = model if model.startswith("models/") else f"models/{model}"

    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")

    return {
        "key": f"page_{page_num:03d}",
        "request": {
            "model": model_full,
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "application/pdf",
                                "data": pdf_b64,
                            }
                        },
                        {"text": prompt},
                    ],
                }
            ],
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        },
    }


# ---------------------------------------------------------------------------
# Job metadata helpers
# ---------------------------------------------------------------------------
def save_job(job_file: Path, metadata: dict[str, Any]) -> None:
    job_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_job(job_file: Path) -> dict[str, Any]:
    return json.loads(job_file.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Failure / success log helpers
# ---------------------------------------------------------------------------
def log_success(page_num: int, chars: int) -> None:
    with open(RESULTS_DIR / "success.log", "a", encoding="utf-8") as f:
        f.write(f"page_{page_num:03d}\tok\t{chars} chars\n")


def log_failure(page_num: int, reason: str) -> None:
    with open(RESULTS_DIR / "failures.log", "a", encoding="utf-8") as f:
        f.write(f"page_{page_num:03d}\tFAIL\t{reason}\n")