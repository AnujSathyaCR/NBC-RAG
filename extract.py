"""
Per-page extraction of PDFs into Markdown using Gemini 3 Pro.

Writes one `<stem>.md` next to each `<stem>.pdf` found (directly, non-recursive)
in --pages-dir. Works for a single-document directory (page_001.pdf, ...) as
well as a flattened multi-document directory (e.g. 04-07-2025_page_001.pdf, ...).

Usage:
    python extract.py                # extract every PDF (skips already-done)
    python extract.py 1 6 31 81 101  # extract PDFs whose filename ends _page_NNN.pdf for these N
    python extract.py --model gemini-2.5-pro 1 6
    python extract.py --rebuild      # re-extract even if .md exists
    python extract.py --pages-dir output/gosplit --prompt extract_amendment_page.md
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
os.environ["PYTHONUTF8"] = "1"  # ensure UTF-8 output on Windows
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent
PAGES_DIR = ROOT / "output" / "pages"
PROMPT_PATH = ROOT / "extract_page.md"
DEFAULT_MODEL = "gemini-2.5-pro"
MAX_RETRY_WAIT_S = 60  # If server suggests longer (e.g. daily-quota reset), give up.


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8", errors="ignore")


def load_api_key() -> str:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit(
            "ERROR: GEMINI_API_KEY not found. Add it to .env:\n"
            "  echo 'GEMINI_API_KEY=your_key' > .env"
        )
    return key


def normalize_output(raw: str) -> str:
    """Coerce the model's reply into `---\\nfrontmatter\\n---\\n\\nbody`.

    Models sometimes wrap the whole reply in a ```markdown fence, or wrap only
    the YAML frontmatter in a ```yaml fence. The latter is corrosive: stripping
    the opening fence leaves bare YAML with no `---` delimiters and a stray
    closing ``` between frontmatter and body, which breaks combine.py's
    frontmatter parser. Handle both.
    """
    text = (raw or "").strip()
    # Drop a leading fence line (``` / ```yaml / ```markdown) ...
    if text.startswith("```"):
        nl = text.find("\n")
        text = text[nl + 1:] if nl != -1 else ""
        # ... and a matching trailing fence if the whole reply was fenced.
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    # Opening `---` missing: the reply starts with bare YAML. Two variants seen:
    #   (a) model fenced only the YAML -> a stray ``` closes the frontmatter;
    #   (b) model just dropped the opening `---` but kept the closing `---`.
    # Find whichever delimiter closes the frontmatter and rebuild proper fences.
    if not text.startswith("---") and re.match(r"^[A-Za-z_][\w ]*:", text):
        m = re.search(r"^(?:---|```)\s*$", text, re.MULTILINE)
        if m:
            fm = text[: m.start()].strip()
            body = text[m.end():].strip()
            text = f"---\n{fm}\n---\n\n{body}"
    return text.strip()


PAGE_NUM_RE = re.compile(r"page_(\d+)\.pdf$", re.IGNORECASE)


def page_number(pdf_path: Path) -> int | None:
    m = PAGE_NUM_RE.search(pdf_path.name)
    return int(m.group(1)) if m else None


def extract_page(client: genai.Client, model: str, prompt: str, pdf_path: Path) -> str:
    pdf_bytes = pdf_path.read_bytes()

    last_err: Exception | None = None
    for attempt in range(6):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=16384,
                ),
            )
            text = normalize_output(response.text or "")
            if not text:
                raise RuntimeError("empty response")
            return text
        except Exception as e:
            last_err = e
            err_text = str(e)
            # Honor server-suggested retryDelay when present (handles "8s", "8.686s").
            wait = 10 + 10 * attempt  # 10, 20, 30, 40, 50, 60
            m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", err_text)
            if m:
                suggested = float(m.group(1)) + 2
                if suggested > MAX_RETRY_WAIT_S:
                    # Server is telling us to wait hours (daily quota / spend cap).
                    # No point looping — bail so the user can fix it.
                    raise RuntimeError(
                        f"{pdf_path.name}: server retryDelay={suggested:.0f}s exceeds "
                        f"MAX_RETRY_WAIT_S={MAX_RETRY_WAIT_S}s — likely daily quota / "
                        f"spend cap. Aborting. Raw: {err_text[:200]}"
                    ) from e
                wait = max(wait, int(suggested))
            print(
                f"  {pdf_path.name}: attempt {attempt + 1} failed; retrying in {wait}s "
                f"({err_text[:120]})",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(f"{pdf_path.name} failed after retries: {last_err}")


def process_page(
    client: genai.Client, model: str, prompt: str, pdf_path: Path, rebuild: bool
) -> tuple[str, str]:
    md_path = pdf_path.with_suffix(".md")
    if md_path.exists() and not rebuild:
        return pdf_path.name, "skipped"
    text = extract_page(client, model, prompt, pdf_path)
    md_path.write_text(text + "\n", encoding="utf-8")
    return pdf_path.name, f"ok ({len(text)} chars)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "pages",
        nargs="*",
        type=int,
        help="Page numbers to extract, matched against trailing _page_NNN in the "
        "filename (default: all)",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rebuild", action="store_true", help="Re-extract even if .md exists")
    ap.add_argument("--workers", type=int, default=4, help="Parallel API calls")
    ap.add_argument(
        "--pages-dir",
        help="Directory of page_NNN.pdf files (default: output/pages). "
        "Use e.g. output/amendments/pages/go161_2025 for amendment GOs.",
    )
    ap.add_argument(
        "--prompt",
        help="Path to the extraction prompt (default: extract_page.md). "
        "Use extract_amendment_page.md for amendment GOs.",
    )
    args = ap.parse_args()

    # Allow targeting a different page set / prompt (e.g. amendment GOs) without
    # duplicating the retry + parallelism machinery below.
    global PAGES_DIR, PROMPT_PATH
    if args.pages_dir:
        PAGES_DIR = (ROOT / args.pages_dir).resolve()
    if args.prompt:
        PROMPT_PATH = (ROOT / args.prompt).resolve()

    key = load_api_key()
    client = genai.Client(api_key=key)
    prompt = load_prompt()

    all_pdfs = sorted(PAGES_DIR.glob("*.pdf"))
    if args.pages:
        wanted = set(args.pages)
        targets = [p for p in all_pdfs if page_number(p) in wanted]
    else:
        targets = all_pdfs
    if not targets:
        sys.exit(f"No matching PDFs found in {PAGES_DIR}")
    print(f"Model:  {args.model}")
    print(f"Prompt: {PROMPT_PATH.name} ({len(prompt)} chars)")
    print(f"Pages:  {len(targets)} (workers={args.workers}, rebuild={args.rebuild})")

    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_page, client, args.model, prompt, p, args.rebuild): p
            for p in targets
        }
        for fut in concurrent.futures.as_completed(futures):
            p = futures[fut]
            try:
                name, status = fut.result()
                done += 1
                print(f"[{done}/{len(targets)}] {name}: {status}")
            except Exception as e:
                print(f"[ERR] {p.name}: {e}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
