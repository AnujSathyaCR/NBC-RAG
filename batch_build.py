"""
batch_build.py — Build a JSONL batch request file from page PDFs.

Usage:
    # Contiguous range
    python batch_build.py --start 1 --end 1226 --output nbc_vol1

    # Specific pages (e.g. retry failures)
    python batch_build.py --pages 588 592 619 --output nbc_vol1_retry

    # Custom prompt / model
    python batch_build.py --start 1 --end 100 \
        --model gemini-2.5-pro \
        --prompt extract_nbc_pages.md \
        --output nbc_vol1

    # Custom pages dir (e.g. amendment GOs)
    python batch_build.py --start 1 --end 50 \
        --pages-dir output/amendments/pages/go161_2025 \
        --prompt extract_amendment_page.md \
        --output go161_2025

Output:
    batch/requests/<output>.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from batch_common import (
    ROOT,
    PAGES_DIR,
    REQUESTS_DIR,
    build_gemini_request,
    load_prompt,
)


def resolve_pages_dir(cli_value: str | None) -> Path:
    if cli_value:
        p = Path(cli_value)
        return p if p.is_absolute() else ROOT / p
    return PAGES_DIR


def collect_targets(args: argparse.Namespace, pages_dir: Path) -> list[int]:
    """Return sorted list of page numbers to include in the batch."""
    if args.pages:
        return sorted(set(args.pages))
    if args.start is not None and args.end is not None:
        return list(range(args.start, args.end + 1))
    # Default: all pages found in pages_dir
    found = sorted(
        int(p.stem.split("_")[1])
        for p in pages_dir.glob("page_*.pdf")
        if p.stem.split("_")[1].isdigit()
    )
    if not found:
        sys.exit(f"ERROR: No page_NNN.pdf files found in {pages_dir}")
    return found


def build_jsonl(
    targets: list[int],
    pages_dir: Path,
    prompt: str,
    model: str,
    output_path: Path,
    skip_missing: bool = False,
) -> tuple[int, list[int]]:
    """Write JSONL to output_path.  Returns (written_count, missing_pages)."""
    missing: list[int] = []
    written = 0

    with open(output_path, "w", encoding="utf-8") as fh:
        for page_num in targets:
            pdf_path = pages_dir / f"page_{page_num:03d}.pdf"
            if not pdf_path.exists():
                missing.append(page_num)
                if skip_missing:
                    print(f"  [SKIP] page {page_num}: PDF not found", flush=True)
                    continue
                else:
                    fh.close()
                    output_path.unlink(missing_ok=True)
                    sys.exit(
                        f"ERROR: page {page_num}: {pdf_path} not found.\n"
                        "Use --skip-missing to skip absent pages."
                    )

            pdf_bytes = pdf_path.read_bytes()
            record = build_gemini_request(
                page_num=page_num,
                pdf_bytes=pdf_bytes,
                prompt=prompt,
                model=model,
            )
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(f"  queued page {page_num:04d}  ({len(pdf_bytes):,} bytes)", flush=True)

    return written, missing


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a JSONL batch request file for Gemini Batch API."
    )

    # Page selection (--pages and --start are mutually exclusive)
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--start", type=int, help="First page number (use with --end)")
    sel.add_argument("--pages", nargs="+", type=int, metavar="PAGE",
                     help="Explicit page numbers e.g. --pages 588 592 619")

    ap.add_argument("--end", type=int, help="Last page number (use with --start)")
    ap.add_argument("--model", default="gemini-2.5-pro", help="Gemini model name")
    ap.add_argument("--prompt", default=None,
                    help="Path to prompt .md file (default: extract_nbc_pages.md)")
    ap.add_argument("--pages-dir", default=None,
                    help="Directory containing page_NNN.pdf files")
    ap.add_argument("--output", default="batch",
                    help="Base name for the output JSONL (default: batch)")
    ap.add_argument("--skip-missing", action="store_true",
                    help="Silently skip pages whose PDF is absent")

    args = ap.parse_args()

    if args.start is not None and args.end is None:
        ap.error("--start requires --end")
    if args.end is not None and args.start is None:
        ap.error("--end requires --start")

    pages_dir  = resolve_pages_dir(args.pages_dir)
    if not pages_dir.exists():
        sys.exit(f"ERROR: pages directory not found: {pages_dir}")

    targets     = collect_targets(args, pages_dir)
    prompt      = load_prompt(args.prompt)
    output_path = REQUESTS_DIR / f"{args.output}.jsonl"

    print(f"Model:      {args.model}")
    print(f"Prompt:     {args.prompt or 'extract_nbc_pages.md'} ({len(prompt):,} chars)")
    print(f"Pages dir:  {pages_dir}")
    print(f"Pages:      {len(targets)}  ({targets[0]}–{targets[-1]})")
    print(f"Output:     {output_path}")
    print()

    written, missing = build_jsonl(
        targets=targets,
        pages_dir=pages_dir,
        prompt=prompt,
        model=args.model,
        output_path=output_path,
        skip_missing=args.skip_missing,
    )

    print()
    if missing:
        print(f"WARNING: {len(missing)} PDFs not found and skipped: {missing[:20]}")
    print(f"Done. {written} requests written to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1_048_576:.1f} MB")


if __name__ == "__main__":
    main()