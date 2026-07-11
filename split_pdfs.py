"""
Burst PDFs into per-page PDFs, mirroring the single-page layout the
extractor expects.

By default, splits the amendment Government Orders in `data/*.pdf`. For
each `data/Amendment_GO_Ms_No_<NUM>_Dated_<DD>_<MM>_<YYYY>.pdf` we write:

    output/amendments/pages/go<NUM>_<YYYY>/page_001.pdf
    output/amendments/pages/go<NUM>_<YYYY>/page_002.pdf
    ...
    output/amendments/pages/go<NUM>_<YYYY>/meta.json

`meta.json` carries the GO metadata that is authoritative from the filename
(number + date) so downstream stages don't have to trust the model for it.

With --input-dir/--output-dir, any folder of PDFs can be split generically:
each PDF's filename (stem) becomes the output subfolder name, with no
metadata parsing or meta.json (since the amendment naming pattern doesn't
apply):

    output/gosplit/<stem>/page_001.pdf
    output/gosplit/<stem>/page_002.pdf
    ...

Usage:
    python split_pdfs.py                  # split every GO (skips already-done)
    python split_pdfs.py 161 154          # split only these GO numbers
    python split_pdfs.py --rebuild        # re-split even if pages exist
    python split_pdfs.py --input-dir gopdfs --output-dir output/gosplit
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PAGES_ROOT = ROOT / "output" / "amendments" / "pages"

# Amendment_GO_Ms_No_161_Dated_15_10_2025.pdf
NAME_RE = re.compile(
    r"Amendment_GO_Ms_No_(?P<num>\d+)_Dated_(?P<d>\d{2})_(?P<m>\d{2})_(?P<y>\d{4})",
    re.IGNORECASE,
)


def parse_name(path: Path) -> dict | None:
    m = NAME_RE.search(path.stem)
    if not m:
        return None
    num, d, mo, y = m.group("num"), m.group("d"), m.group("m"), m.group("y")
    return {
        "go_number": num,
        "go_date": f"{y}-{mo}-{d}",  # ISO, sortable
        "go_id": f"go{num}_{y}",
        "source": path.name,
    }


def split_one(pdf_path: Path, pages_root: Path, rebuild: bool) -> tuple[str, int, str]:
    meta = parse_name(pdf_path)
    go_id = meta["go_id"] if meta else pdf_path.stem
    out_dir = pages_root / go_id
    existing = sorted(out_dir.glob("page_*.pdf")) if out_dir.exists() else []
    reader = PdfReader(str(pdf_path))
    n = len(reader.pages)
    if existing and len(existing) == n and not rebuild:
        return (go_id, n, "skipped")

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)
        with (out_dir / f"page_{i:03d}.pdf").open("wb") as fh:
            writer.write(fh)
    if meta is not None:
        meta["n_pages"] = n
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return (go_id, n, "ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("numbers", nargs="*", help="GO numbers to split (default: all)")
    ap.add_argument("--rebuild", action="store_true", help="Re-split even if pages exist")
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_DIR,
        help="Folder of PDFs to split (default: data/)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=PAGES_ROOT,
        help="Folder to write per-page PDFs into (default: output/amendments/pages/)",
    )
    args = ap.parse_args()

    input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {input_dir}")

    if args.numbers:
        wanted = set(args.numbers)
        pdfs = [p for p in pdfs if (m := parse_name(p)) and m["go_number"] in wanted]
        if not pdfs:
            sys.exit(f"No GOs matched numbers {sorted(wanted)}")

    print(f"Splitting {len(pdfs)} PDF(s) into {output_dir}/")
    total_pages = 0
    for p in pdfs:
        go_id, n, status = split_one(p, output_dir, args.rebuild)
        total_pages += n
        print(f"  {go_id:>16}: {n:>3} pages  {status}")
    print(f"Done — {total_pages} pages across {len(pdfs)} PDF(s).")


if __name__ == "__main__":
    main()
