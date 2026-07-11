"""
combine_nbc.py
==============
Reads every extracted page markdown file from output/pages/page_XXXX.md,
parses the YAML frontmatter and markdown body, and writes a single
consolidated output/nbc_full.json.

Each page becomes one JSON object. No content is lost; the markdown body
is preserved verbatim alongside all structured frontmatter fields.

Usage
-----
    python combine_nbc.py
    python combine_nbc.py --pages-dir output/pages --out output/nbc_full.json

Dependencies
------------
    PyYAML (pip install pyyaml)
    All other imports are stdlib.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FigureClauseEntry:
    figure: str
    parent_clause: str | None

@dataclass
class FigureMetadataEntry:
    id: str
    caption: str | None

@dataclass
class FigureAssetEntry:
    figure: str
    image_index: int

@dataclass
class PageRecord:
    # ---- file provenance ----
    source_file: str          # e.g. "page_1216.md"
    seq_num: int              # file sequence number (1-1226), always unique
                              # distinct from 'page' which resets per NBC Part

    # ---- YAML frontmatter ----
    page: int                 # NBC printed page number (resets per Part — NOT unique)
    kind: list[str]

    continues_from_prev: bool
    continues_to_next: bool

    parts_on_page: list[str]
    sections_on_page: list[str]
    clauses_on_page: list[str]

    figures_on_page: list[str]
    figure_clause_map: list[FigureClauseEntry]
    figure_metadata: list[FigureMetadataEntry]
    figure_assets: list[FigureAssetEntry]

    xrefs: list[str]
    defined_terms: list[str]

    has_table: bool
    has_figure: bool
    has_drawing: bool

    notes: str | None

    # ---- raw markdown body (everything after frontmatter) ----
    body: str

    # ---- any frontmatter keys not in the schema above ----
    extra_fields: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """
    Split a markdown string into (frontmatter_dict, body_str).
    Returns ({}, raw) if no frontmatter block is found.
    """
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        log.warning("YAML parse error: %s", exc)
        fm = {}
    body = raw[m.end():]
    return fm, body


def _str_list(value: Any, field_name: str, page_hint: Any) -> list[str]:
    """Coerce a frontmatter value to list[str], tolerating None / scalar."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    log.debug("page %s: field '%s' had unexpected type %s", page_hint, field_name, type(value))
    return [str(value)]


def _bool_field(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return default


def _kind_list(value: Any) -> list[str]:
    """'kind' may be a string or a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _parse_figure_clause_map(raw: Any) -> list[FigureClauseEntry]:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append(FigureClauseEntry(
                figure=str(item.get("figure", "")),
                parent_clause=item.get("parent_clause"),
            ))
    return out


def _parse_figure_metadata(raw: Any) -> list[FigureMetadataEntry]:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append(FigureMetadataEntry(
                id=str(item.get("id", "")),
                caption=item.get("caption"),
            ))
    return out


def _parse_figure_assets(raw: Any) -> list[FigureAssetEntry]:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, dict):
            try:
                idx = int(item.get("image_index", 0))
            except (TypeError, ValueError):
                idx = 0
            out.append(FigureAssetEntry(
                figure=str(item.get("figure", "")),
                image_index=idx,
            ))
    return out


# Known frontmatter keys consumed into typed fields.
_KNOWN_KEYS = {
    "page", "kind", "continues_from_prev", "continues_to_next",
    "parts_on_page", "sections_on_page", "clauses_on_page",
    "figures_on_page", "figure_clause_map", "figure_metadata", "figure_assets",
    "xrefs", "defined_terms",
    "has_table", "has_figure", "has_drawing", "notes",
}


# Prompt-leak detection: strings that only appear when Gemini echoed the prompt
_PROMPT_LEAK_MARKERS = (
    "Extract verbatim",
    "YAML FRONTMATTER",
    "## BODY RULES",
    "## HIERARCHY",
)


def parse_page(path: Path) -> PageRecord:
    raw = path.read_text(encoding="utf-8")
    fm, body = split_frontmatter(raw)

    # seq_num: always derived from filename (page_0042.md → 42); always unique
    m_seq = re.search(r"(\d+)", path.stem)
    seq_num = int(m_seq.group(1)) if m_seq else 0

    # page number: NBC printed page (not unique — resets per Part)
    page_num = fm.get("page")
    if page_num is None:
        page_num = seq_num  # fall back to seq_num, not 0
        log.warning("%s: 'page' missing from frontmatter, using seq_num %d", path.name, seq_num)
    else:
        try:
            page_num = int(page_num)
        except (TypeError, ValueError):
            page_num = seq_num

    # Detect prompt leak: Gemini echoed the extraction prompt as body content
    body_stripped = body.strip()
    if any(marker in body_stripped for marker in _PROMPT_LEAK_MARKERS):
        log.warning("%s: prompt leak detected — body contains extraction prompt text; clearing body", path.name)
        body_stripped = ""

    extra = {k: v for k, v in fm.items() if k not in _KNOWN_KEYS}

    return PageRecord(
        source_file=path.name,
        seq_num=seq_num,
        page=page_num,
        kind=_kind_list(fm.get("kind")),
        continues_from_prev=_bool_field(fm.get("continues_from_prev")),
        continues_to_next=_bool_field(fm.get("continues_to_next")),
        parts_on_page=_str_list(fm.get("parts_on_page"), "parts_on_page", page_num),
        sections_on_page=_str_list(fm.get("sections_on_page"), "sections_on_page", page_num),
        clauses_on_page=_str_list(fm.get("clauses_on_page"), "clauses_on_page", page_num),
        figures_on_page=_str_list(fm.get("figures_on_page"), "figures_on_page", page_num),
        figure_clause_map=_parse_figure_clause_map(fm.get("figure_clause_map")),
        figure_metadata=_parse_figure_metadata(fm.get("figure_metadata")),
        figure_assets=_parse_figure_assets(fm.get("figure_assets")),
        xrefs=_str_list(fm.get("xrefs"), "xrefs", page_num),
        defined_terms=_str_list(fm.get("defined_terms"), "defined_terms", page_num),
        has_table=_bool_field(fm.get("has_table")),
        has_figure=_bool_field(fm.get("has_figure")),
        has_drawing=_bool_field(fm.get("has_drawing")),
        notes=fm.get("notes"),
        body=body_stripped,
        extra_fields=extra,
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def page_record_to_dict(rec: PageRecord) -> dict[str, Any]:
    """Convert PageRecord to a plain dict suitable for JSON serialisation."""
    d = asdict(rec)
    # asdict handles nested dataclasses recursively — nothing extra needed.
    return d


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def collect_page_files(pages_dir: Path) -> list[Path]:
    """Return page_XXXX.md files sorted by page number (filename order)."""
    files = sorted(pages_dir.glob("page_*.md"))
    if not files:
        log.error("No page_*.md files found in %s", pages_dir)
        sys.exit(1)
    return files


def run(pages_dir: Path, out_path: Path) -> None:
    log.info("Scanning %s for page files …", pages_dir)
    page_files = collect_page_files(pages_dir)
    log.info("Found %d page files", len(page_files))

    pages: list[dict[str, Any]] = []
    errors: list[str] = []

    for path in page_files:
        try:
            rec = parse_page(path)
            pages.append(page_record_to_dict(rec))
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to parse %s: %s", path.name, exc)
            errors.append(path.name)

    if errors:
        log.warning("%d pages failed to parse: %s", len(errors), errors)

    # Sort by seq_num (file sequence 1-1226) — always unique, unlike 'page' which resets per NBC Part
    pages.sort(key=lambda p: p.get("seq_num", 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"pages": pages}, fh, ensure_ascii=False, indent=2)

    # Summary
    empty_body   = sum(1 for p in pages if not p.get("body", "").strip())
    prompt_leaks = sum(1 for p in pages if p.get("body", "") == "" and p.get("source_file", "") not in errors)
    log.info("─" * 50)
    log.info("Page files found:    %d", len(page_files))
    log.info("Parsed successfully: %d", len(pages))
    log.info("Parse errors:        %d", len(errors))
    log.info("Empty body pages:    %d", empty_body)
    log.info("Output:              %s", out_path)
    log.info("─" * 50)
    if errors:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Combine NBC 2016 extracted page markdown files into nbc_full.json",
    )
    p.add_argument(
        "--pages-dir",
        type=Path,
        default=Path("output/pages"),
        help="Directory containing page_XXXX.md files (default: output/pages)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("output/nbc_full.json"),
        help="Output JSON path (default: output/nbc_full.json)",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(pages_dir=args.pages_dir, out_path=args.out)