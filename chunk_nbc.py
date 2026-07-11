"""
chunk_nbc.py
============
Reads output/nbc_full.json and produces output/chunks.jsonl — one JSON
object per line, each representing one retrieval chunk.

Chunking strategy
-----------------
Each page's markdown body is split on Markdown headings (# through ######).
A heading that matches a clause identifier starts a new chunk.
Table blocks and figure blocks that appear between clauses are attached to
the chunk immediately preceding them; if they precede all clause text they
form their own chunk with kind="table" or kind="figure".

Every chunk carries:
  - full clause path  (PART > Section > Annex > clause)
  - figures + figure assets + derived image_paths
  - tables (raw markdown)
  - xrefs and defined_terms inherited from page frontmatter
    (filtered to those mentioned in the chunk text where possible)

Metadata inheritance (NEW)
--------------------------
  - A global HierarchyStack is maintained across ALL pages, not just within
    a single page. When a page begins without its own PART/Section/clause
    heading, the stack carries forward the last known state (backward
    inheritance / continuation-page support).
  - sub_clause is extracted from enumerated list labels (a), b), iv) …).
  - figure_id and table_id are detected from headings and captions.

Image path derivation
---------------------
Given page=1216 and image_index=1:
  output/images/page_1216/page_1216_img_01.png

Usage
-----
    python chunk_nbc.py
    python chunk_nbc.py --nbc-json output/nbc_full.json --out output/chunks.jsonl

Dependencies
------------
    stdlib only (json, re, pathlib, hashlib, …)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

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
# Constants
# ---------------------------------------------------------------------------

IMAGE_PATH_TEMPLATE = "output/images/page_{page:04d}/page_{page:04d}_img_{idx:02d}.png"

# Heading regex: captures the hashes and the rest of the line
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Patterns used to identify table blocks in markdown
_TABLE_FENCE_RE = re.compile(
    r"(?:(?:\*\*Table[^\n]*\*\*\n)?"   # optional bold caption
    r"(?:<table[\s\S]*?</table>)"       # HTML table
    r"|"
    r"(?:\|[^\n]+\n(?:\|[-:| ]+\n)(?:\|[^\n]+\n)+))",  # GFM table
    re.MULTILINE,
)

# NBC clause identifier patterns (covers most variants)
_CLAUSE_HEADING_RE = re.compile(
    r"""
    (?:
        (?:PART\s+\d+)                          # PART 8
        | (?:Section\s+\d+)                     # Section 2
        | (?:Annex\s+[A-Z])                     # Annex B
        | (?:[A-Z]-\d+(?:\.\d+)*)               # B-9, B-9.2.2.1, A-3, A-3.1.2
        | (?:\d+(?:[A-Z])?\d*(?:\.\d+)+)        # 4.3.2, 5A.1, 3.1, 3.1.1.2
    )
    """,
    re.VERBOSE,
)

# Secondary sub-clause patterns: legal list items that start a new retrieval unit.
_SUBCLAUSE_RE = re.compile(
    r"^("
    r"\*{0,2}[a-z]{1,2}\)\*{0,2}"        # a)  b)  aa)  **a)**
    r"|"
    r"\*{0,2}[ivxlcdmIVXLCDM]{1,6}\)\*{0,2}"  # iv)  viii)  **iv)**
    r"|"
    r"\d{1,2}\)"                            # 1)  2)  12)
    r"|"
    r"\(\d+\)"                              # (1)  (2)
    r")\s",
    re.MULTILINE,
)

# Figure detection: "Fig. 55", "Fig. 76", "Figure 3" etc.
_FIGURE_ID_RE = re.compile(
    r"\b(Fig(?:ure)?\.?\s*\d+[A-Za-z]?)\b",
    re.IGNORECASE,
)

# Table detection: "Table 1", "Table 10", "Table A-1" etc.
_TABLE_ID_RE = re.compile(
    r"\b(Table\s+\d+[A-Za-z0-9.-]*)\b",
    re.IGNORECASE,
)

# Sub-clause label extraction (e.g. "a)" or "iv)" from heading_raw)
_SUBCLAUSE_LABEL_RE = re.compile(
    r"^(?:\*{0,2})([a-z]{1,2}|[ivxlcdmIVXLCDM]{1,6}|\d{1,2})(?:\)|\.)(?:\*{0,2})\s*",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str

    page: int

    # Hierarchy context
    part: str | None
    section: str | None
    clause_id: str | None
    clause_path: str

    # NEW: finer-grained metadata
    sub_clause: str | None
    figure_id: str | None
    table_id: str | None

    # Content
    text: str

    # Figures on this chunk's page that are associated with this clause
    figures: list[str]                    # figure identifiers
    figure_assets: list[dict[str, Any]]   # {figure, image_index}
    image_paths: list[str]                # derived paths

    # Tables extracted from the chunk text — dicts with caption + content
    tables: list[dict[str, str]]          # [{"caption": ..., "content": ...}]

    # Inherited from page
    xrefs: list[str]
    defined_terms: list[str]

    # Source file provenance
    source_file: str = ""

    # Page-level image asset metadata
    page_assets: dict[str, Any] = field(default_factory=dict)

    # Source kind hint
    kind: str = "clause"


def make_chunk_id(page: int, clause_id: str | None, index: int) -> str:
    """Deterministic chunk id: nbc_<page>_<slug>_<index>."""
    slug = re.sub(r"[^a-zA-Z0-9]", "_", clause_id or "unk").strip("_").lower()
    raw = f"nbc_{page:04d}_{slug}_{index:03d}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:6]
    return f"{raw}_{h}"


# ---------------------------------------------------------------------------
# Image path derivation
# ---------------------------------------------------------------------------

def pdf_page_from_source_file(source_file: str) -> int:
    m = re.search(r"page_(\d+)", source_file)
    return int(m.group(1)) if m else 0


def derive_image_paths(source_file: str, figure_assets: list[dict[str, Any]]) -> list[str]:
    paths = []
    pdf_page = pdf_page_from_source_file(source_file)
    for asset in figure_assets:
        idx = asset.get("image_index", 0)
        if idx:
            paths.append(IMAGE_PATH_TEMPLATE.format(page=pdf_page, idx=idx))
    return paths


# ---------------------------------------------------------------------------
# Markdown splitting
# ---------------------------------------------------------------------------

@dataclass
class _Section:
    """Internal: a heading + the body text that follows it."""
    level: int
    heading_raw: str
    body: str
    is_clause: bool


def split_body_into_sections(body: str) -> list[_Section]:
    if not body.strip():
        return []

    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        return [_Section(level=0, heading_raw="", body=body, is_clause=False)]

    sections: list[_Section] = []

    pre = body[: matches[0].start()].strip()
    if pre:
        sections.append(_Section(level=0, heading_raw="", body=pre, is_clause=False))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading_raw = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_body = body[start:end].strip()
        is_clause = bool(_CLAUSE_HEADING_RE.search(heading_raw))
        sections.append(_Section(
            level=level,
            heading_raw=heading_raw,
            body=section_body,
            is_clause=is_clause,
        ))

    return sections


def split_section_into_subclauses(sec: "_Section") -> list["_Section"]:
    body = sec.body
    if not body:
        return [sec]

    matches = list(_SUBCLAUSE_RE.finditer(body))
    if not matches:
        return [sec]

    results: list[_Section] = []
    child_level = sec.level + 1 if sec.level > 0 else 1

    preamble = body[: matches[0].start()].strip()
    if preamble:
        results.append(_Section(
            level=sec.level,
            heading_raw=sec.heading_raw,
            body=preamble,
            is_clause=sec.is_clause,
        ))

    for i, m in enumerate(matches):
        label = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sub_body = body[start:end].strip()
        results.append(_Section(
            level=child_level,
            heading_raw=label,
            body=sub_body,
            is_clause=False,
        ))

    return results if results else [sec]


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

_TABLE_CAPTION_RE = re.compile(r"^[*][*]Table\b[^\n]*[*][*]\s*$", re.MULTILINE)


def extract_tables_from_text(text: str) -> list[dict[str, str]]:
    tables: list[dict[str, str]] = []

    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        if lines[i].startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            table_content = "".join(lines[i:j]).strip()

            caption = ""
            k = i - 1
            while k >= 0 and lines[k].strip() == "":
                k -= 1
            if k >= 0:
                m = _TABLE_CAPTION_RE.match(lines[k].rstrip())
                if m:
                    caption = lines[k].strip().strip("*").strip()

            tables.append({"caption": caption, "content": table_content})
            i = j
        else:
            i += 1

    for m in re.finditer(r"<table[\s\S]*?</table>", text, re.IGNORECASE):
        table_content = m.group(0).strip()
        pre = text[max(0, m.start() - 200): m.start()]
        cap_match = _TABLE_CAPTION_RE.search(pre)
        caption = cap_match.group(0).strip().strip("*").strip() if cap_match else ""
        tables.append({"caption": caption, "content": table_content})

    return tables


# ---------------------------------------------------------------------------
# Hierarchy tracker (IMPROVED: cross-page stack + sub_clause / figure / table)
# ---------------------------------------------------------------------------

@dataclass
class HierarchyStack:
    """
    Tracks the running PART / Section / Annex / Clause context ACROSS pages.

    Unlike the old per-page HierarchyState, this object lives for the entire
    pipeline run and is passed into chunks_from_page(), so every page
    automatically inherits the last known context (backward inheritance /
    continuation-page support).

    The clause_stack allows deeper nesting to be represented:
        PART 3 > B-9 > B-9.2 > B-9.2.1
    Items are pushed when a deeper heading is seen and popped when a shallower
    (or same-level) sibling appears.
    """
    part: str | None = None
    section: str | None = None
    annex: str | None = None

    # Stack of (level, clause_id) tuples for deep nesting
    clause_stack: list[tuple[int, str]] = field(default_factory=list)

    # Current sub-clause label (e.g. "j)" — reset when a new clause heading appears)
    sub_clause: str | None = None

    @staticmethod
    def _part_number(h: str) -> str | None:
        """Extract the bare PART number token, e.g. 'PART 4' from any PART heading."""
        m = re.match(r"^PART\s+(\d+)", h, re.IGNORECASE)
        return m.group(1) if m else None

    def update(self, heading: str, heading_level: int) -> None:
        """Update hierarchy state given a new heading and its markdown level.

        KEY RULE — PART / Section repeat on continuation pages
        -------------------------------------------------------
        NBC often prints the running PART/Section header at the top of every
        physical page even when the page is a continuation of the previous one.
        If we reset clause_stack every time we see that header we destroy the
        inherited clause_id that sub-clause content needs.

        Fix: only reset clause_stack when the PART *changes* (different number),
        not when the same PART header is repeated.  Same logic for Section.
        """
        h = heading.strip()

        if re.match(r"^PART\s+\d+", h, re.IGNORECASE):
            new_part_num = self._part_number(h)
            old_part_num = self._part_number(self.part or "")
            if new_part_num != old_part_num:
                # Genuinely different PART — reset everything
                self.section = None
                self.annex = None
                self.clause_stack = []
                self.sub_clause = None
            # Always record the new (possibly more verbose) heading text
            self.part = h
            return

        if re.match(r"^Section\s+\d+", h, re.IGNORECASE):
            if h != self.section:
                # New section — reset clause stack
                self.annex = None
                self.clause_stack = []
                self.sub_clause = None
            self.section = h
            return

        if re.match(r"^Annex\s+[A-Z]", h, re.IGNORECASE):
            if h != self.annex:
                self.clause_stack = []
                self.sub_clause = None
            self.annex = h
            return

        # Look for a clause id in the heading
        m = _CLAUSE_HEADING_RE.search(h)
        if m:
            clause_id = m.group(0)
            # Pop deeper or same-level items off the stack
            while self.clause_stack and self.clause_stack[-1][0] >= heading_level:
                self.clause_stack.pop()
            self.clause_stack.append((heading_level, clause_id))
            self.sub_clause = None  # new clause resets sub-clause

    def update_subclause(self, label: str) -> None:
        """Record the current sub-clause label (e.g. 'j)' 'iv)')."""
        m = _SUBCLAUSE_LABEL_RE.match(label + " ")
        if m:
            self.sub_clause = m.group(1) + ")"
        else:
            self.sub_clause = label.strip()

    @property
    def current_clause(self) -> str | None:
        """The innermost (deepest) clause id currently on the stack."""
        return self.clause_stack[-1][1] if self.clause_stack else None

    def clause_path(self) -> str:
        parts: list[str] = []
        if self.part:
            parts.append(self.part)
        if self.section:
            parts.append(self.section)
        if self.annex:
            parts.append(self.annex)
        for _, cid in self.clause_stack:
            parts.append(cid)
        return " > ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Figure + table ID detection in text
# ---------------------------------------------------------------------------

def detect_figure_id(text: str) -> str | None:
    """Return the first figure reference found in text, normalised to 'Fig. N' form.

    Handles all-caps headings like "FIG. 76" and mixed-case "Fig. 76" / "Figure 3".
    Always returns 'Fig. <N>' so the value matches figure_assets keys from frontmatter.
    """
    m = _FIGURE_ID_RE.search(text)
    if not m:
        return None
    raw = m.group(1)
    # Normalise: extract the number and rebuild as "Fig. N"
    num_m = re.search(r"\d+[A-Za-z]?", raw)
    if num_m:
        return f"Fig. {num_m.group(0)}"
    return raw


def detect_table_id(text: str) -> str | None:
    """Return the first 'Table N' reference found in text."""
    m = _TABLE_ID_RE.search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Figure filtering: which figure assets belong to a clause
# ---------------------------------------------------------------------------

def figures_for_clause(
    clause_id: str | None,
    clause_text: str,
    page_figure_clause_map: list[dict[str, Any]],
    page_figure_assets: list[dict[str, Any]],
    page_figure_metadata: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    parent_map: dict[str, str | None] = {
        e.get("figure", ""): e.get("parent_clause")
        for e in page_figure_clause_map
    }

    tier1: list[str] = []
    tier2: list[str] = []
    tier3: list[str] = []

    for asset in page_figure_assets:
        fig_id = asset.get("figure", "")
        if not fig_id:
            continue
        parent = parent_map.get(fig_id)

        if clause_id and parent == clause_id:
            tier1.append(fig_id)
        elif fig_id.lower() in clause_text.lower():
            # Case-insensitive: "Fig. 76" matches "FIG. 76" in headings
            tier2.append(fig_id)
        elif parent is None:
            tier3.append(fig_id)

    seen: set[str] = set()
    unique_figs: list[str] = []
    for fig_id in tier1 + tier2 + tier3:
        if fig_id not in seen:
            seen.add(fig_id)
            unique_figs.append(fig_id)

    matched_assets = [a for a in page_figure_assets if a.get("figure") in seen]
    return unique_figs, matched_assets


# ---------------------------------------------------------------------------
# Core chunker  (now accepts shared HierarchyStack)
# ---------------------------------------------------------------------------

def chunks_from_page(
    page_dict: dict[str, Any],
    hierarchy: HierarchyStack,
    debug: bool = False,
) -> list[Chunk]:
    """Produce all chunks for a single page dict.

    The hierarchy stack is shared across all pages so that context from page N
    is automatically inherited by page N+1 (backward metadata inheritance /
    continuation-page support).
    """
    page_num: int = page_dict.get("page", 0)
    source_file: str = page_dict.get("source_file", "")
    body: str = page_dict.get("body", "")

    page_figure_clause_map: list[dict] = page_dict.get("figure_clause_map", [])
    page_figure_assets: list[dict] = page_dict.get("figure_assets", [])
    page_figure_metadata: list[dict] = page_dict.get("figure_metadata", [])
    page_xrefs: list[str] = page_dict.get("xrefs", [])
    page_defined_terms: list[str] = page_dict.get("defined_terms", [])
    page_assets: dict[str, Any] = page_dict.get("page_assets") or {}

    # --- record clause_id BEFORE processing this page (for debug output) ---
    clause_before = hierarchy.current_clause

    sections = split_body_into_sections(body)
    if not sections:
        return []

    # Override part/section if the page frontmatter explicitly provides them
    # (only if they are more specific than what we already have)
    fm_parts = page_dict.get("parts_on_page") or []
    fm_sections = page_dict.get("sections_on_page") or []
    if fm_parts and fm_parts[0]:
        hierarchy.part = fm_parts[0]
    if fm_sections and fm_sections[0]:
        hierarchy.section = fm_sections[0]

    # Expand sections with secondary sub-clause splits
    expanded_sections: list[_Section] = []
    for sec in sections:
        expanded_sections.extend(split_section_into_subclauses(sec))

    chunks: list[Chunk] = []
    chunk_index = 0

    for sec in expanded_sections:
        is_subclause_label = bool(_SUBCLAUSE_RE.match(sec.heading_raw + " ")) if sec.heading_raw else False

        if sec.heading_raw and not is_subclause_label:
            # Real heading — update the shared hierarchy stack
            hierarchy.update(sec.heading_raw, sec.level)
        elif is_subclause_label:
            # Sub-clause enumeration label: record it but don't reset clause
            hierarchy.update_subclause(sec.heading_raw)

        heading_line = ("#" * sec.level + " " + sec.heading_raw).strip() if sec.heading_raw else ""
        full_text = (heading_line + "\n\n" + sec.body).strip() if heading_line else sec.body.strip()

        if not full_text:
            continue

        clause_id = hierarchy.current_clause

        # --- Improvement #3: figure_id from text ---
        fig_id_in_text = detect_figure_id(full_text)

        # --- Improvement #4: table_id from text / caption ---
        tbl_id_in_text = detect_table_id(full_text)

        fig_ids, fig_assets = figures_for_clause(
            clause_id=clause_id,
            clause_text=full_text,
            page_figure_clause_map=page_figure_clause_map,
            page_figure_assets=page_figure_assets,
            page_figure_metadata=page_figure_metadata,
        )

        image_paths = derive_image_paths(source_file, fig_assets)
        tables = extract_tables_from_text(full_text)

        # If no table_id found inline, try from extracted table captions
        if not tbl_id_in_text and tables:
            for t in tables:
                m = _TABLE_ID_RE.search(t.get("caption", ""))
                if m:
                    tbl_id_in_text = m.group(1)
                    break

        # Determine chunk kind
        if not sec.heading_raw:
            kind = "continuation"
        elif is_subclause_label:
            kind = "subclause"
        elif re.match(r"#{4,}", "#" * sec.level) and re.search(r"\bFig", sec.heading_raw, re.IGNORECASE):
            kind = "figure"
        elif tables and not sec.is_clause:
            kind = "table"
        else:
            kind = "clause"

        chunk = Chunk(
            chunk_id=make_chunk_id(page_num, clause_id, chunk_index),
            page=page_num,
            part=hierarchy.part,
            section=hierarchy.section,
            clause_id=clause_id,
            clause_path=hierarchy.clause_path(),
            sub_clause=hierarchy.sub_clause if is_subclause_label else None,
            figure_id=fig_id_in_text,
            table_id=tbl_id_in_text,
            text=full_text,
            figures=fig_ids,
            figure_assets=fig_assets,
            image_paths=image_paths,
            tables=tables,
            xrefs=page_xrefs,
            defined_terms=page_defined_terms,
            source_file=source_file,
            page_assets=page_assets,
            kind=kind,
        )
        chunks.append(chunk)
        chunk_index += 1

    # --- Debug output ---
    if debug:
        clause_after = hierarchy.current_clause
        label = source_file or f"page_{page_num}"
        if clause_before is None and clause_after is not None:
            log.info("[inherit] %s  Inherited clause = %s", label, clause_after)
        elif clause_after is None:
            log.warning("[no-clause] %s  clause_id still None", label)
        # Per-chunk figure and table detections
        for c in chunks:
            if c.figure_id:
                log.info("[figure]  %s  Detected figure = %s", label, c.figure_id)
            if c.table_id:
                log.info("[table]   %s  Detected table  = %s", label, c.table_id)

    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

_PROMPT_LEAK_MARKERS = (
    "Extract verbatim",
    "YAML FRONTMATTER",
    "## BODY RULES",
    "## HIERARCHY",
)


def should_skip_page(page_dict: dict) -> tuple[bool, str]:
    body = page_dict.get("body", "") or ""
    kind = page_dict.get("kind") or []

    if not body.strip():
        return True, "empty body"
    if any(marker in body for marker in _PROMPT_LEAK_MARKERS):
        return True, "prompt leak"
    if kind == [] or kind == "[]":
        return True, "empty kind"
    return False, ""


def run(nbc_json: Path, out_path: Path, debug: bool = False) -> None:
    log.info("Reading %s …", nbc_json)
    with nbc_json.open(encoding="utf-8") as fh:
        data = json.load(fh)

    all_pages: list[dict] = data.get("pages", [])
    all_pages.sort(key=lambda p: p.get("seq_num") or p.get("page", 0))

    pages: list[dict] = []
    skipped: dict[str, int] = {}
    for p in all_pages:
        skip, reason = should_skip_page(p)
        if skip:
            skipped[reason] = skipped.get(reason, 0) + 1
        else:
            pages.append(p)

    log.info("Total pages in JSON: %d", len(all_pages))
    for reason, count in skipped.items():
        log.info("  Skipped (%s): %d", reason, count)
    log.info("Processing %d valid pages …", len(pages))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # SHARED hierarchy stack — lives across all pages (backward inheritance)
    # -----------------------------------------------------------------------
    hierarchy = HierarchyStack()

    total_pages = len(pages)
    total_chunks = 0
    total_figures = 0
    total_tables = 0
    total_image_paths = 0
    chunks_with_clause = 0
    chunks_without_clause = 0
    figures_detected = 0
    tables_detected = 0

    with out_path.open("w", encoding="utf-8") as fh:
        for page_dict in pages:
            try:
                page_chunks = chunks_from_page(page_dict, hierarchy, debug=debug)
            except Exception as exc:
                log.error("Error chunking page %s: %s", page_dict.get("page"), exc)
                continue

            for chunk in page_chunks:
                line = json.dumps(asdict(chunk), ensure_ascii=False)
                fh.write(line + "\n")
                total_chunks += 1
                total_figures += len(chunk.figures)
                total_tables += len(chunk.tables)
                total_image_paths += len(chunk.image_paths)

                # Validation counters
                if chunk.clause_id:
                    chunks_with_clause += 1
                else:
                    chunks_without_clause += 1
                if chunk.figure_id:
                    figures_detected += 1
                if chunk.table_id:
                    tables_detected += 1

    # Summary
    log.info("─" * 60)
    log.info("Processed           %d pages", total_pages)
    log.info("Generated           %d chunks", total_chunks)
    log.info("  Chunks with clause_id   : %d (%.1f%%)",
             chunks_with_clause,
             100 * chunks_with_clause / total_chunks if total_chunks else 0)
    log.info("  Chunks without clause_id: %d (%.1f%%)",
             chunks_without_clause,
             100 * chunks_without_clause / total_chunks if total_chunks else 0)
    log.info("Detected            %d figures (figure_id field)", figures_detected)
    log.info("Detected            %d tables  (table_id field)", tables_detected)
    log.info("Figure assets       %d", total_figures)
    log.info("Table blocks        %d", total_tables)
    log.info("Image paths         %d", total_image_paths)
    log.info("Output              %s", out_path)
    log.info("─" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Chunk nbc_full.json into retrieval units → chunks.jsonl",
    )
    p.add_argument(
        "--nbc-json",
        type=Path,
        default=Path("output/nbc_full.json"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("output/chunks.jsonl"),
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable per-page clause inheritance debug logging",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(nbc_json=args.nbc_json, out_path=args.out, debug=args.debug)