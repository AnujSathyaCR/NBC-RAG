"""
extract_nbc_v2.py

Rule-based (no Gemini / no LLM) page extraction for NBC 2016 Volume 2.

Reads the single-page PDFs already split into
output/pages_v2/NBC_2016_Volume_2/page_NNN.pdf and writes one
page_NNN.md next to each PDF, in the same YAML-frontmatter + Markdown-body
format used by output/pages/*.md (the Gemini-based Volume 1 extraction):

  page, kind, continues_from_prev, continues_to_next,
  parts_on_page, sections_on_page, clauses_on_page,
  figures_on_page, figure_clause_map, figure_metadata, figure_assets,
  xrefs, defined_terms, has_table, has_figure, has_drawing, notes

Everything derivable from the text/table layer (headings, clause numbers,
tables, figure captions, cross-references) is filled in with regex/layout
heuristics via PyMuPDF + pdfplumber. Fields that genuinely require visual
interpretation of a figure (figure_metadata, figure_clause_map, defined_terms
on non-glossary pages) are left empty rather than guessed -- an honest gap
beats an invented one. Labels and callouts printed *inside* an embedded
image (drawings, scanned photos, legend text) have no PDF text layer at
all, so every embedded image under output/images_v2/page_NNN/ is OCR'd
locally with Tesseract and appended to the page body -- no external API
call.

Usage:
    python extract_nbc_v2.py             # all pages (skips already-done)
    python extract_nbc_v2.py 1 250       # only pages 1..250 (inclusive)
    python extract_nbc_v2.py --rebuild   # re-extract even if .md exists
"""

from __future__ import annotations

import html
import re
import shutil
import sys
import json
from pathlib import Path
from collections import Counter

import fitz
import pdfplumber
import pytesseract
import yaml
from PIL import Image

if not shutil.which("tesseract"):
    _fallback_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if _fallback_tesseract.exists():
        pytesseract.pytesseract.tesseract_cmd = str(_fallback_tesseract)

ROOT = Path(__file__).resolve().parent
PAGES_DIR = ROOT / "output" / "pages_v2"
IMAGES_DIR = ROOT / "output" / "images_v2"
BOILERPLATE_CACHE = ROOT / "output" / "pages_v2" / ".boilerplate_cache_v2.json"

PAGE_NUM_FROM_NAME = re.compile(r"(\d+)")

CLAUSE_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,3}){0,4})(?:\s+|$)(.*)")
PART_RE = re.compile(r"^PART\s+\d+", re.IGNORECASE)
SECTION_RE = re.compile(r"^Section\s+\d+[A-Za-z]?\b", re.IGNORECASE)
PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
FIG_CAPTION_RE = re.compile(r"^FIG\.?\s*(\d+[A-Za-z]?)\b\.?\s*(.*)$")
FIG_MENTION_RE = re.compile(r"\bFig\.?\s*(\d+[A-Za-z]?)\b", re.IGNORECASE)
LEGEND_ROW_RE = re.compile(r"^[A-Za-z]{1,3}\s*[\x96\x97\u2013\u2014-]\s*\S")
LEGEND_DESC_CONTINUATION_RE = re.compile(r"^[\x96\x97\u2013\u2014-]\s*\S")
LEGEND_MAX_GAP = 20      # max vertical gap (pt) between consecutive figure-block lines
LEGEND_MAX_BELOW = 260   # max distance (pt) below an image the caption/legend can sit
TOC_LEADER_RE = re.compile(r"\.{4,}\s*\d+\s*$")
PREAMBLE_RE = re.compile(
    r"(?i)^(foreword|preface|committee|acknowledg|composition of|list of)"
)
BOLD_TERM_RE = re.compile(r"^(.{2,60}?)\s*[—–-]\s*(Part\s+\d+.*)$")

XREF_PATTERNS = [
    re.compile(r"\bPart\s+\d+(?:/Section\s+\d+[A-Za-z]?)?\b", re.IGNORECASE),
    re.compile(r"\bSection\s+\d+[A-Za-z]?\b", re.IGNORECASE),
    re.compile(r"\bTable\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bFig\.?\s*\d+[A-Za-z]?\b", re.IGNORECASE),
    re.compile(r"\bIS(?:/[A-Z]+)?\s*:?\s*\d+(?:-\d+)?\b"),
    re.compile(r"\bClause\s+[\d.]+\b", re.IGNORECASE),
]

MARGIN_TOP = 100
MARGIN_BOTTOM = 680


def is_bold(span_font: str) -> bool:
    return "bold" in span_font.lower()


def get_page_lines(page):
    """Return list of {text, size, bold, y0, y1, col} per visual line.

    Two-column layouts interleave left/right column text at the same
    vertical position in PyMuPDF's raw ordering, so blocks are split by
    x-midpoint and each column is sorted top-to-bottom independently.
    """
    d = page.get_text("dict")
    page_width = page.rect.width
    mid_x = page_width / 2

    raw_lines = []
    for block in d["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            s0 = line["spans"][0]
            x0 = line["bbox"][0]
            raw_lines.append({
                "text": text,
                "size": round(s0["size"], 1),
                "bold": is_bold(s0.get("font", "")),
                "y0": line["bbox"][1],
                "y1": line["bbox"][3],
                "x0": x0,
                "col": 0 if x0 < mid_x else 1,
            })

    left = [l for l in raw_lines if l["col"] == 0]
    right = [l for l in raw_lines if l["col"] == 1]
    if len(left) > 3 and len(right) > 3:
        left.sort(key=lambda l: l["y0"])
        right.sort(key=lambda l: l["y0"])
        return left + right
    else:
        raw_lines.sort(key=lambda l: l["y0"])
        return raw_lines


def find_boilerplate(all_pages_lines, n_pages):
    """Lines repeated across many pages = running headers/footers -> strip.

    Margin-band lines (running Part/Section titles, book title, page
    numbers) get a much lower repeat threshold than main-body lines, since
    a running header only repeats within its own Part's page range, not
    across the whole sampled set.
    """
    main_counter = Counter()
    margin_counter = Counter()
    for lines in all_pages_lines:
        seen_main, seen_margin = set(), set()
        for l in lines:
            key = l["text"]
            in_margin = l["y0"] < MARGIN_TOP or l["y0"] > MARGIN_BOTTOM
            target_counter = margin_counter if in_margin else main_counter
            seen = seen_margin if in_margin else seen_main
            if key not in seen:
                target_counter[key] += 1
                seen.add(key)

    main_threshold = max(5, int(n_pages * 0.25))
    margin_threshold = max(3, int(n_pages * 0.03))
    boilerplate = {t for t, c in main_counter.items() if c >= main_threshold}
    boilerplate |= {t for t, c in margin_counter.items() if c >= margin_threshold}
    return boilerplate


def body_font_size(all_pages_lines):
    sizes = Counter()
    for lines in all_pages_lines:
        for l in lines:
            sizes[l["size"]] += 1
    return sizes.most_common(1)[0][0] if sizes else 10.0


def line_in_any_bbox(line, bboxes):
    ymid = (line["y0"] + line["y1"]) / 2
    for (x0, top, x1, bottom) in bboxes:
        if (top - 2 <= ymid <= bottom + 2) and (x0 - 2 <= line["x0"] <= x1 + 2):
            return True
    return False


def table_to_md(table):
    if not table or not table[0]:
        return ""
    def clean(c):
        return html.escape((c or "").replace("\n", " ").strip())
    header = table[0]
    rows = table[1:]
    out = ["<table>", "  <tr>" + "".join(f"<th>{clean(c)}</th>" for c in header) + "</tr>"]
    for row in rows:
        out.append("  <tr>" + "".join(f"<td>{clean(c)}</td>" for c in row) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def table_quality_ok(table):
    if not table or len(table) < 2:
        return False
    total = sum(len(r) for r in table)
    empty = sum(1 for r in table for c in r if not c or not c.strip())
    if total == 0:
        return False
    return (empty / total) < 0.4


# ---------------------------------------------------------------------------
# Borderless-table reconstruction (best-effort geometry, no ruling lines)
#
# Most NBC tables have no drawn cell borders at all -- pdfplumber's default
# line-based table finder collapses them into one giant garbled cell. These
# tables follow a strong convention instead: a header row of bare column
# indices "(1) (2) (3) ...", whose word x-centres give the true column grid.
# Everything above that row is header (grouped into colspan/rowspan by
# horizontal contiguity and vertical blankness); everything below is one
# data record per line, bucketed into the same column grid, with a blank
# cell inheriting (as rowspan) from the non-blank cell above it -- the usual
# convention for a repeated group label.
#
# This is a geometric approximation, not a re-derivation of the original
# layout: exact rowspan boundaries in a hand/vision-transcribed table can
# reflect a stylistic choice (e.g. an explicit blank header cell vs. a
# rowspan that swallows it) that plain coordinates can't disambiguate.
# ---------------------------------------------------------------------------

INDEX_CELL_RE = re.compile(r"^\(\s*(\d+)\s*\)$")
TABLE_CAPTION_RE = re.compile(r"^Table\s+\d+[A-Za-z]?\b")


def _words_to_table_lines(words):
    """Bucket pdfplumber words into visual rows by vertical position."""
    rows: dict[float, list] = {}
    for w in words:
        key = round(w["top"] / 3) * 3
        rows.setdefault(key, []).append(w)
    lines = []
    for key in sorted(rows):
        ws = sorted(rows[key], key=lambda w: w["x0"])
        lines.append({"top": key, "words": ws})
    return lines


def _find_index_row(lines):
    """Locate the '(1) (2) (3) ...' column-index row, if present.

    Only the *count* of index cells is used (to fix the column grid), not
    their printed values -- some source pages misprint the last index (e.g.
    "(1) (2) (3) (4) (2)" instead of "...(5)"), so the sequence itself isn't
    required to be strictly 1..n.
    """
    for i, line in enumerate(lines):
        toks = [w["text"] for w in line["words"]]
        if len(toks) < 2:
            continue
        if all(INDEX_CELL_RE.match(t) for t in toks):
            return i
    return None


def _col_bounds(centers):
    bounds = [-float("inf")]
    for a, b in zip(centers, centers[1:]):
        bounds.append((a + b) / 2)
    bounds.append(float("inf"))
    return bounds


def _assign_col(x_center, bounds):
    for i in range(len(bounds) - 1):
        if bounds[i] <= x_center < bounds[i + 1]:
            return i
    return len(bounds) - 2


def _refine_col_bounds(body_lines, k, bounds):
    """Recompute column boundaries from the observed extent of body words,
    rather than trusting the (possibly off-centre) index-row label centres."""
    extents = [None] * k
    for line in body_lines:
        for w in line["words"]:
            cx = (w["x0"] + w["x1"]) / 2
            c = _assign_col(cx, bounds)
            lo, hi = extents[c] if extents[c] else (w["x0"], w["x1"])
            extents[c] = (min(lo, w["x0"]), max(hi, w["x1"]))

    new_bounds = [-float("inf")]
    for c in range(k - 1):
        left, right = extents[c], extents[c + 1]
        new_bounds.append((left[1] + right[0]) / 2 if left and right else bounds[c + 1])
    new_bounds.append(float("inf"))
    return new_bounds


def build_table_html(plumber_page, bbox):
    """Reconstruct an HTML table (thead/tbody, colspan/rowspan) for a
    borderless table region. Returns None if the region doesn't match the
    '(1) (2) (3) ...' index-row convention this heuristic depends on."""
    x0, top, x1, bottom = bbox
    crop = plumber_page.crop((x0, top, x1, bottom))
    words = crop.extract_words()
    if not words:
        return None
    lines = _words_to_table_lines(words)
    idx = _find_index_row(lines)
    if idx is None:
        return None

    col_centers = [(w["x0"] + w["x1"]) / 2 for w in lines[idx]["words"]]
    k = len(col_centers)
    bounds = _col_bounds(col_centers)

    # The "(N)" index labels are centred within their column, but the data
    # itself (e.g. left-aligned text) may start well to the left of that
    # centre -- a naive centre-to-centre midpoint boundary can then cut
    # into where real words of the *previous* column actually sit. Refine
    # the boundaries using the observed extent of the body data itself;
    # repeated a few times since one pass can still misclassify a
    # borderline word, which then skews that pass's own extent estimate --
    # each subsequent pass re-buckets with the tighter boundary and
    # converges (same idea as Lloyd's algorithm for 1-D k-means).
    body_word_lines = lines[idx + 1:]
    for _ in range(5):
        bounds = _refine_col_bounds(body_word_lines, k, bounds)

    def clean(t):
        return html.escape((t or "").strip())

    # --- caption lines above the header, e.g. "Table 1 Maximum Allowable
    # Contaminant Concentrations for Ventilation Air" (title, itself often
    # wrapping across 2+ lines in a narrow column) followed by a "(Clause
    # 2.2.9)" reference that may *also* wrap onto its own line. The title
    # has no reliable end-of-line marker, so its extent is inferred by
    # scanning forward for the first line that mentions "Clause" at all. ---
    caption_entries = []  # [kind, text], kind in {"caption", "clause"}
    header_start = 0

    def line_text(i):
        return " ".join(w["text"] for w in lines[i]["words"]).strip()

    if idx > 0 and TABLE_CAPTION_RE.match(line_text(0)):
        scan_limit = min(idx, 6)
        clause_start = next(
            (j for j in range(scan_limit) if re.search(r"\bClause\b", line_text(j), re.IGNORECASE)),
            None,
        )
        title_end = clause_start if clause_start is not None else 1
        title = " ".join(line_text(t) for t in range(title_end)).strip()
        caption_entries.append(["caption", title])
        header_start = title_end

        if clause_start is not None:
            j = clause_start
            clause_text = ""
            while j < idx:
                clause_text = (clause_text + " " + line_text(j)).strip()
                j += 1
                if clause_text.endswith(")"):
                    break
            caption_entries.append(["clause", clause_text])
            header_start = j

    header_lines = lines[header_start:idx]
    n_header = len(header_lines)

    # Pass 1: bucket each header word by nearest column (same grid as the
    # data rows), then only grow a cell into a neighbouring column when that
    # neighbour has no words of its own this row *and* the cell's own bbox
    # geometrically overlaps it -- i.e. colspan is granted from bbox width,
    # never by guessing from word spacing (which is unreliable when column
    # labels sit only a single space apart in a narrow borderless layout).
    row_groups = []
    for line in header_lines:
        col_words = [[] for _ in range(k)]
        for w in line["words"]:
            cx = (w["x0"] + w["x1"]) / 2
            col_words[_assign_col(cx, bounds)].append(w)

        cells = []
        visited = [False] * k
        for c in range(k):
            if visited[c] or not col_words[c]:
                continue
            words = col_words[c]
            gx0 = min(w["x0"] for w in words)
            gx1 = max(w["x1"] for w in words)
            covered = {cc for cc, ctr in enumerate(col_centers) if gx0 - 2 <= ctr <= gx1 + 2}
            covered.add(c)
            lo = hi = c
            while lo - 1 >= 0 and (lo - 1) in covered and not col_words[lo - 1]:
                lo -= 1
            while hi + 1 < k and (hi + 1) in covered and not col_words[hi + 1]:
                hi += 1
            colspan = hi - lo + 1
            text = " ".join(w["text"] for w in words)
            cells.append([lo, colspan, text])
            for cc in range(lo, hi + 1):
                visited[cc] = True
        row_groups.append(cells)

    # Pass 2: rowspan -- extend a header cell downward while its full column
    # range has no competing content in the following header rows.
    consumed = [[False] * k for _ in range(n_header)]
    occupied = []
    for cells in row_groups:
        occ = set()
        for start_col, colspan, _ in cells:
            occ.update(range(start_col, start_col + colspan))
        occupied.append(occ)

    header_cells = [[] for _ in range(n_header)]  # (start_col, colspan, rowspan, text)
    for r, cells in enumerate(row_groups):
        for start_col, colspan, text in cells:
            span = 1
            rr = r + 1
            while rr < n_header and not any(c in occupied[rr] for c in range(start_col, start_col + colspan)):
                span += 1
                rr += 1
            if span > 1:
                for rr2 in range(r + 1, r + span):
                    for c in range(start_col, start_col + colspan):
                        consumed[rr2][c] = True
            header_cells[r].append((start_col, colspan, span, text))

    def render_header_row(cells_here, consumed_row):
        cells_here = sorted(cells_here, key=lambda c: c[0])
        out = []
        col = 0
        by_start = {c[0]: c for c in cells_here}
        while col < k:
            if consumed_row[col]:
                col += 1
                continue
            if col in by_start:
                start_col, colspan, rowspan, text = by_start[col]
                attrs = ""
                if rowspan > 1:
                    attrs += f' rowspan="{rowspan}"'
                if colspan > 1:
                    attrs += f' colspan="{colspan}"'
                out.append(f"<th{attrs}>{clean(text)}</th>")
                col += colspan
            else:
                out.append("<th></th>")
                col += 1
        return "  <tr>" + "".join(out) + "</tr>"

    thead_rows = []
    for r in range(n_header):
        thead_rows.append(render_header_row(header_cells[r], consumed[r]))
    thead_rows.append(
        "  <tr>" + "".join(f"<th>{clean(w['text'])}</th>" for w in lines[idx]["words"]) + "</tr>"
    )

    # --- body: bucket each data line into the k-column grid, blank cells
    # inherit (rowspan) from the non-blank cell above in the same column.
    #
    # A physical line whose column-0 cell (the row-label / serial-number
    # column) is blank is a line-wrap continuation of the previous record's
    # text, not a new grouped row -- e.g. a long description overflowing
    # onto a second line. Such lines are merged into the previous row
    # before the rowspan pass runs, so a genuinely wrapped cell doesn't get
    # mistaken for a blank column meant to inherit a category-label rowspan. ---
    body_lines = lines[idx + 1:]
    grid_rows = []
    for line in body_lines:
        cells = [[] for _ in range(k)]
        for w in line["words"]:
            cx = (w["x0"] + w["x1"]) / 2
            cells[_assign_col(cx, bounds)].append(w["text"])
        cell_texts = [" ".join(c).strip() for c in cells]
        if grid_rows and not cell_texts[0]:
            prev = grid_rows[-1]
            for c in range(k):
                if cell_texts[c]:
                    prev[c] = f"{prev[c]} {cell_texts[c]}".strip() if prev[c] else cell_texts[c]
        else:
            grid_rows.append(cell_texts)

    active = [None] * k  # per column: dict with "text","rowspan" of the open cell
    body_grid = []       # per row: dict col -> cell-dict, or None if spanned-away
    for row_cells in grid_rows:
        row_out = {}
        for c in range(k):
            text = row_cells[c]
            if not text and active[c] is not None:
                active[c]["rowspan"] += 1
                row_out[c] = None
            else:
                cell = {"text": text, "rowspan": 1}
                row_out[c] = cell
                active[c] = cell
        body_grid.append(row_out)

    tbody_rows = []
    for row_out in body_grid:
        out = []
        for c in range(k):
            cell = row_out[c]
            if cell is None:
                continue
            attrs = f' rowspan="{cell["rowspan"]}"' if cell["rowspan"] > 1 else ""
            out.append(f"<td{attrs}>{clean(cell['text'])}</td>")
        tbody_rows.append("  <tr>" + "".join(out) + "</tr>")

    if not tbody_rows:
        return None

    parts = []
    if caption_entries:
        def squeeze_parens(t):
            return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", t))
        cap_md = "\n".join(
            f"**{t}**" if kind == "caption" else f"*{squeeze_parens(t)}*"
            for kind, t in caption_entries
        )
        parts.append(cap_md)
    table_html = "<table>\n  <thead>\n" + "\n".join(thead_rows) + \
                 "\n  </thead>\n  <tbody>\n" + "\n".join(tbody_rows) + "\n  </tbody>\n</table>"
    parts.append(table_html)
    return "\n\n".join(parts)


def list_page_pdfs():
    """Map page number (parsed from filename) -> filepath, for PAGES_DIR."""
    paged = {}
    for f in PAGES_DIR.glob("*.pdf"):
        m = PAGE_NUM_FROM_NAME.search(f.stem)
        if m:
            paged[int(m.group(1))] = f
    return paged


def page_image_dir(page_num: int) -> Path:
    return IMAGES_DIR / f"page_{page_num:03d}"


def page_image_assets(page_num: int) -> list[str]:
    d = page_image_dir(page_num)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_file())


MIN_OCR_DIM = 100  # skip tiny decorative/icon rasters, not real figures


def ocrable_page_images(page_num: int) -> list[Path]:
    """Embedded images on this page worth OCR'ing (filters out tiny icons)."""
    d = page_image_dir(page_num)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        try:
            with Image.open(p) as im:
                if im.width < MIN_OCR_DIM or im.height < MIN_OCR_DIM:
                    continue
        except Exception:
            continue
        out.append(p)
    return out


def get_image_regions(page):
    """Bounding boxes of embedded raster images on this page (the figure
    artwork itself -- diagrams, drawings, photos). Text is never inside
    these; PyMuPDF reports them as a separate object from the text layer."""
    try:
        return [tuple(info["bbox"]) for info in page.get_image_info()]
    except Exception:
        return []


def is_figure_associated(text):
    """True for lines that are part of a figure's caption/legend block
    (not body prose), e.g. 'REFERENCES', 'FIG. 1 ...', 'O \u2013 Observer's
    station'. These sit physically wherever there was layout space, so
    letting them fall into the normal y-sorted paragraph flow splices
    them into whatever body sentence happens to be at that height."""
    t = text.strip()
    if not t:
        return False
    if t.upper() == "REFERENCES":
        return True
    if FIG_CAPTION_RE.match(t):
        return True
    if LEGEND_ROW_RE.match(t):
        return True
    if LEGEND_DESC_CONTINUATION_RE.match(t):
        return True
    # short all-caps line (continuation of a caption, e.g. "CELESTIAL BODY")
    if len(t) < 40 and t == t.upper() and any(c.isalpha() for c in t):
        return True
    return False


def expand_figure_region(image_bbox, lines):
    """Grow an image's bbox downward to also cover its caption/legend text,
    by walking lines below the image while they keep looking like
    caption/legend content and stay vertically contiguous. Returns
    (region_bbox, member_lines) or (image_bbox, []) if nothing attaches."""
    x0, top, x1, bottom = image_bbox
    candidates = sorted(
        (l for l in lines if bottom - 2 <= l["y0"] <= bottom + LEGEND_MAX_BELOW
         and x0 - 30 <= l["x0"] <= x1 + 30),
        key=lambda l: l["y0"],
    )
    member_lines = []
    last_y1 = bottom
    for l in candidates:
        if l["y0"] - last_y1 > LEGEND_MAX_GAP:
            break
        if not is_figure_associated(l["text"]):
            break
        member_lines.append(l)
        last_y1 = l["y1"]

    if not member_lines:
        return image_bbox, []

    region = (x0, top, x1, max(bottom, max(l["y1"] for l in member_lines)))
    return region, member_lines


def reconstruct_legend_rows(legend_lines):
    """Legend rows are printed as a nested 2-column mini-layout ('O \u2013
    Observer's station' | 'S \u2013 Geographical south' side by side), which
    can arrive as 1, 2, or 4 separate line fragments per row depending on
    PyMuPDF's grouping. Bucket by y-row, then split each row into left/right
    clusters at its biggest horizontal gap so label+description stay paired
    regardless of fragment count."""
    from collections import defaultdict
    rows = defaultdict(list)
    for l in legend_lines:
        key = round(l["y0"] / 3) * 3
        rows[key].append(l)

    entries = []
    for key in sorted(rows.keys()):
        frags = sorted(rows[key], key=lambda l: l["x0"])
        if len(frags) == 1:
            entries.append(frags[0]["text"])
            continue
        gaps = sorted(
            ((frags[i + 1]["x0"] - frags[i]["x0"], i) for i in range(len(frags) - 1)),
            reverse=True,
        )
        split_i = gaps[0][1] + 1 if gaps and gaps[0][0] > 40 else len(frags)
        left = " ".join(f["text"] for f in frags[:split_i]).strip()
        right = " ".join(f["text"] for f in frags[split_i:]).strip()
        if left:
            entries.append(left)
        if right:
            entries.append(right)
    return entries


def build_figure_block(member_lines, ocr_text, asset_name):
    """Render a figure's caption + legend + (optional) OCR'd raster text as
    one self-contained markdown block, cleaning up the mis-encoded dash
    byte (\\x96/\\x97) this OCR'd font uses in place of an en dash."""
    def clean(t):
        return t.replace("\x96", "\u2013").replace("\x97", "\u2013").strip()

    caption_lines = []
    in_caption = False
    for l in member_lines:
        if FIG_CAPTION_RE.match(l["text"]):
            caption_lines.append(l)
            in_caption = True
        elif in_caption and len(l["text"].strip()) < 40 and l["text"].strip() == l["text"].strip().upper():
            caption_lines.append(l)
        else:
            in_caption = False
    legend_lines = [l for l in member_lines
                    if l not in caption_lines and l["text"].strip().upper() != "REFERENCES"]

    caption = " ".join(clean(l["text"]) for l in caption_lines).strip()
    legend_entries = reconstruct_legend_rows(legend_lines)

    parts = []
    if caption:
        parts.append(f"**{caption}**")
    if legend_entries:
        parts.append("\n".join(f"- {clean(e)}" for e in legend_entries))
    if ocr_text:
        parts.append(f"<!-- OCR text from {asset_name} -->\n{ocr_text}")
    return "\n\n".join(parts)



def calibrate(pdf_files):
    """Sample across the split pages once to get boilerplate + body font size.
    Cached to disk so re-runs (e.g. chunked over page ranges) reuse it."""
    if BOILERPLATE_CACHE.exists():
        cached = json.loads(BOILERPLATE_CACHE.read_text())
        return set(cached["boilerplate"]), cached["body_size"]

    page_nums = sorted(pdf_files.keys())
    sample_size = 200
    step = max(1, len(page_nums) // sample_size)
    sample_nums = page_nums[::step][:sample_size]

    sample_lines = []
    for pn in sample_nums:
        doc = fitz.open(pdf_files[pn])
        sample_lines.append(get_page_lines(doc[0]))
        doc.close()

    boilerplate = find_boilerplate(sample_lines, len(sample_lines))
    body_size = body_font_size(sample_lines)
    BOILERPLATE_CACHE.write_text(
        json.dumps({"boilerplate": list(boilerplate), "body_size": body_size})
    )
    return boilerplate, body_size


def _mean_confidence(im) -> tuple[float, str]:
    """Run Tesseract on `im`, return (avg word confidence, recognized text).

    Vector figures exported to raster (drawings, flowcharts) are frequently
    white-on-black rather than the dark-on-light OCR expects; giving a
    -1 confidence back means "no usable text found here", not "zero".
    """
    try:
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
    except Exception:
        return -1.0, ""
    confs, words = [], []
    for conf, text in zip(data["conf"], data["text"]):
        text = text.strip()
        if not text:
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c < 0:
            continue
        confs.append(c)
        words.append(text)
    if not confs:
        return -1.0, ""
    return sum(confs) / len(confs), " ".join(words)


def ocr_fallback(image_path: Path | None) -> str | None:
    """Local Tesseract OCR on a pre-extracted embedded image -- no external
    API call.

    Figures rendered to raster often carry inverted colors (light strokes/
    text on a dark fill) and rotated labels (vertical axis captions, sideways
    callouts). Neither is guessable from metadata, so this tries the image
    as-is and inverted, at all four 90-degree rotations, and keeps whichever
    combination Tesseract was most confident about.
    """
    if not image_path or not image_path.exists():
        return None
    try:
        from PIL import ImageOps
        base = Image.open(image_path).convert("L")
        candidates = [base, ImageOps.invert(base)]
    except Exception as e:
        return f"<!-- OCR fallback failed: {e} -->"

    best_conf, best_text = -1.0, ""
    try:
        for variant in candidates:
            for angle in (0, 90, 180, 270):
                rotated = variant.rotate(angle, expand=True) if angle else variant
                conf, text = _mean_confidence(rotated)
                if conf > best_conf:
                    best_conf, best_text = conf, text
    except Exception as e:
        return f"<!-- OCR fallback failed: {e} -->"

    return best_text.strip() or None


def dedupe_keep_order(items):
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def process_one_page(page_num, pdf_path, boilerplate, body_size, prev_continues_to_next):
    doc = fitz.open(pdf_path)
    page = doc[0]
    lines = get_page_lines(page)
    is_two_col = len({l["col"] for l in lines}) > 1

    with pdfplumber.open(pdf_path) as pdf:
        plumber_page = pdf.pages[0]
        page_width = float(plumber_page.width)
        mid_x = page_width / 2
        raw_tables = plumber_page.find_tables()
        tables_data = []
        for t in raw_tables:
            extracted = t.extract()
            x0, top, x1, bottom = t.bbox
            col = 0 if ((x0 + x1) / 2) < mid_x else 1
            html_table = build_table_html(plumber_page, t.bbox)
            tables_data.append({
                "bbox": t.bbox, "data": extracted, "col": col, "html": html_table,
            })

    notes = []
    for t in tables_data:
        if t["html"] is None and not table_quality_ok(t["data"]):
            notes.append("low-confidence table extraction, needs review")

    # --- figure regions: image bbox + its attached caption/legend text,
    # so that block can be pulled out of the main paragraph flow the same
    # way tables already are, instead of splicing into whatever sentence
    # happens to sit at that height in the column. ---
    image_regions_raw = get_image_regions(page)
    figure_regions = []
    for img_bbox in image_regions_raw:
        region_bbox, member_lines = expand_figure_region(img_bbox, lines)
        rx0, rtop, rx1, rbottom = region_bbox
        col = 0 if ((rx0 + rx1) / 2) < (page.rect.width / 2) else 1
        figure_regions.append({
            "bbox": region_bbox, "col": col, "member_lines": member_lines,
        })
    figure_regions.sort(key=lambda r: r["bbox"][1])

    # Match each figure region to an OCR'd image asset by top-to-bottom
    # order (best-effort -- exact filename correspondence isn't tracked
    # upstream, but page order reliably matches layout order in practice).
    ocrable_assets = ocrable_page_images(page_num)
    for i, region in enumerate(figure_regions):
        asset_path = ocrable_assets[i] if i < len(ocrable_assets) else None
        region["ocr_text"] = ocr_fallback(asset_path) if asset_path else None
        region["asset_name"] = asset_path.name if asset_path else "unknown"
    unmatched_assets = ocrable_assets[len(figure_regions):]

    # --- metadata scan over ALL raw lines (parts/sections repeat as running
    # headers in the margin bands and would otherwise be stripped before we
    # get a chance to record them) ---
    parts_on_page, sections_on_page = [], []
    printed_page_num = None
    for l in lines:
        in_margin = l["y0"] < MARGIN_TOP or l["y0"] > MARGIN_BOTTOM
        if PART_RE.match(l["text"]):
            parts_on_page.append(l["text"])
        if SECTION_RE.match(l["text"]):
            sections_on_page.append(l["text"])
        if in_margin and printed_page_num is None and PAGE_NUM_RE.match(l["text"]):
            printed_page_num = int(l["text"])
    parts_on_page = dedupe_keep_order(parts_on_page)
    sections_on_page = dedupe_keep_order(sections_on_page)

    table_bboxes = [t["bbox"] for t in tables_data]
    figure_bboxes = [r["bbox"] for r in figure_regions]
    content_lines = []
    for l in lines:
        if l["text"] in boilerplate:
            continue
        if PART_RE.match(l["text"]) and (l["y0"] < MARGIN_TOP or l["y0"] > MARGIN_BOTTOM):
            continue  # running Part header, already captured above
        if SECTION_RE.match(l["text"]) and (l["y0"] < MARGIN_TOP or l["y0"] > MARGIN_BOTTOM):
            continue  # running Section header, already captured above
        if PAGE_NUM_RE.match(l["text"]) and (l["y0"] < 60 or l["y0"] > 720):
            continue
        if line_in_any_bbox(l, table_bboxes):
            continue
        if line_in_any_bbox(l, figure_bboxes):
            continue  # consumed by a figure's caption/legend block instead
        content_lines.append(l)

    mentions_table = any(re.search(r"\btable\b", l["text"], re.IGNORECASE) for l in content_lines)
    if mentions_table and not tables_data:
        notes.append("page mentions a table but none was extracted")

    events = []
    col_range = [0, 1] if is_two_col else [0]
    for col in col_range:
        col_events = [{"kind": "line", "y": l["y0"], "data": l}
                      for l in content_lines if l["col"] == col]
        col_events += [{"kind": "table", "y": t["bbox"][1], "data": t}
                       for t in tables_data if t["col"] == col]
        col_events.sort(key=lambda e: e["y"])
        events.extend(col_events)
    # Figures are deliberately NOT merged into the column y-sort: a figure
    # floats beside body text in the original layout, so its physical
    # position often falls between a clause and its own continuation (e.g.
    # top of the next column). Inserting it there would still split the
    # sentence even if the block itself is now clean, so figures are
    # collected and appended after all prose instead.

    md_out = []
    paragraph_buf = []
    clauses_on_page = []
    figures_on_page = []
    xrefs = []
    defined_terms = []
    last_heading_level = 1
    first_content_text = None
    last_content_text = None

    def flush_paragraph():
        if paragraph_buf:
            md_out.append(" ".join(paragraph_buf).strip())
            paragraph_buf.clear()

    for e in events:
        if e["kind"] == "table":
            flush_paragraph()
            if e["data"]["html"] is not None:
                md_out.append(e["data"]["html"])
            elif table_quality_ok(e["data"]["data"]):
                md_out.append(table_to_md(e["data"]["data"]))
            else:
                md_out.append("<!-- table extraction low-confidence, needs review -->")
            continue

        l = e["data"]
        text = l["text"]
        if first_content_text is None:
            first_content_text = text
        last_content_text = text

        for pat in XREF_PATTERNS:
            xrefs.extend(m.group(0) for m in pat.finditer(text))

        clause_m = CLAUSE_RE.match(text)
        bold_term = BOLD_TERM_RE.match(text) if l["bold"] else None

        if clause_m and clause_m.group(2) and clause_m.group(2)[:1].isupper():
            flush_paragraph()
            clauses_on_page.append(clause_m.group(1))
            depth = clause_m.group(1).count(".") + 1
            level = min(depth + 1, 6)
            last_heading_level = level
            md_out.append(f"{'#' * level} {text}")
        elif PART_RE.match(text):
            flush_paragraph()
            md_out.append(f"# {text}")
            last_heading_level = 1
        elif SECTION_RE.match(text):
            flush_paragraph()
            md_out.append(f"## {text}")
            last_heading_level = 2
        else:
            if bold_term:
                defined_terms.append(bold_term.group(1).strip())
            paragraph_buf.append(text)

        for m in FIG_MENTION_RE.finditer(text):
            figures_on_page.append(f"Fig. {m.group(1)}")
    flush_paragraph()

    # Figures appended here, after all prose, so they never split a clause
    # that happens to continue past where the figure physically sits.
    for region in figure_regions:
        block = build_figure_block(region["member_lines"], region["ocr_text"], region["asset_name"])
        if block:
            md_out.append(block)
        fig_cap_line = next((ml for ml in region["member_lines"] if FIG_CAPTION_RE.match(ml["text"])), None)
        if fig_cap_line:
            fm = FIG_CAPTION_RE.match(fig_cap_line["text"])
            figures_on_page.append(f"Fig. {fm.group(1)}")

    figures_on_page = dedupe_keep_order(figures_on_page)
    clauses_on_page = dedupe_keep_order(clauses_on_page)
    xrefs = dedupe_keep_order(xrefs)
    defined_terms = dedupe_keep_order(defined_terms)

    image_assets = page_image_assets(page_num)

    # Sparse page (little/no text layer) -> likely a scanned figure/photo,
    # flagged for review even though the OCR pass below covers it the same
    # as any other embedded image.
    is_sparse = len(content_lines) < 5 and not tables_data
    if is_sparse:
        notes.append("sparse page, possible scanned figure/photo")



    # Any embedded image not matched to a caption/legend region above
    # (e.g. a bare photo with no adjacent text) still gets its local OCR
    # text appended here, tagged by filename, rather than being dropped.
    for img_path in unmatched_assets:
        ocr_text = ocr_fallback(img_path)
        if ocr_text:
            md_out.append(f"<!-- OCR text from {img_path.name} -->\n{ocr_text}")

    has_table = bool(tables_data)
    has_figure = bool(figures_on_page)
    has_drawing = bool(image_assets) and not has_figure

    if not lines and not tables_data and not image_assets:
        kind = "blank"
    elif not content_lines and not tables_data and not image_assets:
        kind = "boilerplate"
    elif first_content_text and PREAMBLE_RE.match(first_content_text):
        kind = "preamble"
    else:
        toc_hits = sum(1 for l in content_lines if TOC_LEADER_RE.search(l["text"]))
        kind = "toc" if content_lines and toc_hits / len(content_lines) > 0.3 else "content"

    continues_from_prev = bool(prev_continues_to_next) or bool(
        first_content_text and first_content_text[:1].islower()
    )
    continues_to_next = bool(
        last_content_text and not re.search(r'[.!?:;"”)]\s*$', last_content_text)
    )

    frontmatter = {
        "page": printed_page_num if printed_page_num is not None else page_num,
        "kind": kind,
        "continues_from_prev": continues_from_prev,
        "continues_to_next": continues_to_next,
        "parts_on_page": parts_on_page,
        "sections_on_page": sections_on_page,
        "clauses_on_page": clauses_on_page,
        "figures_on_page": figures_on_page,
        "figure_clause_map": [],
        "figure_metadata": [],
        "figure_assets": image_assets,
        "xrefs": xrefs,
        "defined_terms": defined_terms,
        "has_table": has_table,
        "has_figure": has_figure,
        "has_drawing": has_drawing,
        "notes": "; ".join(notes) if notes else None,
    }

    fm_yaml = yaml.safe_dump(
        frontmatter, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).strip()
    body = "\n\n".join(md_out).strip()
    doc.close()
    return f"---\n{fm_yaml}\n---\n\n{body}\n", continues_to_next


def run(start_page=None, end_page=None, rebuild=False):
    pdf_files = list_page_pdfs()
    if not pdf_files:
        sys.exit(f"No page PDFs found in {PAGES_DIR}")

    page_nums = sorted(pdf_files.keys())
    if start_page is not None:
        page_nums = [p for p in page_nums if start_page <= p <= end_page]

    boilerplate, body_size = calibrate(pdf_files)
    print(f"Body font size: {body_size}, boilerplate lines: {len(boilerplate)}")
    print(f"Found {len(pdf_files)} page PDFs. Processing {len(page_nums)} pages this run...")

    prev_continues_to_next = False
    done = 0
    skipped = 0
    for pn in page_nums:
        pdf_path = pdf_files[pn]
        md_path = pdf_path.with_suffix(".md")
        if md_path.exists() and not rebuild:
            skipped += 1
            continue
        md_text, continues_to_next = process_one_page(
            pn, pdf_path, boilerplate, body_size, prev_continues_to_next
        )
        prev_continues_to_next = continues_to_next
        md_path.write_text(md_text, encoding="utf-8")
        done += 1
        if done % 50 == 0:
            print(f"  ...{done} pages written")

    print(f"Done. {done} pages written, {skipped} skipped (already existed).")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--rebuild"]
    rebuild = "--rebuild" in sys.argv
    if len(args) >= 2:
        run(int(args[0]), int(args[1]), rebuild=rebuild)
    else:
        run(rebuild=rebuild)