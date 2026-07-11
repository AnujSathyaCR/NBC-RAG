#!/usr/bin/env python3
"""
evaluate_retrieval.py
=====================
Evaluate the NBC RAG retrieval pipeline using graded relevance (0-3) with
separate PAGE-level, CLAUSE-level, FIGURE-level, TABLE-level, and COMBINED
metrics, plus NDCG at K ∈ {1, 3, 5, 7, 10}.

Graded Relevance Model (NBC-specific)
--------------------------------------
  3  — Exact match on clause_id, figure_id, or table_id.
  2  — Parent-child relationship between expected and retrieved identifier,
       or the expected figure/table belongs to the retrieved clause.
  1  — Same section / annex / domain (sibling clause, nearby sub-clause).
  0  — Unrelated chunk.

Two relevance tracks are maintained in parallel:

PAGE MATCH        retrieved_page == expected_page           (binary, legacy)
GRADED MATCH      compute_relevance_score() → 0 / 1 / 2 / 3

The GRADED track drives all new metrics (Precision, Recall, NDCG, HitRate,
MRR for clauses / figures / tables / combined).  The PAGE track is kept for
backward-compatible "found the right neighbourhood" diagnostics.

Usage
-----
    python evaluate_retrieval.py --eval evaluation_set.json --top-k 10

Output
------
  • Per-question ranked result table with graded relevance
  • Aggregate NBC RETRIEVAL PERFORMANCE SUMMARY
  • Page / Clause / Figure / Table / Combined sub-reports
  • Failure Analysis and Page-Hit / Clause-Miss Gap Analysis
  • evaluation_report.csv

Backward Compatibility
----------------------
Old eval schema:  {"relevant_clauses": [...]}
New eval schema:  {"relevant_clauses": [...],
                   "relevant_figures": [...],
                   "relevant_tables":  [...]}
Missing fields default to empty lists.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import production retriever (do NOT modify search_nbc.py)
# ---------------------------------------------------------------------------
try:
    from search_nbc import NBCSearcher, SearchResult  # type: ignore
except ModuleNotFoundError as exc:
    log.error(
        "Cannot import search_nbc.  Ensure it is on PYTHONPATH and all "
        "dependencies (voyageai, numpy, etc.) are installed."
    )
    raise SystemExit(1) from exc

# ---------------------------------------------------------------------------
# K values for all metrics
# ---------------------------------------------------------------------------
K_VALUES = (1, 3, 5, 7, 10)

# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------

def _get(result: SearchResult, field: str, default: Any = "") -> Any:
    """Read an attribute from a SearchResult dataclass safely."""
    return getattr(result, field, default)


def _norm(value: Any) -> str:
    """Lower-case and strip whitespace for case-insensitive comparison."""
    return str(value or "").strip().lower()


# ---------------------------------------------------------------------------
# Identity extraction helpers
# ---------------------------------------------------------------------------

def _clause_id(r: SearchResult) -> str:
    return _norm(_get(r, "effective_clause_id") or _get(r, "clause_id"))


def _figure_id(r: SearchResult) -> str:
    """
    Try dedicated figure_id / figure fields; fall back to clause_id when the
    chunk is obviously a figure chunk (contains 'fig.' or 'figure').
    """
    fid = _norm(_get(r, "figure_id") or _get(r, "figure"))
    if fid:
        return fid
    cid = _clause_id(r)
    if any(h in cid for h in ("fig.", "figure", "fig ")):
        return cid
    return ""


def _table_id(r: SearchResult) -> str:
    """
    Try dedicated table_id / table fields; fall back to clause_id when the
    chunk is obviously a table chunk.
    """
    tid = _norm(_get(r, "table_id") or _get(r, "table"))
    if tid:
        return tid
    cid = _clause_id(r)
    if "table" in cid:
        return cid
    return ""


def _page_num(r: SearchResult) -> int:
    try:
        return int(_get(r, "page", -1))
    except (TypeError, ValueError):
        return -1


# ---------------------------------------------------------------------------
# Parent-child heuristic
# ---------------------------------------------------------------------------

def _is_parent_child(a: str, b: str) -> bool:
    """
    True iff `a` and `b` are in a parent-child relationship under NBC
    clause-numbering conventions.

    Examples:
        B-9  / B-9.3       → True
        B-9.3 / B-9.3.1    → True
        J-5   / J-5.2      → True
    """
    if not a or not b or a == b:
        return False
    # child starts with parent followed by '.' or '-'
    return b.startswith(a + ".") or b.startswith(a + "-") \
        or a.startswith(b + ".") or a.startswith(b + "-")


def _same_section(a: str, b: str) -> bool:
    """
    True iff `a` and `b` share the same top-level section prefix.

    Examples:
        B-9.3 / B-9.4  → share 'b-9'  → True
        B-6.2.5 / B-6.2.1 → share 'b-6.2' → True
        B-9.3 / J-5    → False
    """
    if not a or not b:
        return False
    # Split on '.' and check that at least the first two segments match
    pa = a.split(".")
    pb = b.split(".")
    if len(pa) < 2 or len(pb) < 2:
        # Single-segment: compare the alphabetic prefix (annex letter)
        import re
        prefix_a = re.match(r"[a-z]+", a)
        prefix_b = re.match(r"[a-z]+", b)
        return (prefix_a is not None and prefix_b is not None
                and prefix_a.group() == prefix_b.group())
    return pa[0] == pb[0] and pa[1] == pb[1]


# ---------------------------------------------------------------------------
# Graded relevance scorer (NBC-specific)
# ---------------------------------------------------------------------------

def compute_relevance_score(
    result: SearchResult,
    expected_clauses: list[str],
    expected_figures: list[str],
    expected_tables: list[str],
) -> int:
    """
    Return a relevance grade in {0, 1, 2, 3} for `result` against the
    expected targets for a single evaluation question.

    Grading rules (highest grade wins):

      3  Exact match on clause_id, figure_id, or table_id.

      2  Parent-child relationship between expected and retrieved identifier;
         OR expected figure / table belongs to the retrieved clause
         (figure/table chunk whose clause_id matches an expected clause).

      1  Same section / annex (sibling clause or nearby sub-clause);
         OR retrieved clause's section overlaps the section that contains
         the expected figure or table.

      0  No match.
    """
    r_cid = _clause_id(result)
    r_fid = _figure_id(result)
    r_tid = _table_id(result)

    # --- Relevance 3: exact match ----------------------------------------
    for ec in expected_clauses:
        if ec and r_cid and _norm(ec) == r_cid:
            return 3
    for ef in expected_figures:
        if ef and r_fid and _norm(ef) == r_fid:
            return 3
    for et in expected_tables:
        if et and r_tid and _norm(et) == r_tid:
            return 3

    # --- Relevance 2: parent-child OR figure/table belongs to clause -----
    for ec in expected_clauses:
        if ec and r_cid and _is_parent_child(_norm(ec), r_cid):
            return 2
    for ef in expected_figures:
        if ef:
            # Figure in retrieved clause's chunk
            if r_cid and _is_parent_child(_norm(ef), r_cid):
                return 2
            # Retrieved figure is parent/child of expected figure
            if r_fid and _is_parent_child(_norm(ef), r_fid):
                return 2
    for et in expected_tables:
        if et:
            if r_cid and _is_parent_child(_norm(et), r_cid):
                return 2
            if r_tid and _is_parent_child(_norm(et), r_tid):
                return 2
    # Figure/table belongs to a clause: figure_id-like patterns alongside
    # clause targets already handled by parent-child above.
    # Additionally: if the expected figure references a clause that matches.
    for ef in expected_figures:
        if ef and r_cid:
            # e.g. expected Fig. 76 ↔ retrieved B-9.3 (clause owning fig)
            # We detect this by checking whether the retrieved clause_id is
            # a known parent of the expected figure identifier.
            # Since we don't have a lookup table, we use a heuristic:
            # if the figure id starts with "fig" and the chunk is a clause, rel=2.
            if _norm(ef).startswith("fig") and r_cid and not r_cid.startswith("fig"):
                return 2
    for et in expected_tables:
        if et and r_cid:
            if _norm(et).startswith("table") and r_cid and not r_cid.startswith("table"):
                return 2

    # --- Relevance 1: same section / annex / domain ----------------------
    for ec in expected_clauses:
        if ec and r_cid and _same_section(_norm(ec), r_cid):
            return 1
    for ef in expected_figures:
        if ef and r_cid and _same_section(_norm(ef), r_cid):
            return 1
    for et in expected_tables:
        if et and r_cid and _same_section(_norm(et), r_cid):
            return 1

    return 0


# ---------------------------------------------------------------------------
# Metric functions (typed, documented, no duplication)
# ---------------------------------------------------------------------------

def compute_precision_at_k(relevance_grades: list[int], k: int, threshold: int = 1) -> float:
    """
    Precision@K — fraction of top-K results with relevance >= threshold.

    Parameters
    ----------
    relevance_grades : list of int (0-3)
    k                : cutoff rank
    threshold        : minimum grade to count as relevant (default 1 → any
                       partial relevance counts; use 3 for exact-only)

    Returns
    -------
    float in [0, 1]
    """
    top = relevance_grades[:k]
    if not top:
        return 0.0
    return sum(1 for g in top if g >= threshold) / len(top)


def compute_recall_at_k(relevance_grades: list[int], num_relevant: int, k: int,
                         threshold: int = 1) -> float:
    """
    Recall@K — fraction of gold targets found in the top-K results.

    With a single gold target (num_relevant == 1) this equals HitRate@K.

    Parameters
    ----------
    relevance_grades : list of int (0-3)
    num_relevant     : number of distinct gold targets for this question
    k                : cutoff rank
    threshold        : minimum grade to count as a hit (default 1)

    Returns
    -------
    float in [0, 1]
    """
    if num_relevant == 0:
        return 0.0
    found = sum(1 for g in relevance_grades[:k] if g >= threshold)
    return min(found / num_relevant, 1.0)


def compute_dcg_at_k(relevance_grades: list[int], k: int) -> float:
    """
    DCG@K — Discounted Cumulative Gain at K.

    Formula: Σ_{i=1}^{K} (2^rel_i - 1) / log2(i + 1)

    Parameters
    ----------
    relevance_grades : list of int (0-3) in ranked order
    k                : cutoff rank

    Returns
    -------
    float >= 0
    """
    dcg = 0.0
    for rank, grade in enumerate(relevance_grades[:k], start=1):
        dcg += (2 ** grade - 1) / math.log2(rank + 1)
    return dcg


def compute_ndcg_at_k(relevance_grades: list[int], k: int) -> float:
    """
    NDCG@K — Normalized Discounted Cumulative Gain at K.

    Formula: DCG@K / IDCG@K
    IDCG@K = DCG of the ideal (perfectly sorted) ranking.

    Parameters
    ----------
    relevance_grades : list of int (0-3) in retrieval order
    k                : cutoff rank

    Returns
    -------
    float in [0, 1].  Returns 0.0 when IDCG == 0 (no relevant results).
    """
    ideal = sorted(relevance_grades, reverse=True)
    idcg = compute_dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return compute_dcg_at_k(relevance_grades, k) / idcg


def compute_hitrate_at_k(relevance_grades: list[int], k: int, threshold: int = 1) -> bool:
    """
    HitRate@K — True iff at least one result in top-K has relevance >= threshold.

    Parameters
    ----------
    relevance_grades : list of int (0-3)
    k                : cutoff rank
    threshold        : minimum grade to count (default 1)

    Returns
    -------
    bool
    """
    return any(g >= threshold for g in relevance_grades[:k])


def compute_mrr(relevance_grades: list[int], threshold: int = 1) -> float:
    """
    Reciprocal Rank — 1 / rank of the first result with relevance >= threshold.
    Returns 0.0 if no hit exists.

    Parameters
    ----------
    relevance_grades : list of int (0-3) in ranked order
    threshold        : minimum grade to count (default 1)

    Returns
    -------
    float in [0, 1]
    """
    for rank, grade in enumerate(relevance_grades, start=1):
        if grade >= threshold:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Legacy binary helpers (page-level and exact clause-level)
# ---------------------------------------------------------------------------

def _page_match(result: SearchResult, expected_page: int | str) -> bool:
    """True iff the retrieved chunk's page equals the expected page."""
    try:
        return int(_get(result, "page", -1)) == int(expected_page)
    except (TypeError, ValueError):
        return False


def _exact_clause_match(result: SearchResult, expected_clauses: list[str]) -> bool:
    r_cid = _clause_id(result)
    return bool(r_cid) and any(_norm(ec) == r_cid for ec in expected_clauses if ec)


# ---------------------------------------------------------------------------
# Failure-reason classifiers (diagnostics — unchanged from original)
# ---------------------------------------------------------------------------

_FIGURE_HINTS = ("figure", "fig.", "fig ")
_TABLE_HINTS  = ("table",)


def _looks_like_figure(result: SearchResult) -> bool:
    chunk_type = _norm(
        _get(result, "chunk_type") or _get(result, "content_type") or _get(result, "block_type")
    )
    if chunk_type in ("figure", "image"):
        return True
    return any(h in _clause_id(result) for h in _FIGURE_HINTS)


def _looks_like_table(result: SearchResult) -> bool:
    chunk_type = _norm(
        _get(result, "chunk_type") or _get(result, "content_type") or _get(result, "block_type")
    )
    if chunk_type == "table":
        return True
    return any(h in _clause_id(result) for h in _TABLE_HINTS)


def _is_parent_of(parent: str, child: str) -> bool:
    """Heuristic: True iff `parent` is a clause-numbering ancestor of `child`."""
    p, c = _norm(parent), _norm(child)
    if not p or not c or p == c:
        return False
    return c.startswith(p + ".") or c.startswith(p + "-")


def _clause_path_match(result: SearchResult, clause_path: str) -> bool:
    """DIAGNOSTIC ONLY — never folded into any metric."""
    norm_path = _norm(clause_path)
    if not norm_path:
        return False
    r_cpath = _norm(_get(result, "effective_clause_path") or _get(result, "clause_path"))
    return norm_path in r_cpath


def _classify_failure(
    rank1: SearchResult | None,
    expected_clauses: list[str],
    expected_page: int | str,
) -> str:
    if rank1 is None:
        return "No results retrieved"
    r_cid = str(_get(rank1, "effective_clause_id") or _get(rank1, "clause_id") or "").strip()
    if not r_cid or r_cid.lower() == "none":
        return "Missing clause metadata"
    if _looks_like_figure(rank1):
        return "Figure retrieved instead of clause"
    if _looks_like_table(rank1):
        return "Table retrieved instead of clause"
    for ec in expected_clauses:
        if ec and (_is_parent_of(r_cid, ec) or _is_parent_of(ec, r_cid)):
            return "Parent clause retrieved instead of child clause"
    if _page_match(rank1, expected_page):
        return "Correct page but wrong clause"
    return "Other / unclassified"


# ---------------------------------------------------------------------------
# Eval-item field helpers
# ---------------------------------------------------------------------------

def _expected_clauses_for(item: dict[str, Any]) -> list[str]:
    """
    Collect every clause string that counts as a clause hit.
    Supports both old schema (relevant_clauses only) and new schema
    (relevant_clauses + relevant_figures + relevant_tables).
    """
    clauses: list[str] = list(item.get("relevant_clauses") or [])
    single = item.get("expected_clause") or item.get("clause_id")
    if single and single not in clauses:
        clauses.append(single)
    return clauses


def _expected_figures_for(item: dict[str, Any]) -> list[str]:
    """Return expected figure identifiers; defaults to [] for old schema."""
    return list(item.get("relevant_figures") or [])


def _expected_tables_for(item: dict[str, Any]) -> list[str]:
    """Return expected table identifiers; defaults to [] for old schema."""
    return list(item.get("relevant_tables") or [])


# ---------------------------------------------------------------------------
# Metrics for a single track (e.g. clause-only exact, figure-only, combined)
# ---------------------------------------------------------------------------

def _track_metrics(
    grades: list[int],
    num_relevant: int,
) -> dict[str, Any]:
    """
    Compute all K-wise metrics for one graded relevance vector.

    Returns a dict with keys:
        p_at, r_at, h_at, ndcg_at (each a dict keyed by k), mrr (float)
    """
    p_at    = {k: compute_precision_at_k(grades, k) for k in K_VALUES}
    r_at    = {k: compute_recall_at_k(grades, num_relevant, k) for k in K_VALUES}
    h_at    = {k: compute_hitrate_at_k(grades, k) for k in K_VALUES}
    ndcg_at = {k: compute_ndcg_at_k(grades, k) for k in K_VALUES}
    mrr     = compute_mrr(grades)
    return {"p_at": p_at, "r_at": r_at, "h_at": h_at, "ndcg_at": ndcg_at, "mrr": mrr}


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def _run_evaluation(
    eval_items: list[dict[str, Any]],
    searcher: NBCSearcher,
    retrieve_k: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Run retrieval once per question and compute all metrics.

    Returns
    -------
    rows      : per-question result dicts (console + CSV data)
    aggregate : aggregate metrics across all questions
    """
    n = len(eval_items)

    # Running sums for aggregate — one dict per track
    # Tracks: page (binary), clause (binary exact), combined (graded)
    sums: dict[str, dict] = {
        track: {
            "p":    {k: 0.0 for k in K_VALUES},
            "r":    {k: 0.0 for k in K_VALUES},
            "h":    {k: 0.0 for k in K_VALUES},
            "ndcg": {k: 0.0 for k in K_VALUES},
            "mrr":  0.0,
        }
        for track in ("page", "clause", "figure", "table", "combined")
    }

    rows: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}

    for idx, item in enumerate(eval_items, start=1):
        question          = item.get("question", "")
        expected_clauses  = _expected_clauses_for(item)
        expected_figures  = _expected_figures_for(item)
        expected_tables   = _expected_tables_for(item)
        expected_page     = item.get("source_page", -1)
        clause_path       = item.get("clause_path", "")
        domain            = item.get("domain", "")

        log.info("[%02d/%d] %s", idx, n, question[:72])

        # ------------------------------------------------------------------
        # Retrieve
        # ------------------------------------------------------------------
        try:
            results: list[SearchResult] = searcher.search(
                query=question, top_k=retrieve_k, rerank=True
            )
        except Exception as exc:
            log.warning("Retrieval failed for Q%d: %s", idx, exc)
            results = []

        # Pad to retrieve_k so slice [:k] always works
        pad = retrieve_k - len(results)

        # ------------------------------------------------------------------
        # Relevance vectors
        # ------------------------------------------------------------------
        # PAGE — binary, legacy
        page_rel_bin: list[int] = [
            1 if _page_match(r, expected_page) else 0 for r in results
        ] + [0] * pad

        # GRADED — NBC model
        graded: list[int] = [
            compute_relevance_score(r, expected_clauses, expected_figures, expected_tables)
            for r in results
        ] + [0] * pad

        # Clause-exact — binary (grade 3 only)
        clause_bin: list[int] = [
            1 if _exact_clause_match(r, expected_clauses) else 0 for r in results
        ] + [0] * pad

        # Figure-exact — grade 3 for figure_id match
        figure_bin: list[int] = [
            1 if (
                expected_figures
                and _figure_id(r)
                and any(_norm(ef) == _figure_id(r) for ef in expected_figures if ef)
            ) else 0
            for r in results
        ] + [0] * pad

        # Table-exact — grade 3 for table_id match
        table_bin: list[int] = [
            1 if (
                expected_tables
                and _table_id(r)
                and any(_norm(et) == _table_id(r) for et in expected_tables if et)
            ) else 0
            for r in results
        ] + [0] * pad

        # Number of relevant items per track
        num_rel_clause  = max(len(expected_clauses), 1)
        num_rel_figure  = max(len(expected_figures), 1) if expected_figures else 1
        num_rel_table   = max(len(expected_tables),  1) if expected_tables  else 1
        num_rel_combined = max(
            len(expected_clauses) + len(expected_figures) + len(expected_tables), 1
        )

        # ------------------------------------------------------------------
        # Per-track metrics
        # ------------------------------------------------------------------
        m_page     = _track_metrics(page_rel_bin, 1)
        m_clause   = _track_metrics(clause_bin,   num_rel_clause)
        m_figure   = _track_metrics(figure_bin,   num_rel_figure)
        m_table    = _track_metrics(table_bin,    num_rel_table)
        m_combined = _track_metrics(graded,        num_rel_combined)

        # ------------------------------------------------------------------
        # Accumulate for aggregate
        # ------------------------------------------------------------------
        for k in K_VALUES:
            sums["page"]["p"][k]    += m_page["p_at"][k]
            sums["page"]["r"][k]    += m_page["r_at"][k]
            sums["page"]["h"][k]    += float(m_page["h_at"][k])
            sums["page"]["ndcg"][k] += m_page["ndcg_at"][k]

            sums["clause"]["p"][k]    += m_clause["p_at"][k]
            sums["clause"]["r"][k]    += m_clause["r_at"][k]
            sums["clause"]["h"][k]    += float(m_clause["h_at"][k])
            sums["clause"]["ndcg"][k] += m_clause["ndcg_at"][k]

            sums["figure"]["p"][k]    += m_figure["p_at"][k]
            sums["figure"]["r"][k]    += m_figure["r_at"][k]
            sums["figure"]["h"][k]    += float(m_figure["h_at"][k])
            sums["figure"]["ndcg"][k] += m_figure["ndcg_at"][k]

            sums["table"]["p"][k]    += m_table["p_at"][k]
            sums["table"]["r"][k]    += m_table["r_at"][k]
            sums["table"]["h"][k]    += float(m_table["h_at"][k])
            sums["table"]["ndcg"][k] += m_table["ndcg_at"][k]

            sums["combined"]["p"][k]    += m_combined["p_at"][k]
            sums["combined"]["r"][k]    += m_combined["r_at"][k]
            sums["combined"]["h"][k]    += float(m_combined["h_at"][k])
            sums["combined"]["ndcg"][k] += m_combined["ndcg_at"][k]

        sums["page"]["mrr"]     += m_page["mrr"]
        sums["clause"]["mrr"]   += m_clause["mrr"]
        sums["figure"]["mrr"]   += m_figure["mrr"]
        sums["table"]["mrr"]    += m_table["mrr"]
        sums["combined"]["mrr"] += m_combined["mrr"]

        # ------------------------------------------------------------------
        # Failure diagnostics
        # ------------------------------------------------------------------
        failure_reason = ""
        if not m_clause["h_at"][max(K_VALUES)]:
            failure_reason = _classify_failure(
                results[0] if results else None, expected_clauses, expected_page
            )
            failure_counts[failure_reason] = failure_counts.get(failure_reason, 0) + 1

        # ------------------------------------------------------------------
        # Build per-question row
        # ------------------------------------------------------------------
        display = results[:top_k]
        ret_clauses = [
            str(_get(r, "effective_clause_id") or _get(r, "clause_id") or "None")
            for r in display
        ]
        ret_figures = [str(_figure_id(r) or "None") for r in display]
        ret_tables  = [str(_table_id(r)  or "None") for r in display]
        ret_pages   = [str(_get(r, "page")) for r in display]
        ret_scores  = [
            f"{float(_get(r, 'reranked_score', 0) or _get(r, 'score', 0)):.4f}"
            for r in display
        ]
        ret_grades  = [
            compute_relevance_score(r, expected_clauses, expected_figures, expected_tables)
            for r in display
        ]

        row: dict[str, Any] = {
            "question":          question,
            "domain":            domain,
            "expected_clauses":  "; ".join(expected_clauses),
            "expected_figures":  "; ".join(expected_figures),
            "expected_tables":   "; ".join(expected_tables),
            "expected_page":     expected_page,
            "retrieved_clauses": "; ".join(ret_clauses),
            "retrieved_figures": "; ".join(ret_figures),
            "retrieved_tables":  "; ".join(ret_tables),
            "retrieved_pages":   "; ".join(ret_pages),
            "relevance_scores":  "; ".join(str(g) for g in ret_grades),
            "failure_reason":    failure_reason,
        }

        # Flatten metrics into CSV-friendly columns
        for k in K_VALUES:
            row[f"p_at_{k}"]      = round(m_combined["p_at"][k],    4)
            row[f"r_at_{k}"]      = round(m_combined["r_at"][k],    4)
            row[f"ndcg_at_{k}"]   = round(m_combined["ndcg_at"][k], 4)
            row[f"hit_at_{k}"]    = int(m_combined["h_at"][k])
            row[f"clause_p_{k}"]  = round(m_clause["p_at"][k],      4)
            row[f"clause_r_{k}"]  = round(m_clause["r_at"][k],      4)
            row[f"clause_hit_{k}"]= int(m_clause["h_at"][k])
            row[f"page_p_{k}"]    = round(m_page["p_at"][k],        4)
            row[f"page_r_{k}"]    = round(m_page["r_at"][k],        4)
            row[f"page_hit_{k}"]  = int(m_page["h_at"][k])
        row["mrr"]        = round(m_combined["mrr"], 4)
        row["clause_mrr"] = round(m_clause["mrr"],   4)
        row["page_mrr"]   = round(m_page["mrr"],     4)

        # Private fields (console rendering only — stripped before CSV write)
        row["_display"]    = display
        row["_grades"]     = ret_grades
        row["_scores"]     = ret_scores
        row["_m_combined"] = m_combined
        row["_m_clause"]   = m_clause
        row["_m_figure"]   = m_figure
        row["_m_table"]    = m_table
        row["_m_page"]     = m_page
        row["_clause_bin"] = clause_bin
        row["_page_bin"]   = page_rel_bin

        rows.append(row)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    def _avg(sums_track: dict) -> dict:
        return {
            "p":    {k: sums_track["p"][k]    / n for k in K_VALUES},
            "r":    {k: sums_track["r"][k]    / n for k in K_VALUES},
            "h":    {k: sums_track["h"][k]    / n for k in K_VALUES},
            "ndcg": {k: sums_track["ndcg"][k] / n for k in K_VALUES},
            "mrr":  sums_track["mrr"]              / n,
        }

    aggregate = {
        "n_questions":    n,
        "avgs":           {track: _avg(sums[track]) for track in sums},
        "failure_counts": failure_counts,
    }
    return rows, aggregate


# ---------------------------------------------------------------------------
# Console printing
# ---------------------------------------------------------------------------

def _print_per_question(rows: list[dict[str, Any]]) -> None:
    """Print a ranked result table + per-question metrics for every question."""
    print()
    print("═" * 80)
    print("  PER-QUESTION RESULTS")
    print("═" * 80)

    for idx, row in enumerate(rows, start=1):
        display: list[SearchResult] = row["_display"]
        grades:  list[int]          = row["_grades"]
        scores:  list[str]          = row["_scores"]
        m_comb  = row["_m_combined"]
        m_cl    = row["_m_clause"]

        print()
        print("─" * 80)
        print(f"[{idx:02d}] {row['question']}")
        print(f"  Domain           : {row['domain']}")
        print(f"  Expected Clauses : {row['expected_clauses'] or '(none)'}")
        print(f"  Expected Figures : {row['expected_figures'] or '(none)'}")
        print(f"  Expected Tables  : {row['expected_tables']  or '(none)'}")
        print(f"  Expected Page    : {row['expected_page']}")
        print()
        print("  Retrieved Results:")
        for rank, (r, grade, score) in enumerate(
            zip(display, grades, scores), start=1
        ):
            cid  = str(_get(r, "effective_clause_id") or _get(r, "clause_id") or "—")
            fid  = _figure_id(r) or "—"
            tid  = _table_id(r)  or "—"
            page = str(_get(r, "page") or "—")
            grade_label = {3: "EXACT", 2: "PARENT/CHILD", 1: "SIBLING", 0: "UNRELATED"}
            print(
                f"    Rank {rank:>2}  Clause={cid:<12}  Figure={fid:<10}  "
                f"Table={tid:<10}  Page={page:<5}  "
                f"Score={score}  Rel={grade} ({grade_label.get(grade, '?')})"
            )
        print()
        print("  Combined (graded) metrics:")
        print(
            "    " + "  ".join(
                f"P@{k}={m_comb['p_at'][k]:.3f}" for k in K_VALUES
            )
        )
        print(
            "    " + "  ".join(
                f"R@{k}={m_comb['r_at'][k]:.3f}" for k in K_VALUES
            )
        )
        print(
            "    " + "  ".join(
                f"N@{k}={m_comb['ndcg_at'][k]:.3f}" for k in K_VALUES
            )
        )
        print(
            "    " + "  ".join(
                f"H@{k}={'Y' if m_comb['h_at'][k] else 'N'}" for k in K_VALUES
            )
        )
        print(f"    MRR={m_comb['mrr']:.4f}  (clause-exact MRR={m_cl['mrr']:.4f})")
        if row["failure_reason"]:
            print(f"  Failure reason   : {row['failure_reason']}")


def _print_metric_block(
    title: str,
    n: int,
    retrieve_k: int,
    avgs: dict,
    include_ndcg: bool = True,
) -> None:
    """Generic tabular block for a single retrieval track."""
    W = 9
    ks = [str(k) for k in K_VALUES]
    header = f"{'Metric':<14}" + "".join(f"{'@'+k:>{W}}" for k in ks)
    sep    = "─" * len(header)

    print()
    print("═" * len(header))
    print(f"  {title}")
    print(f"  Questions: {n}  |  retrieve_k: {retrieve_k}")
    print(sep)
    print(header)
    print(sep)
    print(f"{'Precision':<14}" + "".join(f"{avgs['p'][k]:>{W}.4f}" for k in K_VALUES))
    print(f"{'Recall':<14}"    + "".join(f"{avgs['r'][k]:>{W}.4f}" for k in K_VALUES))
    if include_ndcg:
        print(f"{'NDCG':<14}"  + "".join(f"{avgs['ndcg'][k]:>{W}.4f}" for k in K_VALUES))
    print(f"{'HitRate':<14}"   + "".join(f"{avgs['h'][k]:>{W}.4f}" for k in K_VALUES))
    print(sep)
    print(f"{'MRR':<14}{avgs['mrr']:>{W}.4f}")
    print("═" * len(header))


def _print_aggregate(agg: dict[str, Any], retrieve_k: int) -> None:
    n    = agg["n_questions"]
    avgs = agg["avgs"]

    print()
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  NBC RETRIEVAL PERFORMANCE SUMMARY".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"  Questions evaluated: {n}")

    _print_metric_block("COMBINED RETRIEVAL PERFORMANCE (Clause OR Figure OR Table — graded)",
                        n, retrieve_k, avgs["combined"], include_ndcg=True)
    _print_metric_block("CLAUSE RETRIEVAL PERFORMANCE (exact match)",
                        n, retrieve_k, avgs["clause"],   include_ndcg=False)
    _print_metric_block("FIGURE RETRIEVAL PERFORMANCE (exact match)",
                        n, retrieve_k, avgs["figure"],   include_ndcg=False)
    _print_metric_block("TABLE RETRIEVAL PERFORMANCE (exact match)",
                        n, retrieve_k, avgs["table"],    include_ndcg=False)
    _print_metric_block("PAGE RETRIEVAL PERFORMANCE (legacy binary)",
                        n, retrieve_k, avgs["page"],     include_ndcg=False)


def _print_failure_reasons(agg: dict[str, Any]) -> None:
    failure_counts: dict[str, int] = agg["failure_counts"]
    n = agg["n_questions"]
    total = sum(failure_counts.values())

    print()
    print("═" * 76)
    print(f"  FAILURE ANALYSIS  (questions with Clause Hit@{max(K_VALUES)} = NO)")
    print("─" * 76)
    if not failure_counts:
        print(f"  None — every question had a clause hit within top-{max(K_VALUES)}.")
    else:
        for reason, count in sorted(failure_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>3}  ({count/n:5.1%})  {reason}")
    print("─" * 76)
    print(f"  {total}/{n} questions ({total/n:.1%}) failed clause retrieval @{max(K_VALUES)}")
    print("═" * 76)


def _print_gap_analysis(rows: list[dict[str, Any]]) -> None:
    """Questions where correct PAGE was found but correct CLAUSE was not (top-5)."""
    gaps = [
        r for r in rows
        if r["_m_page"]["h_at"].get(5, False) and not r["_m_clause"]["h_at"].get(5, False)
    ]
    n = len(rows)

    print()
    print("═" * 76)
    print("  PAGE-HIT / CLAUSE-MISS GAP  (correct page, wrong clause @5)")
    print("─" * 76)
    if not gaps:
        print("  None — every page hit also produced a clause hit within top-5.")
    else:
        for r in gaps:
            print(f"  • {r['question'][:60]}")
            print(f"      expected={r['expected_clauses'] or '(none)'}  "
                  f"retrieved={r['retrieved_clauses']}")
    print("─" * 76)
    print(f"  {len(gaps)}/{n} questions ({len(gaps)/n:.1%}) have this gap")
    print("═" * 76)


# ---------------------------------------------------------------------------
# Top-level evaluate()
# ---------------------------------------------------------------------------

def evaluate(eval_path: Path, top_k: int, report_csv: Path) -> None:
    """Load eval set, run evaluation, print all reports, write CSV."""

    log.info("Loading evaluation set from %s", eval_path)
    try:
        eval_items: list[dict[str, Any]] = json.loads(
            eval_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as exc:
        log.error("Failed to read evaluation set: %s", exc)
        raise SystemExit(1) from exc

    if not eval_items:
        log.error("Evaluation set is empty.")
        raise SystemExit(1)
    log.info("Loaded %d questions.", len(eval_items))

    log.info("Initialising NBCSearcher …")
    try:
        searcher = NBCSearcher()
        searcher.load()
    except Exception as exc:
        log.error("Failed to initialise NBCSearcher: %s", exc)
        raise SystemExit(1) from exc
    log.info("NBCSearcher ready.")

    # Always retrieve at least max(K_VALUES) results so all @K metrics work
    retrieve_k = max(top_k, max(K_VALUES))

    rows, agg = _run_evaluation(eval_items, searcher, retrieve_k, top_k)

    _print_per_question(rows)
    _print_aggregate(agg, retrieve_k)
    _print_failure_reasons(agg)
    _print_gap_analysis(rows)

    # ------------------------------------------------------------------
    # Write CSV (strip private _* keys)
    # ------------------------------------------------------------------
    csv_rows = [
        {k: v for k, v in row.items() if not k.startswith("_")}
        for row in rows
    ]
    if csv_rows:
        log.info("Writing CSV → %s", report_csv)
        try:
            with report_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
            log.info("CSV written.")
        except OSError as exc:
            log.error("Failed to write CSV: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate NBC RAG retrieval quality using graded relevance (0-3) "
            "with NDCG, Precision, Recall, HitRate, and MRR at K ∈ {1,3,5,7,10}."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--eval", type=Path, default=Path("evaluation_set.json"),
        help="Path to evaluation_set.json",
    )
    p.add_argument(
        "--top-k", type=int, default=10,
        help="Number of results to display per question in the per-question report",
    )
    p.add_argument(
        "--report-csv", type=Path, default=Path("evaluation_report.csv"),
        help="Output path for per-question CSV report",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if not args.eval.exists():
        log.error("Evaluation file not found: %s", args.eval)
        sys.exit(1)
    if args.top_k < 1:
        log.error("--top-k must be >= 1")
        sys.exit(1)

    evaluate(eval_path=args.eval, top_k=args.top_k, report_csv=args.report_csv)


if __name__ == "__main__":
    main()