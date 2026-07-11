"""
search_nbc.py — Semantic search over the NBC 2016 Vol-1 corpus.

Loads pre-built embeddings (output/embeddings.npy) and metadata
(output/embed_meta.json), embeds the query with the same backend used
at index time, and returns the top-k most similar chunks ranked by
cosine similarity.

Supported embedding backends
─────────────────────────────
  voyageai   — VoyageAI voyage-law-2  (default, matches the corpus)
  openai     — OpenAI text-embedding-3-small / -large
  local      — sentence-transformers (any HuggingFace model)

Usage
─────
  python search_nbc.py "accessible toilet layout"
  python search_nbc.py "minimum corridor width" --top-k 20
  python search_nbc.py "fire egress staircase" --backend openai --top-k 5
  python search_nbc.py "ramp gradient" --show-images --show-full-text
  python search_nbc.py "door width" --export-json results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,
)
log = logging.getLogger("search_nbc")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent
OUTPUT_DIR  = ROOT / "output"
EMBED_PATH  = OUTPUT_DIR / "embeddings.npy"
META_PATH   = OUTPUT_DIR / "embed_meta.json"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """One ranked retrieval result."""
    rank:        int
    score:       float
    chunk_id:    str
    source_file: str
    pdf_page:    int | None
    page:        int | str | None
    clause_id:   str
    clause_path: str
    image_paths: list[str]
    figures:     list[str]
    text_preview: str
    full_text:    str   = ""     # populated only when --show-full-text
    reranked_score: float = 0.0  # set by rerank_results(); 0.0 = not reranked
    # Issue 2 — effective clause fields (set by resolve_clause_context())
    effective_clause_id:   str = ""
    effective_clause_path: str = ""
    # Issue 3 — figure caption (resolved from figure_metadata)
    figure_captions: dict[str, str] = field(default_factory=dict)  # fig_id -> caption

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank":                  self.rank,
            "score":                 round(self.score, 6),
            "reranked_score":        round(self.reranked_score, 6),
            "chunk_id":              self.chunk_id,
            "source_file":           self.source_file,
            "pdf_page":              self.pdf_page,
            "page":                  self.page,
            "clause_id":             self.clause_id,
            "clause_path":           self.clause_path,
            "effective_clause_id":   self.effective_clause_id,
            "effective_clause_path": self.effective_clause_path,
            "image_paths":           self.image_paths,
            "figures":               self.figures,
            "figure_captions":       self.figure_captions,
            "text_preview":          self.text_preview,
        }


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

def _embed_voyageai(texts: list[str], model: str = "voyage-law-2") -> np.ndarray:
    """Embed using VoyageAI.  pip install voyageai"""
    try:
        import voyageai  # type: ignore
    except ImportError:
        sys.exit("ERROR: voyageai not installed.  pip install voyageai")

    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        sys.exit("ERROR: VOYAGE_API_KEY not set in environment or .env")

    client   = voyageai.Client(api_key=api_key)
    response = client.embed(texts, model=model, input_type="query")
    return np.array(response.embeddings, dtype=np.float32)


def _embed_openai(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed using OpenAI.  pip install openai"""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        sys.exit("ERROR: openai not installed.  pip install openai")

    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set in environment or .env")

    client   = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=texts, model=model)
    vecs     = [item.embedding for item in response.data]
    return np.array(vecs, dtype=np.float32)


def _embed_local(texts: list[str], model: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed using sentence-transformers.  pip install sentence-transformers"""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        sys.exit("ERROR: sentence-transformers not installed.  pip install sentence-transformers")

    encoder = SentenceTransformer(model)
    vecs    = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# NBCSearcher
# ---------------------------------------------------------------------------

class NBCSearcher:
    """
    Loads the NBC corpus index and answers similarity queries.

    Parameters
    ----------
    embed_path : Path to the .npy embeddings file.
    meta_path  : Path to the embed_meta.json metadata file.
    backend    : Embedding backend — 'voyageai' | 'openai' | 'local'.
    model      : Model name override for the chosen backend.
    """

    def __init__(
        self,
        embed_path: Path = EMBED_PATH,
        meta_path:  Path = META_PATH,
        backend:    str  = "voyageai",
        model:      str  = "",
    ) -> None:
        self.embed_path = embed_path
        self.meta_path  = meta_path
        self.backend    = backend.lower()
        self.model      = model

        self._embeddings: np.ndarray | None = None   # (N, D) float32, L2-normalised
        self._meta:       list[dict]         = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "NBCSearcher":
        """Load embeddings and metadata from disk."""
        if not self.embed_path.exists():
            sys.exit(f"ERROR: embeddings not found at {self.embed_path}")
        if not self.meta_path.exists():
            sys.exit(f"ERROR: metadata not found at {self.meta_path}")

        log.info("Loading embeddings from %s", self.embed_path)
        raw = np.load(self.embed_path).astype(np.float32)

        # Ensure unit-norm (cosine similarity == dot product)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings = raw / norms

        log.info("Loading metadata from %s", self.meta_path)
        with open(self.meta_path, encoding="utf-8") as f:
            self._meta = json.load(f)

        if len(self._meta) != self._embeddings.shape[0]:
            sys.exit(
                f"ERROR: embeddings has {self._embeddings.shape[0]} rows but "
                f"metadata has {len(self._meta)} entries — index is inconsistent."
            )

        log.info(
            "Index loaded: %d chunks, embedding dim=%d",
            self._embeddings.shape[0],
            self._embeddings.shape[1],
        )
        return self

    def embed_query(self, query: str) -> np.ndarray:
        """Embed *query* with the configured backend.  Returns shape (1, D)."""
        log.info("Embedding query via backend=%s", self.backend)

        if self.backend == "voyageai":
            model = self.model or "voyage-law-2"
            vec   = _embed_voyageai([query], model=model)
        elif self.backend == "openai":
            model = self.model or "text-embedding-3-small"
            vec   = _embed_openai([query], model=model)
        elif self.backend == "local":
            model = self.model or "all-MiniLM-L6-v2"
            vec   = _embed_local([query], model=model)
        else:
            sys.exit(f"ERROR: unknown backend '{self.backend}'.  Choose voyageai / openai / local.")

        # Normalise
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        return (vec / norm).astype(np.float32)

    def search(
        self,
        query:         str,
        top_k:         int  = 10,
        include_text:  bool = False,
        rerank:        bool = True,   # Issue 4: heuristic reranking (default on)
        debug:         bool = False,  # Issue 7: print debug diagnostics
    ) -> list[SearchResult]:
        """
        Embed *query* and return the top-*k* chunks ranked by cosine similarity,
        optionally followed by heuristic reranking.

        Parameters
        ----------
        query        : Natural-language query string.
        top_k        : Number of results to return.
        include_text : If True, populate SearchResult.full_text from metadata.
        rerank       : Apply heuristic reranking after cosine similarity.
        debug        : Print per-result diagnostic lines to stdout.
        """
        if self._embeddings is None:
            self.load()

        expanded = expand_query(query)                         # Issue 1: domain expansion
        q_vec = self.embed_query(expanded)                     # (1, D)
        scores = (self._embeddings @ q_vec.T).squeeze()        # (N,)

        top_k   = min(top_k, len(scores))
        indices = np.argpartition(scores, -top_k)[-top_k:]     # fast top-k
        indices = indices[np.argsort(scores[indices])[::-1]]   # sort descending

        results: list[SearchResult] = []
        for rank, idx in enumerate(indices, start=1):
            m     = self._meta[int(idx)]
            score = float(scores[idx])

            results.append(SearchResult(
                rank         = rank,
                score        = score,
                chunk_id     = m.get("chunk_id",    ""),
                source_file  = m.get("source_file", ""),
                pdf_page     = m.get("pdf_page") or m.get("page"),
                page         = m.get("page"),
                clause_id    = m.get("clause_id",   ""),
                clause_path  = m.get("clause_path", ""),
                image_paths  = m.get("image_paths", []),
                figures      = m.get("figures",     []),
                text_preview = m.get("text_preview", ""),
                full_text    = m.get("text", "") if include_text else "",
            ))

        # Issue 2: resolve effective clause IDs for chunks missing clause context
        results = resolve_clause_context(results)

        # Issue 3: attach figure captions from metadata
        results = resolve_figure_captions(results, self._meta)

        # Issue 4: optional heuristic reranking
        if rerank:
            results = rerank_results(results, query=query)

        # Issue 5: structured debug output
        if debug:
            print("[DEBUG] Per-result diagnostics:")
            for r in results:
                rs  = getattr(r, "reranked_score", r.score)
                inh = getattr(r, "_inheritance_source", "unknown")
                dbg = getattr(r, "_rerank_debug", {})

                boost_lines = []
                for key, val in dbg.items():
                    label = key.replace("_", " ")
                    boost_lines.append(f"  +{val:.2f} {label}" if val >= 0 else f"  {val:.2f} {label}")

                boosts_str = "\n".join(boost_lines) if boost_lines else "  (none)"
                print(
                    f"\n  [DEBUG] chunk_id={r.chunk_id}\n"
                    f"    similarity : {r.score:.4f}\n"
                    f"    boosts:\n{boosts_str}\n"
                    f"    final      : {rs:.4f}\n"
                    f"    inheritance: {inh}\n"
                    f"    clause     : {r.effective_clause_id or r.clause_id or '—'}\n"
                    f"    source     : {r.source_file}  pdf_page={r.pdf_page}"
                )

        log.info("Search complete: %d results for query=%r", len(results), query[:60])
        return results



# ---------------------------------------------------------------------------
# Issue 2 — Retrieval-time clause inheritance
# ---------------------------------------------------------------------------

def _section_prefix(clause_path: str) -> str:
    """Return the top-level Section/Annex/Part prefix of a clause_path.

    Used to prevent cross-section inheritance: a chunk from Annex B must
    never inherit a clause_id from Part 4 just because it appears later
    in the result list.

    Examples:
        "PART 6 > Section 5 > 38.7" -> "PART 6 > Section 5"
        "Annex B > B-9"              -> "Annex B"
        "B-9.2"                      -> "B-9"         (single-level annex)
        ""                           -> ""
    """
    if not clause_path:
        return ""
    parts = [p.strip() for p in clause_path.split(">")]
    # Keep up to first two breadcrumb components (Part/Section or Annex)
    return " > ".join(parts[:2]) if len(parts) >= 2 else parts[0]


def resolve_clause_context(results: list[SearchResult]) -> list[SearchResult]:
    """Infer effective_clause_id / effective_clause_path for chunks that lack them.

    Four-tier priority — evaluated in order, stopping at first match.

    Tier 1  chunk.clause_id / clause_path already set (explicit).
    Tier 2  Nearest previous result from the SAME source_file
            that has a clause_id, provided it shares the same section prefix
            (prevents Annex B inheriting from a Part 6 Section 5 result).
    Tier 3  Nearest previous result with the same pdf_page that has a
            clause_id and the same section prefix.
    Tier 4  Nearest previous result with the same clause_path prefix and a
            clause_id (sibling inheritance within the same structural section).
    Fallback: leave effective_clause_id / path blank rather than inherit
              from a different section when a same-page parent exists.

    Sets effective_clause_id and effective_clause_path on every result
    without mutating the original clause_id / clause_path fields.
    Also records _inheritance_source for --debug diagnostics.
    """
    # Tier 2: source_file -> (clause_id, clause_path)
    last_by_source: dict[str, tuple[str, str]] = {}
    # Tier 3: pdf_page -> (clause_id, clause_path)
    last_by_page:   dict[int | None, tuple[str, str]] = {}
    # Tier 4: section_prefix -> (clause_id, clause_path)
    last_by_section: dict[str, tuple[str, str]] = {}

    for r in results:
        cid   = (r.clause_id   or "").strip()
        cpath = (r.clause_path or "").strip()
        sec   = _section_prefix(cpath)

        if cid:
            # Tier 1 — explicit clause on chunk
            r.effective_clause_id   = cid
            r.effective_clause_path = cpath
            r._inheritance_source   = "explicit"          # type: ignore[attr-defined]
            # Update all inheritance caches
            last_by_source[r.source_file]  = (cid, cpath)
            last_by_page[r.pdf_page]       = (cid, cpath)
            if sec:
                last_by_section[sec]       = (cid, cpath)
        else:
            r_sec = _section_prefix(r.clause_path or "")

            # Tier 2 — same source_file, same section
            if r.source_file in last_by_source:
                prev_id, prev_path = last_by_source[r.source_file]
                if not r_sec or _section_prefix(prev_path) == r_sec:
                    r.effective_clause_id   = prev_id
                    r.effective_clause_path = prev_path
                    r._inheritance_source   = "same_source_file"   # type: ignore[attr-defined]
                    continue

            # Tier 3 — same pdf_page, same section
            if r.pdf_page in last_by_page:
                prev_id, prev_path = last_by_page[r.pdf_page]
                if not r_sec or _section_prefix(prev_path) == r_sec:
                    r.effective_clause_id   = prev_id
                    r.effective_clause_path = prev_path
                    r._inheritance_source   = "same_pdf_page"      # type: ignore[attr-defined]
                    continue

            # Tier 4 — same section prefix
            if r_sec and r_sec in last_by_section:
                prev_id, prev_path = last_by_section[r_sec]
                r.effective_clause_id   = prev_id
                r.effective_clause_path = prev_path
                r._inheritance_source   = "same_section_prefix"   # type: ignore[attr-defined]
                continue

            # No valid parent found in any tier
            r.effective_clause_id   = ""
            r.effective_clause_path = ""
            r._inheritance_source   = "none"                       # type: ignore[attr-defined]

    return results


# ---------------------------------------------------------------------------
# Issue 3 — Figure caption resolution
# ---------------------------------------------------------------------------

def resolve_figure_captions(
    results: list[SearchResult],
    meta:    list[dict],
) -> list[SearchResult]:
    """Attach figure captions to each SearchResult from embed_meta figure_metadata.

    embed_meta entries may contain a 'figure_metadata' list:
        [{"id": "Fig. 80", "caption": "Accessible Toilet Layout"}, ...]
    """
    caption_map: dict[str, str] = {}
    for m in meta:
        for fm in (m.get("figure_metadata") or []):
            fid = (fm.get("id") or "").strip()
            cap = (fm.get("caption") or "").strip()
            if fid and cap and fid not in caption_map:
                caption_map[fid] = cap

    for r in results:
        r.figure_captions = {
            fig: caption_map[fig]
            for fig in r.figures
            if fig in caption_map
        }
    return results


# ---------------------------------------------------------------------------
# Issue 4 — Heuristic reranker
# ---------------------------------------------------------------------------

def _query_tokens(query: str) -> set[str]:
    """Lowercase, split on whitespace/punctuation, return non-trivial tokens."""
    import re as _re
    return {t.lower() for t in _re.split(r"[\s\-_/,;.]+", query) if len(t) >= 3}


# ---------------------------------------------------------------------------
# Issue 1 — Domain-Aware Query Expansion
# ---------------------------------------------------------------------------

# Expansion seed terms keyed by topic.  Each entry maps a trigger word to a
# list of NBC-specific synonyms appended to the embedding query.
_EXPANSION_MAP: dict[str, list[str]] = {
    "residential": [
        "residential", "dwelling", "housing", "group a", "group a-1",
        "layout", "subdivision", "residential layout",
    ],
    "road": [
        "road", "street", "means of access", "access road", "road width",
        "street width", "layout road", "subdivision", "cul-de-sac",
        "residential layout",
    ],
    "street": [
        "street", "road", "means of access", "road width", "street width",
        "access road", "layout",
    ],
    "access": [
        "access", "means of access", "access road", "road width", "street width",
        "layout road",
    ],
    "hospital": [
        "hospital", "institutional", "institutional occupancy", "group c",
        "group c-1", "c-1", "medical facility", "nursing home",
    ],
    "school": [
        "school", "educational", "group b", "group b-1", "classroom",
        "educational occupancy",
    ],
    "hotel": [
        "hotel", "lodging", "group f", "group f-1", "boarding house",
    ],
    "office": [
        "office", "business", "group d", "group d-1", "commercial",
    ],
    "mercantile": [
        "mercantile", "shop", "group e", "retail", "market",
    ],
    "assembly": [
        "assembly", "group a-5", "auditorium", "cinema", "theatre",
    ],
    "corridor": [
        "corridor", "passageway", "internal corridor", "exit access",
    ],
    "staircase": [
        "staircase", "stair", "exit stair", "flight", "tread", "riser",
    ],
    "ramp": [
        "ramp", "gradient", "slope", "accessible ramp", "wheelchair ramp",
    ],
    "toilet": [
        "toilet", "water closet", "wc", "accessible toilet", "sanitary",
        "washroom", "lavatory",
    ],
    "fire": [
        "fire", "fire safety", "means of escape", "egress", "exit",
        "fire resistance", "fire compartment",
    ],
    "parking": [
        "parking", "car park", "vehicle", "off-street parking", "parking space",
    ],
}


def expand_query(query: str) -> str:
    """Append NBC domain-specific synonym terms to *query* for richer embedding.

    The original query is always preserved at the front.  Expansion terms are
    appended only once (deduplication) and only when a trigger word in the query
    matches an entry in ``_EXPANSION_MAP``.

    Parameters
    ----------
    query : The raw user query string.

    Returns
    -------
    Expanded query string with synonym terms appended, e.g.
        "residential access road width road street means of access subdivision …"
    """
    import re as _re
    tokens = {t.lower() for t in _re.split(r"[\s\-_/,;.]+", query) if t}

    extra: list[str] = []
    seen:  set[str]  = set(tokens)          # avoid repeating original words

    for trigger, synonyms in _EXPANSION_MAP.items():
        if trigger in tokens:
            for term in synonyms:
                term_lower = term.lower()
                if term_lower not in seen:
                    extra.append(term)
                    seen.add(term_lower)

    if not extra:
        return query
    return query + " " + " ".join(extra)


def rerank_results(
    results: list[SearchResult],
    query: str = "",
) -> list[SearchResult]:
    """Lightweight heuristic reranking applied after cosine similarity.

    Boosts / penalises each result based on metadata quality signals,
    then re-sorts.  The original similarity score is preserved unchanged;
    a new ``reranked_score`` attribute is attached to each result.

    A per-result ``_rerank_debug`` dict is also attached for --debug output.

    Boost table
    ───────────
    +0.03  clause_id present and non-empty
    +0.02  clause_path present and non-empty
    +0.02  text length > 250 chars
    +0.01  figures list non-empty
    -0.02  text length < 80 chars
    --- query-term boosts (only when query provided) ---
    +0.05  any query term appears in clause_path / heading (Issue 4)
    +0.05  any query term appears in clause_id title (Issue 4)
    +0.03  any query term appears in a figure caption / title
    +0.08  occupancy keyword match — domain boosting (Issue 2)
    +0.10  road-width intent match (Issue 3)
    """
    tokens = _query_tokens(query) if query else set()
    q_lower = query.lower() if query else ""

    # Issue 2 — Occupancy-aware keyword boost map
    DOMAIN_BOOSTS: dict[str, list[str]] = {
        "hospital": [
            "hospital", "institutional", "institutional occupancy",
            "group c", "c-1", "medical facility", "nursing home",
        ],
        "school": [
            "school", "educational", "group b", "classroom",
        ],
        "hotel": [
            "hotel", "lodging", "group f",
        ],
        "residential": [
            "residential", "group a", "dwelling", "housing",
        ],
        "office": [
            "office", "business", "group d",
        ],
        "mercantile": [
            "mercantile", "shop", "group e", "retail",
        ],
        "assembly": [
            "assembly", "group a-5", "auditorium", "cinema",
        ],
    }

    # Determine which domain(s) the query mentions
    active_domains: list[list[str]] = []
    for domain, kws in DOMAIN_BOOSTS.items():
        if any(kw in q_lower for kw in [domain] + kws):
            active_domains.append(kws)

    # Issue 3 — Road-width intent keywords
    ROAD_WIDTH_QUERY_KW = {
        "road width", "street width", "means of access", "road", "street",
        "access road", "minimum width", "layout road",
    }
    ROAD_WIDTH_CHUNK_KW = {
        "road width", "street width", "means of access", "subdivision",
        "layout road", "residential layout", "cul-de-sac", "access road",
    }
    road_width_query = any(kw in q_lower for kw in ROAD_WIDTH_QUERY_KW)

    scored: list[tuple[float, SearchResult]] = []
    for r in results:
        delta  = 0.0
        debug: dict[str, float] = {}
        text   = (r.text_preview or r.full_text or "").lower()

        # --- baseline metadata quality boosts ---
        if r.clause_id:
            delta += 0.03; debug["clause_id_present"] = 0.03
        if r.clause_path:
            delta += 0.02; debug["clause_path_present"] = 0.02
        if len(text) > 250:
            delta += 0.02; debug["long_text"] = 0.02
        if r.figures:
            delta += 0.01; debug["has_figures"] = 0.01
        if len(text) < 80:
            delta -= 0.02; debug["short_text"] = -0.02

        # --- Issue 4: heading-aware query-term boosts ---
        if tokens:
            # +0.05: query term in clause_path heading text
            path_lower = (r.clause_path or "").lower()
            if any(t in path_lower for t in tokens):
                delta += 0.05; debug["heading_match"] = 0.05

            # +0.05: query term in clause_id title
            cid_lower = (r.clause_id or "").lower()
            if any(t in cid_lower for t in tokens):
                delta += 0.05; debug["clause_id_heading_match"] = 0.05

            # +0.03: query term in any figure caption
            captions_lower = " ".join(
                v.lower() for v in (r.figure_captions or {}).values()
            )
            if captions_lower and any(t in captions_lower for t in tokens):
                delta += 0.03; debug["figure_caption_boost"] = 0.03

        # --- Issue 2: occupancy/domain boosting ---
        for domain_kws in active_domains:
            if any(kw in text for kw in domain_kws):
                delta += 0.08; debug["occupancy_match"] = 0.08
                break  # only one +0.08 even if multiple domains match

        # --- Issue 3: road-width intent boosting ---
        if road_width_query and any(kw in text for kw in ROAD_WIDTH_CHUNK_KW):
            delta += 0.10; debug["road_width_match"] = 0.10

        r.reranked_score = round(r.score + delta, 6)   # type: ignore[attr-defined]
        r._rerank_debug  = debug                        # type: ignore[attr-defined]
        scored.append((r.reranked_score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    for new_rank, (_, r) in enumerate(scored, start=1):
        r.rank = new_rank
    return [r for _, r in scored]


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _divider(char: str = "─", width: int = 72) -> str:
    return char * width


def print_results(
    results:        list[SearchResult],
    show_images:    bool = False,
    show_full_text: bool = False,
    show_debug:     bool = False,  # Issue 7
) -> None:
    """Print results to stdout in human-readable form."""
    if not results:
        print("No results found.")
        return

    for r in results:
        print(_divider())
        print(f"  Rank        : {r.rank}")
        # Issue 7: show both scores when reranking was applied
        if r.reranked_score and r.reranked_score != r.score:
            print(f"  Score       : {r.score:.4f}  →  reranked: {r.reranked_score:.4f}")
        else:
            print(f"  Score       : {r.score:.4f}")
        # Issue 2: show effective clause (falls back gracefully)
        eff_path = r.effective_clause_path or r.clause_path or "(no clause path)"
        eff_id   = r.effective_clause_id   or r.clause_id   or "(no clause id)"
        print(f"  Path        : {eff_path}")
        print(f"  Clause      : {eff_id}")
        print(f"  Page        : {r.page}   (PDF page: {r.pdf_page})")
        print(f"  Source      : {r.source_file}")
        print(f"  Preview     : {r.text_preview[:200]}")

        if show_debug:
            print(f"  [DEBUG] chunk_id={r.chunk_id}")
            print(f"  [DEBUG] raw_clause_id={r.clause_id!r}  raw_clause_path={r.clause_path!r}")

        if show_images and r.image_paths:
            print("  Images:")
            for img in r.image_paths:
                print(f"    {img}")

        if show_full_text and r.full_text:
            print("  Full text:")
            for line in r.full_text.splitlines():
                print(f"    {line}")

    print(_divider())
    print(f"  {len(results)} result(s) shown.")
    print(_divider())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Semantic search over the NBC 2016 Vol-1 corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("query", help="Natural-language query string")
    ap.add_argument("--top-k",   type=int,  default=10,       help="Number of results (default: 10)")
    ap.add_argument("--backend", default="voyageai",
                    choices=["voyageai", "openai", "local"],
                    help="Embedding backend (default: voyageai)")
    ap.add_argument("--model",   default="",
                    help="Model name override for the chosen backend")
    ap.add_argument("--embed-path", default=str(EMBED_PATH),
                    help="Path to embeddings .npy file")
    ap.add_argument("--meta-path",  default=str(META_PATH),
                    help="Path to embed_meta.json")
    ap.add_argument("--show-images",    action="store_true",
                    help="Print image paths in results")
    ap.add_argument("--show-full-text", action="store_true",
                    help="Print full chunk text in results")
    ap.add_argument("--export-json", default="",
                    help="Export results to a JSON file")
    # Issue 4
    ap.add_argument("--no-rerank", action="store_true",
                    help="Disable heuristic reranking (reranking is on by default)")
    # Issue 7
    ap.add_argument("--debug", action="store_true",
                    help="Show per-result diagnostics: similarity, reranked score, chunk ID, clause")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Enable debug logging")
    return ap


def main() -> None:
    ap   = _build_arg_parser()
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    searcher = NBCSearcher(
        embed_path = Path(args.embed_path),
        meta_path  = Path(args.meta_path),
        backend    = args.backend,
        model      = args.model,
    ).load()

    results = searcher.search(
        query        = args.query,
        top_k        = args.top_k,
        include_text = args.show_full_text,
        rerank       = not args.no_rerank,
        debug        = args.debug,
    )

    print_results(
        results,
        show_images    = args.show_images,
        show_full_text = args.show_full_text,
        show_debug     = args.debug,
    )

    if args.export_json:
        out_path = Path(args.export_json)
        payload  = {
            "query":   args.query,
            "top_k":   args.top_k,
            "backend": args.backend,
            "results": [r.to_dict() for r in results],
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResults exported to {out_path}")


if __name__ == "__main__":
    main()