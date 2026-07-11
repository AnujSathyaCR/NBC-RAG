"""
ask_nbc.py — RAG question-answering over the NBC 2016 Vol-1 corpus.

Retrieves the most relevant chunks via search_nbc.NBCSearcher, then
sends them as grounded context to an LLM that must answer ONLY from
the supplied text.  Responses always include clause-level citations
and figure references.

Supported LLM backends
───────────────────────
  gemini   — Google Gemini 2.5 Pro  (default)
  openai   — OpenAI GPT-4o / GPT-4-turbo

Usage
─────
  python ask_nbc.py "What are the requirements for accessible toilets?"
  python ask_nbc.py "Minimum corridor width for hospitals" --top-k 15
  python ask_nbc.py "fire egress staircase" --llm openai --llm-model gpt-4o
  python ask_nbc.py "ramp gradient" --backend local --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# search_nbc must be in the same directory (or on PYTHONPATH)
try:
    from search_nbc import NBCSearcher, SearchResult
except ImportError:
    sys.exit(
        "ERROR: search_nbc.py not found.  "
        "Place ask_nbc.py and search_nbc.py in the same directory."
    )

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,
)
log = logging.getLogger("ask_nbc")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a precise technical expert on the National Building Code of India 2016 (NBC 2016), Volume 1.

You will be given:
1. A user question.
2. A set of numbered context chunks extracted directly from the NBC.
   Each chunk includes its clause path, clause ID, page number, and text.

STRICT RULES — you MUST follow all of these:
- Answer ONLY from the supplied NBC context chunks. Do NOT invent requirements.
- If information is missing from the context, write exactly: "Not specified in retrieved context."
- Do NOT cite clause numbers or figures absent from the context.
- Use exact NBC terminology. Preserve dimensions and units as written.
- Every factual statement must carry an inline citation [N] matching the context number.

ANSWER FORMAT — use exactly these six section headings in order.
If a section has no information in the context, write "Not specified in retrieved context."

Definition
Provision Requirements
Dimensional Requirements
Equipment / Fixture Requirements
Exceptions / Notes
References
""".strip()

_USER_TEMPLATE = """\
QUESTION:
{question}

NBC CONTEXT:
{context}

Respond in this exact format:

ANSWER
<your answer here, with inline citations like [1], [2] …>

CITATIONS
<one citation block per referenced chunk, numbered to match your inline citations>

FIGURES
<list any figure paths referenced in the cited chunks; write NONE if there are none>
""".strip()

# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(chunks: list[SearchResult]) -> str:
    """Convert retrieved chunks into a sequentially numbered context block.

    Issues 1, 2, 8:
    - Numbers are always sequential [1]…[N] regardless of retrieval rank.
    - effective_clause_id / effective_clause_path used instead of raw fields.
    - source_file, pdf_page, page always present for GroundRules navigation.
    """
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        text = c.full_text.strip() if c.full_text else c.text_preview.strip()

        # Issue 2: use effective clause fields (never show "(unknown)")
        eff_id   = c.effective_clause_id   or c.clause_id   or "—"
        eff_path = c.effective_clause_path or c.clause_path or "—"

        fig_line = ""
        if c.figures:
            fig_line = f"  Figures referenced: {', '.join(c.figures)}\n"

        # Issue 8: always include source_file + pdf_page for GroundRules
        parts.append(
            f"[{i}]\n"
            f"  Clause path : {eff_path}\n"
            f"  Clause ID   : {eff_id}\n"
            f"  NBC page    : {c.page}\n"
            f"  PDF page    : {c.pdf_page}\n"
            f"  Source      : {c.source_file}\n"
            f"{fig_line}"
            f"  Text:\n"
            + "\n".join(f"    {line}" for line in text.splitlines())
        )
    return "\n\n".join(parts)


def _clause_display(clause_id: str, clause_path: str) -> str:
    """Format a clause reference as "B-9.1 General" when the path carries a title.

    clause_path examples:
        "PART 6 > Annex B > B-9 > B-9.1 Accessible Toilet"
        "PART 6 > Section 5 > 38.7 Slender Compression Members"

    The last breadcrumb usually contains the numeric id + title.
    If clause_id alone matches the last breadcrumb prefix we return the
    full last breadcrumb (id + title).  Otherwise return clause_id only.
    """
    if not clause_id:
        return "—"
    if clause_path:
        last = clause_path.split(">")[-1].strip()
        # last breadcrumb starts with the clause_id (e.g. "B-9.1 Accessible Toilet")
        if last.startswith(clause_id):
            return last
    return clause_id


def build_citations_section(chunks: list[SearchResult]) -> str:
    """Build the rich CITATIONS block.

    Format per entry:
        [N] <clause_with_title>  [<sequential index>]
            <other clause titles if multiple from same source>
            NBC page  : X
            PDF page  : Y
            Source    : page_NNN.md

    Issues 1, 2, 8:
    - Sequential [1]…[N] regardless of retrieval rank.
    - Effective clause fields used; never shows blank/unknown.
    - Clause title is extracted from clause_path last breadcrumb.
    - source_file + pdf_page + page always present for GroundRules navigation.
    """
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        eff_id   = c.effective_clause_id   or c.clause_id   or ""
        eff_path = c.effective_clause_path or c.clause_path or "NBC 2016"
        clause_display = _clause_display(eff_id, eff_path)
        lines.append(
            f"[{i}] {clause_display}\n"
            f"    NBC page  : {c.page}\n"
            f"    PDF page  : {c.pdf_page}\n"
            f"    Source    : {c.source_file}"
        )
    return "\n\n".join(lines)


def build_figures_section(chunks: list[SearchResult]) -> str:
    """Build a rich FIGURES block for GroundRules UI rendering.

    Issues 3 + 6: each figure entry includes:
      Fig. N
      Caption: <caption from figure_metadata, or omitted if unavailable>
      Clause: <effective_clause_id>
      PDF Page: <pdf_page>
      Image: <image path>
    Entries separated by --- for UI parsing.
    """
    seen_figs: set[str]          = set()
    # Build a map from image_path -> (fig_id, chunk) for rich rendering
    # Also build fig_id -> (chunk, image_path) for each figure
    fig_entries: list[dict] = []

    for c in chunks:
        eff_id = c.effective_clause_id or c.clause_id or "—"
        pdf_p  = c.pdf_page or "—"

        # Pair each figure id with its image path (by position in lists)
        fig_list = c.figures or []
        img_list = c.image_paths or []

        for idx, fig_id in enumerate(fig_list):
            if not fig_id or fig_id in seen_figs:
                continue
            seen_figs.add(fig_id)
            img_path = img_list[idx] if idx < len(img_list) else ""
            caption  = (c.figure_captions or {}).get(fig_id, "")
            fig_entries.append({
                "fig_id":   fig_id,
                "caption":  caption,
                "clause":   eff_id,
                "pdf_page": pdf_p,
                "img_path": img_path,
            })

        # Any extra image paths not paired with a figure id
        for ip in img_list[len(fig_list):]:
            if ip and ip not in seen_figs:
                seen_figs.add(ip)
                fig_entries.append({
                    "fig_id":   "",
                    "caption":  "",
                    "clause":   eff_id,
                    "pdf_page": pdf_p,
                    "img_path": ip,
                })

    if not fig_entries:
        return "NONE"

    blocks: list[str] = []
    for e in fig_entries:
        lines: list[str] = []
        # Header line: "Fig. 72 — Accessible Unisex Toilet Layout"  (spec format)
        if e["fig_id"] and e["caption"]:
            lines.append(f"{e['fig_id']} — {e['caption']}")
        elif e["fig_id"]:
            lines.append(e["fig_id"])
        # Sub-fields
        if e["clause"] and e["clause"] != "—":
            lines.append(f"Clause: {e['clause']}")
        if e["pdf_page"] and e["pdf_page"] != "—":
            lines.append(f"PDF Page: {e['pdf_page']}")
        if e["img_path"]:
            lines.append(f"Image: {e['img_path']}")
        if lines:
            blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Issue 1 — Citation renumbering
# ---------------------------------------------------------------------------

def remap_citations(answer_text: str, used_indices: list[int]) -> str:
    """Renumber citations in answer_text so they are sequential [1]…[N].

    The LLM receives context numbered [1]…[K] where K == len(chunks).
    Because all chunks are sequentially numbered before sending to the LLM,
    citations in the raw answer are already sequential.

    However if the LLM skips citation numbers (e.g. uses [1][3][5]),
    this function compresses them to [1][2][3] so the user never sees gaps.

    Parameters
    ----------
    answer_text  : Raw answer text containing [N] citation markers.
    used_indices : Sorted list of citation numbers the LLM actually used,
                   extracted from answer_text.  If empty, answer is returned
                   unchanged.

    Returns the answer text with remapped citation numbers.
    """
    import re as _re
    if not used_indices:
        return answer_text

    remap = {old: new for new, old in enumerate(sorted(set(used_indices)), start=1)}

    def _replace(m: "_re.Match[str]") -> str:
        n = int(m.group(1))
        return f"[{remap[n]}]" if n in remap else m.group(0)

    return _re.sub(r"\[(\d+)\]", _replace, answer_text)


def extract_citation_indices(text: str) -> list[int]:
    """Return sorted list of [N] citation numbers found in *text*."""
    import re as _re
    return sorted(set(int(m) for m in _re.findall(r"\[(\d+)\]", text)))


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str, model: str) -> str:
    """Call Gemini via the google-genai SDK."""
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError:
        sys.exit("ERROR: google-genai not installed.  pip install google-genai")

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: GEMINI_API_KEY not set in environment or .env")

    client   = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model    = model,
        contents = prompt,
        config   = gtypes.GenerateContentConfig(
            system_instruction = _SYSTEM_PROMPT,
            temperature        = 0.1,
            max_output_tokens  = 8192,
        ),
    )
    return (response.text or "").strip()


def _call_openai(prompt: str, model: str) -> str:
    """Call OpenAI chat completions."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        sys.exit("ERROR: openai not installed.  pip install openai")

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set in environment or .env")

    client = OpenAI(api_key=api_key)
    resp   = client.chat.completions.create(
        model    = model,
        messages = [
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": prompt},
        ],
        temperature = 0.1,
        max_tokens  = 8192,
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# NBCQA
# ---------------------------------------------------------------------------

@dataclass
class QAResult:
    """The structured result of one ask_nbc query."""
    question:   str
    chunks:     list[SearchResult]
    raw_answer: str               # Full LLM response text
    answer:     str               # Extracted ANSWER section
    citations:  str               # Formatted CITATIONS block
    figures:    str               # Formatted FIGURES block
    confidence: str  = "UNKNOWN" # Issue 6: HIGH / MEDIUM / LOW
    avg_score:  float = 0.0      # Issue 6: avg reranked score of top-5 chunks

    def to_dict(self) -> dict[str, Any]:
        return {
            "question":   self.question,
            "answer":     self.answer,
            "citations":  self.citations,
            "figures":    self.figures,
            "confidence": self.confidence,
            "avg_score":  self.avg_score,
            "chunks":     [c.to_dict() for c in self.chunks],
        }


class NBCQA:
    """
    Retrieval-augmented QA system for the NBC 2016 Vol-1 corpus.

    Parameters
    ----------
    searcher   : A loaded NBCSearcher instance.
    llm        : LLM backend — 'gemini' | 'openai'.
    llm_model  : Model name for the chosen LLM backend.
    top_k      : Number of chunks to retrieve per query.
    """

    def __init__(
        self,
        searcher:  NBCSearcher,
        llm:       str = "gemini",
        llm_model: str = "",
        top_k:     int = 10,
    ) -> None:
        self.searcher  = searcher
        self.llm       = llm.lower()
        self.llm_model = llm_model
        self.top_k     = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> list[SearchResult]:
        """Return the top-k NBC chunks most relevant to *question*."""
        return self.searcher.search(
            query        = question,
            top_k        = self.top_k,
            include_text = True,   # need full text for the LLM context
        )

    def build_context(self, chunks: list[SearchResult]) -> str:
        """Delegate to the module-level build_context() function."""
        return build_context(chunks)

    def answer(self, question: str, debug: bool = False) -> QAResult:
        """End-to-end: retrieve → build prompt → call LLM → parse response.

        Issues 1, 2, 7, 8:
        - Retrieval passes rerank/debug flags through.
        - Citation numbers are remapped to be sequential after LLM response.
        - Effective clause fields used in citations.
        - debug flag prints diagnostics to stdout.
        """
        log.info("Retrieving chunks for question: %s", question[:80])
        chunks  = self.searcher.search(
            query        = question,
            top_k        = self.top_k,
            include_text = True,
            rerank       = True,
            debug        = debug,
        )
        context = build_context(chunks)
        prompt  = _USER_TEMPLATE.format(question=question, context=context)

        log.info("Calling LLM backend=%s model=%s", self.llm, self.llm_model or "(default)")

        if self.llm == "gemini":
            model = self.llm_model or "gemini-2.5-pro"
            raw   = _call_gemini(prompt, model)
        elif self.llm == "openai":
            model = self.llm_model or "gpt-4o"
            raw   = _call_openai(prompt, model)
        else:
            sys.exit(f"ERROR: unknown LLM backend '{self.llm}'.  Choose gemini / openai.")

        # Parse the structured response
        answer_text = _extract_section(raw, "ANSWER",    next_section="CITATIONS")
        citations   = _extract_section(raw, "CITATIONS", next_section="FIGURES")
        figures_raw = _extract_section(raw, "FIGURES",   next_section=None)

        # Issue 1: remap citations to sequential [1]…[N] (removes gaps)
        used_indices = extract_citation_indices(answer_text)
        answer_text  = remap_citations(answer_text, used_indices)

        # Build rich figures block (issues 3+6)
        figures_combined = build_figures_section(chunks)
        if figures_combined == "NONE" and figures_raw and figures_raw.upper() != "NONE":
            # LLM mentioned figures we couldn't map to image paths — include raw
            figures_combined = figures_raw.strip()

        # Build citations from chunks (issue 8: always includes source/page)
        final_citations = build_citations_section(chunks)

        # Issue 6: answer confidence based on avg reranked score of top-5 chunks
        top5  = chunks[:5]
        scores = [
            (c.reranked_score if c.reranked_score else c.score) for c in top5
        ]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        if avg_score >= 0.70:
            confidence = "HIGH"
        elif avg_score >= 0.55:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return QAResult(
            question   = question,
            chunks     = chunks,
            raw_answer = raw,
            answer     = answer_text.strip(),
            citations  = final_citations,
            figures    = figures_combined.strip(),
            confidence = confidence,
            avg_score  = round(avg_score, 4),
        )


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------

def _extract_section(text: str, header: str, next_section: str | None) -> str:
    """Extract content between *header* and *next_section* in the LLM response."""
    start_marker = header + "\n"
    start        = text.find(start_marker)
    if start == -1:
        # Try without newline (model might write "ANSWER:" etc.)
        start = text.find(header)
        if start == -1:
            return ""
        start += len(header)
    else:
        start += len(start_marker)

    if next_section:
        end = text.find(next_section, start)
        return text[start:end].strip() if end != -1 else text[start:].strip()
    return text[start:].strip()


def _merge_figures(llm_figures: str, chunks: list[SearchResult]) -> str:
    """
    Merge figures mentioned by the LLM with image_paths from the retrieved chunks.
    Returns a deduplicated, newline-separated string.
    """
    seen:  set[str]  = set()
    lines: list[str] = []

    for raw in (llm_figures or "").splitlines():
        raw = raw.strip()
        if raw and raw.upper() != "NONE" and raw not in seen:
            lines.append(raw)
            seen.add(raw)

    for c in chunks:
        for fig in c.figures:
            if fig and fig not in seen:
                lines.append(fig)
                seen.add(fig)
        for img in c.image_paths:
            if img and img not in seen:
                lines.append(img)
                seen.add(img)

    return "\n".join(lines) if lines else "NONE"


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _divider(char: str = "═", width: int = 72) -> str:
    return char * width


def print_qa_result(result: QAResult) -> None:
    """Print the QA result in the documented format."""
    print(_divider())
    print(f"QUESTION\n{result.question}")
    print(_divider())
    # Issue 6: confidence indicator above the answer
    print(f"Confidence: {result.confidence}  (avg score: {result.avg_score:.4f})")
    print(_divider())
    print(f"ANSWER\n{result.answer}")
    print(_divider())
    print(f"CITATIONS\n{result.citations}")
    print(_divider())
    print(f"FIGURES\n{result.figures}")
    print(_divider())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="RAG question-answering over the NBC 2016 Vol-1 corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("question", help="Natural-language question about the NBC")

    # Retrieval
    ap.add_argument("--top-k",   type=int, default=10,
                    help="Number of chunks to retrieve (default: 10)")
    ap.add_argument("--backend", default="voyageai",
                    choices=["voyageai", "openai", "local"],
                    help="Embedding backend for retrieval (default: voyageai)")
    ap.add_argument("--embed-model", default="",
                    help="Embedding model name override")
    ap.add_argument("--embed-path", default=str(ROOT / "output" / "embeddings.npy"))
    ap.add_argument("--meta-path",  default=str(ROOT / "output" / "embed_meta.json"))

    # Generation
    ap.add_argument("--llm", default="gemini",
                    choices=["gemini", "openai"],
                    help="LLM backend for answer generation (default: gemini)")
    ap.add_argument("--llm-model", default="",
                    help="LLM model name override (e.g. gpt-4o, gemini-2.5-pro)")

    # Output
    ap.add_argument("--export-json", default="",
                    help="Export full QA result to a JSON file")
    ap.add_argument("--show-chunks", action="store_true",
                    help="Print retrieved chunks before the answer")
    # Issue 7
    ap.add_argument("--debug", action="store_true",
                    help="Show retrieval diagnostics: similarity, reranked score, clause IDs, source")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Enable debug logging")
    return ap


def main() -> None:
    ap   = _build_arg_parser()
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build and load the searcher
    searcher = NBCSearcher(
        embed_path = Path(args.embed_path),
        meta_path  = Path(args.meta_path),
        backend    = args.backend,
        model      = args.embed_model,
    ).load()

    qa = NBCQA(
        searcher  = searcher,
        llm       = args.llm,
        llm_model = args.llm_model,
        top_k     = args.top_k,
    )

    if args.show_chunks:
        print("Retrieving chunks…")
        chunks = qa.retrieve(args.question)
        from search_nbc import print_results
        print_results(chunks, show_images=True, show_full_text=False)
        print()

    print(f"Asking the NBC ({args.llm})…\n")
    result = qa.answer(args.question, debug=args.debug)
    print_qa_result(result)

    if args.export_json:
        out = Path(args.export_json)
        out.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nResult exported to {out}")


if __name__ == "__main__":
    main()