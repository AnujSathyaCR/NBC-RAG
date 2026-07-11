"""
embed_nbc.py
============
Reads output/chunks.jsonl, generates embeddings for each chunk, and writes:
  output/embeddings.npy   — float32 array of shape (N, D)
  output/embed_meta.json  — list of N metadata dicts (no raw text)

Architecture
------------
Three concerns are kept strictly separate:

  1. TEXT PAYLOAD  — build_text_payload(chunk) → str
       What text is fed to the embedding model.
       Combines clause_path + text + figure captions for dense retrieval.

  2. EMBEDDING BACKEND  — EmbedBackend protocol
       Pluggable.  Swap VoyageAI ↔ OpenAI ↔ local model by changing
       --backend without touching any other code.

  3. METADATA STORAGE  — chunk → metadata dict
       Stores everything the retrieval layer needs (chunk_id, clause_path,
       image_paths, figures, tables …) without duplicating the embedding array.

Supported backends
------------------
  voyage   — VoyageAI (default, same as GroundRules TNCDBR pipeline)
               Requires: pip install voyageai
               Env var:  VOYAGE_API_KEY
  openai   — OpenAI text-embedding-3-small / large
               Requires: pip install openai
               Env var:  OPENAI_API_KEY
  local    — sentence-transformers (CPU/GPU, no API key)
               Requires: pip install sentence-transformers
               Model:    --local-model  (default: BAAI/bge-m3)

Usage
-----
    python embed_nbc.py                               # VoyageAI default
    python embed_nbc.py --backend openai
    python embed_nbc.py --backend local --local-model BAAI/bge-m3
    python embed_nbc.py --chunks output/chunks.jsonl \\
                        --out-dir output/ \\
                        --batch-size 64

Dependencies
------------
    numpy (always required)
    voyageai | openai | sentence-transformers  (one depending on backend)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

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
# 1. TEXT PAYLOAD
# ---------------------------------------------------------------------------

def build_text_payload(chunk: dict[str, Any]) -> str:
    """
    Build the string that will be embedded for a chunk.

    Strategy (retrieval-optimised):
      - clause_path  provides hierarchical context for section-level queries
      - clause_id    improves direct clause-identifier retrieval
      - text         is the primary content
      - figure ids   ensure figure queries surface the right clause
      - figure caps  improve semantic retrieval of NBC figures
      - table hint   helps table-specific queries
    """
    parts: list[str] = []

    # Hierarchical breadcrumb (high weight for structural queries)
    clause_path = chunk.get("clause_path", "")
    if clause_path:
        parts.append(f"Location: {clause_path}")

    # Direct clause identifier — improves "what does clause X say" queries
    clause_id = chunk.get("clause_id")
    if clause_id:
        parts.append(f"Clause: {clause_id}")

    # Main clause text
    text = chunk.get("text", "").strip()
    if text:
        parts.append(text)

    # Figure references — embed figure ids so "Fig. 20" queries hit correctly
    figures = chunk.get("figures", [])
    if figures:
        parts.append("Figures: " + ", ".join(figures))

    # Figure captions — improve semantic retrieval of NBC figures.
    # figure_assets entries may carry a "caption" key populated from figure_metadata.
    figure_assets = chunk.get("figure_assets", [])
    captions = [
        fa["caption"]
        for fa in figure_assets
        if fa.get("caption")
    ]
    for caption in captions:
        parts.append(f"Figure Caption: {caption}")

    # Defined terms — boost definition retrieval
    defined_terms = chunk.get("defined_terms", [])
    if defined_terms:
        parts.append("Defined terms: " + ", ".join(defined_terms))

    return "\n\n".join(parts).strip()


# ---------------------------------------------------------------------------
# 2. EMBEDDING BACKENDS
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbedBackend(Protocol):
    """Any object with an embed(texts) method is a valid backend."""
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimension(self) -> int: ...


# --- VoyageAI ---------------------------------------------------------------

class VoyageBackend:
    """
    VoyageAI backend — mirrors the GroundRules TNCDBR embedding setup.
    Model: voyage-law-2 (legal domain; change to voyage-large-2 for general)
    """

    MODEL = "voyage-law-2"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        try:
            import voyageai  # type: ignore
        except ImportError:
            log.error("voyageai not installed. Run: pip install voyageai")
            sys.exit(1)
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            log.error("VOYAGE_API_KEY not set")
            sys.exit(1)
        self._model = model or self.MODEL
        self._client = voyageai.Client(api_key=key)
        self._dim: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=self._model, input_type="document")
        vecs = result.embeddings
        if self._dim is None and vecs:
            self._dim = len(vecs[0])
        return vecs

    @property
    def dimension(self) -> int:
        return self._dim or 1024  # voyage-law-2 default


# --- OpenAI -----------------------------------------------------------------

class OpenAIBackend:
    MODEL = "text-embedding-3-small"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            log.error("openai not installed. Run: pip install openai")
            sys.exit(1)
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            log.error("OPENAI_API_KEY not set")
            sys.exit(1)
        self._model = model or self.MODEL
        self._client = OpenAI(api_key=key)
        self._dim: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        vecs = [item.embedding for item in response.data]
        if self._dim is None and vecs:
            self._dim = len(vecs[0])
        return vecs

    @property
    def dimension(self) -> int:
        return self._dim or 1536


# --- Local (sentence-transformers) ------------------------------------------

class LocalBackend:
    DEFAULT_MODEL = "BAAI/bge-m3"

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            log.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            sys.exit(1)
        name = model_name or self.DEFAULT_MODEL
        log.info("Loading local model %s …", name)
        self._model = SentenceTransformer(name)
        self._dim: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        result = vecs.tolist()
        if self._dim is None and result:
            self._dim = len(result[0])
        return result

    @property
    def dimension(self) -> int:
        return self._dim or 1024


def build_backend(
    backend: str,
    api_key: str | None = None,
    model: str | None = None,
    local_model: str | None = None,
) -> EmbedBackend:
    if backend == "voyage":
        return VoyageBackend(api_key=api_key, model=model)
    if backend == "openai":
        return OpenAIBackend(api_key=api_key, model=model)
    if backend == "local":
        return LocalBackend(model_name=local_model)
    log.error("Unknown backend '%s'. Choose: voyage | openai | local", backend)
    sys.exit(1)


# ---------------------------------------------------------------------------
# 3. METADATA
# ---------------------------------------------------------------------------

def _pdf_page_from_source_file(source_file: str | None) -> int | None:
    """Derive the physical PDF page number from a source filename.

    Examples:
        "page_114.md"  -> 114
        "page_1216.md" -> 1216

    Returns None if source_file is absent or does not match the pattern.
    """
    if not source_file:
        return None
    import re
    m = re.search(r"page_(\d+)", source_file)
    return int(m.group(1)) if m else None


def chunk_to_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the fields the retrieval layer needs at query time.
    Raw text is NOT stored here (it's in the .npy embeddings by position).
    """
    source_file: str | None = chunk.get("source_file")
    return {
        "chunk_id":        chunk.get("chunk_id"),
        "page":            chunk.get("page"),          # printed NBC page number
        "pdf_page":        _pdf_page_from_source_file(source_file),  # physical PDF page
        "source_file":     source_file,
        "part":            chunk.get("part"),
        "section":         chunk.get("section"),
        "clause_id":       chunk.get("clause_id"),
        "clause_path":     chunk.get("clause_path"),
        "kind":            chunk.get("kind"),
        "figures":         chunk.get("figures", []),
        "figure_assets":   chunk.get("figure_assets", []),
        "image_paths":     chunk.get("image_paths", []),
        "tables":          chunk.get("tables", []),
        "xrefs":           chunk.get("xrefs", []),
        "defined_terms":   chunk.get("defined_terms", []),
        # store a truncated text preview for debugging / display
        "text_preview":    (chunk.get("text") or "")[:300],
        # populated later by the search layer; None until a query is run
        "retrieval_score": None,
    }


# ---------------------------------------------------------------------------
# Batching helper
# ---------------------------------------------------------------------------

def batched(items: list[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    chunks_path: Path,
    out_dir: Path,
    backend: EmbedBackend,
    batch_size: int = 64,
    sleep_between_batches: float = 0.5,
) -> None:
    # Load chunks
    log.info("Reading chunks from %s …", chunks_path)
    chunks: list[dict[str, Any]] = []
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    log.info("Loaded %d chunks", len(chunks))

    if not chunks:
        log.error("No chunks found — nothing to embed")
        sys.exit(1)

    # Build text payloads
    payloads = [build_text_payload(c) for c in chunks]

    # Embed in batches
    all_vectors: list[list[float]] = []
    n_batches = (len(payloads) + batch_size - 1) // batch_size

    for i, batch in enumerate(batched(payloads, batch_size)):
        log.info("Embedding batch %d/%d (%d texts) …", i + 1, n_batches, len(batch))
        try:
            vecs = backend.embed(batch)
        except Exception as exc:  # noqa: BLE001
            log.error("Embedding failed on batch %d: %s", i + 1, exc)
            raise
        all_vectors.extend(vecs)
        if sleep_between_batches > 0 and i < n_batches - 1:
            time.sleep(sleep_between_batches)

    # Validate shape
    if len(all_vectors) != len(chunks):
        log.error(
            "Vector count mismatch: expected %d, got %d", len(chunks), len(all_vectors)
        )
        sys.exit(1)

    # Write embeddings
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.npy"
    arr = np.array(all_vectors, dtype=np.float32)
    np.save(str(emb_path), arr)
    log.info("Saved embeddings %s → %s", arr.shape, emb_path)

    # Write metadata
    meta_path = out_dir / "embed_meta.json"
    metadata = [chunk_to_metadata(c) for c in chunks]
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    log.info("Saved metadata (%d entries) → %s", len(metadata), meta_path)

    log.info("Done. Backend dimension: %d", backend.dimension)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Embed NBC 2016 chunks → embeddings.npy + embed_meta.json",
    )
    p.add_argument(
        "--chunks",
        type=Path,
        default=Path("output/chunks.jsonl"),
        help="Input JSONL from chunk_nbc.py (default: output/chunks.jsonl)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="Directory for embeddings.npy and embed_meta.json (default: output/)",
    )
    p.add_argument(
        "--backend",
        choices=["voyage", "openai", "local"],
        default="voyage",
        help="Embedding backend (default: voyage)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override model name for voyage or openai backend",
    )
    p.add_argument(
        "--local-model",
        default=None,
        help="HuggingFace model name for --backend local (default: BAAI/bge-m3)",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key (overrides env var VOYAGE_API_KEY / OPENAI_API_KEY)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Texts per embedding API call (default: 64)",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between batches (default: 0.5)",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    backend = build_backend(
        backend=args.backend,
        api_key=args.api_key,
        model=args.model,
        local_model=args.local_model,
    )
    run(
        chunks_path=args.chunks,
        out_dir=args.out_dir,
        backend=backend,
        batch_size=args.batch_size,
        sleep_between_batches=args.sleep,
    )