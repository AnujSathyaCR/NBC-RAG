# 📘 NBC RAG – National Building Code 2016 Retrieval-Augmented Generation System

An AI-powered Retrieval-Augmented Generation (RAG) system built over the **National Building Code (NBC) 2016** to enable semantic search, clause-level retrieval, and citation-backed question answering for architects, engineers, planners, and regulatory professionals.

---

## ✨ Features

- 🔍 Semantic search across the complete NBC 2016
- 📖 Clause-aware retrieval
- 📑 Figure and table retrieval
- 🤖 AI-generated answers using Gemini
- 📚 Citation-backed responses
- 📈 Retrieval evaluation using standard Information Retrieval metrics
- ⚡ FAISS vector search with reranking
- 🏗️ Designed for real-world architect and building approval queries

---

# System Architecture

```
                User Question
                      │
                      ▼
            Voyage AI Embedding
                      │
                      ▼
             FAISS Vector Search
                      │
                      ▼
              Top-K Candidate Chunks
                      │
                      ▼
          Cross-Encoder Reranking
                      │
                      ▼
          Top Relevant NBC Chunks
                      │
                      ▼
               Gemini 2.5 Pro
                      │
                      ▼
        Structured Answer + Citations
```

---

# Dataset

| Property | Value |
|----------|------:|
| Document | National Building Code (NBC) 2016 |
| Total Pages | 1226 |
| Extracted Pages | 1191 |
| Chunks Generated | 12,663 |
| Chunks with Clause IDs | 11,869 |
| Figures | 999 |
| Tables | 658 |

---

# Project Structure

```
NBC_RAG/
│
├── output/
│   ├── nbc_full.json
│   ├── chunks.jsonl
│   ├── embeddings.npy
│
├── search_nbc.py
├── ask_nbc.py
├──combine_nbc.py
├── chunk_nbc.py
├── embed_nbc.py
├── evaluate_retrieval.py
├──extract.py
├──split_pdfs.py
├──extract_images.py
├── batch_submit.py
├── batch_status.py
├── batch_download.py
├── batch_build.py
├── batch_common.py
└── README.md
```

---

# Retrieval Pipeline

## 1. Document Extraction

NBC PDF pages are extracted into structured Markdown.

Each page preserves:

- headings
- clause hierarchy
- tables
- figures
- metadata

---

## 2. Chunking

The Markdown pages are converted into semantic chunks.

Each chunk stores metadata including:

- Clause ID
- Clause Path
- NBC Page
- PDF Page
- Figure IDs
- Table IDs
- Source File

Example:

```json
{
  "clause_id": "B-9.2",
  "clause_path": "PART 3 > Annex B > B-9.2",
  "page": 207,
  "text": "...",
  "figure_ids": ["Fig.76"]
}
```

---

## 3. Embedding Generation

Each chunk is converted into dense vector embeddings using **Voyage AI Law Embeddings**.

These embeddings capture semantic meaning instead of keyword similarity.

---

## 4. Vector Search

Queries are embedded using the same embedding model.

FAISS performs Approximate Nearest Neighbor (ANN) search to retrieve the most relevant chunks.

---

## 5. Reranking

Retrieved chunks are reranked using a Cross-Encoder.

This improves ranking quality by considering the query and chunk together.

---

## 6. Answer Generation

The highest-ranked chunks are sent to Gemini 2.5 Pro.

Gemini generates:

- structured answers
- citations
- referenced clauses
- figure references
- table references

---

# Example Query

```
What are the requirements for accessible toilets?
```

Returns

- Structured answer
- Relevant clauses
- NBC page references
- Figures
- Citations

---

# Retrieval Evaluation

The retriever is evaluated independently using a manually curated benchmark.

Evaluation metrics include:

- Precision@K
- Recall@K
- Hit Rate@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

---

## Latest Evaluation Results

### Combined Retrieval Performance

| Metric | @1 | @3 | @5 | @7 | @10 |
|--------|----:|----:|----:|----:|-----:|
| Precision | 0.8438 | 0.6146 | 0.5438 | 0.5112 | 0.4438 |
| Recall | 0.7135 | 0.9219 | 0.9688 | 0.9844 | 0.9844 |
| Hit Rate | 0.8438 | 0.9219 | 0.9688 | 0.9844 | 0.9844 |
| NDCG | 0.7813 | 0.7871 | 0.8194 | 0.8467 | 0.8708 |

**MRR:** **0.8945**

---

# Technologies Used

- Python
- FAISS
- Voyage AI Embeddings
- Gemini 2.5 Pro
- NumPy
- JSONL
- Markdown
- Rich
- tqdm

---

# Key Capabilities

- Semantic search
- Clause-aware retrieval
- Figure retrieval
- Table retrieval
- Citation-aware answers
- Architect-oriented queries
- NBC-wide knowledge search
- Retrieval benchmarking

---

# Example Questions

- What are the ramp requirements?
- What is the minimum width of an access road?
- Explain accessible toilet requirements.
- What are the fire exit staircase requirements?
- What are setback requirements?
- What is the required parking provision?
- What is the minimum corridor width?
- What are the occupancy classifications?

---

# Future Improvements

- Hybrid Retrieval (BM25 + Dense Retrieval)
- Query Expansion
- Multi-query Retrieval
- Metadata-aware Filtering
- Streaming Responses
- Interactive Web Interface
- Hallucination Detection
- Multi-document Retrieval
- Support for TNCDBR integration

---

# Acknowledgements

- National Building Code of India 2016
- FAISS
- Voyage AI
- Google Gemini
