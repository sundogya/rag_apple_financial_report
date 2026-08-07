# rag_apple_financial_report

# 📈 Financial-RAG-Engine: Advanced QA & Evaluation System for Apple 10-K Reports

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RAGAS](https://img.shields.io/badge/Evaluation-RAGAS-green.svg)](https://github.com/explodinggradients/ragas)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red.svg)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-ready, production-grade **Advanced RAG (Retrieval-Augmented Generation)** engine specifically engineered to tackle complex financial documents, multi-page financial tables, and dense numerical queries from the **Apple Annual Form 10-K Report**.

---

## 🌟 Key Features

* **📄 Table-Aware & Parent-Document Processing:** Uses `Unstructured`/`LlamaParse` to extract financial tables as Markdown/HTML, maintaining table integrity with a **Parent-Child Document Retriever** to avoid context truncation.
* **🔍 Hybrid Search Engine:** Combines **Dense Vector Search** (semantic matching) with **Sparse BM25 Search** (exact keyword matching) using **Reciprocal Rank Fusion (RRF)**.
* **🎯 Cross-Encoder Re-Ranking:** Integrated `BGE-Reranker-v2` / `Cohere Rerank` to refine Top-20 candidates down to the Top-3/5 most relevant contexts, dramatically reducing hallucination.
* **⚡ Semantic Caching:** Integrated `GPTCache`/Redis to achieve sub-50ms TTFT (Time-To-First-Token) for semantically similar financial queries.
* **🛡️ Metadata Filtering & ACL:** Pre-filtering by document section, financial year, and user permissions before vector retrieval.
* **📊 Quantitative Evaluation (RAGAS):** Automated evaluation suite tracking **Context Precision**, **Context Recall**, and **Faithfulness**.
* **🔭 Observability:** Full-chain tracing integrated with **LangSmith / Phoenix** for latency and token profiling.

---

## 🏗️ Architecture Pipeline

```text
[ Apple 10-K PDF ]
       │
       ▼
 [ Structural Parser ] ──(Table Extraction & Parent-Child Chunking)──┐
                                                                     ▼
 [ Hybrid Retrieval ] ◄── (Query) ── [ Sparse BM25 ] + [ Dense Vector (Qdrant) ]
       │
       ▼
 [ RRF Fusion ] ──► [ Cross-Encoder Re-Ranker ] ──► [ Metadata/ACL Filter ]
                                                              │
                                                              ▼
 [ Answer Generation ] ◄── [ LLM (DeepSeek/GPT-4o) ] ◄── [ Semantic Cache ]
       │
       ▼
 [ Evaluation (RAGAS) & Observability (LangSmith) ]
```
# Additional operation

---

## Software configuration

* poppler
* tesseract

---

## Clear data

* remove index page (manual)

---