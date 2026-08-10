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
* add table tag to md file (to optimize chunks of table content)

---

## Prompt to convert pdf to .md file

* pdfplumber (Not convert all Table conent)
* Unstructured (Table content problem)
* Gemini(Not all Content)
* Claude(Ok)
```
你是一位专业的金融数据解析助手。请阅读我提供的pdf，将其中的内容转化为干净的 Markdown 文件：
    【核心要求】：
    1. 重点识别其中的所有表格,并且保留表格的描述信息。如果表格跨页，请自动拼接为一个完整连续的 Markdown 表格，绝对不要截断。
    2. 自动过滤掉重复的页眉、页脚（如 Apple Inc. | 2025 Form 10-K）和页码。
    3. 确保所有数据列精准对齐。直接输出 Markdown文件，不要写任何开场白或解释文字。
```