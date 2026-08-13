# 📈 Financial-RAG-Engine: Advanced QA & Evaluation System for Apple 10-K Reports

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RAGAS](https://img.shields.io/badge/Evaluation-RAGAS-green.svg)](https://github.com/explodinggradients/ragas)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-blue.svg)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama--Llama3.1-orange.svg)](https://ollama.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-ready, production-grade **Advanced RAG (Retrieval-Augmented Generation)** engine specifically engineered to tackle complex financial documents, multi-page financial tables, and dense numerical queries from the **Apple Annual Form 10-K Report (Fiscal Year 2025)**.

---

## 🌟 Key Features

* **📄 Table-Aware & Parent-Child Chunking:** Maintains financial table integrity by wrapping tables with explicit tags (`<!-- TABLE_START -->` / `<!-- TABLE_END -->`). Implements a **Small-to-Big (Parent-Child) Retriever** that matches fine-grained child chunks while feeding complete parent context to LLMs.
* **🔍 Hybrid Search Engine:** Combines **Dense Vector Search** (Ollama `nomic-embed-text`) with **Sparse BM25 Search** (exact keyword/ticker matching) using **Ensemble Fusion (0.5 / 0.5 Weighting)** for maximum retrieval coverage.
* **🎯 Cross-Encoder Re-Ranking:** Integrated `BAAI/bge-reranker-base` to re-rank Top-30 hybrid candidates down to the Top-4/5 most relevant parent contexts, drastically reducing context noise and hallucination.
* **🛡️ Strict Financial Guardrails:** System prompts optimized for financial time-series and metric alignment (Row Header vs. Column Year Verification), enforcing strict "Reject-if-Absent" rules to guarantee zero-hallucination.
* **📊 Robust Benchmark & Quantitative Evaluation:** Tested against a 5-suite **Golden Dataset (100+ QA Pairs)** including table breakdowns, direct factoids, specific legal clauses, and adversarial out-of-scope queries.
  * **Factual Accuracy:** **96.2%**
  * **Hallucination Robustness:** **96.2%**
* **🔭 Observability & UX:** Native **Streamlit** Web UI with progress tracking, Query Rewriting, and streaming responses, backed by **LangSmith** for full-chain latency and token profiling.

---

## 🏗️ System Architecture

```text
[ Apple 2025 10-K PDF ] ──► [ Claude Parsing & Table Tagging ] ──► [ Clean Markdown ]
                                                                       │
                                                                       ▼
                                                          [ Parent-Child Chunking ]
                                                                       │
[ User Query ] ──► [ Query Rewriter ]                                   ▼
                         │                        [ ChromaDB (Child) ] + [ BM25 Index ]
                         ▼                                             │
               [ Hybrid Retrieval (k=30) ] ◄───────────────────────────┘
                         │
                         ▼
             [ BGE Cross-Encoder Reranker ] ──► [ Top Child Chunks ]
                                                       │
                                                       ▼
                                            [ Parent Context Fetch ]
                                                       │
                                                       ▼
[ Final Answer ] ◄── [ Ollama (Llama 3.1 8B) ] ◄── [ Strict Financial Alignment Prompt ]
```

---

## 📊 Evaluation & Performance Benchmark

Evaluated using a 5-suite **Golden Dataset** tailored for Apple's 10-K report (covering Product Performance, Segment Breakdown, Legal Proceedings, and Adversarial Out-of-Scope Questions):

| Metric | Score | Performance & Insights |
| :--- | :--- | :--- |
| **Factual Accuracy** | **96.2%** | Precise decimal and monetary extraction (exact matching for EPS, Debt Schedules, Revenues). |
| **Hallucination Robustness** | **96.2%** | Successfully rejected out-of-scope questions without fabricating information. |
| **Context Recall (In-Scope)** | **> 90.0%** | Hybrid search + Reranker reliably fetches multi-year financial footnotes. |
| **TTFT (Time-To-First-Token)**| **~1.5 - 2.5s** | Optimized for local consumer hardware running Ollama Llama 3.1 8B. |

---

## 🛠️ Environment & Prerequisites

### System Dependencies
Ensure the following system tools are installed for PDF parsing and OCR support:
* **Poppler** (PDF rendering)
* **Tesseract-OCR** (Optical Character Recognition)

```bash
# macOS
brew install poppler tesseract

# Ubuntu/Debian
sudo apt-get install -y poppler-utils tesseract-ocr
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧹 Data Preprocessing Pipeline

To achieve high table retrieval precision, we apply a strict preprocessing workflow before ingestion:

1. **Manual Index & Cover Page Removal:** Clean unnecessary front-matter to reduce vector space noise.
2. **PDF to Markdown Conversion (Claude Prompt):**
   We leverage **Claude** for structural PDF-to-Markdown conversion due to superior multi-page table alignment compared to `pdfplumber` or `Unstructured`.

   > **Prompt used for PDF Conversion:**
   ```text
      你是一位专业的金融数据解析助手。请阅读我提供的pdf，将其中的内容转化为干净的 Markdown 文件：
            【核心要求】：
            1. 重点识别其中的所有表格,并且保留表格的描述信息。如果表格跨页，请自动拼接为一个完整连续的 Markdown 表格，绝对不要截断。
            2. 自动过滤掉重复的页眉、页脚（如 Apple Inc. | 2025 Form 10-K）和页码。
            3. 确保所有数据列精准对齐。直接输出 Markdown文件，不要写任何开场白或解释文字。
   ```

3. **Table Tagging:** Wrap extracted Markdown tables with `<!-- TABLE_START: id=tbl_x -->` and `<!-- TABLE_END: id=tbl_x -->` markers to prevent splitter truncation.

---

## 🚀 Quick Start

### 1. Ingest Data & Build Vector Index
Ensure Ollama is running (`ollama serve`) and pull the required models:
```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

Build the ChromaDB vector database and BM25 index:
```bash
python scripts/build_index.py --input data/apple_10k_2025_claude_with_table_tags.md
```

### 2. Run Terminal RAG Chain
```bash
python src/rag_chat.py
```

### 3. Launch Streamlit Web UI
```bash
streamlit run app.py
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
