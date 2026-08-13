import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from pathlib import Path
import pickle

from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

# 项目根路径 (rag_apple_financial_report)
BASE_DIR = Path(__file__).resolve().parent.parent
# 1. 优化后的 Query 重写 Prompt
query_transform_prompt = ChatPromptTemplate.from_template("""You are an expert retrieval query optimizer for Apple's 10-K financial filings.
Your task is to convert the user's question into a highly effective, concise search query for retrieving exact text or table chunks from 10-K reports.

Rules:
- Preserve core financial metrics, specific fiscal years (e.g., 2025, 2024), and key business terms (e.g., "Operating Income", "Net Sales", "Segment").
- Remove conversational filler words (e.g., "Can you tell me...", "What were the...").
- Output ONLY the optimized search query text, with no explanations, markdown formatting, or quotes.

User Question: {query}
Optimized Search Query:""")

# ... (中间函数保持不变) ...

    # 2. 优化后的主回答 Prompt
prompt = ChatPromptTemplate.from_template("""You are a senior financial analyst assistant. Answer the user's question accurately and completely based ONLY on the provided [Reference Context].

【STRICT ZERO-HALLUCINATION & ABSENT DATA RULES】:
1. Grounding Rule: Rely ONLY on clear facts directly mentioned in the [Reference Context]. Do NOT assume or extrapolate.
2. Out-of-Scope / Absence Rule: If the context does not contain sufficient information (e.g., missing specific target year data or unmentioned topics), explicitly state that the provided document does not contain this information. NEVER guess or substitute with other years.

【FINANCIAL & TABLE ANALYSIS RULES】:
1. ⚠️ STRICT TIME-SERIES & METRIC ALIGNMENT (CRITICAL):
   - When extracting data from tables, strictly align each metric (Row Header) with its exact fiscal year (Column Header).
   - NEVER use historical data (e.g., 2024) to represent a requested target year (e.g., 2025) if the target year column is missing in the retrieved table.
2. Table Generation: ONLY generate a Markdown table if actual numerical financial data is present. If the user asks for breakdowns, extract them into a clear Markdown table.
3. ⚠️ MANDATORY MONETARY UNIT RULE (CRITICAL):
   - You MUST explicitly state the monetary unit (e.g., "in millions" or "dollars in millions") in either the table title, column headers, or summary text.
   - NEVER output standalone figures like "$209,586" without indicating whether it is in thousands, millions, or billions.

【Reference Context】:
{context}

【User Query】:
{query}
Answer:
""")


def get_top_unique_parent_docs(
    child_docs: list[Document], 
    parent_store: dict[str, Document], 
    target_parent_count: int = 4
) -> list[Document]:
    seen_ids = set()
    parent_docs = []
    
    for child in child_docs:
        parent_id = child.metadata.get("parent_id")
        if parent_id and parent_id in parent_store and parent_id not in seen_ids:
            seen_ids.add(parent_id)
            parent_docs.append(parent_store[parent_id])
            if len(parent_docs) >= target_parent_count:
                break

    return parent_docs


def create_rag_chain(
    persist_dir: str = None,
    parent_store_path: str = None,
    bm25_store_path: str = None,
    embedding_model: str = "nomic-embed-text",
    llm_model: str = "llama3.1:8b"
):
    """构建 RAG 链的核心逻辑（纯 Python，零 Streamlit 依赖）"""
    persist_dir = persist_dir or str(BASE_DIR / "data" / "chroma_db_ollama_parent_child")
    parent_store_path = parent_store_path or (BASE_DIR / "data" / "store" / "parent_store.pkl")
    bm25_store_path = bm25_store_path or (BASE_DIR / "data" / "store" / "bm25_retriever.pkl")

    if not Path(parent_store_path).exists():
        raise FileNotFoundError(f"未找到父级存储文件: {parent_store_path}")
    if not Path(bm25_store_path).exists():
        raise FileNotFoundError(f"未找到 BM25 检索器文件: {bm25_store_path}")

    embeddings = OllamaEmbeddings(
        model=embedding_model, 
        base_url="http://127.0.0.1:11434"
    )
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    query_rewriter = query_transform_prompt | ChatOllama(model="llama3.1:8b", temperature=0.0) | StrOutputParser()
    with open(parent_store_path, "rb") as f:
        parent_store = pickle.load(f)
    with open(bm25_store_path, "rb") as f:
        bm25_retriever: BM25Retriever = pickle.load(f)
    retriever = vector_store.as_retriever(search_kwargs={"k": 30})
    hybrid_retriever = EnsembleRetriever(
        retrievers=[retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=encoder_model, top_n=5)
    child_reranker = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=hybrid_retriever
    )

    def rerank_then_fetch_unique_parent(query_str):
        search_str = query_rewriter.invoke({"query": query_str}).strip()
        top_child_docs = child_reranker.invoke(search_str)
        # print("\n" + "="*20 + f" [DEBUG] Reranker 筛选出的 Top {min(10, len(top_child_docs))} 子块 " + "="*20)
        # for i, doc in enumerate(top_child_docs[:10]):
        #     parent_id = doc.metadata.get("parent_id", "Unknown")
        #     # 替换换行符，限制只预览前 120 个字，防止控制台刷屏
        #     preview = doc.page_content.replace('\n', ' ')[:120]
        #     print(f"[{i+1}] Parent_ID: {parent_id}")
        #     print(f"    内容: {preview}...\n")
        # print("="*65 + "\n")
        return get_top_unique_parent_docs(top_child_docs, parent_store, target_parent_count=4)

    rerank_retriever = RunnableLambda(rerank_then_fetch_unique_parent)
    
    llm = ChatOllama(
        model=llm_model,
        temperature=0.0,
        base_url="http://127.0.0.1:11434",
        num_ctx=8192
    )

    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs):
            part = doc.metadata.get("Part", "N/A")
            item = doc.metadata.get("Item", "N/A")
            section = doc.metadata.get("Section", "N/A")
            formatted.append(f"--- [Chunk {i+1}] (Source: {part} -> {item} -> {section}) ---\n{doc.page_content}")
        return "\n\n".join(formatted)
    rag_chain = (
        {
            "context": RunnableLambda(lambda x: format_docs(x["context"])),
            "query": lambda x: x["query"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rerank_retriever, rag_chain
