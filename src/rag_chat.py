import os
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

    with open(parent_store_path, "rb") as f:
        parent_store = pickle.load(f)
    with open(bm25_store_path, "rb") as f:
        bm25_retriever: BM25Retriever = pickle.load(f)

    retriever = vector_store.as_retriever(search_kwargs={"k": 50})
    hybrid_retriever = EnsembleRetriever(
        retrievers=[retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=encoder_model, top_n=3)
    child_reranker = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=hybrid_retriever
    )

    def rerank_then_fetch_unique_parent(query_str):
        top_child_docs = child_reranker.invoke(query_str)
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

    prompt = ChatPromptTemplate.from_template("""You are a senior financial analyst assistant. Answer the user's question accurately and completely based on the provided [Reference Context].

【FINANCIAL ANALYSIS RULES】:
1. Carefully inspect ALL provided reference chunks. If tables split metrics across rows or chunks, combine them to give a complete summary.
2. If the user asks for breakdowns (e.g., by product, region, segment), extract the itemized figures into a clear Markdown table.
3. Keep exact monetary units (e.g., "in millions").
4. If information is partially missing, provide what is available in the context and note what is missing.

【Reference Context】:
{context}

【User Query】:
{query}
Answer:
""")

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
