from copyreg import pickle
import os
import pickle


import streamlit as st
from src.chunker import chunk_markdown_file_new

# 1. 设置环境变量，绕过本地代理
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

# 2. 页面基础属性设置
st.set_page_config(page_title="Apple 财报 AI 助手", page_icon="🍎", layout="wide")
st.title("🍎 Apple 10-K 财报 AI 智能分析系统")
st.caption("基于 Ollama (Llama 3.1 8B + Nomic Embed) & ChromaDB 本地私有化 RAG 架构")
def get_top_unique_parent_docs(
    child_docs: list[Document], 
    parent_store: dict[str, Document], 
    target_parent_count: int = 4  # 🎯 确保必须拿到 4 个不同的完整父块上下文！
) -> list[Document]:
    """顺次遍历重排后的 Child Docs，直到凑满 target_parent_count 个不重复的 Parent Doc 为止。"""
    seen_ids = set()
    parent_docs = []
    
    for child in child_docs:
        parent_id = child.metadata.get("parent_id")
        if parent_id and parent_id in parent_store and parent_id not in seen_ids:
            seen_ids.add(parent_id)
            parent_docs.append(parent_store[parent_id])
            
            # 凑齐目标数量的独立父块后立即停止，防止上下文过载
            if len(parent_docs) >= target_parent_count:
                break

    return parent_docs

# 3. 初始化服务
@st.cache_resource
def load_rag_chain(
    persist_dir="./data/chroma_db_ollama_parent_child",
    embedding_model="nomic-embed-text",
    llm_model="llama3.1:8b"
):
    embeddings = OllamaEmbeddings(
        model=embedding_model, 
        base_url="http://127.0.0.1:11434"
    )
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    with open("./data/store/parent_store.pkl", "rb") as f:
        parent_store = pickle.load(f)
    with open("./data/store/bm25_retriever.pkl", "rb") as f:
        bm25_retriever: BM25Retriever = pickle.load(f)
    retriever = vector_store.as_retriever(search_kwargs={"k": 30})
    hybrid_retriever = EnsembleRetriever(
        retrievers=[retriever, bm25_retriever],
        weights=[0.3, 0.7]
    )
    encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=encoder_model, top_n=3)
    child_reranker = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=hybrid_retriever
    )
    def rerank_then_fetch_unique_parent(query_str):
        top_child_docs = child_reranker.invoke(query_str)
        # 确保 LLM 一定能拿到 4 个不同维度的完整大表/章节
        return get_top_unique_parent_docs(top_child_docs, parent_store, target_parent_count=4)

    rerank_retriever = RunnableLambda(rerank_then_fetch_unique_parent)
    
    # 💡 修复 1：将 num_ctx 扩大到 8192，防止 Top 5 截断！
    llm = ChatOllama(
        model=llm_model,
        temperature=0.0,
        base_url="http://127.0.0.1:11434",
        num_ctx=8192
    )

    # 💡 修复 2：在 format_docs 中增加 [Chunk X] 编号，与 Prompt 的引用对齐！
    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs):
            part = doc.metadata.get("Part", "N/A")
            item = doc.metadata.get("Item", "N/A")
            section = doc.metadata.get("Section", "N/A")
            formatted.append(f"--- [Chunk {i+1}] (Source: {part} -> {item} -> {section}) ---\n{doc.page_content}")
        return "\n\n".join(formatted)

    # 💡 修复 3：补全了 【User Query】 缺失的右括号 】
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

    # 💡 修复 4：链条直接接收已经格式化好/或检索出来的 docs，避免二次检索逻辑冲突
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

# 加载 RAG 链对象
rerank_retriever, rag_chain = load_rag_chain()

# 4. 初始化对话历史消息列表
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. 在页面渲染之前的历史聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 参考来源出处"):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")

# 6. 处理新的用户输入
if prompt_input := st.chat_input("输入关于苹果财报的问题 (如: What are the primary supply chain risks?)..."):
    # 6.1 在前端渲染用户的问题
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # 6.2 渲染 AI 气泡并执行打字机流式输出
    with st.chat_message("assistant"):
        # 1. 显式进行一次精准检索
        retrieved_docs = rerank_retriever.invoke(prompt_input)
        # with st.expander("🔍 调试：查看送给 LLM 的真实 Parent Context"):
        #     for doc in retrieved_docs:
        #         st.text(
        #             f"【ID: {doc.metadata.get('parent_id')} | Type: {doc.metadata.get('type')}】\n"
        #             f"{doc.page_content[:500]}..."
        #         )
        # 2. 提取前端用于展开展现的出处
        sources = [
            f"[Chunk {i+1}] {doc.metadata.get('Part', 'N/A')} -> {doc.metadata.get('Item', 'N/A')} -> {doc.metadata.get('Section', 'N/A')}" 
            for i, doc in enumerate(retrieved_docs)
        ]

        # 3. 将现成的 retrieved_docs 塞给管道，顺畅流式打印！
        def generate_response():
            for chunk in rag_chain.stream({"context": retrieved_docs, "query": prompt_input}):
                yield chunk

        full_response = st.write_stream(generate_response)

        # 展开显示参考来源
        if sources:
            with st.expander("📚 参考来源出处"):
                for src in sources:
                    st.markdown(f"- `{src}`")

        # 存入历史记录
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })
