import os
from langchain.retrievers import BM25Retriever, ContextualCompressionRetriever
import streamlit as st
from src.chunker import chunk_markdown_file_new

# 1. 设置环境变量，绕过本地代理
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

from langchain.retrievers import EnsembleRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

# 2. 页面基础属性设置
st.set_page_config(page_title="Apple 财报 AI 助手", page_icon="🍎", layout="wide")
st.title("🍎 Apple 10-K 财报 AI 智能分析系统")
st.caption("基于 Ollama (Llama 3.1 8B + Nomic Embed) & ChromaDB 本地私有化 RAG 架构")

# 3. 初始化服务
@st.cache_resource
def load_rag_chain(
    persist_dir="./data/chroma_db_ollama",
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
    
    docs = chunk_markdown_file_new("./data/apple_10k_2025_claude_with_table_tags.md")
    bm25_retriever = BM25Retriever.from_documents(docs, k=30)
    retriever = vector_store.as_retriever(search_kwargs={"k": 30})
    hybrid_retriever = EnsembleRetriever(
        retrievers=[retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=encoder_model, top_n=5)
    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=hybrid_retriever
    )
    
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
    prompt = ChatPromptTemplate.from_template("""You are a senior financial analyst assistant. Answer the user's question accurately, completely, and STRICTLY based on the provided [Reference Context].

【CRITICAL FINANCIAL CONSTRAINTS】:
1. 📖 FULL CONTEXT SCAN: Carefully inspect ALL provided context chunks. Do not stop or draw conclusions after reading only the first chunk.
2. 🎯 GRANULARITY & BREAKDOWN MATCHING:
   - If the query requests a breakdown or disaggregation (e.g., "by category", "by segment", "by region"), you MUST locate and extract line-item details (e.g., iPhone, Mac, Services).
   - STRICTLY FORBIDDEN: Returning only high-level "Total" figures when itemized breakdown tables are present in the context.
3. 📊 MANDATORY MARKDOWN FORMATTING:
   - Present multi-year numerical metrics using a clean Markdown table format:
     | Category / Line Item | 2025 | 2024 | 2023 |
   - Preserve units of measure (e.g., "$ in millions") exactly as reported in the context.
4. 🛡️ STRICT GROUNDING & NO HALLUCINATION:
   - Rely EXCLUSIVELY on the provided context. Do not calculate, extrapolate, or infer missing figures using external knowledge.
   - If the context lacks sufficient information, state clearly: "The provided context does not contain sufficient details to answer this question."

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
