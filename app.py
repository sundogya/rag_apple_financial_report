import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

import streamlit as st
from src.rag_chat import create_rag_chain

# Streamlit 专用的缓存包装
@st.cache_resource(show_spinner=False)
def load_rag_chain():
    return create_rag_chain()

st.set_page_config(page_title="Apple 财报 AI 助手", page_icon="🍎", layout="wide")
st.title("🍎 Apple 10-K 财报 AI 智能分析系统")
st.caption("基于 Ollama (Llama 3.1 8B + Nomic Embed) & ChromaDB 本地私有化 RAG 架构")

# 加载 RAG 链
rerank_retriever, rag_chain = load_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 参考来源出处"):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")

if prompt_input := st.chat_input("输入关于苹果财报的问题 (如: What are the primary supply chain risks?)..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        retrieved_docs = rerank_retriever.invoke(prompt_input)
        sources = [
            f"[Chunk {i+1}] {doc.metadata.get('Part', 'N/A')} -> {doc.metadata.get('Item', 'N/A')} -> {doc.metadata.get('Section', 'N/A')}" 
            for i, doc in enumerate(retrieved_docs)
        ]

        def generate_response():
            for chunk in rag_chain.stream({"context": retrieved_docs, "query": prompt_input}):
                yield chunk

        full_response = st.write_stream(generate_response)

        if sources:
            with st.expander("📚 参考来源出处"):
                for src in sources:
                    st.markdown(f"- `{src}`")

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })
