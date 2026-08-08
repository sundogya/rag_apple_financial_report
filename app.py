import os
import streamlit as st

# 1. 设置环境变量，绕过本地代理
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 2. 页面基础属性设置
st.set_page_config(page_title="Apple 财报 AI 助手", page_icon="🍎", layout="wide")
st.title("🍎 Apple 10-K 财报 AI 智能分析系统")
st.caption("基于 Ollama (Llama 3.1 8B + Nomic Embed) & ChromaDB 本地私有化 RAG 架构")

# 3. 初始化服务（使用 @st.cache_resource 确保加载后的对象驻留内存，防止每次点击都重新加载）
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
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    llm = ChatOllama(
        model=llm_model,
        temperature=0.0,
        base_url="http://127.0.0.1:11434",
        num_ctx=2048
    )

    def format_docs(docs):
        formatted = []
        for doc in docs:
            part = doc.metadata.get("Part", "N/A")
            item = doc.metadata.get("Item", "N/A")
            formatted.append(f"【Source: {part} -> {item}】\n{doc.page_content}")
        return "\n\n" + "=" * 40 + "\n\n".join(formatted)

    prompt = ChatPromptTemplate.from_template("""You are an expert financial analyst. Answer the question strictly based on the following context.

Context:
{context}

Question: 
{question}

Requirements:
1. Base your answer strictly on the provided context. Cite sources (e.g. Part I -> Item 1A) where applicable.
2. If the answer cannot be found in the context, explicitly state "Based on the provided sections, I cannot answer this question."

Answer:""")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, rag_chain

# 加载 RAG 链对象
retriever, rag_chain = load_rag_chain()

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
        # 先检索文档，提取出处的元数据
        retrieved_docs = retriever.invoke(prompt_input)
        sources = [f"{doc.metadata.get('Part', 'N/A')} -> {doc.metadata.get('Item', 'N/A')}" for doc in retrieved_docs]

        # 定义生成器，把 LLM 吐出来的 Chunk 传递给 Streamlit
        def generate_response():
            for chunk in rag_chain.stream(prompt_input):
                yield chunk

        # 💡 关键点：用 st.write_stream 替代 print()，这样才会打字吐到浏览器上！
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