import os
import time
import streamlit as st
from dotenv import load_dotenv

# 1. 加载环境变量 (.env 文件)
load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

# 2. 页面全局配置
st.set_page_config(
    page_title="Apple 10-K 财报 AI 智能助手",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. 兼容模块导入路径
@st.cache_resource(show_spinner="正在初始化 RAG 检索引擎与 ChromaDB...")
def load_rag_chain():
    try:
        from src.rag_chat import create_rag_chain
    except ImportError:
        try:
            from rag_chat import create_rag_chain
        except ImportError:
            from rag_chat_2 import create_rag_chain
    return create_rag_chain()

# 4. 侧边栏控制面板
with st.sidebar:
    st.markdown("### 🍎 控制面板与诊断")
    
    st.markdown("""
    **系统架构信息:**
    - **LLM**: Ollama (`llama3.1:8b`)
    - **Embedding**: `BAAI/bge-small-en-v1.5` (Nomic Embed)
    - **Vector DB**: ChromaDB (Parent-Child)
    - **Reranker**: `bge-reranker-base`
    """)
    
    st.markdown("---")
    
    # 显示 LangSmith 跟踪状态
    langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2") == "true"
    status_icon = "🟢" if langsmith_enabled else "⚪"
    st.markdown(f"**LangSmith 链路诊断:** {status_icon} {'已启用' if langsmith_enabled else '未开启'}")
    
    st.markdown("---")
    
    # 清空对话按钮
    if st.button("🗑️ 清空对话历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("Financial-RAG Engine v1.0 | Apple FY2025 10-K")

# 5. 主页面 Header
st.title("🍎 Apple 10-K 财报 AI 智能分析系统")
st.caption("基于 Ollama (Llama 3.1 8B + Nomic Embed) & ChromaDB 本地私有化 RAG 架构 | 支持混合检索与 BGE 重排")

# 6. 加载 RAG 链
rerank_retriever, rag_chain = load_rag_chain()

# 7. 初始化对话历史 (带默认欢迎语)
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "你好！我是苹果公司 2025 财年 Form 10-K 财报分析助手。你可以向我提问关于苹果公司的财务表现、产品营收拆分、法律诉讼（如 DMA/DOJ 案件）或供应链风险等问题！",
            "sources": []
        }
    ]

# 8. 渲染历史对话
for msg in st.session_state.messages:
    avatar = "🍎" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 查看参考来源出处"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['title']}**")
                    if src.get("preview"):
                        st.caption(f"> {src['preview']}...")

# 9. 处理用户提问与交互
if prompt_input := st.chat_input("输入关于苹果财报的问题 (如: What are the primary supply chain risks?)..."):
    # 记录并渲染用户输入
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_input)

    # 助手回答生成流程
    with st.chat_message("assistant", avatar="🍎"):
        # 优化点 1：使用 st.status 展示动态检索全过程
        with st.status("正在检索 10-K 财报并分析...", expanded=True) as status:
            st.write("🔍 执行 Query 重写与 Vector + BM25 混合检索...")
            retrieved_docs = rerank_retriever.invoke(prompt_input)
            
            st.write(f"⚖️ BGE Cross-Encoder 完成重排，锁定 {len(retrieved_docs)} 个核心父切片...")
            
            # 格式化出处与预览片段
            source_records = []
            for i, doc in enumerate(retrieved_docs):
                part = doc.metadata.get("Part", "N/A")
                item = doc.metadata.get("Item", "N/A")
                sec = doc.metadata.get("Section", "N/A")
                # 提取前 200 字做摘要预览
                preview_text = doc.page_content.replace("\n", " ")[:200]
                source_records.append({
                    "title": f"[Chunk {i+1}] {part} -> {item} -> {sec}",
                    "preview": preview_text
                })
                
            time.sleep(0.1)  # 仅用于平滑动画展示
            status.update(label="检索完成，大模型思考生成中...", state="complete", expanded=False)

        # 优化点 2：打字机流式输出
        def generate_response():
            for chunk in rag_chain.stream({"context": retrieved_docs, "query": prompt_input}):
                yield chunk

        full_response = st.write_stream(generate_response)

        # 优化点 3：回答下方挂载可展开的源文档切片与预览
        if source_records:
            with st.expander("📚 查看参考来源出处"):
                for src in source_records:
                    st.markdown(f"**{src['title']}**")
                    st.caption(f"> {src['preview']}...")

        # 保存到 session_state 持久化
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": source_records
        })
