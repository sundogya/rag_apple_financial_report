import os

# 1. 核心：设置环境变量绕过全局代理拦截
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def build_vector_store(chunks, persist_dir="./chroma_db_nomic"):
    print("⏳ 初始化 Ollama Nomic Embedding 模型...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text", 
        base_url="http://127.0.0.1:11434"
    )

    print(f"⏳ 正在一次性将 {len(chunks)} 个片段向量化并存入磁盘 ({persist_dir})...")
    
    # 直接一行代码全部写入，无需手动批处理循环
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"🎉 向量数据库构建成功！保存在：{persist_dir}")
    return vector_store
