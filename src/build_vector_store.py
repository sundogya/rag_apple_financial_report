import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import pickle
import shutil

# 1. 核心：设置环境变量绕过全局代理拦截
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

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


def persist_rag_database(
    child_docs: list,
    parent_store: dict,
    chroma_dir: str = "./data/chroma_db_ollama",
    parent_store_path: str = "./data/store/parent_store.pkl",
    bm25_store_path: str = "./data/store/bm25_retriever.pkl",
    embedding_model_name: str = "nomic-embed-text",
):
    """将 Child Chunks 写入向量库与 BM25，并将 Parent Store 持久化至磁盘。"""
    # --------------------------------------------------
    # 防坑步骤 1：清空历史向量库磁盘目录，防止数据污染
    # --------------------------------------------------
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
        print(f"🧹 已彻底清空历史向量库目录: {chroma_dir}")

    # --------------------------------------------------
    # 步骤 2：持久化 Parent Store 字典
    # --------------------------------------------------
    with open(parent_store_path, "wb") as f:
        pickle.dump(parent_store, f)
    print(f"💾 Parent Store 已持久化保存至: {parent_store_path}")

    # --------------------------------------------------
    # 步骤 3：初始化 Embedding 模型与 ChromaDB 入库
    # --------------------------------------------------
    print(
        f"⏳ 正在加载 Embedding 模型 [{embedding_model_name}] 并写入向量库..."
    )
    embeddings = OllamaEmbeddings(
        model=embedding_model_name, 
        base_url="http://127.0.0.1:11434"
    )

    # 批量构建 ChromaDB（将 child_docs 写入磁盘）
    vectorstore = Chroma.from_documents(
        documents=child_docs,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name="apple_2025_10k_child",
    )
    print(
        f"✅ ChromaDB 构建完成！共写入 {vectorstore._collection.count()} 条子块向量。"
    )

    # --------------------------------------------------
    # 步骤 4：构建并持久化 BM25 关键词检索器
    # --------------------------------------------------
    print("🔍 正在构建 BM25 关键词索引...")
    bm25_retriever = BM25Retriever.from_documents(child_docs)
    bm25_retriever.k = 50  # 设置粗召回 Top 50

    with open(bm25_store_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"✅ BM25 检索器已持久化保存至: {bm25_store_path}")

    print("\n🎉 所有索引构建与落地完成！现在可以随时加载并执行 Hybrid Search 了。")
    return vectorstore, bm25_retriever

