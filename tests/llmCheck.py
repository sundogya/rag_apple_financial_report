from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

print("🔍【阶段 1】测试向量数据库检索 (nomic-embed-text)...")
try:
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text", 
        base_url="http://localhost:11434"
    )
    vector_store = Chroma(
        persist_directory="../data/chroma_db_ollama",
        embedding_function=embeddings
    )
    docs = vector_store.similarity_search("Apple 供应链", k=1)
    print(f"✅ 阶段 1 成功！检索到了 {len(docs)} 个 Chunk。")
except Exception as e:
    print(f"❌ 阶段 1 报错: {e}")

print("\n" + "="*40 + "\n")

print("🔍【阶段 2】测试 Gemma4 单独对话...")
try:
    llm = ChatOllama(
        model="gemma4",
        base_url="http://localhost:11434"
    )
    res = llm.invoke("用一句话介绍苹果公司")
    print(f"✅ 阶段 2 成功！Gemma4 回复: {res.content}")
except Exception as e:
    print(f"❌ 阶段 2 报错: {e}")
