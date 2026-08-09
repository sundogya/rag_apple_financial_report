import os
import shutil
import uuid
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 假设你之前写好的切片函数
from src.chunker import chunk_markdown_file_new
def check_save_vector(persist_dir):
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434"
    )
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # ================= 3. 针对第一组数据发起向量打分检索 =================
    # 针对 Exhibit 3.1 提问
    query = "Restated Articles of Exhibit Number 3.1"

    print(f"🔎 发起 Chroma 向量匹配检索, 提问词: '{query}'")
    # 使用 similarity_search_with_score 获取与查询词的距离/得分
    results_and_scores = vector_store.similarity_search_with_score(query, k=5)

    # ================= 4. 打印向量匹配结果与得分 =================
    print("\n" + "="*70)
    print(f"🎯 Chroma 检索结果（Top 5 匹配项）:")
    print("💡 注意：Chroma 默认返回 L2 距离，【数值越小】代表越相似！")
    print("="*70)

    for i, (doc, score) in enumerate(results_and_scores):
        print(f"\n【 Top {i+1} 】 ─── L2 距离得分 (Distance): {score:.4f} (越小越匹配)")
        print(f"📌 Chunk 类型: {doc.metadata.get('type')} | 行范围: {doc.metadata.get('row_range', 'N/A')}")
        print(f"📌 归属章节: {doc.metadata.get('Item', doc.metadata.get('Part', 'N/A'))}")
        print("📝 文本前 150 字预览:")
        print("-" * 40)
        print(doc.page_content[:150].replace("\n", " "))
        print("-" * 40)
