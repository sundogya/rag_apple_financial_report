import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import re
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from src.chunker import chunk_markdown_file_new
def financial_tokenizer(text: str) -> list[str]:
    """
    保留英文单词、下划线、以及带小数点的完整编号（如 4.9, 21.1）
    """
    # 转换为小写，匹配字母数字组合或带小数点的数字（如 4.9）
    tokens = re.findall(r'\b\w+(?:\.\w+)*\b', text.lower())
    return tokens
def bm25_retrieval_test(file_path):
    

    # 1. 载入切片好的 Documents
    docs = chunk_markdown_file_new(file_path)
    print(f"📄 正在为 {len(docs)} 个 Chunk 构建 BM25 关键词索引...")
    # 2. 实例化 BM25 检索器
    # k=5 代表检索最匹配的前 5 个 Chunk
    bm25_retriever = BM25Retriever.from_documents(docs, k=3, tokenizer=financial_tokenizer)

    print("✅ BM25 检索器构建完成！\n")

    # 3. 🎯 测试精确代码检索
    # 针对向量检索容易飘移的精确编号进行压测
    query = "Net sales 2025"  # 或试一下 "Form 8-K", "9/17/15"

    print(f"🔎 [BM25 检索] 提问词: '{query}'")
    bm25_results = bm25_retriever.invoke(query)

    # 4. 打印 BM25 召回结果
    print("\n" + "="*70)
    print(f"🎯 BM25 精确匹配结果（Top {len(bm25_results)}）:")
    print("="*70)

    for i, doc in enumerate(bm25_results):
        print(f"\n【 BM25 Top {i+1} 】")
        print(f"📌 Chunk 类型: {doc.metadata.get('type')} | 行范围: {doc.metadata.get('row_range', 'N/A')}")
        print(f"📌 归属章节: {doc.metadata.get('Item', doc.metadata.get('Part', 'N/A'))}")
        print("📝 文本前 150 字预览:")
        print("-" * 40)
        print(doc.page_content[:150].replace("\n", " "))
        print("-" * 40)
def hybrid_retrieval_test(file_path,persist_dir="./data/chroma_db_ollama",query="Exhibit 4.9"):
    # 1. 载入切片好的 Documents
    docs = chunk_markdown_file_new(file_path)
    print(f"📄 正在为 {len(docs)} 个 Chunk 构建 BM25 关键词索引...")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        persist_directory=persist_dir, 
        embedding_function=embeddings,
    )
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    bm25_retriever = BM25Retriever.from_documents(docs, k=10)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.8, 0.2]
    )

    print("✅ Hybrid Search 混合检索器构建完成！\n")
    print(f"🔎 [Hybrid Search 发起检索] 提问词: '{query}'")
    hybrid_docs = hybrid_retriever.invoke(query)

    # ================= 5. 🖨️ 打印前 3 个混合召回结果 =================
    print("\n" + "="*70)
    print(f"🎯 混合检索成功融合并召回 Top {len(hybrid_docs)} 个 Chunk:")
    print("="*70)

    for i, doc in enumerate(hybrid_docs[:3]):
        print(f"\n【 Hybrid Top {i+1} 】")
        print(f"📌 Chunk 类型: {doc.metadata.get('type')} | 行范围: {doc.metadata.get('row_range', 'N/A')}")
        print(f"📌 归属章节: {doc.metadata.get('Item', doc.metadata.get('Part', 'N/A'))}")
        print("📝 文本前 150 字预览:")
        print("-" * 40)
        print(doc.page_content[:150].replace("\n", " "))
        print("-" * 40)
def hybrid_retrieval_with_rerank_test(file_path,persist_dir="./data/chroma_db_ollama",query="Exhibit 4.9"):
    docs = chunk_markdown_file_new(file_path)
    print(f"📄 正在为 {len(docs)} 个 Chunk 构建 BM25 关键词索引...")
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        persist_directory=persist_dir, 
        embedding_function=embeddings,
    )
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(docs, k=5)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=encoder_model, top_n=3)

    # ================= 3. 组装终极 RAG 检索器 (Compressor Retriever) =================
    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=hybrid_retriever
    )

    print("✅ Day 7: 混合检索 + Re-ranker 终极管道构建完成！\n")

    print(f"🔎 [终极管道检索] 提问词: '{query}'")
    final_docs = rerank_retriever.invoke(query)

    # ================= 5. 🖨️ 打印重排后的 Top 3 结果 =================
    print("\n" + "="*70)
    print(f"🎯 重排后最终送给 LLM 的 Top {len(final_docs)} 个黄金 Chunk:")
    print("="*70)

    for i, doc in enumerate(final_docs):
        print(f"\n【 Rerank Top {i+1} 】")
        print(f"📌 Chunk 类型: {doc.metadata.get('type')} | 行范围: {doc.metadata.get('row_range', 'N/A')}")
        print("📝 文本前 150 字预览:")
        print("-" * 40)
        print(doc.page_content[:150].replace("\n", " "))
        print("-" * 40)

