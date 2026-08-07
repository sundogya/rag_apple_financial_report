import src

def check_chunk_health(chunk_path):
    final_chunks = src.chunker.chunk_markdown_file(chunk_path)
    # 4. 打印一个 Chunk 验证元数据
    if final_chunks:
            # 1. 看看有多少 Chunk 成功抓到了 Metadata
            chunks_with_meta = [c for c in final_chunks if c.metadata]
            print(f"📊 总 Chunk 数量: {len(final_chunks)}")
            print(f"✅ 成功带上 Part/Item 元数据的 Chunk 数量: {len(chunks_with_meta)}")
    
            # 2. 抽查一个带有 Metadata 的正文 Chunk
            if chunks_with_meta:
                print("\n--- 🔍 正文 Chunk 抽查 ---")
                print("【Metadata】:", chunks_with_meta[4].metadata)
                print("【Content】:\n", chunks_with_meta[4].page_content[:200], "...")
            else:
                print("❌ 警告：所有 Chunk 的 Metadata 都是空的，说明 md 文件里可能还没有成功注入 '#' 或 '##'！")