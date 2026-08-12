import src.chunker

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
def check_chunk_missed(file_path):
    chunks = src.chunker.chunk_markdown_file_new(file_path)
    # 2. 搜索文本里包含 "3.1" 或 "3.2" 的所有 Chunk
    first_group_chunks = [c for c in chunks if "3.1" in c.page_content or "3.2" in c.page_content]

    print(f"🔍 搜寻到包含 Exhibit 3.1/3.2 的 Chunk 数量: {len(first_group_chunks)}")

    if first_group_chunks:
        print("\n--- 🎯 找到的第一组 Chunk 内容如下 ---")
        print(first_group_chunks[0].page_content)
        print("Metadata:", first_group_chunks[0].metadata)
    else:
        print("❌ 警告：切片列表里根本没有包含 3.1 / 3.2 的 Chunk！说明切片算法在处理第一组数据时跳过了它。")
