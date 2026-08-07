from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
def chunk_markdown_file(file_path: str):
    # 1. 读取手动清理完美的 Markdown
    with open(file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # 2. 第一层切片：按标题抽取层级元数据
    headers_to_split_on = [
        ("#", "Part"),
        ("##", "Item"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    header_splits = markdown_splitter.split_text(markdown_text)

    # 3. 第二层切片：微调 Chunk 长度（推荐 1000~1200 字符）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
        separators=["\n\n", "\n", " ", ""],
    )
    final_chunks = text_splitter.split_documents(header_splits)
    print(f"🎉 切片顺利完成！共得到 {len(final_chunks)} 个带元数据的 Chunks。")
    return final_chunks

