import os
import re
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from .build_vector_store import build_vector_store
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
        chunk_size=3000,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""],
    )
    final_chunks = text_splitter.split_documents(header_splits)
    print(f"🎉 切片顺利完成！共得到 {len(final_chunks)} 个带元数据的 Chunks。")
    return final_chunks


def chunk_markdown_file_new(
    file_path: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    max_table_rows: int = 5
) -> list[Document]:
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 找不到文件: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # 1. 第一层切片：按标题抽取层级元数据
    headers_to_split_on = [
        ("#", "Part"),
        ("##", "Item"),
        ("###", "Section")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    header_splits = markdown_splitter.split_text(markdown_text)

    # 🛡️ 修复原因 3：如果文件开头在第一个 Header 之前有内容，强行补充防护
    if not header_splits:
        header_splits = [Document(page_content=markdown_text, metadata={})]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )

    final_chunks = []

    # 匹配带标签的表格块 OR 未打标的标准 Markdown 表格（双重保障）
    table_block_pattern = re.compile(
        r'(?:<!--\s*TABLE_START:\s*id=(tbl_\d+)\s*-->\n?(.*?)\n?<!--\s*TABLE_END:\s*id=\1\s*-->|((?:^[ \t]*\|[^\n]+\|[ \t]*\n){2,}))',
        re.DOTALL | re.MULTILINE
    )

    for doc in header_splits:
        text = doc.page_content
        base_metadata = doc.metadata.copy()

        last_idx = 0
        matches = list(table_block_pattern.finditer(text))

        if not matches:
            sub_docs = text_splitter.create_documents(
                texts=[text],
                metadatas=[{**base_metadata, "type": "text"}]
            )
            final_chunks.extend(sub_docs)
            continue

        for match in matches:
            start, end = match.span()
            # 兼容有标签和无标签两种匹配情况
            table_id = match.group(1) if match.group(1) else "tbl_auto"
            raw_table_content = match.group(2) if match.group(2) is not None else match.group(3)
            raw_table_content = raw_table_content.strip()

            # A. 处理表格前的普通文本
            non_table_text = text[last_idx:start].strip()
            if non_table_text:
                # 过滤残留的单边标签
                clean_text = re.sub(r'<!--\s*TABLE_(?:START|END):.*?-->', '', non_table_text).strip()
                if clean_text:
                    sub_docs = text_splitter.create_documents(
                        texts=[clean_text],
                        metadatas=[{**base_metadata, "type": "text"}]
                    )
                    final_chunks.extend(sub_docs)

            # B. 处理表格内容（智能识别表头与分隔线）
            table_lines = [l.strip() for l in raw_table_content.split("\n") if l.strip()]
            if len(table_lines) >= 2 and "|" in table_lines[0]:
                # 🛡️ 修复原因 1：检查第二行是否确实是 |---| 分隔线
                has_divider = "---" in table_lines[1] or ":---" in table_lines[1]
                
                if has_divider:
                    header_lines = table_lines[:2]
                    data_lines = table_lines[2:]
                else:
                    # 如果没有分隔线，只把第一行当表头，避免吃掉数据行！
                    header_lines = [table_lines[0], "| " + " | ".join(["---"] * table_lines[0].count("|")) + " |"]
                    data_lines = table_lines[1:]

                total_rows = len(data_lines)

                if total_rows == 0:
                    final_chunks.append(Document(
                        page_content=raw_table_content,
                        metadata={**base_metadata, "type": "table", "table_id": table_id}
                    ))
                else:
                    for i in range(0, total_rows, max_table_rows):
                        sub_data = data_lines[i : i + max_table_rows]
                        chunk_text = "\n".join(header_lines + sub_data)
                        metadata = {
                            **base_metadata,
                            "type": "table",
                            "table_id": table_id,
                            "row_range": f"{i + 1}-{min(i + max_table_rows, total_rows)}"
                        }
                        # print(f"📦 切分表格 {table_id}， 行范围: {metadata['row_range']}, section: {base_metadata.get('Section', '')}")
                        final_chunks.append(Document(
                            page_content=chunk_text,
                            metadata=metadata
                        ))
            else:
                sub_docs = text_splitter.create_documents(
                    texts=[raw_table_content],
                    metadatas=[{**base_metadata, "type": "text"}]
                )
                final_chunks.extend(sub_docs)
            last_idx = end

        # C. 处理尾部剩余文本
        remaining_text = text[last_idx:].strip()
        if remaining_text:
            clean_text = re.sub(r'<!--\s*TABLE_(?:START|END):.*?-->', '', remaining_text).strip()
            if clean_text:
                sub_docs = text_splitter.create_documents(
                    texts=[clean_text],
                    metadatas=[{**base_metadata, "type": "text"}]
                )
                final_chunks.extend(sub_docs)

    print(f"🎉 修复版切片完成！共得到 {len(final_chunks)} 个 Chunks。")
    return final_chunks


def save_chunks_as_vector_store(file_path: str, persist_dir="./data/chroma_db_ollama"):
    final_chunks = chunk_markdown_file_new(file_path=file_path)
    vector_store = build_vector_store(final_chunks, persist_dir=persist_dir)
    return vector_store