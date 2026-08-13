import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import re
import uuid
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from .build_vector_store import (
    build_vector_store,
    persist_rag_database,
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
def chunk_markdown_file_to_parent_child(file_path: str):
    parent_docs,parent_store = chunk_markdown_file_to_parent(file_path)
    child_docs = chunk_parent_docs_to_child_optimized(parent_docs)
    return parent_docs, parent_store, child_docs

def is_junk_text(text: str) -> bool:
    """过滤无效垃圾块：如纯 '---'、空白符或字数过少的分割线"""
    cleaned = re.sub(r"[\s\-\*\=\#|]", "", text)
    return len(cleaned) < 30  # 剥离所有标点/分割符后，有效字符少于 30 的直接过滤
def clean_exhibit_tail(text: str) -> str:
    """如果在正文末尾碰到了 Exhibit 列表，直接从 Exhibit 处截断，防止混入附件噪声"""
    exhibit_match = re.search(
        r"(?i)(?:exhibit\s+number|exhibit\s+index|description\s+of\s+exhibit)",
        text,
    )
    if exhibit_match:
        # 只保留 Exhibit 出现之前的正文内容
        return text[: exhibit_match.start()].strip()
    return text.strip()
def convert_table_rows_to_text(header_line: str, data_lines: list[str]) -> str:
    """将 Markdown 表格的数据行转化为自然语言陈述句，极大地帮助向量模型理解数字"""
    headers = [h.strip() for h in header_line.split("|") if h.strip()]
    summaries = []
    
    for line in data_lines:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) >= 2 and len(headers) >= 2:
            row_name = cells[0]
            row_details = []
            for h, c in zip(headers[1:], cells[1:]):
                if c and c != "-":
                    row_details.append(f"{h} is {c}")
            if row_details:
                summaries.append(f"- {row_name}: " + ", ".join(row_details))
                
    if summaries:
        return "【Row Summaries】:\n" + "\n".join(summaries)
    return ""
# ==========================================
# 模块一：父块切分函数 (chunk_markdown_file_to_parent)
# ==========================================
def chunk_markdown_file_to_parent(
    file_path: str,
    max_parent_text_size: int = 1300,
    parent_text_overlap: int = 200,
) -> tuple[list[Document], dict[str, Document]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 找不到文件: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # 1. 按标题提取层级元数据
    headers_to_split_on = [
        ("#", "Part"),
        ("##", "Item"),
        ("###", "Section"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    header_splits = markdown_splitter.split_text(markdown_text)

    if not header_splits:
        header_splits = [Document(page_content=markdown_text, metadata={})]

    table_block_pattern = re.compile(
        r"(?:<!--\s*TABLE_START:\s*id=(tbl_\d+)\s*-->\n?(.*?)\n?<!--\s*TABLE_END:\s*id=\1\s*-->|((?:^[ \t]*\|[^\n]+\|[ \t]*\n){2,}))",
        re.DOTALL | re.MULTILINE,
    )

    parent_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_parent_text_size,
        chunk_overlap=parent_text_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )

    parent_docs: list[Document] = []
    parent_store: dict[str, Document] = {}

    for doc in header_splits:
        text = doc.page_content
        base_metadata = doc.metadata.copy()

        # 🛡️ 防噪 1：元数据层面过滤 Exhibit
        # item_meta = str(base_metadata.get("Item", "")).lower()
        # part_meta = str(base_metadata.get("Part", "")).lower()
        # if "exhibit" in item_meta or "exhibit" in part_meta:
        #     continue

        last_idx = 0
        matches = list(table_block_pattern.finditer(text))

        if not matches:
            # 🛡️ 防噪 2：内容截断 + 垃圾块过滤
            clean_text = re.sub(
                r"<!--\s*TABLE_(?:START|END):.*?-->", "", text
            )
            # clean_text = clean_exhibit_tail(clean_text)

            if not is_junk_text(clean_text):
                p_docs = parent_text_splitter.create_documents(
                    texts=[clean_text],
                    metadatas=[{**base_metadata, "type": "text"}],
                )
                for p_doc in p_docs:
                    if not is_junk_text(p_doc.page_content):
                        p_id = f"parent_{uuid.uuid4().hex[:8]}"
                        p_doc.metadata["parent_id"] = p_id
                        parent_docs.append(p_doc)
                        parent_store[p_id] = p_doc
            continue

        for match in matches:
            start, end = match.span()
            table_id = match.group(1) if match.group(1) else "tbl_auto"
            raw_table_content = (
                match.group(2) if match.group(2) is not None else match.group(3)
            ).strip()

            non_table_text = text[last_idx:start].strip()
            table_prefix_text = ""

            if non_table_text:
                clean_text = re.sub(r"<!--\s*TABLE_(?:START|END):.*?-->", "", non_table_text)

                if not is_junk_text(clean_text):
                    text_lines = clean_text.split("\n")
                    split_pos = len(text_lines)

                    # 从后往前找，找到紧挨着表格的那一两句引言（比如 "The following table shows..."）
                    for i in range(len(text_lines) - 1, -1, -1):
                        line_str = text_lines[i].strip().lower()
                        if not line_str:
                            continue
                        if (
                            any(
                                k in line_str
                                for k in [
                                    "following table",
                                    "shows net sales",
                                    "as follows",
                                    "dollars in millions",
                                    "net sales by",
                                ]
                            )
                            or line_str.startswith("###")
                        ):
                            split_pos = i
                            break

                    normal_text = "\n".join(text_lines[:split_pos]).strip()
                    # 💡 提取出这句引言，做为表格的“灵魂前缀”
                    table_prefix_text = "\n".join(text_lines[split_pos:]).strip()

                    if not is_junk_text(normal_text):
                        p_docs = parent_text_splitter.create_documents(
                            texts=[normal_text],
                            metadatas=[{**base_metadata, "type": "text"}],
                        )
                        for p_doc in p_docs:
                            if not is_junk_text(p_doc.page_content):
                                p_id = f"parent_{uuid.uuid4().hex[:8]}"
                                p_doc.metadata["parent_id"] = p_id
                                parent_docs.append(p_doc)
                                parent_store[p_id] = p_doc

            # 💡 核心组合：[引言前缀] + \n\n + [表格数据]，合成不可分割的 Table Parent
            combined_table_content = (
                f"{table_prefix_text}\n\n{raw_table_content}".strip()
                if table_prefix_text
                else raw_table_content
            )

            p_id = f"parent_{uuid.uuid4().hex[:8]}"
            table_parent_doc = Document(
                page_content=combined_table_content,
                metadata={
                    **base_metadata,
                    "parent_id": p_id,
                    "type": "table",      # 强制标明是 table 类型！
                    "table_id": table_id,
                },
            )
            parent_docs.append(table_parent_doc)
            parent_store[p_id] = table_parent_doc

            last_idx = end

        remaining_text = text[last_idx:].strip()
        if remaining_text:
            clean_text = re.sub(
                r"<!--\s*TABLE_(?:START|END):.*?-->", "", remaining_text
            )
            # clean_text = clean_exhibit_tail(clean_text)

            if not is_junk_text(clean_text):
                p_docs = parent_text_splitter.create_documents(
                    texts=[clean_text],
                    metadatas=[{**base_metadata, "type": "text"}],
                )
                for p_doc in p_docs:
                    if not is_junk_text(p_doc.page_content):
                        p_id = f"parent_{uuid.uuid4().hex[:8]}"
                        p_doc.metadata["parent_id"] = p_id
                        parent_docs.append(p_doc)
                        parent_store[p_id] = p_doc

    print(
        f"🧹 清洗完成！删除了所有垃圾符号块与 Exhibit 尾部噪声，剩余 {len(parent_docs)} 个干净 Parent Documents。"
    )
    return parent_docs, parent_store
# ==========================================
# 模块二：子块切分函数 (chunk_parent_docs_to_child)
# ==========================================
def chunk_parent_docs_to_child_optimized(
    parent_docs: list[Document],
    child_chunk_size: int = 400,
    child_chunk_overlap: int = 80,
    max_table_rows: int = 4,
) -> list[Document]:
    """
    优化版 Child 切片逻辑：
    1. 增加 Section Header 注入：给每一个 Child 顶部添加 [Location Context]
    2. 增加 Table-to-Text 增强：在 Child 表格底部自动生成自然语言摘要
    3. 保留引言句：将 Parent 中的 table_prefix_text 完整带给 Child
    """
    child_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )

    child_chunks: list[Document] = []

    for p_doc in parent_docs:
        base_metadata = p_doc.metadata.copy()
        doc_type = base_metadata.get("type", "text")
        parent_id = base_metadata.get("parent_id")
        
        # 📌 构建统一的位置元数据前缀（包含 Part -> Item -> Section）
        location_parts = []
        if base_metadata.get("Part"):
            location_parts.append(base_metadata['Part'])
        if base_metadata.get("Item"):
            location_parts.append(base_metadata['Item'])
        if base_metadata.get("Section"):
            location_parts.append(base_metadata['Section'])
            
        header_context_str = f"【Context: {' > '.join(location_parts)}】\n" if location_parts else ""

        # --------------------------------------------------
        # 分支 A： Parent 为【表格】 -> 行切分 + 表头前缀 + 语义摘要注入
        # --------------------------------------------------
        if doc_type == "table":
            table_lines = [l.strip() for l in p_doc.page_content.split("\n") if l.strip()]

            if len(table_lines) >= 2 and "|" in table_lines[0]:
                has_divider = "---" in table_lines[1] or ":---" in table_lines[1]

                if has_divider:
                    header_lines = table_lines[:2]
                    data_lines = table_lines[2:]
                else:
                    header_lines = [
                        table_lines[0],
                        "| " + " | ".join(["---"] * table_lines[0].count("|")) + " |",
                    ]
                    data_lines = table_lines[1:]

                total_rows = len(data_lines)

                if total_rows == 0:
                    # 表格无数据行，直接做全量 Child
                    full_content = f"{header_context_str}{p_doc.page_content}"
                    child_chunks.append(
                        Document(
                            page_content=full_content,
                            metadata={**base_metadata, "is_child": True, "row_range": "full"},
                        )
                    )
                else:
                    # 按 max_table_rows 切分，并强化 Child 的文本描述
                    for i in range(0, total_rows, max_table_rows):
                        sub_data = data_lines[i : i + max_table_rows]
                        markdown_table = "\n".join(header_lines + sub_data)
                        
                        # 💡 核心强化 1：生成自然语言摘要
                        semantic_summary = convert_table_rows_to_text(header_lines[0], sub_data)
                        
                        # 💡 核心强化 2：拼接 [层级前缀] + [Markdown 表格] + [语义摘要]
                        child_text_components = [header_context_str, markdown_table]
                        if semantic_summary:
                            child_text_components.append(semantic_summary)
                            
                        enhanced_child_text = "\n\n".join(child_text_components)
                        row_range_str = f"{i + 1}-{min(i + max_table_rows, total_rows)}"

                        child_chunks.append(
                            Document(
                                page_content=enhanced_child_text,
                                metadata={
                                    **base_metadata,
                                    "is_child": True,
                                    "row_range": row_range_str,
                                },
                            )
                        )
            else:
                # 降级处理
                sub_docs = child_text_splitter.create_documents(
                    texts=[f"{header_context_str}{p_doc.page_content}"],
                    metadatas=[{**base_metadata, "is_child": True}],
                )
                child_chunks.extend(sub_docs)

        # --------------------------------------------------
        # 分支 B： Parent 为【正文】 -> 注入 Header 前缀后做滑动切分
        # --------------------------------------------------
        else:
            # 在切分前给大段文本附加上 Context 头
            text_to_split = f"{header_context_str}{p_doc.page_content}"
            sub_docs = child_text_splitter.create_documents(
                texts=[text_to_split],
                metadatas=[{**base_metadata, "is_child": True}],
            )
            child_chunks.extend(sub_docs)

    print(f"🚀 深度优化版 Child 切片完成！共得到 {len(child_chunks)} 个带语义增强的 Child Chunks。")
    return child_chunks
def chunk_parent_docs_to_child(
    parent_docs: list[Document],
    child_chunk_size: int = 300,
    child_chunk_overlap: int = 50,
    max_table_rows: int = 3,
) -> list[Document]:
    """第二步：接收 parent_docs 列表，将其打碎生成可供向量与 BM25 索引的 Child Document 列表。

    1. 表格 Child：智能解析表头/分隔线，按 max_table_rows 切分并强制注入原始表头。
    2. 正文 Child：按小粒度（如 300 Token）滑动切分。
    3. 所有 Child 继承 Parent 的元数据（Part/Item/Section/parent_id）。
    """
    child_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )

    child_chunks: list[Document] = []

    for p_doc in parent_docs:
        base_metadata = p_doc.metadata.copy()
        doc_type = base_metadata.get("type", "text")
        parent_id = base_metadata.get("parent_id")

        # --------------------------------------------------
        # 分支 A： Parent 为【表格】 -> 行切分并重续表头
        # --------------------------------------------------
        if doc_type == "table":
            table_id = base_metadata.get("table_id", "tbl_auto")
            table_lines = [
                l.strip()
                for l in p_doc.page_content.split("\n")
                if l.strip()
            ]

            if len(table_lines) >= 2 and "|" in table_lines[0]:
                # 🛡️ 检查第二行是否确实是 |---| 分隔线
                has_divider = (
                    "---" in table_lines[1] or ":---" in table_lines[1]
                )

                if has_divider:
                    header_lines = table_lines[:2]
                    data_lines = table_lines[2:]
                else:
                    # 如果没有分隔线，只把第一行当表头，自动补全合成分隔线
                    header_lines = [
                        table_lines[0],
                        "| "
                        + " | ".join(["---"] * table_lines[0].count("|"))
                        + " |",
                    ]
                    data_lines = table_lines[1:]

                total_rows = len(data_lines)

                if total_rows == 0:
                    child_chunks.append(
                        Document(
                            page_content=p_doc.page_content,
                            metadata={
                                **base_metadata,
                                "is_child": True,
                                "row_range": "full",
                            },
                        )
                    )
                else:
                    # 按 max_table_rows 切割子表格并粘贴表头
                    for i in range(0, total_rows, max_table_rows):
                        sub_data = data_lines[i : i + max_table_rows]
                        chunk_text = "\n".join(header_lines + sub_data)
                        row_range_str = f"{i + 1}-{min(i + max_table_rows, total_rows)}"

                        child_chunks.append(
                            Document(
                                page_content=chunk_text,
                                metadata={
                                    **base_metadata,
                                    "is_child": True,
                                    "row_range": row_range_str,
                                },
                            )
                        )
            else:
                # 无法按标准 Markdown 表格解析的降级处理
                sub_docs = child_text_splitter.create_documents(
                    texts=[p_doc.page_content],
                    metadatas=[{**base_metadata, "is_child": True}],
                )
                child_chunks.extend(sub_docs)

        # --------------------------------------------------
        # 分支 B： Parent 为【正文】 -> 小粒度滑动窗口切分
        # --------------------------------------------------
        else:
            sub_docs = child_text_splitter.create_documents(
                texts=[p_doc.page_content],
                metadatas=[{**base_metadata, "is_child": True}],
            )
            child_chunks.extend(sub_docs)

    print(
        f"🎉 Child 切片完成！共得到 {len(child_chunks)} 个 Chunks (准备入库 ChromaDB/BM25)。"
    )
    return child_chunks
def save_parent_child_chunks_as_vector_store(file_path: str, persist_dir="./data/chroma_db_ollama_parent_child"):
    parent_docs, parent_store, child_docs = chunk_markdown_file_to_parent_child(file_path)
    vector_store = persist_rag_database(child_docs, parent_store=parent_store, chroma_dir=persist_dir)
    return parent_docs, parent_store, child_docs, vector_store