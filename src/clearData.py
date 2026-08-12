import re

def clean_markdown_noise(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    clean_lines = []
    for line in lines:
        # 只要这行同时包含 "Apple Inc." 和 "Form 10-K"，直接丢弃（即页脚行）
        # 这种逻辑比正则表达式更轻量、更高效，且 100% 不会漏掉
        if "Apple Inc." in line and "Form 10-K" in line:
            continue
        clean_lines.append(line)
        
    # 将清洗后的文本拼接起来
    clean_text = "".join(clean_lines)
    
    # 覆盖保存
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(clean_text)
        
    print("🎉 清洗成功！所有类似 'Apple Inc. | 2025 Form 10-K | X' 的页脚已被精准剔除！")
def detect_footer_in_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 精准匹配形如 "Apple Inc. | 2025 Form 10-K | 23" 的页脚
    footer_pattern = r"Apple\s+Inc\.\s*[\|\|]?\s*\d{4}\s+Form\s+10-K\s*[\|\|]?\s*\d+"
    
    matches = re.findall(footer_pattern, content, flags=re.IGNORECASE)
    
    if matches:
        print("🔍 侦测到以下页脚样式:")
        for match in matches:
            print(f" - {match}")
    else:
        print("❌ 未侦测到页脚样式。")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print("🔍 正在侦测页脚在文件里的真实形态...\n")
    for idx, line in enumerate(lines):
        # 只要包含 10-K 或 Apple 就打印出来
        if "10-K" in line or "Apple" in line:
            print(f"第 {idx+1} 行原始文本: {repr(line)}")

def inject_10k_headers(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. 匹配 Part I, Part II, Part III, Part IV，替换为一级标题 "# Part X"
    # 匹配独占一行的 Part I/II/III/IV
    content = re.sub(
        r"^\s*(Part\s+[I|V|X]+)\s*$", 
        r"\n# \1\n", 
        content, 
        flags=re.MULTILINE | re.IGNORECASE
    )

    # 2. 匹配 Item 1., Item 1A., Item 7A. 等，替换为二级标题 "## Item X. Title"
    # 匹配形如 "Item 1.", "Item 1A.", "Item 7A." 开头的独立行
    content = re.sub(
        r"^\s*(Item\s+\d+[A-Z]?\..*?)$", 
        r"\n## \1\n", 
        content, 
        flags=re.MULTILINE | re.IGNORECASE
    )

    # 清理可能产生的多余三换行
    content = re.sub(r"\n{3,}", "\n\n", content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("🎉 针对 10-K 的 Part/Item 结构注入完成！")

def auto_fix_broken_tables(md_content: str) -> str:
    """
    全局通用表格缝合器：自动寻找所有被切割的表格并合并，无需硬编码页码
    """
    # 模式说明：
    # 匹配 </table> 闭合标签，到下一个 <table> 或 <tbody> 之间的所有游离文本/重复表头
    pattern = r'</table>\s*(?:[^\n]*\n){0,10}?\s*<table[^>]*>\s*(?:<thead>[\s\S]*?</thead>)?\s*<tbody>'
    
    # 将跨页截断的表格直接连贯起来
    cleaned_md = re.sub(pattern, '', md_content, flags=re.IGNORECASE)
    
    # 自动清理单独夹在表格中间的游离表头文字（如 Filing Date / Period End 等）
    cleaned_md = re.sub(r'Filing Date/\s*Period End', '', cleaned_md)
    
    return cleaned_md
def auto_clean_markdown_file(file_path: str,output_path: str = None):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_md = f.read()
        
    # 一键全自动清洗
    clean_md = auto_fix_broken_tables(raw_md)
    
    if output_path is None:
        output_path = file_path
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(clean_md)
        
    print("✅ 自动化管道处理完成！已被分割的表格已全自动缝合。")

import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import re

def add_table_tags_to_file(
    input_file_path: str, 
    output_file_path: str, 
    encoding: str = "utf-8"
) -> int:
    """
    读取输入的 Markdown 文件，自动识别其中的所有表格，
    打上带有唯一 ID 的 HTML 锚点标记 (TABLE_START / TABLE_END)，
    并将打标后的内容保存到输出文件路径。

    :param input_file_path: 输入的原始 .md 文件路径
    :param output_file_path: 打标完成后保存的新 .md 文件路径
    :param encoding: 文件编码，默认 utf-8
    :return: 成功打标的表格总数量
    """
    # 1. 校验输入文件是否存在
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"❌ 找不到指定的输入文件: {input_file_path}")

    # 2. 读取原始 Markdown 文件内容
    with open(input_file_path, "r", encoding=encoding) as f:
        markdown_text = f.read()

    # 3. 正则表达式：匹配 Markdown 表格（至少连续 2 行包含 | 的结构）
    table_pattern = re.compile(
        r'((?:^[ \t]*\|[^\n]+\|[ \t]*\n){2,})', re.MULTILINE
    )

    table_count = 0

    def replace_with_tags(match):
        nonlocal table_count
        table_count += 1
        table_content = match.group(0).strip()
        # 插入包含唯一 ID 的表格起始与结束锚点
        return (
            f"\n\n<!-- TABLE_START: id=tbl_{table_count} -->\n"
            f"{table_content}\n"
            f"<!-- TABLE_END: id=tbl_{table_count} -->\n\n"
        )

    # 4. 执行替换打标
    tagged_markdown = table_pattern.sub(replace_with_tags, markdown_text)

    # 5. 自动确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 6. 将打标后的文本写入输出文件
    with open(output_file_path, "w", encoding=encoding) as f:
        f.write(tagged_markdown)

    print(f"✅ 处理完成！成功为文件中的 {table_count} 个表格打上标记。")
    print(f"📄 输出文件已保存至: [{output_file_path}]")

    return table_count
