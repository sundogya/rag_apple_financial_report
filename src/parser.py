import pdfplumber
from unstructured.partition.pdf import partition_pdf

def convert_pdf_to_markdown(pdf_path, output_md_path):
    md_content = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # 1. 尝试提取表格
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    # 将二维列表转为 Markdown 表格字符串
                    md_table = ""
                    for row_idx, row in enumerate(table):
                        # 过滤 None 值
                        clean_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                        md_table += "| " + " | ".join(clean_row) + " |\n"
                        if row_idx == 0:  # 加上表头分隔线
                            md_table += "| " + " | ".join(["---"] * len(clean_row)) + " |\n"
                    md_content.append(f"\n{md_table}\n")
            # 2. 提取普通文本（简单版）
            text = page.extract_text()
            if text and not tables: # 如果没有表格，按纯文本处理
                md_content.append(text)         
    # 写入 md 文件
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(md_content))
    print(f"转换完成，已保存至: {output_md_path}")
def parse_pdf_to_elements(pdf_path, output_md_path):
    # 自动识别文本块、标题、表格元素
    elements = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,  # 自动用 AI 提取表格结构
        strategy="hi_res"            # 高精度模式
    )
    md_text = ""
    for element in elements:
        if element.category == "Table":
            # 自动转成 html/markdown 表格
            md_text += f"\n\n{element.metadata.text_as_html}\n\n"
        elif element.category == "Header":
            md_text += f"\n## {element.text}\n"
        else:
            md_text += f"{element.text}\n"
            
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
