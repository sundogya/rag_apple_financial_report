import pdfplumber
from unstructured.partition.pdf import partition_pdf
import pymupdf  # PyMuPDF
import base64
import ollama
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# 1. 全局配置：强制绕过代理
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"
# from docling.document_converter import DocumentConverter
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
        strategy="hi_res" ,           # 高精度模式
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


def pdf_page_to_base64(page) -> str:
    """将单页 PDF 渲染为 Base64 图片"""
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode("utf-8")

def process_full_pdf_with_vlm(pdf_path: str, output_md_path: str, batch_size: int = 2):
    """
    全自动分批管道：按 batch_size（如每 2 页一组）循环调用 VLM 并自动拼接 Markdown
    """
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    print(f"📄 开始全自动解析 PDF，共 {total_pages} 页...")

    # 如果旧文件存在，先清空
    if os.path.exists(output_md_path):
        os.remove(output_md_path)

    prompt = """
    你是一位专业的金融数据解析助手。请阅读我提供的这几页财报截图，将其中的内容转化为干净的 Markdown 格式：
    
    【核心要求】：
    1. 重点识别其中的所有表格。如果表格跨页，请自动拼接为一个完整连续的 Markdown 表格，绝对不要截断。
    2. 自动过滤掉重复的页眉、页脚（如 Apple Inc. | 2025 Form 10-K）和页码。
    3. 确保所有数据列精准对齐。直接输出 Markdown，不要写任何开场白或解释文字。
    """

    # 按 batch_size 步进循环（例如 0~1 页，2~3 页，4~5 页...）
    for start_idx in range(0, total_pages, batch_size):
        end_idx = min(start_idx + batch_size, total_pages)
        print(f"🤖 正在处理第 {start_idx + 1} ~ {end_idx} 页 / 共 {total_pages} 页...")

        images_payload = []
        for p in range(start_idx, end_idx):
            page = doc.load_page(p)
            images_payload.append(pdf_page_to_base64(page))

        # 调用本地 VLM 模型
        response = ollama.chat(
            model="qwen3-vl:8b",
            messages=[{
                "role": "user",
                "content": prompt,
                "images": images_payload
            }]
        )

        page_md = response["message"]["content"]

        # 追加写入总 MD 文件
        with open(output_md_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n<!-- Page {start_idx + 1} to {end_idx} -->\n\n")
            f.write(page_md)

    print(f"🎉 整份文档解析完成！完整 Markdown 已保存至: {output_md_path}")

