import re

def check_markdown_health(md_path):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = content.split("\n")
    total_lines = len(lines)
    
    # 1. 统计表格行数
    table_lines = [line for line in lines if "|" in line]
    
    # 2. 统计 Markdown 标题数 (#, ##, ###)
    header_lines = [line for line in lines if re.match(r"^#{1,4}\s", line)]
    
    # 3. 检查是否有常见的页眉噪音
    noise_count = len(re.findall(r"Form 10-K", content, re.IGNORECASE))

    print("=== 📊 Markdown 文件健康度报告 ===")
    print(f"📄 总行数: {total_lines}")
    print(f"📊 识别到的表格行数: {len(table_lines)} (占比 {len(table_lines)/total_lines:.1%})")
    print(f"🏷️ 识别到的 Markdown 标题数: {len(header_lines)}")
    print(f"⚠️ 'Form 10-K' 关键字出现次数: {noise_count}")
    
    # 诊断结论
    print("\n=== 💡 诊断建议 ===")
    if len(table_lines) > 50 and len(header_lines) > 10:
        print("✅ 表现良好！表格和标题均被成功识别，可以放心进入下一个【切片 (Chunking)】步骤。")
    else:
        print("⚠️ 警告：表格或标题数量偏少，可能出现了表格塌陷或标题层级丢失的情况！")

