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
import re

import re

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



