import src.parser
import test.fileCheck
import test.chunkCheck
import src.clearData
import src.chunker


# src.parser.convert_pdf_to_markdown("data/10k.pdf", "data/apple_10k_2025_pdfplumber.md")
# src.parser.parse_pdf_to_elements("data/10k.pdf", "data/apple_10k_2025_unstructured.md")
# test.fileCheck.check_markdown_health("data/apple_10k_2025_pdfplumber.md")
# test.fileCheck.check_markdown_health("data/apple_10k_2025_unstructured.md")
# src.clearData.clean_markdown_noise("data/apple_10k_2025_unstructured.md")
# src.clearData.inject_10k_headers("data/apple_10k_2025_unstructured.md")
# src.chunker.chunk_markdown_file("data/apple_10k_2025_unstructured.md")
test.chunkCheck.check_chunk_health("data/apple_10k_2025_unstructured.md")