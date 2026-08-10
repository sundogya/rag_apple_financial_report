import src.parser
import test.fileCheck
import test.chunkCheck
import test.vectorCheck
import src.clearData
import src.chunker
import src.rag_chat
import src.bm25Key

# src.parser.convert_pdf_to_markdown("data/10k.pdf", "data/apple_10k_2025_pdfplumber.md")
# src.parser.parse_pdf_to_elements("data/10k.pdf", "data/apple_10k_2025_unstructured.md")
# src.parser.process_full_pdf_with_vlm("data/10k.pdf", "data/apple_10k_2025_vlm.md", batch_size=2)
# test.fileCheck.check_markdown_health("data/apple_10k_2025_pdfplumber.md")
# test.fileCheck.check_markdown_health("data/apple_10k_2025_unstructured.md")
# src.clearData.clean_markdown_noise("data/apple_10k_2025_unstructured.md")
# src.clearData.inject_10k_headers("data/apple_10k_2025_unstructured.md")
# src.clearData.auto_clean_markdown_file("data/apple_10k_2025_unstructured.md", "data/apple_10k_2025_stitched.md")
# src.clearData.add_table_tags_to_file("data/apple_10k_2025_claude.md", "data/apple_10k_2025_claude_with_table_tags.md")
# src.chunker.chunk_markdown_file("data/apple_10k_2025_unstructured.md")
# src.chunker.save_chunks_as_vector_store("data/apple_10k_2025_unstructured.md", "./data/chroma_db_ollama")
# src.chunker.save_chunks_as_vector_store("data/apple_10k_2025_claude_with_table_tags.md", "./data/chroma_db_ollama")
# test.chunkCheck.check_chunk_health("data/apple_10k_2025_unstructured.md")
# test.chunkCheck.check_chunk_missed("data/apple_10k_2025_claude_with_table_tags.md")
# test.vectorCheck.check_save_vector("./data/chroma_db_ollama")
# src.bm25Key.bm25_retrieval_test("data/apple_10k_2025_claude_with_table_tags.md")
# src.bm25Key.hybrid_retrieval_test("data/apple_10k_2025_claude_with_table_tags.md",persist_dir="./data/chroma_db_ollama",query="Net sales 2025")
src.bm25Key.hybrid_retrieval_with_rerank_test(file_path="data/apple_10k_2025_claude_with_table_tags.md", query="net sales by category for 2025, 2024 and 2023")