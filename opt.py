# import src.parser
# import test.fileCheck
# import test.chunkCheck
# import src.clearData
import src.chunker
# import src.rag_chat

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
src.chunker.save_chunks_as_vector_store("data/apple_10k_2025_claude_with_table_tags.md", "./data/chroma_db_ollama")
# test.chunkCheck.check_chunk_health("data/apple_10k_2025_unstructured.md")