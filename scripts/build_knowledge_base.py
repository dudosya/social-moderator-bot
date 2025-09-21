# scripts/build_knowledge_base.py

import os
import sys
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from markdown_it import MarkdownIt
from mdit_plain.renderer import RendererPlain

# --- Boilerplate ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# --- End of Boilerplate ---

# --- DEFINITIVE MODEL AND OUTPUT PATHS ---
EMBEDDING_MODEL = 'intfloat/multilingual-e5-large' # The new, superior model
FAISS_INDEX_PATH = os.path.join(project_root, 'models', 'company_kb.index')
CHUNKS_FILE_PATH = os.path.join(project_root, 'models', 'chunks.json')

def build_knowledge_base():
    """
    Builds the KB using a SOTA embedding model (E5) and its required
    asymmetric prefixing protocol for passages.
    """
    print(f"--- Starting KB Build (Model: {EMBEDDING_MODEL}) ---")
    
    # --- Step 1: Load, Clean, and Chunk (No changes to this logic) ---
    source_dir = os.path.join(project_root, 'knowledge_base_source')
    full_text = ""
    for filename in os.listdir(source_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(source_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                full_text += f.read() + "\n\n"

    md_parser = MarkdownIt(renderer_cls=RendererPlain)
    cleaned_text = md_parser.render(full_text)
    cleaned_text = cleaned_text.replace('|', ' ').replace('---', ' ')
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, length_function=len
    )
    initial_chunks = text_splitter.split_text(cleaned_text)
    
    final_chunks = [chunk.strip() for chunk in initial_chunks if len(chunk.strip()) > 20]
    print(f"[1] Data preparation complete. Created {len(final_chunks)} high-quality chunks.")

    # --- Step 2: Implement E5 "Passage" Prefixing ---
    print('[2] Applying "passage: " prefix to all chunks for E5 model...')
    prefixed_chunks = ["passage: " + chunk for chunk in final_chunks]
    
    # --- VALIDATION ---
    print("--- Sample Prefixed Chunk ---")
    print(prefixed_chunks[0][:200] + "...") # Show a sample
    print("-----------------------------")

    # --- Step 3: Embed and Index with the new model ---
    print(f"[3] Loading embedding model: '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("[4] Converting prefixed chunks to embeddings...")
    embeddings = model.encode(prefixed_chunks, show_progress_bar=True)
    d = embeddings.shape[1]
    print(f"✅ Created {len(embeddings)} embeddings of dimension {d}.")

    print("[5] Building FAISS index...")
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype('float32'))
    print(f"✅ FAISS index built. Total vectors: {index.ntotal}")

    # --- Step 4: Save Artifacts (We save the ORIGINAL chunks, not the prefixed ones) ---
    print("[6] Saving index and original chunks...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=4)
        
    print("\n--- Knowledge base build process COMPLETED. ---")

if __name__ == "__main__":
    build_knowledge_base()