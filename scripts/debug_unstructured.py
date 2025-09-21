# scripts/debug_unstructured.py

import os
import sys
from unstructured.partition.auto import partition

# --- Boilerplate to make project root accessible ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# --- End of Boilerplate ---

# --- Target File for Investigation ---
# We use the most complex file to ensure our logic is robust.
FILE_TO_DEBUG = os.path.join(project_root, 'knowledge_base_source', '4358_Адреса офисов продаж.txt')

def investigate_unstructured():
    """
    Loads a single file with unstructured and prints the raw element
    details to understand its structure.
    """
    print(f"--- Investigating output of unstructured.partition ---")
    print(f"Processing file: {FILE_TO_DEBUG}")

    if not os.path.exists(FILE_TO_DEBUG):
        print(f"🚨 ERROR: File not found. Please ensure '{os.path.basename(FILE_TO_DEBUG)}' is in the 'knowledge_base_source' directory.")
        return

    # Use partition to break the document into elements
    elements = partition(filename=FILE_TO_DEBUG)

    print(f"\n✅ unstructured found {len(elements)} elements in the document.")
    
    # --- VALIDATION: Inspect the first 15 raw elements ---
    print("\n--- Raw Data of First 15 Elements ---")
    for i, element in enumerate(elements[:15]):
        print(f"[Element #{i+1}]")
        print(f"  - Type: {type(element)}")
        print(f"  - Text: {element.text}")
        print("-" * 20)

if __name__ == "__main__":
    investigate_unstructured()