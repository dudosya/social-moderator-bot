# scripts/test_processor.py

import sys
import os
import json

# --- Boilerplate to make sibling packages accessible ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# ---------------------------------------------------

from app.parsers.youtube_parser import parse_youtube_comments
from app.core.processor import CommentProcessor

# A known good URL for consistent testing
TEST_URL = "https://www.youtube.com/watch?v=aZE5KPgiHIE"

def main():
    """
    Runs a full end-to-end test of the parser and processor,
    printing detailed results for the first 5 comments.
    """
    print(f"--- Running Full Parser-Processor Test ---")
    print(f"Fetching comments from: {TEST_URL}\n")
    
    # 1. Parse Comments
    raw_comments = parse_youtube_comments(TEST_URL)
    if not raw_comments:
        print("🚨 FAILED to fetch comments. Exiting.")
        return
    print(f"✅ Successfully parsed {len(raw_comments)} comments.\n")
    
    # 2. Process Comments
    print("Initializing CommentProcessor (this may take a moment)...")
    try:
        processor = CommentProcessor()
        enriched_comments = processor.process_batch(raw_comments)
        print("✅ Batch processing complete.\n")
    except Exception as e:
        print(f"🚨🚨 FAILED during comment processing: {e}")
        return
    
    # 3. Display detailed results for the first 5 comments
    print("--- Detailed Analysis of First 5 Comments ---")
    for i, comment in enumerate(enriched_comments[:5]):
        print(f"\n--- Comment #{i+1} ---")
        print(f"  Text    : {comment.get('text', 'N/A')[:100]}...") # Truncate long text
        print(f"  Language: {comment.get('language', 'N/A')}")
        print(f"  Sentiment: {comment.get('sentiment', 'N/A')}")
        print(f"  Profanity: {comment.get('has_profanity', 'N/A')}")
        print(f"  Spam     : {comment.get('is_spam', 'N/A')}")
        print(f"  Comment Type: {comment.get('comment_type', 'N/A')}")
        print(f"  Triage Score: {comment.get('triage_score', 'N/A')}")
        # Pretty print suggested response for readability
        response = comment.get('suggested_response', 'N/A').replace('\n', '\n             ')
        print(f"  Response : {response}")
        
    print("\n--- Test Complete ---")

if __name__ == "__main__":
    main()