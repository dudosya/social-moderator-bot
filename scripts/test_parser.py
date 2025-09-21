# scripts/test_parser.py

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.parsers.youtube_parser import parse_youtube_comments

# --- Test URLs ---
# 1. A video that should have comments
URL_SUCCESS = "https://www.youtube.com/watch?v=r456JTqHaJ4"

# 2. A completely invalid video URL
URL_INVALID = "https://www.youtube.com/watch?v=INVALID_VIDEO_ID"

# 3. A real video where comments are turned off
URL_COMMENTS_DISABLED = "https://www.youtube.com/watch?v=XqZsoesa55w&list=RDXqZsoesa55w&start_radio=1"


def run_test(test_name: str, url: str, should_succeed: bool):
    """Helper function to run a single test case and print the result."""
    print(f"\n--- {test_name} ---")
    print(f"URL: {url}")
    
    comments = parse_youtube_comments(url)
    
    if should_succeed:
        if comments and len(comments) > 0:
            print(f"✅ PASSED: Successfully fetched {len(comments)} comments.")
        else:
            print(f"❌ FAILED: Expected comments but got none.")
    else: # Should fail gracefully
        if not comments:
            print(f"✅ PASSED: Correctly returned an empty list.")
        else:
            print(f"❌ FAILED: Expected an empty list but got {len(comments)} comments.")


def main():
    print("--- Running comprehensive tests for youtube_parser ---")
    run_test("Test Case 1: Happy Path (Valid URL)", URL_SUCCESS, should_succeed=True)
    run_test("Test Case 2: Invalid URL", URL_INVALID, should_succeed=False)
    run_test("Test Case 3: Comments Disabled", URL_COMMENTS_DISABLED, should_succeed=False)
    print("\n--- All tests complete ---")


if __name__ == "__main__":
    main()