# app/main.py
"""
Main entry point for the Intelligent Triage Assistant application.

This script orchestrates the entire process from command-line input to final report.
It uses the components built in other parts of the application to perform the following steps:
1. Parse command-line arguments to get the target YouTube URL.
2. Call the YouTube parser to fetch comments.
3. Instantiate the CommentProcessor to analyze the comments.
4. Sort the enriched comments by the calculated triage_score.
5. Save the final, prioritized list to a CSV file in the 'reports/' directory.

To run this script:
    python -m app.main --url "YOUR_YOUTUBE_URL_HERE"
"""
import argparse
import sys
import os
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import pandas as pd
import time # --- Import the time module ---

# --- Boilerplate to make sibling packages accessible ---
# Ensures that the script can find and import modules from the `app` package
# regardless of where the script is executed from.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# ---------------------------------------------------

from app.parsers.youtube_parser import parse_youtube_comments
from app.core.processor import CommentProcessor
from app.utils.file_handler import save_to_csv

def generate_filename(url: str) -> str:
    """
    Creates a descriptive and unique filename from a YouTube URL.
    
    Extracts the video ID for clarity and appends a timestamp to prevent
    overwriting previous reports for the same video.

    :param url: The full YouTube URL.
    :return: A safe, descriptive string for use as a filename (e.g., "report_aZE5KPgiHIE_20240921_143000.csv").
    """
    try:
        parsed_url = urlparse(url)
        if "youtube.com" in parsed_url.netloc:
            video_id = parse_qs(parsed_url.query).get('v', [None])[0]
        elif "youtu.be" in parsed_url.netloc:
            video_id = parsed_url.path.strip('/')
        else:
            video_id = "unknown_video"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"report_{video_id}_{timestamp}.csv"
    except Exception:
        # Fallback for any unexpected URL parsing errors.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"report_generic_{timestamp}.csv"

def main():
    """Main function to orchestrate the entire application workflow."""
    
    # --- Start the timer ---
    start_time = time.time()
    
    # 1. Setup Argument Parser for a clean command-line interface.
    parser = argparse.ArgumentParser(description="Intelligent Triage Assistant for Social Media Moderation")
    parser.add_argument("--url", type=str, required=True, help="The full URL of the YouTube video to process.")
    args = parser.parse_args()
    
    video_url = args.url
    print(f"--- Starting Intelligent Triage Assistant for URL: {video_url} ---")
    
    # 2. Parse Comments (Data Ingestion)
    print("\n[Step 1/4] Parsing comments...")
    raw_comments = parse_youtube_comments(video_url)
    if not raw_comments:
        print("✅ No comments found or an error occurred during parsing. Exiting.")
        return
    print(f"✅ Successfully parsed {len(raw_comments)} comments.")
    
    # 3. Process Comments (AI Analysis)
    print("\n[Step 2/4] Initializing and running the Comment Processor...")
    try:
        processor = CommentProcessor()
        enriched_comments = processor.process_batch(raw_comments)
        print("✅ Batch processing complete.")
    except Exception as e:
        print(f"🚨🚨 FAILED during comment processing: {e}")
        return
        
    # 4. Sort by Triage Score (Core Value Proposition)
    print("\n[Step 3/4] Sorting comments by triage score (descending)...")
    # This is the key step that transforms a chaotic list into an actionable one.
    # The lambda function safely gets the triage_score, defaulting to 0.0 if it's missing.
    sorted_comments = sorted(enriched_comments, key=lambda c: c.get('triage_score', 0.0), reverse=True)
    print("✅ Sorting complete.")
    
    # 5. Save to CSV (Output Generation)
    print("\n[Step 4/4] Saving prioritized report to CSV...")
    output_directory = "reports"
    output_filename = generate_filename(video_url)
    
    save_to_csv(sorted_comments, output_directory, output_filename)
    
    # --- CALCULATE AND DISPLAY STATISTICS ---
    print("\n--- Triage Report Summary ---")

    # To calculate stats, we first create the pandas DataFrame
    # This is the same DataFrame we use for saving the CSV
    df = pd.DataFrame(sorted_comments)

    # 1. Language Statistics
    print("\n[+] Comments by Language:")
    lang_counts = df['language'].value_counts()
    for lang, count in lang_counts.items():
        print(f"  - {lang}: {count}")

    # 2. Sentiment Statistics
    print("\n[+] Comments by Sentiment:")
    # We now directly use our clean 'sentiment_label' column.
    # We also handle comments with no sentiment (e.g., in English) by filling them with 'N/A'.
    sentiment_counts = df['sentiment_label'].fillna('N/A').value_counts()
    for label, count in sentiment_counts.items():
        print(f"  - {label}: {count}")

    # 3. Comment Type Statistics
    print("\n[+] Comments by Type:")
    type_counts = df['comment_type'].value_counts()
    for comment_type, count in type_counts.items():
        print(f"  - {comment_type}: {count}")
    
    print("-" * 30)
    # ---------------------------------------------------------
    
    # --- End the timer and display the result ---
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n[+] Total processing time: {total_time:.2f} seconds")
    
    print("\n--- Intelligent Triage Assistant finished successfully! ---")
    print(f"Find your report in the '{output_directory}/' folder.")


if __name__ == "__main__":
    main()