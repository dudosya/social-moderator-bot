# app/utils/file_handler.py

import pandas as pd
from typing import List, Dict, Any
import os

def save_to_csv(data: List[Dict[str, Any]], output_path: str, filename: str):
    """
    Saves a list of dictionaries to a CSV file using pandas.

    Args:
        data: The list of dictionaries to save.
        output_path: The directory where the file will be saved (e.g., 'reports').
        filename: The name of the file (e.g., 'report_12345.csv').
    """
    if not data:
        print("⚠️ Warning: No data provided to save_to_csv. Skipping file creation.")
        return

    # Ensure the output directory exists
    try:
        os.makedirs(output_path, exist_ok=True)
        print(f"Directory '{output_path}' is ready.")
    except OSError as e:
        print(f"🚨 Error creating directory {output_path}: {e}")
        return

    full_path = os.path.join(output_path, filename)
    
    print(f"Attempting to save {len(data)} rows to {full_path}...")

    try:
        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(data)
        
        # --- MODIFIED: Updated column order for the new data structure ---
        # Define the order of columns for the output CSV for better readability
        column_order = [
            'triage_score',
            'suggested_response',
            'text',
            'language',
            'sentiment_label',    # NEW
            'sentiment_score',    # NEW
            'comment_type',       # NEW
            'has_profanity',
            'is_spam',
            'cid',
            'author',
            'time',
        ]
        # --- End of MODIFICATION ---
        
        # Reorder DataFrame columns, only including those that exist
        # This prevents errors if a comment is missing an expected key
        df_reordered = df[[col for col in column_order if col in df.columns]]
        
        # Save the DataFrame to a CSV file
        df_reordered.to_csv(full_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ Successfully saved report to {full_path}")
    
    except Exception as e:
        print(f"🚨🚨 FAILED to save data to CSV: {e}")