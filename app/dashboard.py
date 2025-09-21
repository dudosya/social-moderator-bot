# app/dashboard.py
import streamlit as st
import pandas as pd
import sys
import os
import time

# --- Boilerplate ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.parsers.youtube_parser import parse_youtube_comments
from app.core.processor import CommentProcessor

# --- Page Config ---
st.set_page_config(page_title="Intelligent Triage Assistant", layout="wide")

# --- Model Caching ---
@st.cache_resource
def get_processor():
    print("--- LOADING MODELS (this will only run once) ---")
    return CommentProcessor()
processor = get_processor()

# --- Header ---
st.title("🤖 Intelligent Triage Assistant")
st.write("Enter a YouTube URL to parse, analyze, and prioritize its comments.")

# --- Session State Initialization ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

# --- Main UI ---
youtube_url = st.text_input("Enter YouTube URL", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")

if st.button("Analyze Comments", type="primary"):
    st.session_state.results_df = None 
    st.session_state.metrics = {}
    
    if youtube_url:
        try:
            start_time = time.time()
            
            with st.spinner("Step 1/2: Parsing comments..."):
                raw_comments = parse_youtube_comments(youtube_url)
            
            if not raw_comments:
                st.error("Could not retrieve comments. Check URL or if comments are enabled.")
            else:
                st.info(f"Successfully parsed {len(raw_comments)} comments.")
                with st.spinner("Step 2/2: Analyzing comments with AI models..."):
                    enriched_comments = processor.process_batch(raw_comments)
                    sorted_comments = sorted(enriched_comments, key=lambda c: c.get('triage_score', 0.0), reverse=True)
                
                end_time = time.time()
                total_time = end_time - start_time

                st.session_state.results_df = pd.DataFrame(sorted_comments)
                st.session_state.metrics = {
                    'total_comments': len(sorted_comments),
                    'high_priority': len([c for c in sorted_comments if c['triage_score'] > 0.6]),
                    'questions': len([c for c in sorted_comments if c['comment_type'] == 'question']),
                    'time_taken': f"{total_time:.2f}s"
                }
                st.success("Analysis Complete!")
                
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.error("Please enter a YouTube URL.")

# --- Display Results ---
if st.session_state.results_df is not None:
    st.subheader("Analysis Results")
    df = st.session_state.results_df
    metrics = st.session_state.metrics
    
    # --- Summary Metrics Display ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Comments", metrics.get('total_comments', 0))
    col2.metric("High Priority Issues", metrics.get('high_priority', 0))
    col3.metric("Questions Found", metrics.get('questions', 0))
    col4.metric("Processing Time", metrics.get('time_taken', '0.00s'))

    st.divider()
    
    # --- THE FINAL, POLISHED LIST ---
    for index, row in df.iterrows():
        st.divider()
        col1, col2, col3 = st.columns([2, 5, 3]) # Adjusted column widths slightly
        
        with col1: # Column for Scores and Labels
            st.caption(f"Triage Score")
            st.progress(row['triage_score'], text=f"{row['triage_score']:.2f}")
            st.info(f"Sentiment: **{row['sentiment_label']}** ({row['sentiment_score']:.2f})")
            st.warning(f"Type: **{row['comment_type']}**")

            # --- NEW: Display Profanity and Spam Flags ---
            if row['has_profanity']:
                st.error("Contains Profanity", icon="🚫")
            if row['is_spam']:
                st.error("Contains Link (Spam)", icon="🔗")
            # --- END NEW ---

        with col2: # Column for Comment Text
            st.caption(f"Comment by **{row['author']}**")
            st.write(row['text'])

        with col3: # Column for Suggested Response
            st.caption("Suggested Response")
            if row['comment_type'] == 'question':
                with st.expander("View RAG-generated answer..."):
                    st.markdown(row['suggested_response'])
            else:
                st.text_area(
                    label=f"response_{row['cid']}",
                    value=row['suggested_response'], 
                    height=150, 
                    disabled=True, 
                    key=f"response_{row['cid']}",
                    label_visibility="collapsed"
                )