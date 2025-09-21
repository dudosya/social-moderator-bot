# app/parsers/youtube_parser.py

from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
from typing import List, Dict
import sys


def parse_youtube_comments(url: str) -> List[Dict]:
    """
    Fetches ALL comments from a given YouTube URL.
    WARNING: This can be very slow and memory-intensive for popular videos.

    Args:
        url: The full URL of the YouTube video.

    Returns:
        A list of dictionaries, where each dictionary represents a comment.
        Returns an empty list if an error occurs.
    """
    try:
        downloader = YoutubeCommentDownloader()
        print("Fetching ALL comments from URL. This may take a moment...")
        
        # The downloader returns a generator, which is memory-efficient.
        comments_generator = downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT)
        
        # Directly convert the entire generator to a list. NO LIMIT.
        comments = list(comments_generator)
        
        return comments
    
    except Exception as e:
        # For a hackathon, printing the error helps with debugging.
        # In a production app, you might use a proper logger.
        print(f"⚠️  Could not parse comments (this is normal for invalid URLs or disabled comments): {e}", file=sys.stderr)
        return []