# Social Media Comment Triage Pipeline

A Python-based application for parsing, analyzing, and prioritizing YouTube comments. This project was developed for Datathon 2025.

The system processes comments via a multi-stage pipeline and presents the results in an interactive web-based dashboard or as a CSV report.

## System Components

The pipeline consists of three main analytical components that process each comment:

### 1. Multilingual Sentiment Analysis
-   **Function:** Classifies comment tonality (Positive, Negative, Neutral) for Russian and Kazakh languages.
-   **Implementation:** A "Specialist Router" pattern is used. First, the `fastText lid.176` model identifies the language. Then, the comment is routed to a language-specific, fine-tuned transformer model (`sismetanin/rubert-ru-sentiment` for Russian, `issai/rembert-sentiment-analysis-kazakh` for Kazakh) for more accurate classification.

### 2. Knowledge-Based Question Answering
-   **Function:** Provides context-aware answers to user questions based on a provided document set.
-   **Implementation:** A Retrieval-Augmented Generation (RAG) system was built. Documents are chunked, vectorized using the `intfloat/multilingual-e5-large` embedding model, and stored in a `FAISS` index. The `e5-large` model was selected for its effectiveness in asymmetric search tasks. Retrieval relevance is improved by adding `"query: "` and `"passage: "` prefixes to text before embedding, as required by the model.

### 3. Triage Score Calculation
-   **Function:** Assigns a numerical score from 0.0 to 1.0 to each comment to quantify its urgency for moderation.
-   **Implementation:** The score is calculated based on a weighted combination of the sentiment analysis output and rule-based flags (e.g., presence of profanity, URL links). The final output is sorted by this score in descending order.

## Setup and Usage

### Prerequisites
-   Python 3.11
-   Git

### 1. Setup Environment
Clone the repository and set up a Python virtual environment.
```bash
git clone <your-repo-url>
cd social-moderator-bot

# Create and activate venv
# Windows: py -3.11 -m venv venv && .\venv\Scripts\activate
# macOS/Linux: python3.11 -m venv venv && source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Large Model File (Manual Step)
The main language detection model (`lid.176.bin`, ~125MB) is managed outside of this Git repository.

-   **Download the file:** **[Click here to download lid.176.bin](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin)**
-   **Place the file:** Move the downloaded `lid.176.bin` into the `models/` directory.

### 4. Build Knowledge Base
This one-time command processes the documents in the `data/` folder and creates the `FAISS` vector index required for the RAG component.
```bash
python -m scripts.build_knowledge_base
```

## Running the Application

The application can be run in two modes.

### Option A: Interactive Dashboard (Recommended)
Launches a web-based UI built with Streamlit.
```bash
streamlit run app/dashboard.py
```
A browser tab will open. Enter a YouTube URL to begin analysis.

### Option B: Command-Line Report Generation
Generates a CSV report in the `reports/` directory.
```bash
python -m app.main --url "YOUR_YOUTUBE_URL_HERE"
```

## Project Structure
```
social-moderator-bot/
├── app/                  # Main application source code
│   ├── core/
│   │   └── processor.py  # Core analysis pipeline logic
│   ├── parsers/
│   │   └── youtube_parser.py # YouTube comment scraping
│   ├── utils/
│   │   └── file_handler.py   # CSV export utility
│   ├── dashboard.py      # Streamlit UI application
│   └── main.py           # Command-line entry point
├── data/                 # Source documents for the RAG knowledge base
├── models/               # Directory for models and the FAISS index
├── reports/              # Output directory for CSV reports
├── scripts/              # Helper scripts
│   └── build_knowledge_base.py
├── .gitignore
└── requirements.txt
```