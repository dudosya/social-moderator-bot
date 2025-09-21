
# Intelligent Triage Assistant for Social Media

This project, developed for the ALTEL/TELE2 Hackathon, is an AI-powered tool designed to help social media managers triage and moderate YouTube comment sections efficiently. It transforms chaotic comment feeds into a prioritized, actionable list, allowing teams to address the most critical issues first.

The application features a command-line interface for generating CSV reports and a powerful interactive web dashboard for real-time analysis and visualization.

## ✨ Core Features

Our solution is built on three pillars: **Prioritize, Analyze, and Assist.**

### 1. **Triage Score Prioritization**
Instead of just classifying comments, our core innovation is the `triage_score`. This score (from 0.0 to 1.0) synthesizes multiple data points (negative sentiment, profanity, spam) into a single, actionable metric of urgency. All outputs are automatically sorted by this score, ensuring moderators always see the most critical comments first.

### 2. **"Specialist Router" AI Architecture**
To achieve the highest accuracy for both Russian and Kazakh languages, we rejected a "one-size-fits-all" approach. Our system uses a high-speed language detector to route each comment to a dedicated, fine-tuned sentiment analysis model. This results in faster, more accurate, and culturally aware analysis.

### 3. **RAG-Powered Question Answering**
For comments identified as questions, the assistant employs a state-of-the-art Retrieval-Augmented Generation (RAG) system (`intfloat/multilingual-e5-large`). It consults a knowledge base of provided company documents to generate factually correct and contextually relevant answers, turning support queries into instantly resolved tickets.

## 🚀 Getting Started

### Prerequisites
*   Python 3.11
*   A virtual environment is highly recommended.

### 1. Setup

First, clone the repository and navigate into the project directory.

```bash
git clone <your-repo-url>
cd social-moderator-bot
```

Next, create and activate a virtual environment:

```bash
# For Windows
py -3.11 -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

Install all required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Build the Knowledge Base

Before running the main application, you must build the vector knowledge base from the provided documents. This only needs to be done once.

```bash
python -m scripts.build_knowledge_base
```
This will create `company_kb.index` and `chunks.json` inside the `models/` directory.

##  Usage

The application has two primary modes of operation: an interactive web dashboard (recommended) and a command-line tool for generating CSV reports.

### Option A: Interactive Web Dashboard (Recommended)

This is the best way to visualize and interact with the analysis results.

To launch the dashboard, run the following command from the project root:

```bash
streamlit run app/dashboard.py
```

Your web browser will automatically open a new tab with the application. Simply paste a YouTube URL and click "Analyze Comments".

### Option B: Command-Line CSV Generation

To generate a prioritized CSV report directly, use the `main.py` script.

```bash
python -m app.main --url "YOUR_YOUTUBE_URL_HERE"
```

**Example:**
```bash
python -m app.main --url "https://www.youtube.com/watch?v=M_Xb8cfRm_w"
```

The script will process the comments and save a sorted CSV file inside the `reports/` directory.

## Project Structure
```
social-moderator-bot/
├── app/                  # Main application source code
│   ├── core/
│   │   └── processor.py  # The core AI "brain" of the application
│   ├── parsers/
│   │   └── youtube_parser.py # Logic for scraping comments
│   ├── utils/
│   │   └── file_handler.py   # Utility for saving CSV files
│   ├── dashboard.py      # The Streamlit web application
│   └── main.py           # The command-line interface entry point
├── data/                 # Raw data for building the knowledge base
├── models/               # Pre-trained models and generated knowledge base index
├── reports/              # Output directory for CSV reports
├── scripts/              # Standalone scripts for tasks like building the KB
│   └── build_knowledge_base.py
└── requirements.txt      # Project dependencies
```
````