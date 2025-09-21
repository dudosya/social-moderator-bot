# app/core/processor.py
"""
Core processing module for the Intelligent Triage Assistant.

This module defines the CommentProcessor class, which is the main "brain" of the
application. It handles loading all necessary AI models, orchestrating the
analysis of each comment (language detection, sentiment, rule-based checks),
and calculating the final triage score.
"""

import os
import re
import json
import faiss
import fasttext
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import pipeline, logging
from sentence_transformers import SentenceTransformer

# Suppress verbose logging from transformers to keep the output clean.
logging.set_verbosity_error()

# --- Constants for Model and File Paths ---
# Using constants makes the code cleaner and easier to update.
FASTTEXT_MODEL_PATH = "models/lid.176.bin"
RUSSIAN_SENTIMENT_MODEL = "sismetanin/rubert-ru-sentiment-rusentiment"
KAZAKH_SENTIMENT_MODEL = "issai/rembert-sentiment-analysis-polarity-classification-kazakh"
EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'
FAISS_INDEX_PATH = os.path.join('models', 'company_kb.index')
CHUNKS_FILE_PATH = os.path.join('models', 'chunks.json')

# --- Rule-Based Moderation Keywords ---
PROFANITY_KEYWORDS = ['блин', 'сука', 'дерьмо', 'хрен']

# --- FINALIZED: Human-readable sentiment label mapping ---
# This dictionary translates cryptic model outputs into clean, demo-ready labels.
SENTIMENT_LABEL_MAP = {
    # Russian Model (sismetanin/...) - 5 labels
    'LABEL_0': 'Negative',      # Speech (e.g. insults)
    'LABEL_1': 'Neutral',       # Neutral
    'LABEL_2': 'Positive',      # Positive
    'LABEL_3': 'Neutral',       # Skip (often neutral questions/statements)
    'LABEL_4': 'Positive',      # Speech (e.g. gratitude)
    
    # Kazakh Model (issai/...) - 3 labels
    'NEGATIVE': 'Negative',
    'POSITIVE': 'Positive',
    'NEUTRAL': 'Neutral',
    
    # Fallbacks for other potential models
    'NEG': 'Negative',
    'POS': 'Positive',
    'NEU': 'Neutral'
}

class CommentProcessor:
    """
    A class to process batches of social media comments, enriching them with AI analysis.

    This class encapsulates all the logic for language detection, sentiment analysis
    (using a "Specialist Router" pattern), profanity/spam checks, and triage scoring.
    It also integrates a Retrieval-Augmented Generation (RAG) system to answer questions.
    """
    def __init__(self):
        """
        Initializes the CommentProcessor and loads all required AI models into memory.
        
        This "load-once" approach is critical for performance, as model loading is a
        time-consuming operation.
        """
        print("Initializing CommentProcessor...")
        try:
            print(f"Loading fasttext model from: {FASTTEXT_MODEL_PATH}")
            self.lang_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
            
            print(f"Loading Russian sentiment model: {RUSSIAN_SENTIMENT_MODEL}")
            self.ru_sentiment_model = pipeline("sentiment-analysis", model=RUSSIAN_SENTIMENT_MODEL)
            
            print(f"Loading Kazakh sentiment model: {KAZAKH_SENTIMENT_MODEL}")
            self.kz_sentiment_model = pipeline("sentiment-analysis", model=KAZAKH_SENTIMENT_MODEL)

            print(f"Loading RAG embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

            print(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
            self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)

            print(f"Loading text chunks from: {CHUNKS_FILE_PATH}")
            with open(CHUNKS_FILE_PATH, 'r', encoding='utf-8') as f:
                self.text_chunks = json.load(f)

        except Exception as e:
            print(f"🚨 Error loading models: {e}")
            raise
        print("✅ CommentProcessor initialized and all models are ready.")

    def _answer_question_with_rag(self, question: str, lang: str = 'ru') -> str:
        """
        Answers a user's question using the RAG knowledge base.
        """
        prefixed_question = "query: " + question
        
        question_embedding = self.embedding_model.encode([prefixed_question])
        
        k = 1
        distances, indices = self.faiss_index.search(question_embedding.astype('float32'), k)
        
        if not indices.size:
            return None 

        retrieved_chunk = self.text_chunks[indices[0][0]]
        
        templates = {
            'ru': "Похоже, вы задали вопрос. Вот информация, которая может быть полезна:\n\n             ---\n             {context}\n             ---",
            'kk': "Сіз сұрақ қойған сияқтысыз. Міне, пайдалы болуы мүмкін ақпарат:\n\n             ---\n             {context}\n             ---"
        }
        response_template = templates.get(lang, templates['ru'])
        return response_template.format(context=retrieved_chunk)

    def _generate_response(self, comment: dict) -> str:
        """
        Generates a suggested moderator response based on the comment's analysis.
        """
        lang = comment.get('language', 'ru')
        text = comment.get('text', '')
        
        if comment.get('comment_type') == 'question':
            rag_answer = self._answer_question_with_rag(text, lang)
            if rag_answer:
                return rag_answer
        
        sentiment_label = comment.get('sentiment_label')
        
        templates = {
            'ru': {
                'profanity': "Мы удалили ваш комментарий за нарушение правил сообщества. Пожалуйста, будьте вежливы.",
                'highly_negative': "Сожалеем, что у вас сложилось такое впечатление. Мы передали ваш отзыв команде для улучшения нашего сервиса.",
                'positive': "Спасибо за ваш положительный отзыв! Мы рады, что вам понравилось.",
                'default': "Благодарим за ваш комментарий."
            },'kk': {
                'profanity': "Қауымдастық ережелерін бұзғаныңыз үшін сіздің пікіріңіз жойылды. Сыпайы болыңыз.",
                'highly_negative': "Сізде осындай әсер қалғанына өкінеміз. Қызметімізді жақсарту үшін пікіріңізді командаға жолдадық.",
                'positive': "Оң пікіріңіз үшін рахмет! Сізге ұнағанына қуаныштымыз.",
                'default': "Пікіріңіз үшін рахмет."
            }
        }
        lang_templates = templates.get(lang, templates['ru'])
        
        if comment.get('has_profanity'): return lang_templates['profanity']
        if sentiment_label == 'Negative': return lang_templates['highly_negative']
        if sentiment_label == 'Positive': return lang_templates['positive']
        return lang_templates['default']

    def _calculate_triage_score(self, comment: Dict) -> float:
        """
        Calculates a triage score to quantify the urgency of a comment.
        """
        score = 0.0
        sentiment_label = comment.get('sentiment_label')
        sentiment_score = comment.get('sentiment_score', 0.0)
        
        if sentiment_label == 'Negative':
            score = 0.5 + (sentiment_score * 0.2)
        elif sentiment_label == 'Neutral':
            score = 0.2
        elif sentiment_label == 'Positive':
            score = 0.0

        if comment.get('has_profanity'):
            score = max(score, 0.8) 
        
        if comment.get('is_spam'):
            score += 0.5
        
        if comment.get('comment_type') == 'question':
            score += 0.15

        return min(score, 1.0)
    
    def _classify_comment_type(self, comment: dict) -> str:
        """
        Classifies the comment into a specific type based on keywords.
        """
        text = comment.get('text', '').lower()
        
        padded_text = f" {text} "

        complaint_keywords = ["жалоба", "проблема", "не работает", "ужасно", "плохо", "недоволен", "отвратительно", "какого черта", "безобразие", "неудовлетворительно"]
        gratitude_keywords = ["спасибо", "благодарю", "отлично", "супер", "полезно", "лучший", "круто", "прекрасно", "молодец", "уважаю", "добро", "благодарность", "хорошо"]
        question_keywords = ["почему", "когда", "как", "где", "что", "кто", "сколько", "какой", "ли?", "можно ли", "вы можете", "?"]
        feedback_keywords = ["отзыв", "мнение", "комментарий", "предложение", "хотелось бы", "было бы хорошо"]

        if any(f" {keyword} " in padded_text for keyword in complaint_keywords):
            return "complaint"
        if any(f" {keyword} " in padded_text for keyword in gratitude_keywords):
            return "gratitude"
        if '?' in text or any(f" {keyword} " in padded_text for keyword in question_keywords):
            return "question"
        if any(f" {keyword} " in padded_text for keyword in feedback_keywords):
            return "feedback"

        return "other"
            
    def process_batch(self, comments: List[Dict]) -> List[Dict]:
        """
        Processes a list of raw comment dictionaries.
        """
        enriched_comments = []
        for raw_comment in tqdm(comments, desc="Processing Comments"):
            comment = raw_comment.copy()
            
            text = comment.get("text", "").replace("\n", " ")
            if not text.strip():
                continue

            comment['language'] = self.lang_model.predict(text, k=1)[0][0].replace('__label__', '')
            
            comment['sentiment_label'] = None
            comment['sentiment_score'] = None
            
            try:
                sentiment_result = None
                truncated_text = text[:512]
                if comment['language'] == 'ru': 
                    sentiment_result = self.ru_sentiment_model(truncated_text)
                elif comment['language'] == 'kk': 
                    sentiment_result = self.kz_sentiment_model(truncated_text)
                
                if sentiment_result:
                    raw_label = sentiment_result[0]['label']
                    comment['sentiment_label'] = SENTIMENT_LABEL_MAP.get(raw_label, 'Unknown')
                    comment['sentiment_score'] = round(sentiment_result[0]['score'], 4)

            except Exception as e:
                print(f"Warning: Could not process sentiment for text: '{text[:30]}...'. Error: {e}")
                pass

            lower_text = text.lower()
            comment['is_spam'] = 'http' in lower_text
            comment['has_profanity'] = any(word in lower_text for word in PROFANITY_KEYWORDS)

            comment['comment_type'] = self._classify_comment_type(comment)
            
            comment['triage_score'] = self._calculate_triage_score(comment)
            comment['suggested_response'] = self._generate_response(comment)
            
            comment.pop('sentiment', None) 
            
            enriched_comments.append(comment)
            
        return enriched_comments