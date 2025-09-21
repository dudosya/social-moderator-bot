# scripts/test_nlp.py
# The final, integrated Litmus Test script for our complete NLP pipeline.

import os
import pandas as pd
import fasttext
from transformers import pipeline

# --- Configuration ---
# 1. Define paths and model names for all three of our validated models.
LANG_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'lid.176.bin')
RU_SENTIMENT_MODEL = 'sismetanin/rubert-ru-sentiment-rusentiment'
KZ_SENTIMENT_MODEL = 'issai/rembert-sentiment-analysis-polarity-classification-kazakh'

# --- Sample Data ---
sample_comments = [
    # Russian examples
    "Отличный товар, всем рекомендую! Качество на высоте.", # ru, positive
    "Ужасный сервис, никогда больше не вернусь. Очень разочарован.", # ru, negative
    "Подскажите, пожалуйста, сколько будет стоить доставка в Астану?", # ru, neutral
    # Kazakh examples
    "Керемет! Маған өте ұнады, рахмет сізге.", # kk, positive
    "Бұл өте нашар, сапасы төмен. Ақшаны босқа жұмсадым.", # kk, negative
    "Осы кітап өте қызық сияқты.", # kk, positive
    # Edge cases
    "лол", # other lang, unknown sentiment
    "This is a test comment in English.", # other lang, unknown sentiment
]

# --- Main Test Logic ---
print("--- Starting Full Pipeline Logic Test ---")

# Step 1: Load all models into memory once.
try:
    print("Loading language detection model...")
    lang_model = fasttext.load_model(LANG_MODEL_PATH)

    print("Loading Russian sentiment model...")
    ru_sentiment_pipeline = pipeline("sentiment-analysis", model=RU_SENTIMENT_MODEL)

    print("Loading Kazakh sentiment model...")
    kz_sentiment_pipeline = pipeline("text-classification", model=KZ_SENTIMENT_MODEL)
    
    print("\n--- All models loaded successfully ---\n")
except Exception as e:
    print(f"FATAL ERROR: A model failed to load. Details: {e}")
    exit()
    
# Step 2: Process each comment through the pipeline.
results = []
for i, comment in enumerate(sample_comments):
    cleaned_comment = comment.replace("\n", " ")
    
    # ---> Stage 1: Language Detection
    lang_predictions = lang_model.predict(cleaned_comment, k=1)
    lang_code = lang_predictions[0][0].replace('__label__', '')
    
    sentiment_label = "unknown" # Default sentiment
    sentiment_score = 0.0
    
    # ---> Stage 2: Routing and Sentiment Analysis
    if lang_code == 'ru':
        prediction = ru_sentiment_pipeline(cleaned_comment)[0]
        # Get the human-readable label from the model's config
        label_id = int(prediction['label'].split('_')[1])
        sentiment_label = ru_sentiment_pipeline.model.config.id2label[label_id]
        sentiment_score = prediction['score']
    elif lang_code == 'kk':
        prediction = kz_sentiment_pipeline(cleaned_comment)[0]
        sentiment_label = prediction['label']
        sentiment_score = prediction['score']
    # If lang_code is neither 'ru' nor 'kk', sentiment remains 'unknown'.

    results.append({
        'comment': comment,
        'detected_lang': lang_code,
        'sentiment': sentiment_label,
        'score': f"{sentiment_score:.2f}"
    })

# Step 3: Display the final, comprehensive report.
print("\n--- Test Complete. Final Report ---")
df = pd.DataFrame(results)

# A little bit of post-processing for readability on the Russian model's output
# This is a bit of a hack for the test script, we'll handle it more elegantly later.
# This assumes the labels are 'negative', 'neutral', 'positive' which might not be true.
# Let's adjust this based on the real labels after we run it.
label_remap = {
    'LABEL_0': 'negative',
    'LABEL_1': 'neutral',
    'LABEL_2': 'positive',
    'LABEL_3': 'speech', # from our previous test
    'LABEL_4': 'skip'    # from our previous test
}
df['sentiment'] = df['sentiment'].replace(label_remap)

print(df)