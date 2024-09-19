import logging
import asyncio
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from common.models import ReviewInput
from common.database import execute_query

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

# Set up device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sentiment Analysis
sentiment_model_name = "finiteautomata/bertweet-base-sentiment-analysis"
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer, device=0 if torch.cuda.is_available() else -1)

# Emotion Detection
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=0 if torch.cuda.is_available() else -1)

# Named Entity Recognition
ner_pipeline = pipeline("ner", model="dslim/bert-large-NER", device=0 if torch.cuda.is_available() else -1)

# Text Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# Zero-shot Classification for Intent
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)

def convert_rating(rating):
    try:
        return int(float(rating))
    except (ValueError, TypeError):
        return 0

async def analyze_sentiment(text):
    try:
        # Ensure the text is not empty and is a string
        if not text or not isinstance(text, str):
            return "NEUTRAL", 0.5

        # Limit to 512 tokens and remove any problematic characters
        cleaned_text = re.sub(r'[^\w\s]', '', text[:512])
        
        result = sentiment_pipeline(cleaned_text)
        
        if not result:
            return "NEUTRAL", 0.5

        label = result[0]['label']
        score = result[0]['score']
        
        if label == 'POS':
            if score > 0.8:
                return "VERY_POSITIVE", score
            elif score > 0.6:
                return "POSITIVE", score
            else:
                return "SLIGHTLY_POSITIVE", score
        elif label == 'NEG':
            if score > 0.8:
                return "VERY_NEGATIVE", 1 - score
            elif score > 0.6:
                return "NEGATIVE", 1 - score
            else:
                return "SLIGHTLY_NEGATIVE", 1 - score
        else:
            return "NEUTRAL", 0.5
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {str(e)}", exc_info=True)
        return "NEUTRAL", 0.5

async def detect_emotions(text):
    try:
        result = emotion_pipeline(text[:512])[0]
        return result['label'], result['score']
    except Exception as e:
        logger.error(f"Error in detect_emotions: {str(e)}", exc_info=True)
        return "neutral", 0.5

async def extract_aspects(text):
    try:
        doc = nlp(text[:1000])  # Limit to 1000 characters to avoid processing very long texts
        aspects = []
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ == 'NOUN':
                aspect_text = chunk.text.lower()
                sentiment, score = await analyze_sentiment(chunk.sent.text)
                aspects.append({
                    'aspect': aspect_text[:100],  # Truncate to 100 characters
                    'sentiment': sentiment,
                    'score': score
                })
        return aspects
    except Exception as e:
        logger.error(f"Error in extract_aspects: {str(e)}", exc_info=True)
        return []

async def extract_keywords(text):
    try:
        doc = nlp(text[:1000])  # Limit to 1000 characters
        keywords = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        vectorizer = TfidfVectorizer(max_features=10)
        tfidf_matrix = vectorizer.fit_transform([' '.join(keywords)])
        feature_names = vectorizer.get_feature_names_out()
        
        keyword_sentiments = []
        for keyword in feature_names:
            sentiment, _ = await analyze_sentiment(text)
            keyword_sentiments.append((keyword[:100], sentiment))  # Truncate keyword to 100 characters
        return keyword_sentiments
    except Exception as e:
        logger.error(f"Error in extract_keywords: {str(e)}", exc_info=True)
        return []

async def extract_entities(text):
    try:
        results = ner_pipeline(text[:512])  # Limit to 512 tokens
        entities = []
        current_entity = None
        for ent in results:
            if current_entity and ent['entity'].startswith('I-') and ent['entity'][2:] == current_entity[1]:
                current_entity[0] += ' ' + ent['word']
            else:
                if current_entity:
                    entities.append((current_entity[0][:100], current_entity[1][:100]))  # Truncate to 100 characters
                current_entity = [ent['word'], ent['entity'][2:]]
        if current_entity:
            entities.append((current_entity[0][:100], current_entity[1][:100]))  # Truncate to 100 characters
        return entities
    except Exception as e:
        logger.error(f"Error in extract_entities: {str(e)}", exc_info=True)
        return []

async def classify_intent(text):
    try:
        candidate_labels = ["inquiry", "complaint", "feedback", "purchase", "support"]
        result = intent_classifier(text[:512], candidate_labels)  # Limit to 512 tokens
        return [{"label": label[:100], "score": score} for label, score in zip(result['labels'], result['scores'])]
    except Exception as e:
        logger.error(f"Error in classify_intent: {str(e)}", exc_info=True)
        return [{"label": "unknown", "score": 1.0}]

async def extract_key_points(text):
    try:
        summary = summarizer(text[:1024], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        sentences = nlp(summary).sents
        positive_points = []
        negative_points = []
        for sentence in sentences:
            sentiment, _ = await analyze_sentiment(sentence.text)
            if sentiment in ["POSITIVE", "VERY_POSITIVE", "SLIGHTLY_POSITIVE"]:
                positive_points.append(sentence.text[:100])  # Truncate to 100 characters
            elif sentiment in ["NEGATIVE", "VERY_NEGATIVE", "SLIGHTLY_NEGATIVE"]:
                negative_points.append(sentence.text[:100])  # Truncate to 100 characters
        return positive_points, negative_points
    except Exception as e:
        logger.error(f"Error in extract_key_points: {str(e)}", exc_info=True)
        return [], []

async def analyze_single_review(review: ReviewInput):
    try:
        sentiment, sentiment_score = await analyze_sentiment(review.text)
        emotion, emotion_score = await detect_emotions(review.text)
        aspects = await extract_aspects(review.text)
        keywords = await extract_keywords(review.text)
        entities = await extract_entities(review.text)
        intents = await classify_intent(review.text)
        positive_points, negative_points = await extract_key_points(review.text)

        # Convert rating to integer
        converted_rating = convert_rating(review.rating)

        result_id = await insert_analysis_result(review.review_id, sentiment, sentiment_score, 
                                                 converted_rating, emotion, emotion_score)

        await asyncio.gather(
            *[insert_aspect(result_id, aspect['aspect'], aspect['sentiment'], aspect['score']) for aspect in aspects],
            *[insert_keyword(result_id, keyword, sentiment) for keyword, sentiment in keywords],
            *[insert_entity(result_id, entity, entity_type) for entity, entity_type in entities],
            *[insert_intent(result_id, intent['label'], intent['score']) for intent in intents],
            *[insert_key_point(result_id, point, True) for point in positive_points],
            *[insert_key_point(result_id, point, False) for point in negative_points]
        )

        return {
            "review_id": review.review_id,
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "rating": converted_rating,
            "emotion": emotion,
            "emotion_score": emotion_score,
            "aspects": aspects,
            "keywords": keywords,
            "entities": entities,
            "intents": intents,
            "positive_points": positive_points,
            "negative_points": negative_points
        }
    except Exception as e:
        logger.error(f"Error analyzing review: {str(e)}", exc_info=True)
        return None

async def insert_analysis_result(review_id, sentiment, sentiment_score, rating, emotion, emotion_score):
    query = """
    INSERT INTO analysis_results (review_id, sentiment, sentiment_score, rating, emotion, emotion_score)
    VALUES ($1, $2, $3, $4, $5, $6)
    RETURNING result_id
    """
    result = await execute_query(query, review_id, sentiment, sentiment_score, rating, emotion, emotion_score)
    return result[0]['result_id']

async def insert_aspect(result_id, aspect, sentiment, score):
    query = """
    INSERT INTO aspects (result_id, aspect, sentiment, score)
    VALUES ($1, $2, $3, $4)
    """
    await execute_query(query, result_id, aspect[:100], sentiment, score)

async def insert_keyword(result_id, keyword, sentiment):
    query = """
    INSERT INTO keywords (result_id, keyword, sentiment)
    VALUES ($1, $2, $3)
    """
    await execute_query(query, result_id, keyword[:100], sentiment)

async def insert_entity(result_id, entity, entity_type):
    query = """
    INSERT INTO entities (result_id, entity, entity_type)
    VALUES ($1, $2, $3)
    """
    await execute_query(query, result_id, entity[:100], entity_type[:100])

async def insert_intent(result_id, intent, score):
    query = """
    INSERT INTO intents (result_id, intent, score)
    VALUES ($1, $2, $3)
    """
    await execute_query(query, result_id, intent[:100], score)

async def insert_key_point(result_id, point, is_positive):
    query = """
    INSERT INTO key_points (result_id, point_text, is_positive)
    VALUES ($1, $2, $3)
    """
    await execute_query(query, result_id, point[:100], is_positive)