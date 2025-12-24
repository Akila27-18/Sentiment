import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .models import Review

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


# --------------------------------------------------
# Initial Training (Seed Model)
# --------------------------------------------------

def train_model():
    """
    Train the initial sentiment model with seed data.
    This should be run once (or automatically if no model exists).
    """

    texts = [
        "good product",
        "excellent service",
        "very happy",
        "love this",
        "bad experience",
        "terrible service",
        "hate this",
        "very bad"
    ]

    labels = [
        "Positive", "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative", "Negative"
    ]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Initial sentiment model trained")


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_sentiment(text):
    """
    Predict sentiment and confidence score
    Returns: (sentiment, confidence %)
    """

    # Auto-train if model doesn't exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        train_model()

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X).max()

    confidence = round(probability * 100, 2)

    return prediction, confidence


# --------------------------------------------------
# Active Learning (Feedback Retraining)
# --------------------------------------------------

def train_model_from_feedback():
    """
    Retrain model using user-corrected sentiment.
    Safely retrains only when at least 2 sentiment classes exist.
    """

    reviews = Review.objects.exclude(corrected_sentiment__isnull=True)

    # ❌ No feedback yet
    if reviews.count() < 2:
        print("Not enough feedback to retrain model")
        return False

    texts = [r.text for r in reviews]
    labels = [r.corrected_sentiment for r in reviews]

    # ❌ Only one sentiment class present
    if len(set(labels)) < 2:
        print("Need at least 2 sentiment classes to retrain model")
        return False

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Model retrained from user feedback")
    return True
