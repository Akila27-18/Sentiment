import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .models import Review

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


def train_model():
    """
    Initial training with seed data
    """
    texts = [
        "good product", "excellent service", "very happy", "love this",
        "bad experience", "terrible service", "hate this", "very bad"
    ]
    labels = ["Positive", "Positive", "Positive", "Positive",
              "Negative", "Negative", "Negative", "Negative"]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)


def predict_sentiment(text):
    """
    Predict sentiment + confidence
    """
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X).max()

    confidence = round(probability * 100, 2)

    return prediction, confidence


def train_model_from_feedback():
    """
    Retrain model using user-corrected sentiment
    """
    reviews = Review.objects.exclude(corrected_sentiment__isnull=True)

    if not reviews.exists():
        return False

    texts = [r.text for r in reviews]
    labels = [r.corrected_sentiment for r in reviews]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return True
