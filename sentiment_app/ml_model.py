import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


def train_model():
    texts = [
        "good product", "excellent service", "very happy", "love this",
        "bad experience", "terrible service", "hate this", "very bad"
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)


def predict_sentiment(text):
    # Load model safely every time
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector).max()

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = round(probability * 100, 2)

    return sentiment, confidence
from .models import Review
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")


def train_model_from_feedback():
    """
    Retrain model using user feedback
    """

    reviews = Review.objects.exclude(actual_sentiment__isnull=True)

    if not reviews.exists():
        return False

    texts = [r.text for r in reviews]
    labels = [r.actual_sentiment for r in reviews]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return True
