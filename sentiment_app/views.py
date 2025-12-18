from django.shortcuts import render, redirect
from .models import Review
from .ml_model import predict_sentiment, train_model_from_feedback

def index(request):
    if request.method == 'POST':
        text = request.POST.get('review')

        if not text:
            return render(request, 'sentiment_app/index.html', {
                'error': 'Please enter a review'
            })

        sentiment, confidence = predict_sentiment(text)

        review = Review.objects.create(
            text=text,
            predicted_sentiment=sentiment
        )

        return render(request, 'sentiment_app/result.html', {
            'review': review,
            'confidence': confidence
        })

    # GET request (home page)
    return render(request, 'sentiment_app/index.html')



from .ml_model import train_model_from_feedback

def feedback(request, review_id):
    review = Review.objects.get(id=review_id)

    if request.method == 'POST':
        review.corrected_sentiment = request.POST.get('corrected_sentiment')
        review.save()

        # 🔥 Retrain using real user feedback
        train_model_from_feedback()

        return redirect('/')

    return render(request, 'sentiment_app/feedback.html', {'review': review})



def dashboard(request):
    total = Review.objects.count()
    positive = Review.objects.filter(predicted_sentiment="Positive").count()
    negative = Review.objects.filter(predicted_sentiment="Negative").count()
    corrected = Review.objects.filter(corrected_sentiment__isnull=False).count()
    not_corrected = total - corrected
    reviews = Review.objects.exclude(corrected_sentiment__isnull=True)

    tp = reviews.filter(predicted_sentiment="Positive", corrected_sentiment="Positive").count()
    tn = reviews.filter(predicted_sentiment="Negative", corrected_sentiment="Negative").count()
    fp = reviews.filter(predicted_sentiment="Positive", corrected_sentiment="Negative").count()
    fn = reviews.filter(predicted_sentiment="Negative", corrected_sentiment="Positive").count()

    accuracy = model_accuracy()
    return render(request, 'sentiment_app/dashboard.html', {
        'total': total,
        'positive': positive,
        'negative': negative,
        'corrected': corrected,
        'not_corrected': not_corrected
    })
def model_accuracy():
    reviews = Review.objects.exclude(corrected_sentiment__isnull=True)

    if reviews.count() == 0:
        return 0

    correct = 0
    for r in reviews:
        if r.predicted_sentiment == r.corrected_sentiment:
            correct += 1

    accuracy = (correct / reviews.count()) * 100
    return round(accuracy, 2)
