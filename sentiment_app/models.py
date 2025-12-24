from django.db import models


class Review(models.Model):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"

    SENTIMENT_CHOICES = [
        (POSITIVE, "Positive"),
        (NEGATIVE, "Negative"),
    ]

    text = models.TextField()
    predicted_sentiment = models.CharField(
        max_length=20,
        choices=SENTIMENT_CHOICES
    )
    corrected_sentiment = models.CharField(
        max_length=20,
        choices=SENTIMENT_CHOICES,
        null=True,
        blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.text[:30]
