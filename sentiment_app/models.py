from django.db import models

class Review(models.Model):
    text = models.TextField()
    predicted_sentiment = models.CharField(max_length=20)
    corrected_sentiment = models.CharField(
        max_length=20,
        blank=True,
        null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.text[:30]
