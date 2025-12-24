from django.conf import settings
from django.urls import path
from . import views

app_name = 'sentiment_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('feedback/<int:review_id>/', views.feedback, name='feedback'),
    path('dashboard/', views.dashboard, name='dashboard'),
]

urlpatterns += [path(settings.STATIC_URL.lstrip('/'), views.serve_static, name='static')]