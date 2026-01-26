# analyzer/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('reset/', views.reset_session, name='reset_session'),
    # URL Baru untuk download model
    path('download_model/<int:index>/', views.download_model, name='download_model'),
]