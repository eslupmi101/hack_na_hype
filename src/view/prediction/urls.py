from django.urls import path

from . import views

app_name = 'prediction'

urlpatterns = [
    path('', views.index, name='index'),
    path('download/', views.download_result_file, name='download_result_file'),
]
