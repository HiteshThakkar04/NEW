from django.urls import path
from .views import index, api_ecg_data

urlpatterns = [
    path('', index, name='report_index'),
    path('api/', api_ecg_data, name='api_ecg_data'),

]