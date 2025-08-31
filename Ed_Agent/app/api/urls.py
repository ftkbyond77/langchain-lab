from django.urls import path
from .views import MaterialUploadView, StudyPlanView, QuizView, TestDashboardView

urlpatterns = [
    path('upload/', MaterialUploadView.as_view(), name='material-upload'),
    path('study-plan/', StudyPlanView.as_view(), name='study-plan'),
    path('quiz/<int:material_id>/', QuizView.as_view(), name='quiz'),
    path('dashboard/', TestDashboardView.as_view(), name='test-dashboard'),
]
