from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from django.shortcuts import get_object_or_404
from django.views import View
from django.shortcuts import render

from app.materials.models import Material
from app.materials.utils import extract_text_from_file
from app.agents.embedding_utils import create_embedding
from app.planner.agent import generate_study_plan
from app.quiz.agent import generate_quiz


# -------------------------------
# HTML Dashboard View
# -------------------------------
class TestDashboardView(View):
    def get(self, request):
        return render(request, "api/test_dashboard.html")


# -------------------------------
# API Views
# -------------------------------
class MaterialUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        file_obj = request.data.get('file', None)
        if not file_obj:
            return Response({"error": "No file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        title = request.data.get('title', file_obj.name)

        try:
            # Create Material
            material = Material.objects.create(title=title, file=file_obj)
            # Extract text
            material.text_content = extract_text_from_file(material.file.path)
            material.save()
            # Create embedding
            create_embedding(material.id, material.text_content)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({
            "id": material.id,
            "title": material.title,
            "text_content_preview": material.text_content[:200]
        }, status=status.HTTP_201_CREATED)


class MaterialListView(APIView):
    """
    Return list of all materials for dashboard select box
    """
    def get(self, request):
        materials = Material.objects.all()
        data = [{"id": m.id, "title": m.title} for m in materials]
        return Response(data)


class StudyPlanView(APIView):
    """
    Generate study plan based on all uploaded materials
    """
    def get(self, request):
        materials = Material.objects.all()
        if not materials:
            return Response({"error": "No materials found."}, status=status.HTTP_404_NOT_FOUND)
        try:
            plan = generate_study_plan(materials)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"study_plan": plan})


class QuizView(APIView):
    """
    Generate quiz for a specific material by ID
    """
    def get(self, request, material_id):
        material = get_object_or_404(Material, id=material_id)
        if not material.text_content:
            return Response({"error": "Material has no text content."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            quiz_text = generate_quiz(material.text_content)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"quiz": quiz_text})
