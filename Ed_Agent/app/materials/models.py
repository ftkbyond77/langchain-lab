from django.db import models

# Create your models here.
class Material(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="materials/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    text_content = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.title