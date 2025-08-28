FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# copy ทั้งโปรเจกต์ (รวม manage.py, study_ai/, app/)
COPY . /app

EXPOSE 8000

CMD ["gunicorn", "study_ai.wsgi:application", "--bind", "0.0.0.0:8000"]
