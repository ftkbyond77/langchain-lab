FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy project code
COPY . /app

EXPOSE 8000

# Run server
CMD ["gunicorn", "study_agent.wsgi:application", "--bind", "0.0.0.0:8000"]
