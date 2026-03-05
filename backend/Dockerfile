# Minimal Python runtime for the FastAPI backend.
FROM python:3.11-slim

WORKDIR /app

# Keep logs unbuffered and avoid writing .pyc files in containers.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python dependencies first to leverage Docker layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source.
COPY app ./app

EXPOSE 8000

# Run FastAPI with Uvicorn.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
