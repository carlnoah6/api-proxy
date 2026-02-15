FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (read-only in container)
COPY src/ ./src/

# Config files (keys.json, fallback.json, webhook_secret.txt)
# are mounted as volumes at runtime, NOT baked into the image.

EXPOSE 8180

CMD ["python3", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8180"]
