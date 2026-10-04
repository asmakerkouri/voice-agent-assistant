FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl bzip2 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY scripts/download_models.sh scripts/
RUN ./scripts/download_models.sh
COPY src/ src/
RUN useradd -m app && chown -R app /app
USER app
ENV PYTHONPATH=/app/src
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "voiceagent.api:app", "--host", "0.0.0.0", "--port", "8000"]
