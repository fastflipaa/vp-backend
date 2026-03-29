FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for Docker cache efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Copy entrypoint and make scripts executable
COPY entrypoint.sh .
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh scripts/celery_healthcheck.py

EXPOSE 8000

# SERVICE_ROLE env var selects: api (default), worker, beat, flower
ENTRYPOINT ["./entrypoint.sh"]
