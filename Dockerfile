FROM python:3.11-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ ./src/
COPY conf/ ./conf/
COPY setup.py setup.py
COPY README.md README.md
RUN mkdir -p data logs models notebooks && pip install --no-cache-dir -e .
ENV PYTHONUNBUFFERED=1
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
CMD ["kedro", "run"]
