FROM python:3.12-slim

WORKDIR /app

# System dependencies for native extensions (psycopg C accelerator)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU-only torch via extra index)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy backend source (overridden by bind mount in dev)
COPY . .

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
