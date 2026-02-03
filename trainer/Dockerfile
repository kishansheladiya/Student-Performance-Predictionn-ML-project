FROM python:3.11-slim
WORKDIR /app

# Install build deps (safe) and pip requirements
RUN apt-get update && apt-get install -y build-essential git curl && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy repository
COPY . /app

# Ensure script is executable
RUN chmod +x /app/trainer/run_train.sh

CMD ["/app/trainer/run_train.sh"]
