FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-essential.txt .
RUN pip install --no-cache-dir -r requirements-essential.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
