FROM python:3.12-slim

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch + transformers are large; install CPU-only torch first so Docker
# can cache the layer and avoid re-downloading on every code change
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8015

CMD ["uvicorn", "main:app", "--app-dir", "./src", "--host", "0.0.0.0", "--port", "8015"]
