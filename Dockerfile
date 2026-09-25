# Używamy oficjalnego obrazu Ultralytics YOLO ze sterownikami L4T dla Jetsona (AArch64)
FROM ultralytics/ultralytics:latest-jetson

# Instalujemy pakiety systemowe dla kamer V4L2 i bazy danych
RUN apt-get update && apt-get install -y \
    v4l-utils \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalacja lekkich zależności serwera
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn pydantic jinja2 python-multipart mysql-connector-python

# Kopiowanie kodu
COPY . /app

# Expose Dashboard port
EXPOSE 8080

CMD ["python3", "bicycle_counter.py"]