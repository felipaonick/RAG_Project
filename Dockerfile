# Immagine base NVIDIA con CUDA 12.2 runtime
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Imposta la working directory
WORKDIR /app

# Installa Python 3.10 e dipendenze di sistema
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Imposta python e pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Copia requirements.txt prima per caching
COPY requirements.txt .

# Installa pacchetti Python 
RUN --mount=type=cache,target=/root/.cache \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Copia il codice sorgente nel container
COPY . .

# Espone la porta FastAPI
EXPOSE 8091

# Comando di avvio
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8091"]
