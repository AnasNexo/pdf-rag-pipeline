# PDF RAG Pipeline — container image for the Streamlit web app.
#
# Build:  docker build -t pdf-rag .
# Run:    docker run -p 8501:8501 --env-file .env -v rag_db:/app/pdf_data.db pdf-rag
#   (or use docker-compose, which wires the keys and DB volume for you)

FROM python:3.11-slim

# System dependencies the Python libraries need but pip cannot install:
#   ghostscript  -> required by camelot-py for table extraction
#   libgl1 / libglib2.0-0 -> required by opencv-python-headless
# Cleaned up in the same layer to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ghostscript \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only build of PyTorch FIRST. sentence-transformers pulls in
# torch, and by default pip fetches the full CUDA build (~2-3 GB of NVIDIA
# libraries) — useless here since the container has no GPU. Installing the CPU
# wheel first means the CUDA one is never downloaded: much smaller image, much
# faster build, identical behaviour (embeddings/OCR run on CPU regardless).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the remaining Python deps. torch is already satisfied above, so
# sentence-transformers reuses the CPU build instead of pulling CUDA.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY . .

# sentence-transformers / RapidOCR download their models to the home cache on
# first use; give them a writable, persistable location.
ENV HF_HOME=/app/.cache/huggingface

# Streamlit's default port.
EXPOSE 8501

# Bind to 0.0.0.0 so the server is reachable from outside the container.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
