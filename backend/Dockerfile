# Backend image: CPU-only torch.
#
# The host setup uses CUDA for reranking, but a container that expects a GPU
# would only run on machines with the NVIDIA runtime configured. The CPU
# reranker costs ~1-2 s per query instead of ~0.3 s, which is acceptable for
# the docker demo; the host venv remains the fast path (see README).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Torch first, from the CPU index: PyPI's default Linux wheel already is CPU,
# but pinning the index makes the intent explicit and keeps the image smaller.
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend

# SQLite lives on a volume; everything else is stateless.
ENV SQLITE_PATH=/data/localmind.db \
    HOST=0.0.0.0 \
    PORT=8000

VOLUME ["/data"]
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
