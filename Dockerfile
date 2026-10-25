FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_NUM_THREADS=2
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 1000 user
COPY --chown=user:user . .
USER user
# Index the committed corpus snapshot, keeping CI and deployment data identical.
# Model weights are downloaded at build time, so startup works offline.
RUN python -m librarian.engine \
    && python -c "from librarian.engine import RERANKER; from sentence_transformers import CrossEncoder; CrossEncoder(RERANKER)"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "librarian.app:app", "--host", "0.0.0.0", "--port", "8000"]
