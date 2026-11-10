# syntax=docker/dockerfile:1
#
# Slim GPT-SoVITS v2 inference server (no training / ASR / UVR5 / WebUI).
# Models are NOT baked into the image; they are mounted from the host at runtime.
#
#   build: docker build -t ghcr.io/sdlizhan/gpt-sovits-server:v2 .
#   run:   docker run --gpus all -p 50045:50045 \
#            -v /path/pretrained_models:/app/GPT_SoVITS/pretrained_models:ro \
#            -v /path/G2PWModel:/app/GPT_SoVITS/text/G2PWModel:ro \
#            -v /path/voices:/app/voices:ro \
#            ghcr.io/sdlizhan/gpt-sovits-server:v2

ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.9.1
ARG CUDA_INDEX=https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------
# Stage 1: build a self-contained venv
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG TORCH_VERSION
ARG CUDA_INDEX

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# gcc/g++ are only needed to compile jieba-fast (sdist, no wheel).
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip setuptools wheel \
 && pip install "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}" --index-url "${CUDA_INDEX}" \
 && pip install -r /tmp/requirements.txt

# NLTK data used by g2p_en (English fallback). Harmless if English is unused.
RUN python -m nltk.downloader -d /usr/local/share/nltk_data \
      averaged_perceptron_tagger \
      averaged_perceptron_tagger_eng \
      cmudict

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    NUMBA_CACHE_DIR=/tmp/numba \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# libsndfile1: soundfile backend; libgomp1: OpenMP for torch/numba.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 libgomp1 tzdata \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/local/share/nltk_data /usr/local/share/nltk_data

WORKDIR /app

# Application code only. Model directories are runtime mounts:
#   /app/GPT_SoVITS/pretrained_models
#   /app/GPT_SoVITS/text/G2PWModel
#   /app/voices
COPY . /app

EXPOSE 50045

# Models load before uvicorn starts, so /health is only reachable once warm.
CMD ["python", "server_fast.py", "--host", "0.0.0.0", "--port", "50045"]
