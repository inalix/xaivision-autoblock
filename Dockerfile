# --- Build Stage ---
FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install core build toolchain and video frame processing layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring in modern uv bin for fast locking and syncing
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Sync using locked cu132 dependency profiles
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --python python3.12

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --python python3.12


# --- Final Runtime Stage ---
FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive

# Lightweight baseline system requirements for YOLO execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Safely inherit clean binaries and skip native compilation tools
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

ENV MODEL_PATH=/app/xaivision_autoblock/models/26s-c/best.pt
ENV TARGET_FPS=10
ENV PARKING_LABEL=TEST
ENV PARKING_POSITION=200,200|400,200|400,400|200,400
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "main.py"]
