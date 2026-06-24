# ============================================================================
# QECTOR v3 — Dockerfile
# Build multi-étage : extension Rust + package Python + serveur REST
# ============================================================================

# --- Étape 1 : Build de l'extension Rust ----------------------------------
FROM python:3.12-slim AS rust-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    ocl-icd-opencl-dev \
    && rm -rf /var/lib/apt/lists/*

# Installation de Rust via rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir maturin

WORKDIR /build
COPY Cargo.toml Cargo.lock build.rs ./
COPY src/ ./src/
COPY proto/ ./proto/
COPY python/ ./python/
COPY pyproject.toml README.md ./

# The pure-Python ecosystem layer (codes/dem/result/backend/pymatching_compat/
# benchmarking) lives under python/qector_decoder_v3/ and is bundled into the
# wheel automatically via [tool.maturin] python-source = "python".
RUN maturin build --release

# --- Étape 2 : Image finale Python ----------------------------------------
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie et installation de la wheel QECTOR
COPY --from=rust-builder /build/target/wheels/*.whl /wheels/
RUN pip install --no-cache-dir /wheels/*.whl

# Dépendances REST (FastAPI par défaut, Flask fallback si échec)
RUN pip install --no-cache-dir fastapi uvicorn || pip install --no-cache-dir flask

# Reproduction toolchain: test + benchmark deps, the suite, and the benchmark
# drivers. Makes the image a turnkey LINUX reproduction of the report:
#   docker build -t qector .
#   docker run --rm qector pytest python/tests -q          # full suite
#   docker run --rm qector python scripts/competitive_stim_ler.py \
#       --distances 3 5 7 9 11 --shots 40000                # QECTOR vs PyMatching
RUN pip install --no-cache-dir pytest hypothesis stim pymatching sinter ldpc beliefmatching scipy psutil
COPY python/tests/ ./python/tests/
COPY scripts/ ./scripts/
COPY pyproject.toml README.md ./

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV QECTOR_REST_HOST=0.0.0.0
ENV QECTOR_REST_PORT=8000

CMD ["python", "-c", "from qector_decoder_v3.rest_api import run_server; run_server()"]
