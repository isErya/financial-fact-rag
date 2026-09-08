# Multi-stage so the test suite and pytest never ship in the deployed image,
# while the tests still run in nearly the web process's environment. `base`
# is the shared application layer (same Python, same dependencies, same
# source, same corpus, same embedding model); `runtime` is the image compose
# runs as the indexer and the web server; `test` adds pytest and the suite.

FROM python:3.12-slim AS base
WORKDIR /app
COPY service/requirements.txt ./service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

# The embedding model is downloaded here, once, into the image, so the
# containers never reach the network: index.py opens the cache with
# local_files_only and a dense build (DENSE=1) fails fast without it.
# Only config.py is copied first, so the model layer is invalidated by a
# change to the model name and by nothing else; with the whole source
# tree above this line every edit to the page re-downloaded the model.
# The corpus itself is never embedded at build time; DENSE defaults to 0
# and the dense build is an explicit run-time choice in the indexer.
ENV FASTEMBED_CACHE_PATH=/app/models
COPY service/config.py ./service/config.py
RUN cd service && python -c "import config; from fastembed import TextEmbedding; TextEmbedding(config.EMBED_MODEL, cache_dir='/app/models')"

COPY service/ ./service/
COPY data/ ./data/
COPY eval/fixtures/ ./eval/fixtures/

# The runtime user owns the model cache and the index mount point. A fresh
# named volume mounted at /index copies this ownership, which is what lets
# the indexer write there without running as root.
RUN useradd --create-home --shell /usr/sbin/nologin filing \
    && mkdir -p /index \
    && chown -R filing:filing /app/models /index

# Production image. Compose runs it as the indexer (python index.py build)
# and, with the default command, as the web server, both from the source
# directory as a non-root user so a compromised process does not own the
# container. Building this target stops before the test stage, so neither
# the suite nor pytest is present here.
FROM base AS runtime
WORKDIR /app/service
USER filing
CMD ["uvicorn", "web:app", "--host", "0.0.0.0", "--port", "8000"]

# GPU image: the indexer and the tests. onnxruntime-gpu 1.29 is built
# against CUDA 13 and ships the runtime, cuDNN, cuFFT and cuRAND as pip
# extras, so no CUDA base image is needed; the CPU onnxruntime fastembed
# pulled in is removed first so the two builds cannot shadow each other.
# The web service stays on `runtime`: it embeds one query per question on
# CPU in about 130 ms and must run on a laptop without a GPU.
FROM base AS gpu
RUN pip uninstall -y onnxruntime \
    && pip install --no-cache-dir "onnxruntime-gpu[cuda,cudnn]==1.29.0"

FROM gpu AS indexer
WORKDIR /app/service
USER filing
CMD ["python", "index.py", "build"]

# Test image, never deployed. The GPU image plus pytest and the suite, so
# the dense tests run on the device; pytest.ini puts service/ on the import
# path, so it runs from /app, where the config defaults resolve.
#   docker compose run --rm tests
FROM gpu AS test
COPY tests/requirements.txt ./tests/requirements.txt
RUN pip install --no-cache-dir -r tests/requirements.txt
COPY tests/ ./tests/
COPY eval/tuning.jsonl ./eval/tuning.jsonl
COPY pytest.ini .
CMD ["python", "-m", "pytest", "-q"]
