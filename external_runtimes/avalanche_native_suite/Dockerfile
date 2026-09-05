FROM --platform=linux/amd64 python:3.10.14-slim-bookworm@sha256:45360d9eb0ff89a954085c2b36883b65b08d1549aa9ede50e48bdcadf34d2f3f

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONPATH=/opt/avalanche-eb075be393e1f458b2c352514ff6c17b5a2c0f4e

WORKDIR /opt/qualification
COPY requirements.lock requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes --only-binary=:all: \
    --no-binary=gputil \
    --requirement requirements.lock

COPY fetch_source.py fetch_source.py
RUN python fetch_source.py

COPY Dockerfile Dockerfile.source
COPY requirements.in requirements.in
COPY qualification-plan.json qualification-plan.json
COPY verify_runtime.py verify_runtime.py
ENV HOME=/tmp/asi-runtime-home \
    XDG_CACHE_HOME=/tmp/asi-runtime-cache \
    MPLCONFIGDIR=/tmp/asi-matplotlib \
    PYTHON_SETUPTOOLS_VERSION=84.0.0
USER 65532:65532
RUN --network=none python verify_runtime.py

ENTRYPOINT ["python", "/opt/qualification/verify_runtime.py"]
