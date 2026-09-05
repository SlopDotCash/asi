FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

ARG COOM_COMMIT=7929801176c6e2e036c7c1c7dd6ce9b84a9d1f3e
ARG COOM_ARCHIVE_SHA256=a4736e9916468482d75831d53a12a8601c4da91cd40b9b24d313522034a15661

ADD https://github.com/TTomilin/COOM/archive/${COOM_COMMIT}.tar.gz /tmp/coom.tar.gz
RUN mkdir -p /opt/qualification
COPY requirements.lock /opt/qualification/requirements.lock
COPY coom-gymnasium.patch /opt/qualification/coom-gymnasium.patch

RUN printf '%s  %s\n' "$COOM_ARCHIVE_SHA256" /tmp/coom.tar.gz | sha256sum --check --strict \
    && printf '%s  %s\n' \
       '25bc846908e573ff1c7d02909a9bb895570e0beafe1a928dbd0d5fd4b63835a7' \
       /opt/qualification/coom-gymnasium.patch | sha256sum --check --strict \
    && mkdir /opt/coom \
    && tar --extract --gzip --file /tmp/coom.tar.gz --directory /opt/coom --strip-components=1 \
    && python -c "from pathlib import Path; p=Path('/opt/coom/COOM/wrappers/reward.py'); raw=p.read_text(); old='from gym import RewardWrapper'; assert raw.count(old) == 1; p.write_text(raw.replace(old, 'from gymnasium import RewardWrapper'))" \
    && python -m pip install --no-cache-dir --require-hashes \
       --requirement /opt/qualification/requirements.lock \
    && rm /tmp/coom.tar.gz

COPY Dockerfile /opt/qualification/Dockerfile.source
COPY qualification-manifest.json /opt/qualification/qualification-manifest.json
COPY smoke.py /opt/qualification/smoke.py

ENV HOME=/tmp \
    PYTHONPATH=/opt/coom
USER 65532:65532
WORKDIR /opt/coom
ENTRYPOINT ["python", "/opt/qualification/smoke.py"]
