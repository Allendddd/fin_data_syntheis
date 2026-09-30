ARG PYTHON_BASE_IMAGE=python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
FROM ${PYTHON_BASE_IMAGE}

LABEL org.opencontainers.image.title="mint-synth" \
      org.opencontainers.image.description="Provenance-first financial data synthesis for SLURM clusters" \
      org.opencontainers.image.version="2.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --gid 10001 mint && useradd --uid 10001 --gid mint --create-home mint
WORKDIR /opt/mint-synth

COPY constraints.txt pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --constraint constraints.txt setuptools wheel \
    && python -m pip install --no-build-isolation --constraint constraints.txt . \
    && python -m mint_synth --help >/dev/null

USER 10001:10001
ENTRYPOINT ["mint-synth"]
