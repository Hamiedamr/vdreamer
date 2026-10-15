# vdreamer:<ver> — one image, two entrypoints (PRD §2.1, §5.3):
#
#   docker run vdreamer:<ver> serve  --agent pkg.mod:agent --host <internal-ip>
#   docker run vdreamer:<ver> worker --agent pkg.mod:agent
#
# Multi-stage. The builder may use uv (dev-side toolchain, AGENTS.md); the
# runtime stage copies only the virtualenv: no uv, no compiler, no network at
# runtime. Agent code is layered on top by the operator (FROM vdreamer:<ver>,
# then pip install --no-index from their own wheelhouse), or mounted.
#
# Zero egress (§5.2): the offline switches are baked in as ENV (every vdreamer
# command forces them again); the tiktoken encodings are baked into
# /opt/vdreamer/tiktoken (TIKTOKEN_CACHE_DIR); models.yaml's fastembed.cache_dir
# should point at /opt/vdreamer/models — mount `vdreamer bundle --models`'
# fastembed/ there read-only, or bake it into a derived image.
#
# Sandbox (D19): the image creates `vdreamer-sandbox` (uid 10002). A worker that
# should switch sandboxed code to it runs as root with VDREAMER_SANDBOX_UID=10002;
# by default everything runs as the unprivileged `vdreamer` user (uid 10001).

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder

# Pinned uv (builder only). UV_PYTHON_DOWNLOADS=never: use the image's Python.
COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/vdreamer/venv

WORKDIR /src
# Dependencies first (cached unless the lockfile changes), then the project.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project --all-extras
COPY src ./src
RUN uv sync --locked --no-dev --no-editable --all-extras \
    && /opt/vdreamer/venv/bin/vdreamer --version
# The tiktoken encodings (the builder has network; the runtime never does).
RUN TIKTOKEN_CACHE_DIR=/opt/vdreamer/tiktoken /opt/vdreamer/venv/bin/python -c \
    "import tiktoken; [tiktoken.get_encoding(n) for n in ('cl100k_base', 'o200k_base')]"

FROM python:${PYTHON_VERSION}-slim AS runtime

RUN groupadd --system --gid 10001 vdreamer \
    && useradd --system --uid 10001 --gid vdreamer --home-dir /var/lib/vdreamer --create-home vdreamer \
    && groupadd --system --gid 10002 vdreamer-sandbox \
    && useradd --system --uid 10002 --gid vdreamer-sandbox --no-create-home --shell /usr/sbin/nologin vdreamer-sandbox \
    && mkdir -p /opt/vdreamer/models

COPY --from=builder /opt/vdreamer/venv /opt/vdreamer/venv
COPY --from=builder /opt/vdreamer/tiktoken /opt/vdreamer/tiktoken

ENV PATH=/opt/vdreamer/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANGSMITH_TRACING=false \
    LANGCHAIN_TRACING_V2=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    FASTMCP_CHECK_FOR_UPDATES=off \
    DO_NOT_TRACK=1 \
    TIKTOKEN_CACHE_DIR=/opt/vdreamer/tiktoken

# The image must not carry uv (AGENTS.md: vdreamer never depends on uv).
RUN ! command -v uv && vdreamer --version

USER vdreamer
WORKDIR /var/lib/vdreamer
EXPOSE 8000

ENTRYPOINT ["vdreamer"]
CMD ["--help"]
