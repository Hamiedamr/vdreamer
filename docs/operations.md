# Operating vdreamer air-gapped

How a release gets from a connected build host onto the air-gapped VMs of the reference topology (PRD §5.1), and what each process needs there. Operators use standard `pip`; uv is only the dev-side toolchain and never appears on a target host.

## 1. What a release ships

Each `v*` tag's GitHub release carries (CI job `release`):

| File | What it is |
|---|---|
| `vdreamer-<ver>-py3-none-any.whl`, `vdreamer-<ver>.tar.gz` | the package |
| `constraints.txt` | every dependency of every extra, pinned **with sha256 hashes**: the tested set (`uv export --frozen --no-dev --all-extras --no-emit-project`) |
| `vdreamer-schema.tar.gz` | the client contract: `vdreamer export-schema` output (OpenAPI 3.1, event JSON Schema, TS types) |

The image `vdreamer:<ver>` is built from the repo's `Dockerfile` (CI job `image`) and pushed to the internal registry.

## 2. Build the offline install set (connected host)

On any host with internet access and Python 3.11+:

```
python -m venv bundle-env
bundle-env/bin/pip install vdreamer-<ver>-py3-none-any.whl
bundle-env/bin/vdreamer bundle \
    --vdreamer vdreamer-<ver>-py3-none-any.whl \
    --extra postgres --extra minio --extra fastembed \
    -r my-agents/requirements.txt \
    -c constraints.txt \
    --models models.yaml \
    --platform manylinux2014_x86_64 --python-version 3.11 \
    --out vdreamer-bundle
```

- `--extra` picks the vdreamer extras the deployment needs: `postgres` (always, for `serve`/`worker`), `minio`, `fastembed`, `qdrant`, `neo4j`, `mcp`, `sandbox`, `sql`.
- `-r` adds your agents' own requirements. They need not be pinned: the lock below pins and hashes whatever was resolved.
- `-c constraints.txt` keeps vdreamer's dependencies at the tested versions. pip resolves against the pins. Then every downloaded file of a pinned project is checked against the release's sha256s, and a mismatch fails the bundle.
- `--platform`/`--python-version` build for the Linux VMs from a different host (binary wheels only). Leave them out when the build host matches the target.
- `--models models.yaml` downloads every `provider: fastembed` alias into `vdreamer-bundle/fastembed/`.
- The tiktoken encodings go to `vdreamer-bundle/tiktoken/` (`--no-tiktoken` skips them).

Output:

```
vdreamer-bundle/
  wheelhouse/            every distribution
  requirements.lock      name==version --hash=sha256:... for each of them
  tiktoken/              cl100k_base, o200k_base
  fastembed/             (with --models)
  bundle.json            what was bundled, sha256s
  INSTALL.txt            the commands below
```

## 3. Install on the air-gapped host (no uv, no network)

```
python3.11 -m venv /opt/vdreamer/venv
/opt/vdreamer/venv/bin/pip install --no-index --find-links vdreamer-bundle/wheelhouse \
    --require-hashes -r vdreamer-bundle/requirements.lock
export TIKTOKEN_CACHE_DIR=/opt/vdreamer/tiktoken      # vdreamer-bundle/tiktoken, copied
/opt/vdreamer/venv/bin/vdreamer doctor --offline
```

`doctor --offline` must print `doctor: PASS`. It forces the offline environment, imports the whole stack under a socket guard and asserts zero egress. It also checks the vendored assets and the tiktoken cache, runs a sandbox probe, and with `--models`/`--resources` probes every declared model endpoint, fastembed model file, resource and MCP server.

## 4. The image

```
docker run vdreamer:<ver> serve  --agent my_pkg.agent:agent --host 10.10.0.4
docker run vdreamer:<ver> worker --agent my_pkg.agent:agent
docker run vdreamer:<ver> migrate
```

- **Contents:** only the virtualenv with every extra, as user `vdreamer` (uid 10001). There is no uv, no compiler and no network access at runtime.
- **Offline environment:** the offline variables are baked in as `ENV`, and the tiktoken encodings are baked into `/opt/vdreamer/tiktoken`.
- **Agent code:** derive an image (`FROM vdreamer:<ver>`, then `pip install --no-index` from your own wheelhouse) or mount it and set `PYTHONPATH`.
- **fastembed models:** mount `fastembed/` read-only at `/opt/vdreamer/models` and set `fastembed.cache_dir: /opt/vdreamer/models` in `models.yaml`.
- **Sandbox (D19):** the image has a `vdreamer-sandbox` user (uid 10002). For a dedicated sandbox uid, run the worker container as root with `VDREAMER_SANDBOX_UID=10002`.

## 5. Processes and topology

| Process | Host (§5.1) | Needs |
|---|---|---|
| `vdreamer migrate` | once per release | Postgres (`VDREAMER_PG_DSN`), MinIO env to create the bucket |
| `vdreamer serve` | VM-GW, N replicas | Postgres, MinIO, `VDREAMER_INTERNAL_TOKEN`; binds the internal interface only (`--host`, wildcards refused) |
| `vdreamer worker --agent X` | VM-WRK, ≥1 per agent | Postgres, MinIO, `models.yaml`, `resources.yaml`, model servers and resources |

The gateway and the workers talk through Postgres only (A7). Only workers reach model servers, MinIO writes and agent resources.

## 6. Environment reference

All configuration is environment variables. YAML files name secrets through `*_env` keys and never hold them (A23).

| Variable | Used by | Meaning |
|---|---|---|
| `VDREAMER_PG_DSN` | serve, worker, migrate | Postgres connection string |
| `VDREAMER_INTERNAL_TOKEN` (+ `_PREVIOUS`) | serve, dev | the Web Backend's bearer token (two valid during rotation, A14) |
| `VDREAMER_SERVICE_TOKENS` | serve, dev | `name:token,...`: upstream services allowed to call `/trigger` (A16) |
| `CALLBACK_ALLOWLIST` | serve, dev | where `/trigger` callbacks may go: host globs and CIDRs (A26) |
| `VDREAMER_OUTBOX_WEBHOOK_URL`, `VDREAMER_OUTBOX_SECRET` | serve, worker, dev | interrupt notifications (HMAC-signed webhook) |
| `WEB_BASE_URL` | serve, worker, dev | deep link base: `{WEB_BASE_URL}/review/{interrupt_id}` |
| `VDREAMER_SMTP_HOST`/`_PORT`/`_FROM`/`_TO`/`_USER`/`_PASSWORD`/`_STARTTLS` | serve, worker, dev | interrupt notifications by email |
| `HITL_STRICT_SCOPES` | serve, dev | four-eyes scope patterns (A32) |
| `NETWORK_ALLOWLIST` | worker, validate, doctor | every declared endpoint must match (A40) |
| `VDREAMER_MODELS`, `VDREAMER_RESOURCES` | worker, dev, validate, doctor | default `models.yaml` / `resources.yaml` paths |
| `VDREAMER_MINIO_ENDPOINT`/`_ACCESS_KEY`/`_SECRET_KEY`/`_BUCKET`/`_SECURE` | serve, worker, migrate | artifact bytes |
| `VDREAMER_SANDBOX_UID`/`_GID`/`_ROOT` | worker | sandbox identity and scratch root (D19) |
| `TIKTOKEN_CACHE_DIR` | worker | the bundled tiktoken encodings (§5.2) |
| `VDREAMER_STREAM_DRIVER` | worker, dev | `v2` (default) or `v3` (A31) |

Every vdreamer command except `bundle` forces `LANGSMITH_TRACING=false`, `LANGCHAIN_TRACING_V2=false`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `FASTMCP_CHECK_FOR_UPDATES=off` and `DO_NOT_TRACK=1` (§5.2).

## 7. vLLM prerequisites (A21)

vdreamer talks to vLLM through its OpenAI-compatible API and relies on vLLM for tool calling and reasoning. Start each served model with:

- `--enable-auto-tool-choice` (without it, vLLM rejects requests that carry tools);
- `--tool-call-parser <parser>` matching the model family, for example `hermes` for Qwen2.5/Qwen3 and `llama3_json` for Llama 3.x;
- `--reasoning-parser <parser>` for thinking models (for example `qwen3`, `deepseek_r1`). Its alias in `models.yaml` then uses `thinking: reasoning_content`. A model that inlines `<think>` tags without a reasoning parser uses `thinking: think_tags`.

```
vllm serve Qwen/Qwen3-32B --enable-auto-tool-choice --tool-call-parser hermes \
    --reasoning-parser qwen3 --served-model-name qwen3-32b
```

`vdreamer validate --models models.yaml pkg.mod:agent` and `doctor --offline --models models.yaml` check that each endpoint answers and serves the configured model.

## 8. Upgrades

1. Build a new bundle from the new release's `constraints.txt`.
2. Install it into a fresh venv, or roll out the new image.
3. Run `vdreamer migrate`. Every step is idempotent and upgrades tables in place.
4. Restart the workers, then the gateways.

A `langchain`/`langgraph` 2.0 always comes with a vdreamer major version (A41).
