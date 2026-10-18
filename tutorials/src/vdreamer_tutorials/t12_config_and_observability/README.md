# Lesson 12 — Configuration, logging and observability

> **Outcome:** an agent configured from validated environment variables that fails at startup — not on the first request — when misconfigured; server logs that name the run, thread and user of every line; lifecycle hooks that time each run; client-facing progress events; and vdreamer's Prometheus metrics.

## Learning objectives

- Load settings once from the environment (and `.env`) into a validated, immutable object; report every problem at once.
- Keep secrets out of files by naming the environment variable that holds them, as vdreamer's own YAML does.
- Tell apart the three audiences: **users** (events: progress, logs, messages), **operators** (server logs), **dashboards** (metrics).
- Attach run identity to log lines with a `logging.Filter` over `vdreamer.current_context()`.
- Use the `on_run_start` / `on_run_end` hooks and read vdreamer's built-in metrics.

## Prerequisites

- Lesson [11](../t11_package_structure/README.md) (settings objects, package layout) and [9](../t09_tool_selection/README.md) (middleware).
- Python's `logging` module; Prometheus is helpful but not required.

## Concepts

**Configuration.** vdreamer configures *its* part through environment variables (`VDREAMER_*`) and two YAML files (`models.yaml`, `resources.yaml`); every command first reads `.env` from the working directory (variables already set win). Your agent should follow the same rules:

- one settings object, read from the environment **once**, validated with pydantic;
- invalid configuration fails **at startup**: every vdreamer command imports your agent module first and refuses to start if that raises, so loading settings when the registered agent instance is created turns a typo into a refused start instead of a broken first conversation (`build()` would be too late: it runs on the first run);
- secrets are never values in a file you commit: settings hold the *name* of the variable (`orders_api_key_env`), mirroring vdreamer's `*_env` keys.

`BuildContext.settings` exists but is not filled from your environment by `vdreamer dev`/`worker` today, so this lesson reads the environment itself.

**Three audiences, three channels.**

| Audience | Channel | Example |
| --- | --- | --- |
| The user, live | protocol events: `vdreamer.emit_progress`, `vdreamer.emit_log`, messages | "looking up the order 50%" |
| Operators, later | server logs (Python `logging` → stderr → your log collector) | `level=INFO run=run_… user=ada order lookup order_id=A1001` |
| Dashboards and alerts | metrics (vdreamer's Prometheus endpoint) | `vdreamer_tool_latency_seconds`, `vdreamer_tool_errors_total` |

Do not mix them: `emit_log` goes to the client (never put internals there); server logs never go to the client.

**Correlation.** A log line is only useful if you can find the run it belongs to. Inside a run, `vdreamer.current_context()` returns the run id, thread id, user and roles — from any node, tool or middleware, because vdreamer binds it for the whole run (outside a run it raises `RuntimeError`). A `logging.Filter` copies those fields onto every record.

**Hooks.** `BaseAgent.on_run_start(run)` and `on_run_end(run, status)` are called around every run (`run` is a `RunInfo`: run id, thread id, agent name, attempt). They are the place for timing, audit records, or your own metrics.

**Built-in metrics.** The gateway serves Prometheus text at `GET /api/v1/metrics` (no token needed, so scrapers can reach it). It includes queue depth per status, run durations, SSE connections, model latency and token counts per alias, tool latency and errors per source and error code, MCP server health and the outbox lag. In `vdreamer dev`, worker metrics are on the same endpoint; a separate `vdreamer worker` exposes its own with `--metrics-port`.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `.env` loading for every command, `*_env` secret references | vdreamer |
| `current_context()`, `RunInfo`, `on_run_start`/`on_run_end`, `emit_progress`/`emit_log` | vdreamer |
| `/api/v1/metrics` (`vdreamer_*` metrics), `worker --metrics-port` | vdreamer |
| `@before_model` | LangChain |
| Settings model, `logging.Filter`, formatter | pydantic / Python stdlib, this lesson's code |

## How it works

```
vdreamer dev / worker / validate ── reads .env ──► imports the agent ──► ObservableAgent()
                                                    └─ ObservabilitySettings.from_env()
                                                       ✗ invalid → "cannot load …", exit 1
run ── on_run_start ──► log "run started"
   ├─ before_model ──► log DEBUG "model call messages=N"
   ├─ tool ──► emit_progress (client)  +  log INFO "order lookup …" (operators)
   │           governance ──► vdreamer_tool_latency_seconds{source="custom"} (metrics)
   └─ on_run_end ──► log "run ended status=completed duration_ms=18"
every log record ◄── RunContextFilter: run=…, thread=…, user=…
```

## Step by step

**1. A settings model with a strict loader.**

```python
# file: settings.py
class ObservabilitySettings(BaseModel, frozen=True):
    shop_name: str = "Trailhead Outfitters"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    slow_tool_ms: int = Field(default=500, gt=0)
    #: The *name* of the env var holding the order-system key, never the key itself.
    orders_api_key_env: str = "T12_ORDERS_API_KEY"
```

`from_env()` reads `T12_<FIELD>` variables and raises one `SettingsError` listing every invalid one.

**2. Load it where startup failures surface.** The agent accepts settings (tests pass them explicitly) or reads the environment when the instance is created — and the registered instance is created when vdreamer imports the module at startup:

```python
# file: agent.py
    def __init__(self, settings: ObservabilitySettings | None = None) -> None:
        # Loaded when the instance is created, which vdreamer does at startup when it
        # imports the agent: an invalid environment stops `dev`/`worker`/`serve` before
        # any run. Tests pass settings explicitly.
        self.settings = settings if settings is not None else ObservabilitySettings.from_env()
```

**3. Make every log line say which run it is from.**

```python
# file: logs.py
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            context = vdreamer.current_context()
        except RuntimeError:  # not inside a vdreamer run
            record.run_id = record.thread_id = record.user_id = "-"
        else:
            record.run_id = context.run_id
            record.thread_id = context.thread_id
            record.user_id = context.user_id or "-"
        return True
```

**4. Report to each audience on its own channel** — inside the tool:

```python
# file: agent.py
        vdreamer.emit_progress("looking up the order", 50)  # for the user
        status = ORDERS.get(order_id.strip().upper(), "unknown")
        elapsed_ms = (time.monotonic() - started) * 1000
        log.info("order lookup order_id=%s status=%s", order_id, status)  # for operators
```

**5. Time runs with the hooks.**

```python
# file: agent.py
    async def on_run_end(self, run: RunInfo, status: str) -> None:
        started = self._started.pop(run.run_id, time.monotonic())
        get_logger(self.settings.log_level).info(
            "run ended status=%s duration_ms=%.0f", status, (time.monotonic() - started) * 1000
        )
```

The start times are keyed by run id: one agent instance serves many concurrent runs.

## The complete example

- [`settings.py`](settings.py) — `ObservabilitySettings`, `from_env`, `SettingsError`.
- [`logs.py`](logs.py) — `RunContextFilter`, `get_logger`.
- [`agent.py`](agent.py) — the tool, a `before_model` debug hook, the run hooks, `ObservableAgent` (`t12-observable`).

## Run it

From `tutorials/`:

```bash
echo "T12_LOG_LEVEL=DEBUG" >> .env
uv run vdreamer dev --registry vdreamer.yaml --agent t12-observable --models config/models.ollama.yaml
uv run vdreamer run t12-observable --message "Where is order A1001?"     # second terminal
curl -s http://127.0.0.1:8000/api/v1/metrics | grep -E "vdreamer_(tool|model)_"
```

See a startup failure:

```bash
T12_LOG_LEVEL=LOUD uv run vdreamer dev --registry vdreamer.yaml --agent t12-observable \
    --models config/models.ollama.yaml
```

## Expected behavior

The `dev` terminal shows (ids shortened):

```
… level=INFO logger=vdreamer_tutorials.t12 run=run_2abc… thread=thread_… user=you run started agent=t12-observable attempt=1
… level=DEBUG … model call messages=1
… level=INFO … order lookup order_id=A1001 status=shipped
… level=DEBUG … model call messages=3
… level=INFO … run ended status=completed duration_ms=…
```

The `vdreamer run` terminal shows `[progress] looking up the order 50%` — and none of the log lines. The metrics include `vdreamer_tool_latency_seconds_count{source="custom"}`. The bad-config start exits with status 1 before contacting any model:

```
dev: cannot load 't12-observable': SettingsError: invalid configuration: T12_LOG_LEVEL: Input should be 'DEBUG', 'INFO', 'WARNING' or 'ERROR'
```

## Test it

[`test_t12_config_and_observability.py`](../../../../tests/tutorials/test_t12_config_and_observability.py) passes explicit environments to `from_env` (no global state), runs the documented `vdreamer dev` command with a bad environment and checks it refuses to start, collects log records with a test handler, and scrapes the metrics endpoint of a real gateway:

```python
# file: tests/tutorials/test_t12_config_and_observability.py
    run_ids = {r.run_id for r in logs.records}  # type: ignore[attr-defined]
    assert len(run_ids) == 1 and run_ids != {"-"}
    assert {r.user_id for r in logs.records} == {"ada"}  # type: ignore[attr-defined]
    # The client sees progress, not the server's log lines.
    assert result.payloads("progress") == [{"label": "looking up the order", "pct": 50.0}]
    assert result.payloads("log") == []
```

```bash
uv run pytest tests/tutorials/test_t12_config_and_observability.py -v
```

## Common mistakes and debugging

- **Reading settings on first use** (in `build()` or a tool) — vdreamer builds the graph on the first run, so a typo surfaces in the first user's conversation. Load when the agent is created, list every problem in one message, and let tests pass settings explicitly. Keep the rest of module import cheap: only the registered instance should read the environment.
- **`os.environ["X"]` scattered across tools** — undiscoverable configuration and `KeyError`s mid-run. One settings object.
- **Secrets in logs** — log identifiers, not payloads. vdreamer redacts tool arguments in *events*; your own log lines are your responsibility.
- **`emit_log` for debugging** — it goes to the end user. Use `logging` for operators.
- **No log output at all** — vdreamer does not configure your loggers. Configure a handler (as `get_logger` does) or your process's logging at deployment.
- **Per-run data on `self` without a key** (`self.started = time.monotonic()`) — concurrent runs overwrite each other. Key by `run.run_id`.

## Design trade-offs

- **Pydantic model vs. `pydantic-settings`.** `pydantic-settings` reads env vars and `.env` itself; it is one more dependency, and vdreamer already loads `.env`. A twenty-line loader keeps the tutorials dependency-free.
- **Logs vs. metrics vs. traces.** Logs explain one run; metrics show trends and fire alerts; distributed tracing (OpenTelemetry) follows a request across services. vdreamer ships metrics and events and deliberately no external tracing service, because deployments are air-gapped and must not phone home.
- **Key=value text vs. JSON logs.** Text is readable in a terminal; JSON is easier for log collectors. Swap the formatter, keep the filter.

## Exercises

1. Add `T12_LOG_FORMAT=json` and a JSON formatter; keep the existing tests green.
2. Emit a `log` *event* (`vdreamer.emit_log(..., level="warning")`) only when an order is unknown — something the user should know — and test it.
3. Count runs per final status in `on_run_end` with your own `prometheus_client.Counter` and check it shows up at `/api/v1/metrics`.
4. Make the slow-lookup warning fire in a test by injecting a slow order source (lesson 8 style).

Next: Lesson 13 — Agent state and conversation history.
