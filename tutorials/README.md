# vdreamer tutorials

A hands-on series that takes you from a one-tool ReAct agent to multi-agent systems with memory, middleware and human approval — every lesson a working agent served by [vdreamer](../README.md), with tests.

## Lessons

Work through them in order; each builds on the ones before it.

| # | Lesson | You build |
| --- | --- | --- |
| 1 | [How vdreamer serves an agent](src/vdreamer_tutorials/t01_serving_an_agent/README.md) | a one-node agent with no model; threads, runs and events |
| 2 | [A minimal ReAct agent](src/vdreamer_tutorials/t02_minimal_react_agent/README.md) | `create_agent` with one tool and a model alias |
| 3 | [Inside the reasoning-and-action loop](src/vdreamer_tutorials/t03_react_loop/README.md) | the ReAct loop by hand, with a step budget |
| 4 | [Defining and integrating a custom tool](src/vdreamer_tutorials/t04_custom_tool/README.md) | a typed tool and the schema the model sees |
| 5 | [Tool inputs, outputs, errors and validation](src/vdreamer_tutorials/t05_tool_errors/README.md) | validated, contained, time-bounded tools |
| 6 | [Testing agents and reading execution traces](src/vdreamer_tutorials/t06_testing_and_traces/README.md) | tests at every level, from rule to live model |
| 7 | [An agent with multiple tools](src/vdreamer_tutorials/t07_multiple_tools/README.md) | a shop assistant that selects, chains and batches four tools |
| 8 | [Designing reusable tools with clear contracts](src/vdreamer_tutorials/t08_tool_contracts/README.md) | a tool factory over an injected inventory, idempotent reservations |
| 9 | [Tool selection and execution control](src/vdreamer_tutorials/t09_tool_selection/README.md) | role-based tool visibility, execution-time refusal, call limits |
| 10 | [Handling failures, retries and unexpected outputs](src/vdreamer_tutorials/t10_failures_and_retries/README.md) | tool and model retries, a fallback model, an output guard |
| 11 | [Structuring an agent application as a Python package](src/vdreamer_tutorials/t11_package_structure/README.md) | a layered helpdesk package with an architecture test |
| 12 | [Configuration, logging and observability](src/vdreamer_tutorials/t12_config_and_observability/README.md) | validated settings, run-aware logs, hooks, metrics |
| 13 | [Agent state and conversation history](src/vdreamer_tutorials/t13_state_and_history/README.md) | typed per-thread state with reducers, checkpoints and history |
| 14 | [Short-term conversational memory](src/vdreamer_tutorials/t14_short_term_memory/README.md) | a transcript plus working memory written by a tool |
| 15 | [Persistent long-term memory](src/vdreamer_tutorials/t15_long_term_memory/README.md) | per-customer notes in vdreamer's store, across conversations |
| 16 | [Managing context growth and conversation boundaries](src/vdreamer_tutorials/t16_context_management/README.md) | silent summaries of old turns, cut at turn boundaries |
| 17 | [Testing memory persistence, retrieval and isolation](src/vdreamer_tutorials/t17_testing_memory/README.md) | a memory test suite, including a real `vdreamer dev` restart |
| 18 | [Structured outputs and typed data models](src/vdreamer_tutorials/t18_structured_output/README.md) | typed input, a validated triage with retry and fallback, a JSON artifact |
| 19 | [Sequential and conditional workflows](src/vdreamer_tutorials/t19_workflows/README.md) | a headless, triggered ticket workflow with branches and checked facts |

## Setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --all-extras
cd tutorials
cp .env.example .env
uv run vdreamer dev --registry vdreamer.yaml --agent t01-echo \
    --input '{"messages": [{"role": "user", "content": "hello"}]}'
```

Lessons from 2 on use a model through the `chat` alias: `config/models.ollama.yaml` (Ollama, `gemma4:31b-cloud` — runs on ollama.com; use `gemma4:31b` to stay local) or `config/models.vllm.yaml` (any OpenAI-compatible server). The tests need no model server: `uv run pytest tests/tutorials` from the repository root.
