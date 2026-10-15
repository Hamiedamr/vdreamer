# UC2 — Telecom Site & Antenna Validation (headless)

The PRD §7.3 reference headless agent, `site-validation`. It is built on vdreamer's public API only.

```
/trigger {site_id, package: FileRef} → unpack (safe_extract) → inventory (MCP) → inspect (VLM)
  → verdict (json, renderer verdict.v1) → site_verdict gate (site_approver) → record (MCP) → callback
```

1. **Upload and trigger.** A mail worker uploads the survey ZIP to `inputs/` and calls `POST /api/v1/agents/site-validation/trigger` with a service token (A16), an `Idempotency-Key` and a `callback` on `CALLBACK_ALLOWLIST`.
2. **`unpack`.** `ctx.files.safe_extract` unpacks the ZIP in the sandbox (A34). It keeps the photos and `survey.json`; anything else is skipped.
3. **`inventory`.** It asks the site-inventory MCP server (`siteinv`, `ctx.tools.mcp("siteinv")`) for the planned sectors and antenna counts.
4. **`inspect`.** Each photo goes to the VLM alias (`ctx.models["vlm"]`), which answers `{sector, antennas, issues}`.
5. **`verdict`.** Planned and observed are compared: `pass`, `fail` or `needs_review`. The result is a `json` artifact with `renderer="verdict.v1"`.
6. **The approval.** A `site_approver` reviews it in the inbox (`approver_role`, A16; headless runs require one).
   - `approve` (or `edit` the status and a note): `record` writes the decision back through MCP, and the callback hears `completed`.
   - `reject`: the run ends `rejected`, and the callback hears `failed`.

`site_validation.inventory_server` is a small FastMCP server standing in for the operator's inventory system. It is installed as the `site-inventory-mcp` console script, and its demo data is in `inventory.json`.

## Run it

```
vdreamer dev --agent site_validation.agent:agent --models config/models.ollama.yaml --resources config/resources.yaml
# copy the survey ZIP to <db dir>/inputs/surveys/CAI-0042.zip, then:
vdreamer run site-validation --trigger --input '{"site_id": "CAI-0042", "package": {"object_key": "inputs/surveys/CAI-0042.zip"}}' --roles site_approver
```

Set `command` in `config/resources.yaml` to the absolute path of `site-inventory-mcp` in your venv (stdio MCP commands must be absolute, §5.2). `config/models.ollama.yaml` uses Ollama's `gemma4:31b-cloud` (vision; the `-cloud` tag runs on ollama.com).
