# UC3 — Multi-PDF Research & Comparison (chat + headless, EN + AR)

Two reference agents over one Qdrant collection, built on vdreamer's public API only.

**`pdf-ingest`** (headless, `/trigger {documents: [FileRef, ...]}`):
- **Parse.** PDFs are parsed by `pypdf` *in the sandbox*, in page batches; `.txt`/`.md` are split on form feeds.
- **Chunk.** Pages become page-offset chunks (`document, page, start, end, language`).
- **Embed and store.** Each chunk gets `dense` + `sparse` fastembed vectors and is upserted into `ctx.resources["vectors"]` with deterministic point ids, so re-ingesting is safe.
- **Summary.** A `table` artifact summarizes what was ingested.

**`pdf-research`** (chat, `create_agent` with `**ctx.agent_kwargs()`):
- **`search_documents`** prefetches dense + sparse candidates, fuses them with RRF in Qdrant (`query_points`), then the `rerank` cross-encoder orders them. It returns numbered sources.
- **`compare_documents`** puts the best passages per document side by side in a `table` artifact.
- **Answering.** The model answers with `[n]` markers. `CitationMiddleware` emits one `citation` per marker, bound to the answer's `message_id` (`locator.page`, offsets, snippet, score).
- **Thinking.** A reasoning model streams `thinking.delta` (`expose_thinking = True`).

Arabic and English documents share the collection: the embedders and reranker in `config/models.yaml` are multilingual, and the agent answers in the question's language.

## Run it

```
vdreamer bundle --models config/models.yaml ...     # pre-fetch the fastembed models (connected host)
vdreamer dev --agent pdf_research.agents:ingest_agent --models config/models.ollama.yaml --resources config/resources.dev.yaml
vdreamer run pdf-ingest --trigger --input '{"documents": [{"object_key": "inputs/docs/report-en.pdf"}]}'
vdreamer run pdf-research --message "Compare the two reports on revenue growth"   # served by a second dev/worker
```

PDF parsing needs a POSIX worker (the sandbox). On other hosts, PDFs are reported as skipped and text documents still ingest. `config/models.ollama.yaml` uses Ollama's `gemma4:31b-cloud` as the reasoner (the `-cloud` tag runs on ollama.com).
