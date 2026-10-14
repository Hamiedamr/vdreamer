"""The Milestone 7 exit agent: declares models and resources, reaches all of them.

``models = ["reasoner", "local", "dense"]`` — a vLLM chat alias with
``thinking: reasoning_content``, an Ollama alias with ``think_tags``, and a
fastembed embedding alias. ``resources = ["warehouse", "vectors"]`` — a SQL
engine and a qdrant client. ``build()`` only closes over the aliases; the
nodes use the native objects directly (A35).
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from vdreamer import BaseAgent, BuildContext


class State(TypedDict, total=False):
    question: str
    sites: list[str]
    hits: list[int]
    answer: str
    summary: str


class M7Agent(BaseAgent):
    name = "m7"
    models = ["reasoner", "local", "dense"]
    resources = ["warehouse", "vectors"]

    def build(self, ctx: BuildContext) -> Any:
        reasoner = ctx.models["reasoner"]
        local = ctx.models["local"]
        # A subclass may leave fastembed out (the CLI test has no model files).
        dense = ctx.models["dense"] if "dense" in self.models else FakeTextEmbedding()
        warehouse = ctx.resources["warehouse"]
        vectors = ctx.resources["vectors"]

        async def lookup(state: State) -> dict[str, Any]:
            from sqlalchemy import text

            async with warehouse.connect() as conn:
                rows = await conn.execute(text("SELECT name FROM sites ORDER BY id"))
                return {"sites": [row[0] for row in rows]}

        async def search(state: State) -> dict[str, Any]:
            from qdrant_client import models

            if not await vectors.collection_exists("sites"):
                await vectors.create_collection(
                    "sites",
                    vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
                )
                points = [
                    models.PointStruct(id=i, vector=vector)
                    for i, vector in enumerate(dense.embed(state["sites"]), start=1)
                ]
                await vectors.upsert("sites", points)
            (query,) = list(dense.embed([state["question"]]))
            found = await vectors.query_points("sites", query=query, limit=1)
            return {"hits": [point.id for point in found.points]}

        async def answer(state: State) -> dict[str, Any]:
            message = await reasoner.ainvoke(
                [HumanMessage(f"{state['question']} sites={state['sites']} hits={state['hits']}")]
            )
            return {"answer": message.text}

        async def summarize(state: State) -> dict[str, Any]:
            message = await local.ainvoke([HumanMessage(f"Summarize: {state['answer']}")])
            return {"summary": message.text}

        graph = StateGraph(State)
        for name, node in (
            ("lookup", lookup), ("search", search), ("answer", answer), ("summarize", summarize)
        ):
            graph.add_node(name, node)
        graph.add_edge(START, "lookup")
        graph.add_edge("lookup", "search")
        graph.add_edge("search", "answer")
        graph.add_edge("answer", "summarize")
        graph.add_edge("summarize", END)
        return graph


agent = M7Agent()


class FakeTextEmbedding:
    """Stands in for ``fastembed.TextEmbedding`` (no model files ship with tests)."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def embed(self, documents: list[str]) -> Any:
        for document in documents:
            yield [1.0, 0.0] if "cairo" in document.lower() else [0.0, 1.0]
