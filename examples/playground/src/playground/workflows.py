"""Two LangGraph workflow patterns from the docs, served headless by vdreamer.

Source: https://docs.langchain.com/oss/python/langgraph/workflows-agents —
"Prompt chaining" and "Routing", Graph API versions. The node code is the
docs' code with two changes: ``llm`` is ``ctx.models["chat"]``, and node code
reads ``msg.text`` where the docs read ``msg.content``. vdreamer builds every
chat model with ``output_version="v1"`` (PRD §2.4), so ``content`` is a list of
content blocks — ``"?" in msg.content`` would never match — while ``.text`` is
the plain string on every LangChain version. Both take a typed input (a
``HEADLESS``/``BOTH`` agent must declare ``input_schema``), so they run with
``vdreamer run <agent> --trigger --input '{...}'`` or ``POST /agents/{name}/trigger``.

The client sees each LLM call as a streamed message (``token.delta`` →
``message.completed``); the final state keys (``final_joke``, ``output``) are
not events — state is the agent's own, and only messages, artifacts, logs and
progress reach clients (PRD §3.3).
"""

from __future__ import annotations

from typing import Any, Literal

from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from vdreamer import BaseAgent, BuildContext, Mode

# -- prompt chaining ----------------------------------------------------------------------


class JokeInput(BaseModel):
    topic: str


class JokeState(TypedDict, total=False):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str


class PromptChainAgent(BaseAgent):
    name = "joke-chain"
    mode = Mode.HEADLESS
    models = ("chat",)
    input_schema = JokeInput

    def build(self, ctx: BuildContext) -> StateGraph:
        llm = ctx.models["chat"]

        def generate_joke(state: JokeState) -> dict[str, Any]:
            """First LLM call to generate initial joke"""
            msg = llm.invoke(f"Write a short joke about {state['topic']}")
            return {"joke": msg.text}

        def check_punchline(state: JokeState) -> str:
            """Gate function to check if the joke has a punchline"""
            if "?" in state["joke"] or "!" in state["joke"]:
                return "Pass"
            return "Fail"

        def improve_joke(state: JokeState) -> dict[str, Any]:
            """Second LLM call to improve the joke"""
            msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
            return {"improved_joke": msg.text}

        def polish_joke(state: JokeState) -> dict[str, Any]:
            """Third LLM call for final polish"""
            msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
            return {"final_joke": msg.text}

        workflow = StateGraph(JokeState)
        workflow.add_node("generate_joke", generate_joke)
        workflow.add_node("improve_joke", improve_joke)
        workflow.add_node("polish_joke", polish_joke)
        workflow.add_edge(START, "generate_joke")
        workflow.add_conditional_edges(
            "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
        )
        workflow.add_edge("improve_joke", "polish_joke")
        workflow.add_edge("polish_joke", END)
        return workflow


# -- routing ------------------------------------------------------------------------------


class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        description="The next step in the routing process"
    )


class RouterInput(BaseModel):
    input: str


class RouterState(TypedDict, total=False):
    input: str
    decision: str
    output: str


class RouterAgent(BaseAgent):
    name = "router"
    mode = Mode.BOTH
    models = ("chat",)
    input_schema = RouterInput

    def build(self, ctx: BuildContext) -> StateGraph:
        llm = ctx.models["chat"]
        # function_calling, not the default json_schema: Ollama's `-cloud` models
        # ignore the `format` schema and answer in prose; tool calling is honored
        # by every tool-capable model (vLLM and Ollama alike).
        router = llm.with_structured_output(Route, method="function_calling")

        def llm_call_1(state: RouterState) -> dict[str, Any]:
            """Write a story"""
            return {"output": llm.invoke(state["input"]).text}

        def llm_call_2(state: RouterState) -> dict[str, Any]:
            """Write a joke"""
            return {"output": llm.invoke(state["input"]).text}

        def llm_call_3(state: RouterState) -> dict[str, Any]:
            """Write a poem"""
            return {"output": llm.invoke(state["input"]).text}

        def llm_call_router(state: RouterState) -> dict[str, Any]:
            """Route the input to the appropriate node"""
            decision = router.invoke(
                [
                    SystemMessage(
                        content="Route the input to story, joke, or poem "
                        "based on the user's request."
                    ),
                    HumanMessage(content=state["input"]),
                ]
            )
            return {"decision": decision.step}

        def route_decision(state: RouterState) -> str:
            if state["decision"] == "story":
                return "llm_call_1"
            if state["decision"] == "joke":
                return "llm_call_2"
            return "llm_call_3"

        builder = StateGraph(RouterState)
        builder.add_node("llm_call_1", llm_call_1)
        builder.add_node("llm_call_2", llm_call_2)
        builder.add_node("llm_call_3", llm_call_3)
        builder.add_node("llm_call_router", llm_call_router)
        builder.add_edge(START, "llm_call_router")
        builder.add_conditional_edges(
            "llm_call_router",
            route_decision,
            {"llm_call_1": "llm_call_1", "llm_call_2": "llm_call_2", "llm_call_3": "llm_call_3"},
        )
        for node in ("llm_call_1", "llm_call_2", "llm_call_3"):
            builder.add_edge(node, END)
        return builder


chain_agent = PromptChainAgent()
router_agent = RouterAgent()
