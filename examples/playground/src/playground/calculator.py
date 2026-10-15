"""The LangGraph quickstart calculator, both ways, served by vdreamer.

Source: https://docs.langchain.com/oss/python/langgraph/quickstart — the Graph
API and the Functional API versions, kept as close to the docs as possible.
What changed to serve them:

- The model is ``ctx.models["chat"]`` (an alias from ``models.yaml``) instead
  of ``init_chat_model("claude-…")``: vdreamer never talks to cloud providers.
- ``build()`` returns the graph; vdreamer compiles it with its checkpointer.
- The functional ``@entrypoint`` receives the run input, ``{"messages": [...]}``,
  rather than a bare list, and gets ``ctx.checkpointer`` so threads persist.
- Tools are governed (``ctx.tools.govern``) so ``tool.call``/``tool.result``
  events reach the client. The docs' plain ``@tool`` functions are unchanged.

::

    vdreamer dev --agent playground.calculator:graph_agent --models config/models.ollama.yaml
    vdreamer run calculator --message "Add 3 and 4, then multiply by 10."
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from langchain.messages import AnyMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_core.messages import BaseMessage, ToolCall
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from vdreamer import BaseAgent, BuildContext, Mode

SYSTEM = "You are a helpful assistant tasked with performing arithmetic on a set of inputs."


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


class CalculatorAgent(BaseAgent):
    """Graph API version (``llm_call`` ⇄ ``tool_node``)."""

    name = "calculator"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> StateGraph:
        tools = ctx.tools.govern([add, multiply, divide])
        tools_by_name = {t.name: t for t in tools}
        model_with_tools = ctx.models["chat"].bind_tools(tools)

        def llm_call(state: MessagesState) -> dict[str, Any]:
            """LLM decides whether to call a tool or not"""
            return {
                "messages": [
                    model_with_tools.invoke([SystemMessage(content=SYSTEM)] + state["messages"])
                ],
                "llm_calls": state.get("llm_calls", 0) + 1,
            }

        def tool_node(state: MessagesState) -> dict[str, Any]:
            """Performs the tool call"""
            result = []
            for tool_call in state["messages"][-1].tool_calls:  # type: ignore[union-attr]
                observation = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
            return {"messages": result}

        def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
            last_message = state["messages"][-1]
            if getattr(last_message, "tool_calls", None):
                return "tool_node"
            return END

        builder = StateGraph(MessagesState)
        builder.add_node("llm_call", llm_call)
        builder.add_node("tool_node", tool_node)
        builder.add_edge(START, "llm_call")
        builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
        builder.add_edge("tool_node", "llm_call")
        return builder


class FunctionalCalculatorAgent(BaseAgent):
    """Functional API version (``@entrypoint`` + ``@task``)."""

    name = "calculator-functional"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> Any:
        tools = ctx.tools.govern([add, multiply, divide])
        tools_by_name = {t.name: t for t in tools}
        model_with_tools = ctx.models["chat"].bind_tools(tools)

        @task
        def call_llm(messages: list[BaseMessage]) -> BaseMessage:
            """LLM decides whether to call a tool or not"""
            return model_with_tools.invoke([SystemMessage(content=SYSTEM)] + messages)  # type: ignore[no-any-return]

        @task
        def call_tool(tool_call: ToolCall) -> ToolMessage:
            """Performs the tool call"""
            return tools_by_name[tool_call["name"]].invoke(tool_call)  # type: ignore[no-any-return]

        @entrypoint(checkpointer=ctx.checkpointer)
        def calculator(inputs: dict[str, Any]) -> dict[str, Any]:
            messages = add_messages([], inputs["messages"])
            model_response = call_llm(messages).result()
            while model_response.tool_calls:  # type: ignore[attr-defined]
                futures = [call_tool(tc) for tc in model_response.tool_calls]  # type: ignore[attr-defined]
                messages = add_messages(messages, [model_response, *[f.result() for f in futures]])
                model_response = call_llm(messages).result()
            return {"messages": add_messages(messages, model_response)}

        return calculator


graph_agent = CalculatorAgent()
functional_agent = FunctionalCalculatorAgent()
