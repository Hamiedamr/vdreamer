"""``ApprovalGate`` — the prebuilt approval subgraph (PRD §4.1).

``prepare → gate → {approve|edit: execute, reject: on_reject, feedback: replan}``

The gate node calls :func:`~vdreamer.context.require_approval` and routes on
the resolution with a ``Command``, so the agent's state needs no routing key:

- ``approve`` → ``execute``;
- ``edit`` → the edited payload (validated against ``edit_schema`` by the
  gateway and again in the gate) is written to state first, then ``execute``
  runs on the edited state (§4.3: "written to state before execution");
- ``reject`` → ``on_reject`` (chat; the conversation goes on). In a headless
  run the gateway ends the run ``rejected`` instead and nothing resumes;
- ``feedback`` → ``replan``, which reads the reviewer's text from
  ``state[decision_key]`` and loops back through ``prepare`` to the gate.

``execute``, ``on_reject`` and ``replan`` then end the subgraph (``replan``
loops). Pre-work belongs in ``prepare``: a resume restarts the gate node from
the top (§4.1), never ``prepare``.

``vdreamer validate`` finds every gate in an agent's graph (subgraphs
included) through its node metadata and flags offered actions with no node
wired to them (:meth:`ApprovalGate.unwired`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

import pydantic
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from vdreamer.context import DEFAULT_APPROVAL_ACTIONS, require_approval

__all__ = ["APPROVAL_GATE_METADATA_KEY", "ApprovalGate", "approval_gates"]

#: Node-metadata key marking an ``ApprovalGate``'s gate node (read by ``validate``).
APPROVAL_GATE_METADATA_KEY: Final[str] = "vdreamer_approval_gate"

#: The actions a gate routes; ``respond`` answers a tool, not a gate (§4.3).
_GATE_ACTIONS: Final[frozenset[str]] = frozenset({"approve", "edit", "reject", "feedback"})

#: The node each action routes to.
_ROUTE_NODES: Final[Mapping[str, str]] = {
    "approve": "execute",
    "edit": "execute",
    "reject": "on_reject",
    "feedback": "replan",
}

Node = Callable[..., Any]


class ApprovalGate:
    """A prebuilt ``prepare → gate → {execute, on_reject, replan}`` subgraph (§4.1).

    Args:
        state_schema: the state the subgraph runs on (usually the agent's own).
        kind: the interrupt's renderer hint, e.g. ``"sql_review"``.
        payload: ``state → payload`` — what the approver sees.
        execute: the node that runs once approved (or edited).
        prepare: pre-work before the gate (e.g. a dry run); optional.
        on_reject: the chat ``reject`` route.
        replan: the ``feedback`` route; reads the text from ``state[decision_key]``.
        actions: offered actions, default ``approve, reject, edit, feedback``.
        approver_role / scope / ttl_hours: the approval policy (§4.1, A16, A32).
        edit_schema: pydantic model for ``edit`` payloads; its fields are state keys.
        decision_key: a state key that receives the full resolution
            ``{action, payload?, text?}`` — required when ``feedback`` routes
            to ``replan``.
        apply_edit: ``(state, payload) → state update`` for ``edit``; the
            default writes the payload's keys into state.
        name: the gate node's name prefix in traces (cosmetic).
    """

    def __init__(
        self,
        state_schema: type[Any],
        *,
        kind: str,
        payload: Callable[[Any], Any],
        execute: Node | None,
        prepare: Node | None = None,
        on_reject: Node | None = None,
        replan: Node | None = None,
        actions: Sequence[str] = DEFAULT_APPROVAL_ACTIONS,
        approver_role: str | None = None,
        scope: str | None = None,
        edit_schema: type[pydantic.BaseModel] | None = None,
        ttl_hours: int = 24,
        decision_key: str | None = None,
        apply_edit: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        offered = tuple(actions)
        unknown = sorted(set(offered) - _GATE_ACTIONS)
        if not offered or unknown:
            raise ValueError(
                f"ApprovalGate actions must be a non-empty subset of {sorted(_GATE_ACTIONS)}"
                + (f"; unknown: {unknown}" if unknown else "")
            )
        if "feedback" in offered and replan is not None and decision_key is None:
            raise ValueError("an ApprovalGate with a replan route needs decision_key (the text)")
        self.state_schema = state_schema
        self.kind = kind
        self.payload = payload
        self.actions = offered
        self.approver_role = approver_role
        self.scope = scope
        self.edit_schema = edit_schema
        self.ttl_hours = ttl_hours
        self.decision_key = decision_key
        self.apply_edit = apply_edit
        self.nodes: dict[str, Node | None] = {
            "prepare": prepare,
            "execute": execute,
            "on_reject": on_reject,
            "replan": replan,
        }

    # -- routes -------------------------------------------------------------------

    def routes(self) -> dict[str, str | None]:
        """Each offered action → the node it routes to, ``None`` when unwired."""
        return {
            action: _ROUTE_NODES[action] if self.nodes[_ROUTE_NODES[action]] else None
            for action in self.actions
        }

    def unwired(self, *, headless: bool = False) -> list[str]:
        """Offered actions with no node wired (``validate``'s ``approval.routes``).

        A headless ``reject`` needs no route: the run ends ``rejected`` (§4.3).
        """
        return [
            action
            for action, node in self.routes().items()
            if node is None and not (headless and action == "reject")
        ]

    def metadata(self) -> dict[str, Any]:
        """The JSON-safe facts ``validate`` reads off the gate node."""
        return {
            "kind": self.kind,
            "actions": list(self.actions),
            "routes": self.routes(),
            "approver_role": self.approver_role,
        }

    # -- the graph --------------------------------------------------------------------

    def _gate(self, state: Any) -> Command[str]:
        decision = require_approval(
            self.kind,
            self.payload(state),
            self.actions,
            approver_role=self.approver_role,
            scope=self.scope,
            edit_schema=self.edit_schema,
            ttl_hours=self.ttl_hours,
        )
        action = decision.get("action") if isinstance(decision, Mapping) else None
        update: dict[str, Any] = {}
        if self.decision_key is not None:
            update[self.decision_key] = decision
        if action == "edit" and isinstance(decision, Mapping):
            edited = decision.get("payload") or {}
            update.update(
                self.apply_edit(state, edited) if self.apply_edit is not None else dict(edited)
            )
        target = _ROUTE_NODES.get(str(action))
        goto = target if target is not None and self.nodes[target] is not None else END
        return Command(goto=goto, update=update)

    def build(self) -> StateGraph[Any]:
        """The uncompiled subgraph (return it from ``build()``, or compile and nest it)."""
        graph: StateGraph[Any] = StateGraph(self.state_schema)
        wired = {name: node for name, node in self.nodes.items() if node is not None}
        destinations = tuple(
            name for name in ("execute", "on_reject", "replan") if name in wired
        ) + (END,)
        graph.add_node(
            "gate",
            self._gate,
            destinations=destinations,
            metadata={APPROVAL_GATE_METADATA_KEY: self.metadata()},
        )
        for name, node in wired.items():
            graph.add_node(name, node)
        entry = "prepare" if "prepare" in wired else "gate"
        graph.add_edge(START, entry)
        if entry == "prepare":
            graph.add_edge("prepare", "gate")
        for name in ("execute", "on_reject"):
            if name in wired:
                graph.add_edge(name, END)
        if "replan" in wired:
            graph.add_edge("replan", entry)
        return graph

    def compile(self, **kwargs: Any) -> Any:
        """``build().compile(**kwargs)`` — nest the result as a node of a parent graph."""
        return self.build().compile(**kwargs)


def approval_gates(compiled: Any) -> list[dict[str, Any]]:
    """The gate metadata of every ``ApprovalGate`` in a compiled graph and its subgraphs."""
    graphs = [compiled]
    try:
        graphs += [sub for _ns, sub in compiled.get_subgraphs(recurse=True)]
    except Exception:  # not a Pregel graph
        pass
    found: list[dict[str, Any]] = []
    for graph in graphs:
        builder = getattr(graph, "builder", None)
        for spec in (getattr(builder, "nodes", None) or {}).values():
            metadata = getattr(spec, "metadata", None) or {}
            gate = metadata.get(APPROVAL_GATE_METADATA_KEY)
            if isinstance(gate, Mapping) and gate not in found:
                found.append(dict(gate))
    return found
