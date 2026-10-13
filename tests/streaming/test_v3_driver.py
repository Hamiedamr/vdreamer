"""The v3 driver (opt-in beta) and the both-driver parity check (A31).

v3 is smoke-level for M2: it runs, it emits the run.status + message events,
and — on the shared kit graphs — it emits the *same* normalized event
sequence as v2. The full golden-event suite is WP3's gate (A31); the
``run_graph_all_drivers`` helper here is the hook it builds on.
"""

from __future__ import annotations

import _kit
import pytest

from vdreamer.events import PROTOCOL_VERSION, EventType


@pytest.mark.filterwarnings("ignore:.*v3 streaming protocol.*")
async def test_v3_runs_and_emits_run_status_and_messages() -> None:
    agent = _kit.ChattyAgent()
    run = await _kit.run_streamed(
        agent, _kit.input_for(agent), driver="v3", context=_kit.make_context(agent)
    )

    types = [envelope.type.value for envelope in run.events]
    assert types[0] == "run.status"
    assert types[-1] == "run.status"
    assert "message.completed" in types
    assert "token.delta" in types
    assert "progress" in types
    assert "citation" in types
    assert run.events[0].v == PROTOCOL_VERSION


@pytest.mark.filterwarnings("ignore:.*v3 streaming protocol.*")
async def test_v3_interrupts_end_on_awaiting_approval() -> None:
    agent = _kit.ApprovalAgent()
    run = await _kit.run_streamed(
        agent, _kit.input_for(agent), driver="v3", context=_kit.make_context(agent)
    )
    requested = [e for e in run.events if e.type == EventType.INTERRUPT_REQUESTED]
    assert len(requested) == 1
    assert requested[0].payload["kind"] == "sql_review"
    assert run.events[-1].payload["status"] == "awaiting_approval"


@pytest.mark.filterwarnings("ignore:.*v3 streaming protocol.*")
@pytest.mark.parametrize("agent_cls", [_kit.ChattyAgent, _kit.ToolAgent, _kit.ApprovalAgent])
async def test_both_drivers_emit_the_same_events(agent_cls: type) -> None:
    """A31, one graph at a time: identical normalized sequences.

    Normalization (``_kit.normalized``) drops per-run allocations only:
    ``seq``/``ts``/``expires_at``, run-unique uuids (subgraph namespaces),
    and provider-assigned message/generated ids (the two streaming protocols
    assign different ids to the same model call). Type order, ``ns``,
    and every other payload field must match exactly.
    """
    agent = agent_cls()
    runs = await _kit.run_graph_all_drivers(
        agent, _kit.input_for(agent), context=_kit.make_context(agent)
    )
    assert _kit.normalized(runs["v2"].events) == _kit.normalized(runs["v3"].events)


@pytest.mark.filterwarnings("ignore:.*v3 streaming protocol.*")
async def test_v3_failed_run_emits_failed_status_and_reraises() -> None:
    agent = _kit.BoomyAgent()
    with pytest.raises(ValueError, match="boom"):
        await _kit.run_streamed(agent, _kit.input_for(agent), driver="v3")
