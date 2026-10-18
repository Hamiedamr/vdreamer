"""Lesson 26: pausing for information and for authority — and not pausing when headless."""

from __future__ import annotations

from typing import Any, TypedDict

import httpx
from _tutorial_kit import TOKEN, served, vdreamer_run, wait_status
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from vdreamer_tutorials.t26_asking_for_input.agent import MANAGER_ROLE, agent
from vdreamer_tutorials.testing import AgentSession

from vdreamer import BaseAgent, BuildContext

CLAIM = {"message": "My tent pole broke, order A1001"}


async def test_a_missing_detail_is_asked_for() -> None:
    session = AgentSession(agent)
    first = await session.run(CLAIM)

    (question,) = first.interrupts
    assert question["kind"] == "claim_details" and question["actions"] == ["respond"]
    assert "serial number" in question["payload"]["question"]

    second = await session.resume({"action": "respond", "text": "it says sn-123456",
                                   "resolved_by": "ada"})
    (review,) = second.interrupts
    assert review["kind"] == "warranty_replacement"
    assert review["payload"]["serial"] == "SN-123456"


async def test_an_invalid_answer_is_asked_again_then_the_run_gives_up() -> None:
    session = AgentSession(agent)
    await session.run(CLAIM)
    again = await session.resume({"action": "respond", "text": "no idea", "resolved_by": "ada"})
    assert again.interrupts[0]["payload"]["question"].startswith("That did not look like")
    done = await session.resume({"action": "respond", "text": "still no", "resolved_by": "ada"})
    assert done.status == "completed" and "need the product's serial number" in done.answer


async def test_the_replacement_waits_for_a_manager() -> None:
    session = AgentSession(agent)
    first = await session.run({"message": "Order A1001, serial SN-654321, pole broke"})
    (review,) = first.interrupts  # no question: the serial was in the message
    assert review["kind"] == "warranty_replacement"
    state = await session.state()
    assert state.interrupts[0].value["approver_role"] == MANAGER_ROLE

    done = await session.resume({"action": "approve", "resolved_by": "sam"})
    assert done.answer == ("Your claim for A1001 was approved by sam: a replacement is on "
                           "its way.")


async def test_out_of_warranty_needs_no_one() -> None:
    result = await AgentSession(agent).run({"message": "Order A1002, SN-111111, zip broke"})
    assert result.interrupts == [] and "out of warranty" in result.answer


async def test_a_headless_run_never_pauses_for_the_customer() -> None:
    result = await AgentSession(agent).run(CLAIM, headless=True)
    assert result.interrupts == []
    assert "need the product's serial number" in result.answer


def test_the_customer_answers_and_only_a_manager_decides(monkeypatch: Any) -> None:
    with served([agent]) as url:
        # Ada answers the question, then tries to approve her own claim.
        ada = vdreamer_run(url, monkeypatch, "respond", "it says SN-123456", "approve",
                           agent=agent.name, user="ada", once=True, color=False,
                           message=CLAIM["message"])
        sam = vdreamer_run(url, monkeypatch, "approve", agent=agent.name, user="sam",
                           roles=(MANAGER_ROLE,), inbox=True, color=False)
        after = vdreamer_run(url, monkeypatch, agent=agent.name, user="ada",
                             thread=ada.thread_id(), once=True, color=False)
    assert "[interrupt resolved: respond by ada]" in ada.out
    assert "403 forbidden_role: resolving needs the role 'warranty_manager'" in ada.out
    assert sam.code == 0, sam.out
    assert "[interrupt resolved: approve by sam]" in after.out
    assert "approved by sam: a replacement is on its way." in after.out


class _AskState(TypedDict, total=False):
    answer: str


class RawInterruptAgent(BaseAgent):
    """LangGraph's own ``interrupt()`` — no vdreamer helper — resolved through the gateway."""

    name = "t26-raw-interrupt"

    def build(self, ctx: BuildContext) -> StateGraph[Any]:
        def ask(state: _AskState) -> dict[str, Any]:
            return {"answer": interrupt({"question": "Your postcode?"})}

        graph: StateGraph[Any] = StateGraph(_AskState)
        graph.add_node("ask", ask)
        graph.add_edge(START, "ask")
        graph.add_edge("ask", END)
        return graph


def test_a_raw_interrupt_is_resumed_with_the_text_verbatim() -> None:
    raw = RawInterruptAgent()
    headers = {"Authorization": f"Bearer {TOKEN}", "X-User-Id": "alice"}  # as wait_status
    with served([raw]) as url, httpx.Client(base_url=f"{url}/api/v1", headers=headers) as api:
        thread = api.post(f"/agents/{raw.name}/threads").json()["thread_id"]
        run = api.post(f"/threads/{thread}/runs", json={"input": {}}).json()["run_id"]
        wait_status(url, run, "awaiting_approval")
        (pending,) = api.get("/interrupts").json()["interrupts"]
        api.post(f"/interrupts/{pending['interrupt_id']}/resolve",
                 json={"action": "respond", "text": "10115",
                       "expected_version": pending["version"]}).raise_for_status()
        wait_status(url, run, "completed")
        values = api.get(f"/threads/{thread}").json()["values"]
    assert pending["kind"] == "generic"
    assert values == {"answer": "10115"}
