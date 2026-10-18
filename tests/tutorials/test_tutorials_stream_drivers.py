"""The tutorials under both stream drivers (``VDREAMER_STREAM_DRIVER``: v2 default, v3 opt-in).

vdreamer promises identical protocol events from both drivers. Writing the
series found one difference, recorded here as a strict ``xfail`` so it stays
visible and the test starts failing (as XPASS) the day it is fixed: on the
second run of a thread, the v3 driver re-emits the previous turn's AI message
as a fresh ``message.completed`` (and emits no ``token.delta`` for a
non-streaming model). The tutorials themselves run on the default, v2.
"""

from __future__ import annotations

import pytest
from vdreamer_tutorials.t01_serving_an_agent.agent import agent
from vdreamer_tutorials.testing import AgentSession


@pytest.mark.parametrize(
    "driver",
    [
        "v2",
        pytest.param("v3", marks=pytest.mark.xfail(
            strict=True, reason="v3 replays earlier turns' messages on a thread's next run"
        )),
    ],
)
async def test_a_second_turn_emits_only_its_own_message(
    driver: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VDREAMER_STREAM_DRIVER", driver)
    session = AgentSession(agent)
    await session.run("first")
    second = await session.run("second")

    assert [p["text"] for p in second.payloads("message.completed")] == [
        "You said: second (message 2 in this thread)"
    ]
