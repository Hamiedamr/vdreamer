"""Protocol v1.0 constants, pinned against the PRD (§3.3 line 211, A22).

Delivery tuning the drivers and (later) the SSE transport consume; if any of
these drifts, every golden sequence and every client breaks — they are part
of the golden contract.
"""

from __future__ import annotations

from vdreamer.events import (
    DELTA_BATCH_INTERVAL_S,
    DELTA_BATCH_MAX_TOKENS,
    HEARTBEAT_INTERVAL_S,
    PROTOCOL_VERSION,
    SSE_BUFFERING_HEADER,
    SSE_BUFFERING_VALUE,
    EventEnvelope,
)


def test_protocol_constants_match_the_prd() -> None:
    assert PROTOCOL_VERSION == "1.0"
    assert HEARTBEAT_INTERVAL_S == 15.0
    assert DELTA_BATCH_INTERVAL_S == 0.05
    assert DELTA_BATCH_MAX_TOKENS == 32
    assert SSE_BUFFERING_HEADER == "X-Accel-Buffering"
    assert SSE_BUFFERING_VALUE == "no"
    # §3.3: the envelope field is literally "v" — {"v":"1.0", …}.
    assert EventEnvelope.model_fields["v"].default == PROTOCOL_VERSION
