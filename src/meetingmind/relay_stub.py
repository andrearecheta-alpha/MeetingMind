"""
relay_stub.py
-------------
Stub for the Sprint 9 cloud relay architecture.

Architecture (planned):
    The host laptop pushes output-only payloads (fact-check cards, key facts)
    to a lightweight cloud relay service.  Remote guests connect to the relay
    over HTTPS/WSS rather than directly to the laptop.

    * Audio and raw transcripts NEVER leave the local machine.
    * Only structured, anonymised output payloads are relayed.
    * The relay is stateless — it fans out to connected guest WebSockets
      and drops payloads when no guests are listening.

    This stub exists so that call-sites in main.py can be wired up now;
    the real implementation will replace this module in Sprint 9.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class RelayStub:
    """No-op relay that logs calls.  Replace with real implementation in S9."""

    def push_card(self, token: str, payload: dict) -> None:
        logger.info("RELAY STUB: push_card called — not implemented (token=%s)", token)

    def push_key_facts(self, token: str, payload: dict) -> None:
        logger.info("RELAY STUB: push_key_facts called — not implemented (token=%s)", token)

    def deactivate_session(self, token: str) -> None:
        logger.info("RELAY STUB: deactivate_session called — not implemented (token=%s)", token)
