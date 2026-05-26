"""
Runtime errors.

Kept narrow on purpose. The most important error class is
InvalidAgentOutputError, which is what fires when an agent returns content that
does not parse and validate against its declared output schema. We want this
to be a *typed* failure, not a string match in a try/except, because the
metric layer (Layer 2) will eventually count validation failures as a separate
class of event.
"""

from __future__ import annotations


class InvalidAgentOutputError(Exception):
    """Raised when an agent's LLM output cannot be parsed and validated.

    Carries enough context for both human debugging and (eventual) automated
    metric attribution: which agent, what was returned, why it failed.
    """

    def __init__(self, agent_name: str, raw_content: str, reason: str) -> None:
        self.agent_name = agent_name
        self.raw_content = raw_content
        self.reason = reason
        super().__init__(
            f"{agent_name} produced invalid output: {reason}. "
            f"First 200 chars of raw content: {raw_content[:200]!r}"
        )
