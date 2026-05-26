"""
Runtime-level types.

These are kept separate from `agensflow.schema` because they describe runtime
inputs (documents to retrieve from, configuration overrides) rather than the
orchestration-policy primitives that schema.py covers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    A retrievable document for the memory agent.

    For the demo, the memory agent receives a list of these and selects relevant
    facts. In production deployments, documents typically come from a vector
    store, BM25 index, or other retrieval system.
    """

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}
