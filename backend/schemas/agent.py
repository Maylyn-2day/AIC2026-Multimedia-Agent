"""Structured schemas for the provider-independent Agent Router."""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from backend.schemas.query import QueryFilters
from backend.schemas.submission import TaskType

StrictPositiveInt = Annotated[int, Field(strict=True, gt=0)]
StrictTopK = Annotated[int, Field(strict=True, ge=1, le=100)]


class FusionParams(BaseModel):
    """Parameters forwarded to hybrid search; the Agent never computes RRF."""

    model_config = ConfigDict(extra="forbid")

    k: StrictPositiveInt = 60
    weight_dense: float = Field(default=1.0, strict=True, ge=0, allow_inf_nan=False)
    weight_sparse: float = Field(default=1.0, strict=True, ge=0, allow_inf_nan=False)

    @field_validator("weight_dense", "weight_sparse", mode="before")
    @classmethod
    def reject_boolean_weights(cls, value: object) -> object:
        """Reject booleans even though Python treats them as numbers."""
        if isinstance(value, bool):
            raise ValueError("fusion weights must be finite numbers greater than or equal to 0")
        return value

    @model_validator(mode="after")
    def validate_weights(self) -> FusionParams:
        """Match the invariants enforced by the weighted RRF service."""
        if not math.isfinite(self.weight_dense) or not math.isfinite(self.weight_sparse):
            raise ValueError("fusion weights must be finite")
        if self.weight_dense == 0 and self.weight_sparse == 0:
            raise ValueError("weight_dense and weight_sparse cannot both be 0")
        return self


class TemporalEvent(BaseModel):
    """One semantic event in an arbitrary-length TRAKE sequence."""

    model_config = ConfigDict(extra="forbid")

    order: StrictPositiveInt
    query: str = Field(min_length=1, max_length=2000)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only event queries."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("event query must not be blank")
        return stripped


class AgentPlan(BaseModel):
    """Validated, safe-to-display routing decision produced by an agent provider.

    ``VQA`` is the canonical value for the competition's Q&A task. Input values
    ``Q&A`` and ``QA`` are accepted as aliases and normalized to ``VQA``.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    task_type: TaskType
    raw_query: str = Field(min_length=1, max_length=2000)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    fusion_params: FusionParams = Field(default_factory=FusionParams)
    top_k: StrictTopK = 100
    requires_rerank: bool = False
    requires_temporal_alignment: bool = False
    events: list[TemporalEvent] = Field(default_factory=list)
    clarification_question: str | None = Field(default=None, max_length=1000)
    decision_summary: str = Field(min_length=1, max_length=1000)

    @computed_field
    @property
    def should_search(self) -> bool:
        """Indicate whether downstream orchestration may execute this plan."""
        return self.clarification_question is None

    @field_validator("task_type", mode="before")
    @classmethod
    def normalize_qa_alias(cls, value: object) -> object:
        """Normalize human-facing Q&A spellings to canonical VQA."""
        if isinstance(value, str) and value.strip().upper() in {"Q&A", "QA"}:
            return TaskType.VQA
        return value

    @field_validator("raw_query", "decision_summary")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only required text fields."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be blank")
        return stripped

    @field_validator("clarification_question")
    @classmethod
    def optional_text_must_not_be_blank(cls, value: str | None) -> str | None:
        """Normalize absent clarification and reject a blank question."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("clarification_question must not be blank")
        return stripped

    @field_validator("decision_summary")
    @classmethod
    def reject_internal_reasoning_markers(cls, value: str) -> str:
        """Prevent common internal-prompt or chain-of-thought disclosures."""
        lowered = value.casefold()
        forbidden = ("chain-of-thought", "chain of thought", "internal prompt", "system prompt", "hidden reasoning")
        if any(marker in lowered for marker in forbidden):
            raise ValueError("decision_summary must not expose internal prompts or reasoning")
        return value

    @model_validator(mode="after")
    def validate_task_semantics(self) -> AgentPlan:
        """Keep temporal and reranking flags consistent with the selected task."""
        if self.task_type is TaskType.TRAKE:
            if not self.requires_temporal_alignment:
                raise ValueError("TRAKE requires temporal alignment")
            if not self.events:
                raise ValueError("TRAKE requires at least one temporal event")
            orders = [event.order for event in self.events]
            expected_orders = list(range(1, len(orders) + 1))
            if orders != expected_orders:
                raise ValueError("temporal event order values must be exactly 1..N")
        elif self.requires_temporal_alignment or self.events:
            raise ValueError("non-TRAKE plans cannot request temporal alignment or temporal events")

        if self.task_type is TaskType.VQA and not self.requires_rerank:
            raise ValueError("VQA requires reranking")
        return self


class AgentRequest(BaseModel):
    """Input accepted by the Agent Router service."""

    model_config = ConfigDict(extra="forbid")

    raw_query: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=200)
    task_type: TaskType | None = None

    @field_validator("raw_query", "session_id")
    @classmethod
    def value_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only queries and session identifiers."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped
