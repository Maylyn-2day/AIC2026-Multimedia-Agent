"""Provider-independent Agent Router and bounded conversational memory."""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from threading import RLock
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, ValidationError

from backend.schemas.agent import AgentPlan, AgentRequest
from backend.schemas.submission import TaskType


class AgentProviderError(ValueError):
    """Raised when an Agent provider fails to return a valid structured plan."""


class ConversationTurn(BaseModel):
    """Safe conversational context retained for one completed routing turn."""

    model_config = ConfigDict(extra="forbid")

    raw_query: str
    plan: AgentPlan


@runtime_checkable
class AgentProvider(Protocol):
    """Abstraction implemented by deterministic and future hosted providers.

    Implementations must be genuinely asynchronous or offload blocking SDK and
    network calls to a worker thread so they never block the event loop.
    """

    async def create_plan(
        self,
        *,
        raw_query: str,
        session_id: str,
        history: Sequence[ConversationTurn],
        task_type: TaskType | None = None,
    ) -> AgentPlan | Mapping[str, object]:
        """Return structured plan data without raw chain-of-thought."""
        ...


class ConversationMemory:
    """Thread-safe, bounded, process-local conversation storage.

    Storage is intentionally local to one Python process and is not shared
    between Uvicorn workers. Sessions are evicted deterministically by least
    recent write when ``max_sessions`` is reached. A distributed deployment
    must provide an external memory implementation in a later integration.
    """

    def __init__(self, max_turns: int = 10, max_sessions: int = 1000) -> None:
        if type(max_turns) is not int or max_turns <= 0:
            raise ValueError("max_turns must be an integer greater than 0")
        if type(max_sessions) is not int or max_sessions <= 0:
            raise ValueError("max_sessions must be an integer greater than 0")
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, list[ConversationTurn]] = OrderedDict()
        self._lock = RLock()

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        """Return a deep copy of one session's history."""
        normalized = _validate_session_id(session_id)
        with self._lock:
            return deepcopy(self._sessions.get(normalized, []))

    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        """Append a copied turn and evict the oldest turn beyond the limit."""
        normalized = _validate_session_id(session_id)
        with self._lock:
            if normalized not in self._sessions and len(self._sessions) >= self._max_sessions:
                self._sessions.popitem(last=False)
            history = self._sessions.setdefault(normalized, [])
            history.append(deepcopy(turn))
            del history[: max(0, len(history) - self._max_turns)]
            self._sessions.move_to_end(normalized)

    def clear_session(self, session_id: str) -> None:
        """Remove all retained context for one session only."""
        normalized = _validate_session_id(session_id)
        with self._lock:
            self._sessions.pop(normalized, None)


class MockAgentProvider:
    """Deterministic provider backed only by caller-supplied structured fixtures."""

    def __init__(self, responses: Sequence[AgentPlan | Mapping[str, object]]) -> None:
        self._responses = deepcopy(list(responses))
        self._next_response = 0
        self.calls: list[dict[str, object]] = []

    async def create_plan(
        self,
        *,
        raw_query: str,
        session_id: str,
        history: Sequence[ConversationTurn],
        task_type: TaskType | None = None,
    ) -> AgentPlan | Mapping[str, object]:
        """Return the next fixture and record copied public call inputs."""
        self.calls.append(
            {
                "raw_query": raw_query,
                "session_id": session_id,
                "history": deepcopy(list(history)),
                "task_type": task_type,
            }
        )
        if self._next_response >= len(self._responses):
            raise AgentProviderError("mock provider has no response configured for this call")
        response = deepcopy(self._responses[self._next_response])
        self._next_response += 1
        return response


class LocalRuleBasedAgentProvider:
    """Deterministic offline fallback using small, imperfect heuristics."""

    _temporal_pattern = re.compile(
        r"\b(?:trước khi|sau khi|rồi|tiếp theo|trước đó|sau đó|before|after|then|next)\b",
        flags=re.IGNORECASE,
    )
    _question_pattern = re.compile(
        r"\b(?:ai|gì|nào|bao nhiêu|màu gì|ở đâu|khi nào|who|what|which|how many|where|when)\b",
        flags=re.IGNORECASE,
    )

    async def create_plan(
        self,
        *,
        raw_query: str,
        session_id: str,
        history: Sequence[ConversationTurn],
        task_type: TaskType | None = None,
    ) -> AgentPlan:
        """Classify a trimmed query without network or hidden reasoning."""
        del session_id, history
        query = raw_query.strip()
        event_queries = [part.strip(" ,.;:") for part in self._temporal_pattern.split(query) if part.strip(" ,.;:")]
        selected_task = task_type
        if selected_task is None and self._temporal_pattern.search(query) and len(event_queries) >= 2:
            selected_task = TaskType.TRAKE
        elif selected_task is None and "?" in query and self._question_pattern.search(query):
            selected_task = TaskType.VQA
        elif selected_task is None:
            selected_task = TaskType.KIS

        if selected_task is TaskType.TRAKE:
            return AgentPlan(
                task_type=TaskType.TRAKE,
                raw_query=query,
                requires_temporal_alignment=True,
                events=[{"order": index, "query": event} for index, event in enumerate(event_queries, start=1)],
                decision_summary="Classified as TRAKE; search ordered events with temporal alignment.",
            )
        if selected_task is TaskType.VQA:
            return AgentPlan(
                task_type=TaskType.VQA,
                raw_query=query,
                requires_rerank=True,
                decision_summary="Classified as VQA; retrieve candidates and verify the answer with reranking.",
            )
        return AgentPlan(
            task_type=TaskType.KIS,
            raw_query=query,
            decision_summary="Classified as KIS; retrieve matching video frames.",
        )


class AgentRouter:
    """Validate input, delegate planning, and retain bounded session context."""

    def __init__(self, provider: AgentProvider, memory: ConversationMemory | None = None) -> None:
        self._provider = provider
        self._memory = memory if memory is not None else ConversationMemory()

    async def route(self, raw_query: str, session_id: str, task_type: TaskType | None = None) -> AgentPlan:
        """Create a validated plan without executing search, fusion, or reranking."""
        request = AgentRequest(raw_query=raw_query, session_id=session_id, task_type=task_type)
        history = self._memory.get_history(request.session_id)
        try:
            provider_output = await self._provider.create_plan(
                raw_query=request.raw_query,
                session_id=request.session_id,
                history=history,
                task_type=request.task_type,
            )
            if isinstance(provider_output, AgentPlan):
                plan = provider_output.model_copy(deep=True)
            elif isinstance(provider_output, Mapping):
                plan = AgentPlan.model_validate(provider_output)
            else:
                raise AgentProviderError("agent provider must return AgentPlan or mapping data")
        except ValidationError:
            raise AgentProviderError("agent provider returned an invalid structured plan") from None

        if plan.raw_query != request.raw_query:
            raise AgentProviderError("agent provider plan raw_query must match the requested query")
        self._memory.append_turn(request.session_id, ConversationTurn(raw_query=request.raw_query, plan=plan))
        return plan.model_copy(deep=True)

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        """Return copied conversational context for a session."""
        return self._memory.get_history(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear conversational context for a session."""
        self._memory.clear_session(session_id)


def _validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must not be blank")
    normalized = session_id.strip()
    if len(normalized) > 200:
        raise ValueError("session_id must contain at most 200 characters")
    return normalized
