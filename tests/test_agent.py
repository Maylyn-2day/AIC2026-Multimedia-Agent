"""Tests for the structured, provider-independent Agent Router."""

from __future__ import annotations

import asyncio
import importlib
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from pydantic import ValidationError

from backend.schemas.agent import AgentPlan, FusionParams
from backend.schemas.submission import TaskType
from backend.services.agent import (
    AgentProviderError,
    AgentRouter,
    ConversationMemory,
    ConversationTurn,
    MockAgentProvider,
)


def plan_data(raw_query: str = "Find the red car", **overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "task_type": "KIS",
        "raw_query": raw_query,
        "decision_summary": "Retrieve frames matching the visual description.",
    }
    data.update(overrides)
    return data


def test_valid_kis_plan() -> None:
    router = AgentRouter(MockAgentProvider([plan_data()]))
    plan = asyncio.run(router.route("Find the red car", "session-1"))
    assert plan.task_type is TaskType.KIS
    assert plan.top_k == 100
    assert plan.events == []
    assert not plan.requires_rerank
    assert not plan.requires_temporal_alignment


def test_valid_vqa_and_qa_alias_require_rerank() -> None:
    router = AgentRouter(
        MockAgentProvider([plan_data("What color is the car?", task_type="Q&A", requires_rerank=True)])
    )
    plan = asyncio.run(router.route("What color is the car?", "session-1"))
    assert plan.task_type is TaskType.VQA
    assert plan.requires_rerank


@pytest.mark.parametrize("event_count", [1, 4])
def test_trake_supports_one_or_four_events(event_count: int) -> None:
    events = [{"order": index, "query": f"semantic event {index}"} for index in range(1, event_count + 1)]
    plan = AgentPlan.model_validate(
        plan_data(
            "A sequence of activities",
            task_type="TRAKE",
            requires_temporal_alignment=True,
            events=events,
        )
    )
    assert len(plan.events) == event_count


@pytest.mark.parametrize(
    "events",
    [
        [{"order": 1, "query": "first"}, {"order": 1, "query": "duplicate"}],
        [{"order": 2, "query": "later"}, {"order": 1, "query": "earlier"}],
        [{"order": 1, "query": "first"}, {"order": 3, "query": "gap"}],
    ],
)
def test_invalid_event_order_is_rejected(events: list[dict[str, object]]) -> None:
    with pytest.raises(ValidationError, match="exactly 1..N"):
        AgentPlan.model_validate(
            plan_data("Sequence", task_type="TRAKE", requires_temporal_alignment=True, events=events)
        )


@pytest.mark.parametrize("raw_query", ["", "   "])
def test_blank_raw_query_is_rejected(raw_query: str) -> None:
    router = AgentRouter(MockAgentProvider([]))
    with pytest.raises(ValidationError):
        asyncio.run(router.route(raw_query, "session"))


@pytest.mark.parametrize("session_id", ["", "   "])
def test_blank_session_id_is_rejected(session_id: str) -> None:
    router = AgentRouter(MockAgentProvider([]))
    with pytest.raises(ValidationError):
        asyncio.run(router.route("query", session_id))


@pytest.mark.parametrize("top_k", [0, 101, True])
def test_invalid_top_k_is_rejected(top_k: object) -> None:
    with pytest.raises(ValidationError):
        AgentPlan.model_validate(plan_data(top_k=top_k))


def test_default_and_valid_override_fusion_params() -> None:
    default = AgentPlan.model_validate(plan_data()).fusion_params
    overridden = AgentPlan.model_validate(
        plan_data(fusion_params={"k": 10, "weight_dense": 2.5, "weight_sparse": 0.25})
    ).fusion_params
    assert default == FusionParams(k=60, weight_dense=1.0, weight_sparse=1.0)
    assert overridden == FusionParams(k=10, weight_dense=2.5, weight_sparse=0.25)


@pytest.mark.parametrize(
    "fusion_params",
    [
        {"k": 0},
        {"k": True},
        {"weight_dense": -1},
        {"weight_sparse": float("nan")},
        {"weight_dense": float("inf")},
        {"weight_dense": "1.0"},
        {"weight_dense": 0, "weight_sparse": 0},
    ],
)
def test_invalid_fusion_params_are_rejected(fusion_params: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AgentPlan.model_validate(plan_data(fusion_params=fusion_params))


def test_plan_has_no_rrf_score_and_agent_module_does_not_import_fusion() -> None:
    plan = AgentPlan.model_validate(plan_data())
    agent_module = importlib.import_module("backend.services.agent")
    assert "rrf_score" not in AgentPlan.model_fields
    assert "weighted_rrf" not in vars(agent_module)
    assert "fusion" not in vars(agent_module)
    assert "rrf_score" not in plan.model_dump()


def test_clarification_plan_does_not_execute_search() -> None:
    provider = MockAgentProvider(
        [plan_data("Find it", clarification_question="Which person or object should be located?")]
    )
    router = AgentRouter(provider)
    plan = asyncio.run(router.route("Find it", "session"))
    assert plan.clarification_question is not None
    assert not plan.should_search
    assert len(provider.calls) == 1


def test_plan_without_clarification_is_searchable() -> None:
    assert AgentPlan.model_validate(plan_data()).should_search


def test_sessions_are_isolated() -> None:
    provider = MockAgentProvider([plan_data("first"), plan_data("second")])
    router = AgentRouter(provider)
    asyncio.run(router.route("first", "A"))
    asyncio.run(router.route("second", "B"))
    assert [turn.raw_query for turn in router.get_history("A")] == ["first"]
    assert [turn.raw_query for turn in router.get_history("B")] == ["second"]
    assert provider.calls[1]["history"] == []


def test_memory_limits_turns_and_clear_session() -> None:
    provider = MockAgentProvider([plan_data(str(index)) for index in range(3)])
    router = AgentRouter(provider, ConversationMemory(max_turns=2))
    for index in range(3):
        asyncio.run(router.route(str(index), "session"))
    assert [turn.raw_query for turn in router.get_history("session")] == ["1", "2"]
    router.clear_session("session")
    assert router.get_history("session") == []


def test_memory_evicts_least_recently_written_session() -> None:
    provider = MockAgentProvider([plan_data("A1"), plan_data("B1"), plan_data("A2"), plan_data("C1")])
    router = AgentRouter(provider, ConversationMemory(max_sessions=2))
    asyncio.run(router.route("A1", "A"))
    asyncio.run(router.route("B1", "B"))
    asyncio.run(router.route("A2", "A"))
    asyncio.run(router.route("C1", "C"))
    assert [turn.raw_query for turn in router.get_history("A")] == ["A1", "A2"]
    assert router.get_history("B") == []
    assert [turn.raw_query for turn in router.get_history("C")] == ["C1"]


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("max_turns", True),
        ("max_turns", 0),
        ("max_turns", -1),
        ("max_sessions", True),
        ("max_sessions", 0),
        ("max_sessions", -1),
    ],
)
def test_memory_bounds_reject_invalid_integers(parameter: str, value: object) -> None:
    with pytest.raises(ValueError, match=parameter):
        ConversationMemory(**{parameter: value})  # type: ignore[arg-type]


def test_concurrent_memory_writes_to_same_session_are_not_lost() -> None:
    memory = ConversationMemory(max_turns=100, max_sessions=2)

    def append(index: int) -> None:
        plan = AgentPlan.model_validate(plan_data(str(index)))
        memory.append_turn("shared", ConversationTurn(raw_query=str(index), plan=plan))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(50)))
    assert len(memory.get_history("shared")) == 50


def test_memory_is_deep_copied_on_write_and_read() -> None:
    fixture = plan_data(filters={"objects": ["car"]})
    original = deepcopy(fixture)
    router = AgentRouter(MockAgentProvider([fixture]))
    returned = asyncio.run(router.route("Find the red car", "session"))
    returned.filters.objects.append("mutated")  # type: ignore[union-attr]
    history = router.get_history("session")
    history[0].plan.filters.objects.append("also-mutated")  # type: ignore[union-attr]
    assert fixture == original
    assert router.get_history("session")[0].plan.filters.objects == ["car"]


def test_import_and_mock_provider_do_not_require_gemini_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    module = importlib.import_module("backend.services.agent")
    assert module.MockAgentProvider([]) is not None


def test_invalid_provider_data_has_clear_error() -> None:
    secret = "credential-secret-that-must-not-leak"
    router = AgentRouter(MockAgentProvider([{"task_type": "KIS", "credential": secret}]))
    with pytest.raises(AgentProviderError, match="invalid structured plan") as captured:
        asyncio.run(router.route("query", "session"))
    assert secret not in str(captured.value)
    assert captured.value.__cause__ is None


def test_provider_cannot_expose_chain_of_thought() -> None:
    leaked = plan_data(chain_of_thought="private reasoning")
    router = AgentRouter(MockAgentProvider([leaked]))
    with pytest.raises(AgentProviderError, match="invalid structured plan"):
        asyncio.run(router.route("Find the red car", "session"))


@pytest.mark.parametrize("task_type", ["KIS", "VQA"])
def test_non_trake_cannot_create_temporal_work(task_type: str) -> None:
    overrides: dict[str, object] = {
        "task_type": task_type,
        "requires_temporal_alignment": True,
        "events": [{"order": 1, "query": "unneeded"}],
    }
    if task_type == "VQA":
        overrides["requires_rerank"] = True
    with pytest.raises(ValidationError, match="non-TRAKE"):
        AgentPlan.model_validate(plan_data(**overrides))


@pytest.mark.parametrize("task_type", ["QNA", "VQ", "KISS", "TRAKEE"])
def test_near_match_task_types_are_rejected(task_type: str) -> None:
    with pytest.raises(ValidationError):
        AgentPlan.model_validate(plan_data(task_type=task_type))


def test_qa_alias_serializes_to_canonical_vqa() -> None:
    plan = AgentPlan.model_validate(plan_data(task_type="Q&A", requires_rerank=True))
    assert plan.model_dump(mode="json")["task_type"] == "VQA"
