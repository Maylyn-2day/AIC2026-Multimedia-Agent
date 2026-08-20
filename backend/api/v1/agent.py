"""Agent routing endpoints backed by the deterministic local provider."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Header

from backend.core.exceptions import InvalidQueryError
from backend.schemas.agent import AgentRequest
from backend.schemas.common import BaseResponse
from backend.services.agent import AgentProviderError, AgentRouter, ConversationMemory, LocalRuleBasedAgentProvider

router = APIRouter()
agent_router = AgentRouter(LocalRuleBasedAgentProvider(), ConversationMemory(max_turns=10, max_sessions=1000))
SessionHeader = Annotated[
    str,
    Header(alias="X-Session-ID", min_length=1, max_length=200, pattern=r".*\S.*"),
]


@router.post("/agent/route", response_model=BaseResponse, summary="Create an Agent routing plan")
async def route_agent(
    request: AgentRequest,
    x_session_id: SessionHeader,
) -> BaseResponse:
    """Return a public plan without executing search or exposing reasoning."""
    start = time.perf_counter()
    if request.session_id != x_session_id.strip():
        raise InvalidQueryError("body session_id must match X-Session-ID")
    try:
        plan = await agent_router.route(request.raw_query, x_session_id, request.task_type)
    except (AgentProviderError, ValueError) as error:
        raise InvalidQueryError(str(error)) from None
    elapsed = time.perf_counter() - start
    return BaseResponse(
        status="success",
        data=plan.model_dump(mode="json"),
        message="Agent routing plan created",
        execution_time=f"{elapsed:.3f}s",
        agent_reasoning=None,
    )


@router.delete("/agent/session/{session_id}", response_model=BaseResponse, summary="Clear one Agent session")
async def clear_agent_session(
    session_id: str,
    x_session_id: SessionHeader,
) -> BaseResponse:
    """Clear only the caller's matching process-local session."""
    start = time.perf_counter()
    if session_id != x_session_id.strip():
        raise InvalidQueryError("path session_id must match X-Session-ID")
    try:
        agent_router.clear_session(session_id)
    except ValueError as error:
        raise InvalidQueryError(str(error)) from None
    elapsed = time.perf_counter() - start
    return BaseResponse(
        status="success",
        data={"cleared": True, "session_id": session_id},
        message="Agent session cleared",
        execution_time=f"{elapsed:.3f}s",
        agent_reasoning=None,
    )
