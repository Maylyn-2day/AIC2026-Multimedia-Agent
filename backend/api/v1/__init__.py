"""
API v1 Router Aggregator.

Mounts all endpoint routers under the ``/v1`` prefix and exports
a single ``v1_router`` for the FastAPI app to include.
"""

from fastapi import APIRouter

from backend.api.v1.agent import router as agent_router
from backend.api.v1.health import router as health_router
from backend.api.v1.image_query import router as image_query_router
from backend.api.v1.query import router as query_router
from backend.api.v1.rerank import router as rerank_router
from backend.api.v1.sketch import router as sketch_router
from backend.api.v1.submission import router as submission_router
from backend.api.v1.temporal import router as temporal_router

v1_router = APIRouter(prefix="/v1")

v1_router.include_router(health_router, tags=["Health"])
v1_router.include_router(agent_router, tags=["Agent"])
v1_router.include_router(query_router, tags=["Search"])
v1_router.include_router(rerank_router, tags=["Reranking"])
v1_router.include_router(image_query_router, tags=["Search"])
v1_router.include_router(sketch_router, tags=["Search"])
v1_router.include_router(temporal_router, tags=["Temporal"])
v1_router.include_router(submission_router, tags=["Submission"])
