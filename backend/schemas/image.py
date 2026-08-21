"""
Image Query & Sketch Schemas.

Request models for:
- ``POST /v1/query/image-example`` — Image-to-image similarity search.
- ``POST /v1/query/sketch`` — Sketch-to-image search via ControlNet.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImageQueryRequest(BaseModel):
    """
    Request body for ``POST /v1/query/image-example``.

    Used when a user clicks on a keyframe to find visually similar frames
    (exploitation / re-query mechanism).
    """

    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded image data for query-by-example",
    )
    video_id: str | None = Field(
        default=None,
        description="Source video ID of the reference keyframe",
    )
    frame_id: int | None = Field(
        default=None,
        description="Source frame ID of the reference keyframe",
    )
    top_k: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of similar results to return",
    )


class SketchQueryRequest(BaseModel):
    """
    Request body for ``POST /v1/query/sketch``.

    Receives a Base64-encoded sketch from the Streamlit canvas,
    processes it through ControlNet / SDXL-Turbo to generate a
    feature vector, then performs similarity search.
    """

    sketch_base64: str = Field(
        ...,
        description="Base64-encoded sketch image from the drawing canvas",
    )
    prompt: str = Field(
        default="",
        description="Optional text prompt to guide sketch interpretation",
    )
    top_k: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of results to return",
    )
