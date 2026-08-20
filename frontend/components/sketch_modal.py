"""
Sketch Search Modal Component.

Renders a full sketch-input workflow inside the Streamlit dashboard:

1. ``streamlit-drawable-canvas`` provides the drawing surface.
2. A text prompt field and top-k slider refine the query.
3. "Search" button encodes the canvas to Base64, calls
   ``POST /v1/query/sketch``, and renders results via ``render_keyframe_card``.

Usage::

    from frontend.components.sketch_modal import render_sketch_modal

    render_sketch_modal(keyframes_dir="/data/keyframes")

The component is self-contained: it reads/writes only its own ``st.session_state``
keys (prefixed ``sketch_``) and never touches other page state directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Session-state key constants
# ---------------------------------------------------------------------------

_KEY_RESULTS = "sketch_results"
_KEY_QUERY = "sketch_query"
_KEY_SEARCHING = "sketch_searching"


def render_sketch_modal(
    keyframes_dir: str | Path = "/data/keyframes",
    api_base_url: str = "http://localhost:8000",
    default_stroke_width: int = 4,
    default_top_k: int = 50,
    canvas_height: int = 400,
    canvas_width: int = 700,
) -> None:
    """Render the sketch search panel in the active Streamlit page.

    Args:
        keyframes_dir: Root directory used by ``render_keyframe_card`` to
            resolve thumbnail paths.
        api_base_url: Base URL of the FastAPI backend.
        default_stroke_width: Initial pen width (pixels).
        default_top_k: Initial top-k slider value.
        canvas_height: Pixel height of the drawing canvas.
        canvas_width: Pixel width of the drawing canvas.
    """
    import streamlit as st
    from frontend.components.keyframe_card import render_keyframe_card
    from frontend.utils.image_utils import base64_encode

    # ── Initialise session state ─────────────────────────────────────
    if _KEY_RESULTS not in st.session_state:
        st.session_state[_KEY_RESULTS] = []
    if _KEY_QUERY not in st.session_state:
        st.session_state[_KEY_QUERY] = ""

    # ── Section header ───────────────────────────────────────────────
    st.markdown("### ✏️ Sketch Search")
    st.caption(
        "Draw a rough scene on the canvas below. "
        "Add an optional text prompt to guide interpretation."
    )

    # ── Canvas + controls layout ─────────────────────────────────────
    col_canvas, col_controls = st.columns([3, 1])

    with col_controls:
        stroke_width = st.slider(
            "Pen width",
            min_value=1,
            max_value=20,
            value=default_stroke_width,
            key="sketch_stroke_width",
        )
        stroke_color = st.color_picker(
            "Pen colour",
            value="#000000",
            key="sketch_stroke_color",
        )
        bg_color = st.color_picker(
            "Background",
            value="#FFFFFF",
            key="sketch_bg_color",
        )
        drawing_mode = st.selectbox(
            "Mode",
            options=["freedraw", "line", "rect", "circle"],
            index=0,
            key="sketch_drawing_mode",
        )
        top_k = st.slider(
            "Results (top-k)",
            min_value=5,
            max_value=100,
            value=default_top_k,
            step=5,
            key="sketch_top_k",
        )

    with col_canvas:
        canvas_result = _render_canvas(
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            bg_color=bg_color,
            drawing_mode=drawing_mode,
            height=canvas_height,
            width=canvas_width,
        )

    # ── Prompt & search button ───────────────────────────────────────
    prompt = st.text_input(
        "Text prompt (optional)",
        placeholder='e.g. "two people at a news desk"',
        key="sketch_prompt",
    )

    col_btn, col_clear = st.columns([2, 1])
    search_clicked = col_btn.button(
        "🔍 Search by Sketch",
        key="sketch_search_btn",
        type="primary",
        use_container_width=True,
    )
    clear_clicked = col_clear.button(
        "🗑️ Clear results",
        key="sketch_clear_btn",
        use_container_width=True,
    )

    if clear_clicked:
        st.session_state[_KEY_RESULTS] = []
        st.rerun()

    # ── Execute search ───────────────────────────────────────────────
    if search_clicked:
        if canvas_result is None or canvas_result.image_data is None:
            st.warning("Please draw something on the canvas first.")
        else:
            try:
                from PIL import Image
                import numpy as np

                # Convert canvas numpy array to PIL Image then to Base64.
                arr = canvas_result.image_data.astype(np.uint8)
                pil_img = Image.fromarray(arr, mode="RGBA").convert("RGB")
                sketch_b64 = base64_encode(pil_img, fmt="PNG")

                with st.spinner("Searching…"):
                    results = _call_sketch_api(
                        api_base_url=api_base_url,
                        sketch_base64=sketch_b64,
                        prompt=prompt,
                        top_k=top_k,
                    )
                st.session_state[_KEY_RESULTS] = results

            except Exception as exc:  # noqa: BLE001
                st.error(f"Search failed: {exc}")

    # ── Render results ───────────────────────────────────────────────
    results: list[dict[str, Any]] = st.session_state[_KEY_RESULTS]
    if results:
        st.markdown(f"**{len(results)} results**")
        cols_per_row = 4
        for row_start in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, result in enumerate(results[row_start: row_start + cols_per_row]):
                with cols[col_idx]:
                    video_id = result.get("video_id", "")
                    frame_id = int(result.get("frame_id", 0))
                    image_path = (
                        Path(keyframes_dir) / video_id / f"{frame_id:06d}.jpg"
                    )

                    # Build a minimal RerankResultItem-like object for the card.
                    card_item = _ResultProxy(
                        video_id=video_id,
                        frame_id=frame_id,
                        rerank_score=float(result.get("score", 0.0)),
                        original_score=float(result.get("score", 0.0)),
                    )
                    render_keyframe_card(
                        item=card_item,
                        image_path=image_path,
                    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_canvas(
    stroke_width: int,
    stroke_color: str,
    bg_color: str,
    drawing_mode: str,
    height: int,
    width: int,
) -> Any | None:
    """Render the drawable canvas, returning the canvas result or ``None``.

    Gracefully degrades if ``streamlit-drawable-canvas`` is not installed.
    """
    try:
        from streamlit_drawable_canvas import st_canvas
        return st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=height,
            width=width,
            drawing_mode=drawing_mode,
            key="sketch_canvas",
        )
    except ImportError:
        import streamlit as st
        st.info(
            "Install `streamlit-drawable-canvas` to enable the drawing surface: "
            "`pip install streamlit-drawable-canvas`"
        )
        return None


def _call_sketch_api(
    api_base_url: str,
    sketch_base64: str,
    prompt: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Call ``POST /v1/query/sketch`` and return the result list.

    Args:
        api_base_url: Backend URL (e.g. ``"http://localhost:8000"``).
        sketch_base64: Base64-encoded PNG of the canvas.
        prompt: Optional text prompt.
        top_k: Number of results to request.

    Returns:
        List of result dicts from ``data.results``.

    Raises:
        RuntimeError: If the API call fails or returns a non-success status.
    """
    import httpx

    url = f"{api_base_url.rstrip('/')}/v1/query/sketch"
    payload = {
        "sketch_base64": sketch_base64,
        "prompt": prompt,
        "top_k": top_k,
    }

    try:
        resp = httpx.post(url, json=payload, timeout=30.0)
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "success":
            raise RuntimeError(body.get("message", "Unknown API error"))
        return body.get("data", {}).get("results", [])
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"API error {exc.response.status_code}: {exc.response.text}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Cannot reach backend at {api_base_url}: {exc}") from exc


class _ResultProxy:
    """Minimal duck-typed object satisfying render_keyframe_card's interface."""

    def __init__(
        self,
        video_id: str,
        frame_id: int,
        rerank_score: float,
        original_score: float,
    ) -> None:
        self.video_id = video_id
        self.frame_id = frame_id
        self.rerank_score = rerank_score
        self.original_score = original_score
        self.grounding: list = []
        self.reasoning_trace: str | None = None
