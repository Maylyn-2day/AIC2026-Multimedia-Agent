"""
Keyframe Card Streamlit Component.

Renders a single reranked keyframe result as an interactive card inside
the Streamlit dashboard.  Each card shows:

- Keyframe thumbnail with optional Grounding DINO bounding box overlay.
- Metadata row: video_id, frame_id, rerank_score, detected labels.
- Reasoning trace (collapsible expander) when available.
- Action buttons: "Find Similar" (image-to-image search) and "↔ Timeline".

Usage::

    from frontend.components.keyframe_card import render_keyframe_card
    from backend.schemas.rerank import RerankResultItem

    render_keyframe_card(
        item=result,
        image_path="/data/keyframes/L01_V001/001500.jpg",
        on_find_similar=lambda item: ...,
        on_expand_timeline=lambda item: ...,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

# Streamlit is imported lazily inside the render function so this module
# can be imported in test environments without a running Streamlit session.


def render_keyframe_card(
    item: object,
    image_path: Path | str,
    on_find_similar: Callable[[object], None] | None = None,
    on_expand_timeline: Callable[[object], None] | None = None,
    thumbnail_size: tuple[int, int] = (320, 180),
    show_bboxes: bool = True,
) -> None:
    """Render an interactive keyframe card in the Streamlit dashboard.

    The card is rendered using ``st.container()`` and is self-contained:
    it never mutates session state directly.  Instead it calls the
    provided callbacks when the user interacts with the action buttons,
    delegating state management to the parent page.

    Args:
        item: A ``RerankResultItem`` (or any object with ``video_id``,
            ``frame_id``, ``rerank_score``, ``grounding``, and optionally
            ``reasoning_trace`` attributes).
        image_path: Absolute or relative path to the keyframe JPEG.
        on_find_similar: Callback invoked with ``item`` when the user
            clicks "Find Similar".  If ``None`` the button is hidden.
        on_expand_timeline: Callback invoked with ``item`` when the user
            clicks "↔ Timeline".  If ``None`` the button is hidden.
        thumbnail_size: ``(width, height)`` for the displayed thumbnail.
        show_bboxes: Whether to overlay Grounding DINO bounding boxes.
    """
    import streamlit as st
    from PIL import Image, UnidentifiedImageError

    from frontend.utils.image_utils import create_thumbnail, draw_bboxes

    # Build a unique key prefix from the item's identity to avoid
    # Streamlit widget key collisions when multiple cards are rendered.
    video_id: str = getattr(item, "video_id", "unknown")
    frame_id: int = getattr(item, "frame_id", 0)
    rerank_score: float = getattr(item, "rerank_score", 0.0)
    grounding: list = getattr(item, "grounding", [])
    reasoning_trace: str | None = getattr(item, "reasoning_trace", None)
    key_prefix = f"card_{video_id}_{frame_id}"

    with st.container():
        # ── Card border via custom CSS ───────────────────────────────
        st.markdown(
            """
            <style>
            .kf-card {
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 8px;
                padding: 8px;
                margin-bottom: 8px;
                background: rgba(255,255,255,0.02);
            }
            </style>
            <div class="kf-card">
            """,
            unsafe_allow_html=True,
        )

        # ── Thumbnail ────────────────────────────────────────────────
        image_path = Path(image_path)
        img: Image.Image | None = None
        if image_path.exists():
            try:
                img = Image.open(image_path).convert("RGB")
            except (OSError, UnidentifiedImageError):
                img = None

        if img is not None:
            thumb = create_thumbnail(img, size=thumbnail_size)
            if show_bboxes and grounding:
                thumb = draw_bboxes(thumb, grounding)
            st.image(thumb, use_container_width=True)
        else:
            # Placeholder when image is missing.
            st.markdown(
                f"<div style='background:#222;height:{thumbnail_size[1]}px;"
                "display:flex;align-items:center;justify-content:center;"
                "color:#888;border-radius:4px;'>No image</div>",
                unsafe_allow_html=True,
            )

        # ── Metadata ─────────────────────────────────────────────────
        labels = ", ".join(g.label if hasattr(g, "label") else g["label"] for g in grounding)
        score_pct = f"{rerank_score:.4f}"

        col_meta, col_score = st.columns([3, 1])
        with col_meta:
            st.markdown(f"**{video_id}** · `f{frame_id}`")
            if labels:
                st.caption(f"🎯 {labels}")
        with col_score:
            st.metric(label="Score", value=score_pct, label_visibility="collapsed")

        # ── Reasoning trace (collapsible) ────────────────────────────
        if reasoning_trace:
            with st.expander("🧠 Reasoning", expanded=False):
                st.markdown(reasoning_trace)

        # ── Action buttons ───────────────────────────────────────────
        btn_cols = st.columns(2)
        if on_find_similar is not None:
            with btn_cols[0]:
                if st.button(
                    "🔍 Find Similar",
                    key=f"{key_prefix}_similar",
                    use_container_width=True,
                ):
                    on_find_similar(item)
        if on_expand_timeline is not None:
            with btn_cols[1]:
                if st.button(
                    "↔ Timeline",
                    key=f"{key_prefix}_timeline",
                    use_container_width=True,
                ):
                    on_expand_timeline(item)

        st.markdown("</div>", unsafe_allow_html=True)
