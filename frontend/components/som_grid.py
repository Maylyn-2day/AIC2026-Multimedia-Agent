"""
SOM Grid Component.

Renders the Video-Grouped Grid Layout in the center main area
of the dashboard. Groups keyframe results by video_id and displays
them as interactive cards with metadata overlays.
"""

from __future__ import annotations

from itertools import groupby
from operator import itemgetter

import streamlit as st


def render_som_grid(results_data: dict | None = None) -> None:
    """
    Render the results grid in the center main area.

    Groups keyframe results by ``video_id`` and displays them
    in a responsive grid layout. Each card shows the thumbnail,
    score, and metadata.

    Args:
        results_data: The ``data`` field from a query response,
            containing ``results``, ``total``, ``som_coords``, etc.
    """
    if results_data is None:
        results_data = st.session_state.get("latest_results", None)

    if not results_data or not results_data.get("results"):
        st.markdown(
            """
            <div style="text-align: center; padding: 60px 20px; color: #888;">
                <h3>🔍 No Results Yet</h3>
                <p>Use the AI Agent in the sidebar to search for keyframes.</p>
                <p style="font-size: 0.85em; color: #666;">
                    Try: "Người phụ nữ mặc áo đỏ tại HTV9"
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    results = results_data["results"]
    total = results_data.get("total", len(results))

    # Header with result count
    st.markdown(f"### 📊 Search Results ({total} keyframes)")

    # Group by video_id for Video-Grouped Grid Layout
    sorted_results = sorted(results, key=itemgetter("video_id"))

    for video_id, group in groupby(sorted_results, key=itemgetter("video_id")):
        frames = list(group)

        # Video group header
        st.markdown(f"#### 🎬 `{video_id}` — {len(frames)} frame(s)")

        # Render frames in a responsive grid (max 4 columns)
        cols = st.columns(min(len(frames), 4))

        for idx, frame in enumerate(frames):
            col = cols[idx % len(cols)]

            with col:
                # Frame card
                score = frame.get("score", 0.0)
                frame_id = frame.get("frame_id", "?")
                metadata = frame.get("metadata", {})

                # Score badge color
                if score >= 0.9:
                    badge_color = "#10b981"  # green
                elif score >= 0.7:
                    badge_color = "#f59e0b"  # amber
                else:
                    badge_color = "#ef4444"  # red

                # Thumbnail placeholder (since we don't have real images yet)
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #333;
                        border-radius: 8px;
                        padding: 12px;
                        margin-bottom: 8px;
                        background: #1a1a2e;
                    ">
                        <div style="
                            background: #16213e;
                            height: 120px;
                            border-radius: 4px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            color: #4a90d9;
                            font-size: 0.9em;
                            margin-bottom: 8px;
                        ">
                            🖼️ Frame #{frame_id}
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <code style="font-size: 0.75em;">F:{frame_id}</code>
                            <span style="
                                background: {badge_color};
                                color: white;
                                padding: 2px 8px;
                                border-radius: 10px;
                                font-size: 0.75em;
                                font-weight: 600;
                            ">{score:.2f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Metadata expander
                if metadata:
                    with st.expander("📋 Metadata", expanded=False):
                        if metadata.get("ocr_text"):
                            st.caption(f"📝 OCR: {metadata['ocr_text']}")
                        if metadata.get("objects"):
                            st.caption(f"📦 Objects: {', '.join(metadata['objects'])}")
                        if metadata.get("asr_text"):
                            st.caption(f"🎙️ ASR: {metadata['asr_text']}")
                        if metadata.get("channel"):
                            st.caption(f"📺 Channel: {metadata['channel']}")

        st.divider()
