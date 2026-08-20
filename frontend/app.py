"""
AIC2026-Multimedia-Agent — Streamlit Dashboard.

Unified single-page dashboard for the AIC 2026 competition.

Layout:
┌───────────────────────────────────────────────────┐
│ [LEFT SIDEBAR]       │  [CENTER MAIN AREA]        │
│ • AI Agent Chatbot   │  • Search Results Grid     │
│ • Metadata Filters   │  • Video-Grouped Layout    │
│ • Submit Controls    │  • Keyframe Cards          │
│                      │  [MODALS / EXPANDERS]      │
│                      │  • Timeline Viewer         │
│                      │  • Sketch Board            │
└───────────────────────────────────────────────────┘

Run with:
    cd frontend && streamlit run app.py --server.port 8501
"""

from __future__ import annotations

import streamlit as st

from frontend.components.chatbot_sidebar import render_chatbot_sidebar
from frontend.components.som_grid import render_som_grid
from frontend.utils.api_client import check_health

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="AIC2026 Multimedia Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark theme refinements */
    .stApp {
        background-color: #0e1117;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 1px solid #16213e;
    }

    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #0f3460 0%, #533483 50%, #e94560 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2em;
        font-weight: 800;
        margin-bottom: 0;
    }

    .sub-header {
        color: #8892b0;
        font-size: 0.95em;
        margin-top: -8px;
        margin-bottom: 20px;
    }

    /* Status indicator */
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 600;
    }
    .status-online { background: #10b981; color: white; }
    .status-offline { background: #ef4444; color: white; }
    .status-mock { background: #f59e0b; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    """Main dashboard entry point."""

    # ── Header ───────────────────────────────────────────────
    st.markdown('<p class="main-header">🎯 AIC2026 Multimedia Agent</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Intelligent Multimedia Retrieval & Reasoning System</p>',
        unsafe_allow_html=True,
    )

    # ── System Health Status ─────────────────────────────────
    health = check_health()
    health_data = health.get("data", {})

    if isinstance(health_data, dict):
        qdrant_status = health_data.get("qdrant", "unknown")
        es_status = health_data.get("elasticsearch", "unknown")

        col1, col2, col3 = st.columns(3)
        with col1:
            badge = "status-online" if qdrant_status == "connected" else "status-mock"
            st.markdown(
                f'<span class="status-badge {badge}">Qdrant: {qdrant_status}</span>',
                unsafe_allow_html=True,
            )
        with col2:
            badge = "status-online" if es_status == "connected" else "status-mock"
            st.markdown(
                f'<span class="status-badge {badge}">ES: {es_status}</span>',
                unsafe_allow_html=True,
            )
        with col3:
            models = health_data.get("models", {})
            loaded = sum(1 for v in models.values() if v == "loaded")
            total = len(models)
            badge = "status-online" if loaded == total else "status-mock"
            st.markdown(
                f'<span class="status-badge {badge}">Models: {loaded}/{total} loaded</span>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── LEFT SIDEBAR: Agent Chatbot ──────────────────────────
    render_chatbot_sidebar()

    # ── LEFT SIDEBAR: Quick Filters (Phase 2 placeholder) ────
    st.sidebar.markdown("## 🔧 Filters")
    st.sidebar.multiselect(
        "Objects",
        options=["person", "car", "laptop", "cup", "dog", "bicycle", "building"],
        default=[],
        key="filter_objects",
    )
    st.sidebar.text_input("OCR Text", key="filter_ocr", placeholder="e.g. HTV9")
    st.sidebar.text_input("Channel", key="filter_channel", placeholder="e.g. VTV1")
    st.sidebar.divider()

    # ── LEFT SIDEBAR: Submission Controls ────────────────────
    st.sidebar.markdown("## 📤 Submission")
    st.sidebar.selectbox("Task Type", options=["KIS", "VQA", "TRAKE"], key="task_type")
    st.sidebar.text_input("Question ID", key="question_id", placeholder="Q001")
    if st.sidebar.button("📨 Submit Results", use_container_width=True):
        st.sidebar.info("⏳ Submission endpoint ready (connect in Phase 4)")

    # ── CENTER MAIN AREA: Results Grid ───────────────────────
    render_som_grid()


if __name__ == "__main__":
    main()
