"""
Chatbot Sidebar Component.

Renders the System 2 Agent chatbot interface in the Streamlit sidebar.
Displays conversation history, CoT reasoning traces, and supports
multi-turn conversational KIS queries.
"""

from __future__ import annotations

import streamlit as st

from frontend.utils.api_client import hybrid_search


def render_chatbot_sidebar() -> str | None:
    """
    Render the chatbot interface in st.sidebar.

    Returns:
        The latest user query string if a new message was submitted,
        or None if no new input.
    """
    st.sidebar.markdown("## 🧠 AI Agent")
    st.sidebar.caption("System 2 Chain-of-Thought Reasoning")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        role_icon = "🧑" if msg["role"] == "user" else "🤖"
        st.sidebar.markdown(f"{role_icon} **{msg['role'].title()}:** {msg['content']}")

        # Display CoT reasoning trace if present
        if msg.get("reasoning"):
            with st.sidebar.expander("🔍 Agent Reasoning", expanded=False):
                st.sidebar.markdown(f"```\n{msg['reasoning']}\n```")

    # Chat input
    user_query = st.sidebar.text_input(
        "Ask the AI Agent...",
        key="chat_input",
        placeholder="Tìm người phụ nữ mặc áo đỏ tại HTV9",
    )

    submitted_query = None

    if st.sidebar.button("🔍 Search", use_container_width=True, type="primary"):
        if user_query.strip():
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query,
            })

            # Call backend (or mock)
            response = hybrid_search(user_query)
            agent_reasoning = response.get("agent_reasoning", "")
            message = response.get("message", "Search completed")

            # Add agent response to history
            st.session_state.chat_history.append({
                "role": "agent",
                "content": message,
                "reasoning": agent_reasoning,
            })

            # Store results in session state for the grid to display
            st.session_state["latest_results"] = response.get("data", {})
            submitted_query = user_query

    # Clear conversation button
    if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.pop("latest_results", None)
        st.rerun()

    st.sidebar.divider()
    return submitted_query
