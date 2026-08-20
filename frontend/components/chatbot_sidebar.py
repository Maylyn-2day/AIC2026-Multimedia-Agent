"""Safe Agent planning and hybrid-search controls for the sidebar."""

from __future__ import annotations

from uuid import uuid4

import streamlit as st

from frontend.utils.api_client import clear_agent_session, hybrid_search, route_agent


def render_chatbot_sidebar() -> str | None:
    """Route a query, display its public summary, then search when allowed."""
    st.sidebar.markdown("## AI Agent")
    st.sidebar.caption("Decision summary and search plan")
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("agent_session_id", str(uuid4()))
    route_task = st.sidebar.selectbox("Route Task Type", ["Auto", "KIS", "VQA", "TRAKE"], key="route_task_type")
    for message in st.session_state.chat_history:
        st.sidebar.markdown(f"**{message['role'].title()}:** {message['content']}")
        if message.get("decision_summary"):
            with st.sidebar.expander("Decision summary", expanded=False):
                st.sidebar.write(message["decision_summary"])
    user_query = st.sidebar.text_input("Ask the AI Agent...", key="chat_input")
    submitted_query = None
    if st.sidebar.button("Search", use_container_width=True, type="primary") and user_query.strip():
        session_id = st.session_state.agent_session_id
        st.session_state.chat_history.append({"role": "user", "content": user_query.strip()})
        response = route_agent(user_query, session_id, None if route_task == "Auto" else route_task)
        if response.get("status") != "success":
            st.sidebar.error(response.get("message", "Agent routing failed"))
        else:
            plan = response["data"]
            clarification = plan.get("clarification_question")
            st.session_state.chat_history.append(
                {
                    "role": "agent",
                    "content": clarification or "Search plan ready",
                    "decision_summary": plan.get("decision_summary"),
                }
            )
            st.session_state["latest_agent_plan"] = plan
            if clarification:
                st.sidebar.info(clarification)
            elif plan.get("should_search", True):
                search = hybrid_search(
                    plan["raw_query"], plan.get("filters", {}), plan.get("top_k", 100), session_id=session_id
                )
                if search.get("status") == "success":
                    st.session_state["latest_results"] = search.get("data", {})
                    submitted_query = plan["raw_query"]
                else:
                    st.sidebar.error(search.get("message", "Search failed"))
    if st.sidebar.button("Clear Chat", use_container_width=True):
        response = clear_agent_session(st.session_state.agent_session_id)
        if response.get("status") == "success":
            st.session_state.chat_history = []
            st.session_state.pop("latest_results", None)
            st.session_state.pop("latest_agent_plan", None)
            st.rerun()
        else:
            st.sidebar.error(response.get("message", "Unable to clear Agent session"))
    st.sidebar.divider()
    return submitted_query
