"""Validate-only submission controls for currently displayed results."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.utils.api_client import submit_results


def _available_results(data: object) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    candidates = data.get("candidates", data.get("results", []))
    return [
        item
        for item in candidates
        if isinstance(item, dict)
        and isinstance(item.get("video_id"), str)
        and bool(item["video_id"].strip())
        and type(item.get("frame_id")) is int
    ]


def render_submission_sidebar() -> None:
    """Select existing results and request local submission validation."""
    st.sidebar.markdown("## Submission")
    plan = st.session_state.get("latest_agent_plan", {})
    default_task = plan.get("task_type", "KIS") if isinstance(plan, dict) else "KIS"
    tasks = ["KIS", "VQA", "TRAKE"]
    task_type = st.sidebar.selectbox(
        "Task Type", tasks, index=tasks.index(default_task) if default_task in tasks else 0
    )
    if isinstance(plan, dict) and plan.get("task_type") in tasks and task_type != plan["task_type"]:
        st.sidebar.warning(f"Submission task {task_type} differs from routed task {plan['task_type']}.")
    question_id = st.sidebar.text_input("Question ID", key="submission_question_id")
    answer = st.sidebar.text_input("VQA Answer", key="submission_answer") if task_type == "VQA" else None
    available = _available_results(st.session_state.get("latest_results"))
    labels = [f"{item['video_id']} / {item['frame_id']}" for item in available]
    selected_labels = st.sidebar.multiselect("Results", labels, default=labels[: min(len(labels), 100)])
    selected = []
    for label, item in zip(labels, available, strict=True):
        if label in selected_labels:
            packaged = {"video_id": item["video_id"], "frame_id": item["frame_id"]}
            if task_type == "VQA":
                packaged["answer"] = answer
            selected.append(packaged)
    if st.sidebar.button("Validate submission", use_container_width=True):
        response = submit_results(task_type, selected, question_id, st.session_state.get("agent_session_id"))
        if response.get("status") == "success" and response.get("data", {}).get("submitted") is False:
            st.sidebar.success("Đã kiểm tra hợp lệ, chưa gửi đến BTC")
        else:
            st.sidebar.error(response.get("message", "Submission validation failed"))
    st.sidebar.divider()
