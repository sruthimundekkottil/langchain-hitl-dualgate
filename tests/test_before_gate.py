"""Pre-execution gate: the same four decisions as the built-in middleware."""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from conftest import Recorder, ScriptedModel, call_then_finish
from dual_gate_hitl import DualGateHumanInTheLoopMiddleware


def _agent(recorder: Recorder, interrupt_on: dict[str, Any], thread: str = "t"):
    @tool
    def send_email(to: str, body: str) -> str:
        """Send an email."""
        recorder.record("send_email", to=to, body=body)
        return f"sent to {to}"

    @tool
    def draft_email(to: str, body: str) -> str:
        """Save a draft instead of sending."""
        recorder.record("draft_email", to=to, body=body)
        return f"drafted for {to}"

    agent = create_agent(
        model=ScriptedModel(
            script=call_then_finish("send_email", {"to": "a@b.com", "body": "hi"})
        ),
        tools=[send_email, draft_email],
        middleware=[DualGateHumanInTheLoopMiddleware(interrupt_on=interrupt_on)],
        checkpointer=InMemorySaver(),
    )
    return agent, {"configurable": {"thread_id": thread}}


def test_payload_describes_the_pending_call(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"send_email": "before"})
    out = agent.invoke({"messages": [("user", "go")]}, config)

    payload = out["__interrupt__"][0].value
    assert payload["type"] == "tool_call_review"
    assert payload["name"] == "send_email"
    assert payload["args"] == {"to": "a@b.com", "body": "hi"}
    assert payload["tool_call_id"] == "call_1"
    assert set(payload["allowed_decisions"]) == {"approve", "edit", "reject", "respond"}
    assert "Tool execution requires approval" in payload["description"]
    assert recorder.count() == 0


def test_approve_runs_the_original_call(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"send_email": "before"})
    agent.invoke({"messages": [("user", "go")]}, config)
    out = agent.invoke(Command(resume={"type": "approve"}), config)

    assert recorder.executions == [
        {"tool": "send_email", "to": "a@b.com", "body": "hi"}
    ]
    assert out["messages"][-1].content == "all done"


def test_edit_runs_the_reviewers_arguments(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"send_email": "before"})
    agent.invoke({"messages": [("user", "go")]}, config)
    out = agent.invoke(
        Command(
            resume={
                "type": "edit",
                "edited_action": {
                    "name": "send_email",
                    "args": {"to": "legal@b.com", "body": "reviewed"},
                },
            }
        ),
        config,
    )

    assert recorder.executions[0]["to"] == "legal@b.com"
    tool_message = next(m for m in out["messages"] if m.type == "tool")
    assert "A human reviewer edited this tool call." in tool_message.content
    assert "legal@b.com" in tool_message.content


def test_edit_can_redirect_to_another_tool(recorder: Recorder) -> None:
    """A reviewer can downgrade a dangerous call to a safe one."""
    agent, config = _agent(recorder, {"send_email": "before"})
    agent.invoke({"messages": [("user", "go")]}, config)
    agent.invoke(
        Command(
            resume={
                "type": "edit",
                "edited_action": {
                    "name": "draft_email",
                    "args": {"to": "a@b.com", "body": "hi"},
                },
            }
        ),
        config,
    )

    assert recorder.count("send_email") == 0
    assert recorder.count("draft_email") == 1


def test_reject_skips_the_tool(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"send_email": "before"})
    agent.invoke({"messages": [("user", "go")]}, config)
    out = agent.invoke(
        Command(resume={"type": "reject", "message": "wrong recipient"}), config
    )

    assert recorder.count() == 0
    tool_message = next(m for m in out["messages"] if m.type == "tool")
    assert tool_message.status == "error"
    assert "wrong recipient" in tool_message.content


def test_respond_answers_in_the_tools_place(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"send_email": "before"})
    agent.invoke({"messages": [("user", "go")]}, config)
    out = agent.invoke(
        Command(resume={"type": "respond", "message": "Already sent by hand."}), config
    )

    assert recorder.count() == 0
    tool_message = next(m for m in out["messages"] if m.type == "tool")
    assert tool_message.status == "success"
    assert tool_message.content == "Already sent by hand."


def test_when_predicate_can_auto_approve(recorder: Recorder) -> None:
    """A call the predicate rejects runs with no interrupt at all."""
    agent, config = _agent(
        recorder,
        {
            "send_email": {
                "before": {
                    "when": lambda req: req.tool_call["args"]["to"].endswith(".gov")
                }
            }
        },
    )
    out = agent.invoke({"messages": [("user", "go")]}, config)

    assert "__interrupt__" not in out
    assert recorder.count("send_email") == 1


def test_unlisted_tool_is_never_gated(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"draft_email": True})
    out = agent.invoke({"messages": [("user", "go")]}, config)

    assert "__interrupt__" not in out
    assert recorder.count("send_email") == 1


def test_false_disables_gating(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"send_email": False})
    out = agent.invoke({"messages": [("user", "go")]}, config)

    assert "__interrupt__" not in out
    assert recorder.count("send_email") == 1


def test_restricted_decisions_are_enforced(recorder: Recorder) -> None:
    agent, config = _agent(
        recorder, {"send_email": {"before": {"allowed_decisions": ["approve", "reject"]}}}
    )
    out = agent.invoke({"messages": [("user", "go")]}, config)
    assert out["__interrupt__"][0].value["allowed_decisions"] == ["approve", "reject"]

    with pytest.raises(Exception, match="respond"):
        agent.invoke(Command(resume={"type": "respond", "message": "nope"}), config)


def test_custom_description_callable(recorder: Recorder) -> None:
    agent, config = _agent(
        recorder,
        {
            "send_email": {
                "before": {
                    "description": lambda req: f"Email {req.tool_call['args']['to']}?"
                }
            }
        },
    )
    out = agent.invoke({"messages": [("user", "go")]}, config)
    assert out["__interrupt__"][0].value["description"] == "Email a@b.com?"


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="neither gate"):
        DualGateHumanInTheLoopMiddleware(interrupt_on={"x": {}})
    with pytest.raises(ValueError, match="accepts"):
        DualGateHumanInTheLoopMiddleware(
            interrupt_on={"x": {"before": {"allowed_decisions": ["accept"]}}}
        )
    with pytest.raises(ValueError, match="accepts"):
        DualGateHumanInTheLoopMiddleware(
            interrupt_on={"x": {"after": {"allowed_decisions": ["approve"]}}}
        )
    with pytest.raises(TypeError, match="ToolReviewConfig"):
        DualGateHumanInTheLoopMiddleware(interrupt_on={"x": "sometimes"})  # type: ignore[dict-item]
