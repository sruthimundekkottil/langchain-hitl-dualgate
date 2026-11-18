"""Post-execution gate: the part the built-in middleware has no equivalent for."""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from conftest import Recorder, ScriptedModel, call_then_finish
from dual_gate_hitl import DualGateHumanInTheLoopMiddleware

RECORDS = {
    "r-1": "name: Asha Menon, phone: 98xxxxxx12, balance: 4300",
}


def _agent(
    recorder: Recorder,
    interrupt_on: dict[str, Any],
    thread: str = "t",
    *,
    fail_first: bool = False,
    **kwargs: Any,
):
    state = {"attempts": 0}

    @tool
    def lookup_customer(record_id: str) -> str:
        """Look up a customer record."""
        state["attempts"] += 1
        recorder.record("lookup_customer", record_id=record_id)
        if fail_first and state["attempts"] == 1:
            return "ERROR: upstream timeout"
        return RECORDS.get(record_id, "not found")

    script = call_then_finish("lookup_customer", {"record_id": "r-1"})
    middleware = DualGateHumanInTheLoopMiddleware(interrupt_on=interrupt_on, **kwargs)
    agent = create_agent(
        model=ScriptedModel(script=script),
        tools=[lookup_customer],
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    return agent, {"configurable": {"thread_id": thread}}


def _tool_messages(state: dict[str, Any]) -> list[ToolMessage]:
    return [m for m in state["messages"] if m.type == "tool"]


def test_payload_carries_the_result(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"lookup_customer": "after"})
    out = agent.invoke({"messages": [("user", "go")]}, config)

    payload = out["__interrupt__"][0].value
    assert payload["type"] == "tool_result_review"
    assert payload["name"] == "lookup_customer"
    assert payload["result"] == RECORDS["r-1"]
    assert payload["status"] == "success"
    assert payload["attempt"] == 0
    assert payload["retries_left"] == 2
    assert set(payload["allowed_decisions"]) == {"accept", "edit", "retry", "reject"}


def test_accept_passes_the_result_through(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"lookup_customer": "after"})
    agent.invoke({"messages": [("user", "go")]}, config)
    out = agent.invoke(Command(resume={"type": "accept"}), config)

    assert _tool_messages(out)[0].content == RECORDS["r-1"]


def test_edit_redacts_before_the_model_sees_it(recorder: Recorder) -> None:
    """The headline use case: a human strips PII out of a result in flight."""
    agent, config = _agent(recorder, {"lookup_customer": "after"})
    agent.invoke({"messages": [("user", "go")]}, config)
    out = agent.invoke(
        Command(
            resume={
                "type": "edit",
                "content": "name: Asha Menon, phone: [redacted], balance: [redacted]",
            }
        ),
        config,
    )

    message = _tool_messages(out)[0]
    assert "98xxxxxx12" not in message.content
    assert "[redacted]" in message.content
    assert "A human reviewer edited this tool result." in message.content
    assert message.status == "success"


def test_reject_withholds_the_result(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"lookup_customer": "after"})
    agent.invoke({"messages": [("user", "go")]}, config)
    out = agent.invoke(
        Command(resume={"type": "reject", "message": "not cleared for this account"}),
        config,
    )

    message = _tool_messages(out)[0]
    assert message.status == "error"
    assert "98xxxxxx12" not in message.content
    assert "not cleared for this account" in message.content


def test_retry_reruns_and_reviews_again(recorder: Recorder) -> None:
    """A flaky tool can be run again without the model ever seeing the bad result."""
    agent, config = _agent(recorder, {"lookup_customer": "after"}, fail_first=True)
    first = agent.invoke({"messages": [("user", "go")]}, config)
    assert "ERROR: upstream timeout" in first["__interrupt__"][0].value["result"]

    second = agent.invoke(Command(resume={"type": "retry"}), config)
    payload = second["__interrupt__"][0].value
    assert payload["result"] == RECORDS["r-1"]
    assert payload["attempt"] == 1
    assert payload["retries_left"] == 1

    out = agent.invoke(Command(resume={"type": "accept"}), config)
    assert _tool_messages(out)[0].content == RECORDS["r-1"]
    assert recorder.count("lookup_customer") == 2


def test_retry_budget_removes_the_option(recorder: Recorder) -> None:
    """Once the budget is spent, `retry` is not offered - no dead-end choices."""
    agent, config = _agent(
        recorder, {"lookup_customer": {"after": {"max_retries": 1}}}
    )
    first = agent.invoke({"messages": [("user", "go")]}, config)
    assert "retry" in first["__interrupt__"][0].value["allowed_decisions"]

    second = agent.invoke(Command(resume={"type": "retry"}), config)
    payload = second["__interrupt__"][0].value
    assert payload["retries_left"] == 0
    assert "retry" not in payload["allowed_decisions"]

    with pytest.raises(Exception, match="retry"):
        agent.invoke(Command(resume={"type": "retry"}), config)


def test_zero_retries_never_offers_retry(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"lookup_customer": "after"}, max_retries=0)
    out = agent.invoke({"messages": [("user", "go")]}, config)
    assert "retry" not in out["__interrupt__"][0].value["allowed_decisions"]


def test_when_predicate_sees_the_result(recorder: Recorder) -> None:
    """Escalate on the output, not the call: the gate stays quiet on a clean result."""
    gate = {
        "after": {
            "when": lambda req, msg: "ERROR" in str(msg.content),
            "allowed_decisions": ["accept", "retry"],
        }
    }

    clean, config = _agent(recorder, {"lookup_customer": gate}, thread="clean")
    out = clean.invoke({"messages": [("user", "go")]}, config)
    assert "__interrupt__" not in out, "a clean result should not bother a human"
    assert _tool_messages(out)[0].content == RECORDS["r-1"]

    flaky, config = _agent(
        recorder, {"lookup_customer": gate}, thread="flaky", fail_first=True
    )
    out = flaky.invoke({"messages": [("user", "go")]}, config)
    assert out["__interrupt__"][0].value["type"] == "tool_result_review"


def test_custom_description_sees_the_result(recorder: Recorder) -> None:
    agent, config = _agent(
        recorder,
        {
            "lookup_customer": {
                "after": {
                    "description": lambda req, msg: f"{len(str(msg.content))} chars back"
                }
            }
        },
    )
    out = agent.invoke({"messages": [("user", "go")]}, config)
    expected = f"{len(RECORDS['r-1'])} chars back"
    assert out["__interrupt__"][0].value["description"] == expected


def test_notice_can_be_switched_off(recorder: Recorder) -> None:
    agent, config = _agent(
        recorder, {"lookup_customer": "after"}, result_edit_notice=None
    )
    agent.invoke({"messages": [("user", "go")]}, config)
    out = agent.invoke(Command(resume={"type": "edit", "content": "clean"}), config)

    assert _tool_messages(out)[0].content == "clean"


def test_post_gate_requires_a_tool_call_id(recorder: Recorder) -> None:
    """Without an id a result cannot be tracked across a pause, so this fails loudly."""
    from langchain_core.messages import AIMessage

    @tool
    def ping() -> str:
        """Ping."""
        recorder.record("ping")
        return "pong"

    agent = create_agent(
        model=ScriptedModel(
            script=[
                AIMessage(
                    content="", tool_calls=[{"name": "ping", "args": {}, "id": None}]
                ),
                AIMessage(content="all done"),
            ]
        ),
        tools=[ping],
        middleware=[DualGateHumanInTheLoopMiddleware(interrupt_on={"ping": "after"})],
        checkpointer=InMemorySaver(),
    )
    with pytest.raises(ValueError, match="has no id"):
        agent.invoke(
            {"messages": [("user", "go")]}, {"configurable": {"thread_id": "x"}}
        )
