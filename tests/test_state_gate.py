"""Design B: reviewing results by rewriting them in state."""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from conftest import Recorder, ScriptedModel, call_then_finish
from dual_gate_hitl import StateReviewMiddleware

RECORD = "name='Asha Menon' phone=9812345670 balance=4300"


def _agent(
    recorder: Recorder,
    review_on: dict[str, Any],
    thread: str = "t",
    *,
    fail_first: bool = False,
    **kwargs: Any,
):
    attempts = {"n": 0}

    @tool
    def fetch_record(record_id: str) -> str:
        """Fetch a customer record."""
        attempts["n"] += 1
        recorder.record("fetch_record", record_id=record_id)
        if fail_first and attempts["n"] == 1:
            return "ERROR: upstream timeout"
        return RECORD

    agent = create_agent(
        model=ScriptedModel(
            script=call_then_finish("fetch_record", {"record_id": "r-1"})
        ),
        tools=[fetch_record],
        middleware=[StateReviewMiddleware(review_on=review_on, **kwargs)],
        checkpointer=InMemorySaver(),
    )
    return agent, {"configurable": {"thread_id": thread}}


def _tool_messages(state: dict[str, Any]) -> list[ToolMessage]:
    return [m for m in state["messages"] if m.type == "tool"]


def test_payload_carries_the_result(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"fetch_record": True})
    out = agent.invoke({"messages": [("user", "go")]}, config)

    payload = out["__interrupt__"][0].value
    assert payload["type"] == "tool_result_review"
    assert payload["name"] == "fetch_record"
    assert payload["args"] == {"record_id": "r-1"}
    assert payload["result"] == RECORD
    assert payload["attempt"] == 0
    assert set(payload["allowed_decisions"]) == {"accept", "edit", "retry", "reject"}


def test_tool_runs_once_with_no_ledger(recorder: Recorder) -> None:
    """The point of this design: the pause is upstream of nothing, so nothing replays."""
    agent, config = _agent(recorder, {"fetch_record": True})
    agent.invoke({"messages": [("user", "go")]}, config)
    assert recorder.count("fetch_record") == 1

    agent.invoke(Command(resume={"type": "accept"}), config)
    assert recorder.count("fetch_record") == 1


def test_accept_leaves_the_result_alone(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"fetch_record": True})
    agent.invoke({"messages": [("user", "go")]}, config)
    out = agent.invoke(Command(resume={"type": "accept"}), config)

    assert _tool_messages(out)[0].content == RECORD


def test_edit_replaces_the_message_in_place(recorder: Recorder) -> None:
    """One tool message in the history, with the reviewer's content, same id."""
    agent, config = _agent(recorder, {"fetch_record": True})
    first = agent.invoke({"messages": [("user", "go")]}, config)
    original_id = _tool_messages(first)[0].id

    out = agent.invoke(
        Command(resume={"type": "edit", "content": "phone=[redacted]"}), config
    )

    messages = _tool_messages(out)
    assert len(messages) == 1, "editing must not append a second result"
    assert messages[0].id == original_id
    assert "9812345670" not in messages[0].content
    assert "A human reviewer edited this tool result." in messages[0].content


def test_reject_withholds_the_result(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"fetch_record": True})
    agent.invoke({"messages": [("user", "go")]}, config)
    out = agent.invoke(
        Command(resume={"type": "reject", "message": "not cleared"}), config
    )

    message = _tool_messages(out)[0]
    assert message.status == "error"
    assert "9812345670" not in message.content
    assert "not cleared" in message.content


def test_retry_reruns_the_tool(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"fetch_record": True}, fail_first=True)
    first = agent.invoke({"messages": [("user", "go")]}, config)
    assert "ERROR" in first["__interrupt__"][0].value["result"]

    second = agent.invoke(Command(resume={"type": "retry"}), config)
    payload = second["__interrupt__"][0].value
    assert payload["result"] == RECORD
    assert payload["attempt"] == 1
    assert recorder.count("fetch_record") == 2

    out = agent.invoke(Command(resume={"type": "accept"}), config)
    assert len(_tool_messages(out)) == 1, "the discarded result must not linger"
    assert _tool_messages(out)[0].content == RECORD


def test_retry_can_change_arguments(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"fetch_record": True})
    agent.invoke({"messages": [("user", "go")]}, config)
    agent.invoke(Command(resume={"type": "retry", "args": {"record_id": "r-2"}}), config)

    assert recorder.executions[-1]["record_id"] == "r-2"


def test_retry_budget_removes_the_option(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"fetch_record": {"max_retries": 1}})
    agent.invoke({"messages": [("user", "go")]}, config)
    second = agent.invoke(Command(resume={"type": "retry"}), config)

    payload = second["__interrupt__"][0].value
    assert payload["retries_left"] == 0
    assert "retry" not in payload["allowed_decisions"]


def test_retry_is_withheld_when_the_turn_has_siblings(recorder: Recorder) -> None:
    """The measured limitation: re-running the tools node would re-run the sibling too.

    Jumping back to `tools` re-executes every call on the last AI message, which fires
    ungated tools a second time and leaves two results for one tool call id. So `retry`
    is not offered at all when the gated result is not alone.
    """

    @tool
    def fetch_record(record_id: str) -> str:
        """Fetch a customer record."""
        recorder.record("fetch_record", record_id=record_id)
        return RECORD

    @tool
    def charge(amount: int) -> str:
        """Charge a card."""
        recorder.record("charge", amount=amount)
        return f"charged {amount}"

    agent = create_agent(
        model=ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "charge", "args": {"amount": 50}, "id": "c1"},
                        {
                            "name": "fetch_record",
                            "args": {"record_id": "r-1"},
                            "id": "c2",
                        },
                    ],
                ),
                AIMessage(content="all done"),
            ]
        ),
        tools=[fetch_record, charge],
        middleware=[StateReviewMiddleware(review_on={"fetch_record": True})],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "siblings"}}

    out = agent.invoke({"messages": [("user", "go")]}, config)
    payload = out["__interrupt__"][0].value
    assert payload["retries_left"] == 0
    assert "retry" not in payload["allowed_decisions"]
    assert set(payload["allowed_decisions"]) == {"accept", "edit", "reject"}

    with pytest.raises(Exception, match="retry"):
        agent.invoke(Command(resume={"type": "retry"}), config)

    # Editing is still fine, and the sibling is untouched.
    out = agent.invoke(Command(resume={"type": "edit", "content": "redacted"}), config)
    assert recorder.count("charge") == 1
    assert recorder.count("fetch_record") == 1
    ids = [m.tool_call_id for m in _tool_messages(out)]
    assert sorted(ids) == ["c1", "c2"], "no duplicate results in the history"


def test_when_predicate_sees_the_result(recorder: Recorder) -> None:
    review_on = {
        "fetch_record": {
            "when": lambda req, msg: "ERROR" in str(msg.content),
            "allowed_decisions": ["accept", "retry"],
        }
    }
    clean, config = _agent(recorder, review_on, thread="clean")
    out = clean.invoke({"messages": [("user", "go")]}, config)
    assert "__interrupt__" not in out
    assert _tool_messages(out)[0].content == RECORD

    flaky, config = _agent(recorder, review_on, thread="flaky", fail_first=True)
    out = flaky.invoke({"messages": [("user", "go")]}, config)
    assert out["__interrupt__"][0].value["type"] == "tool_result_review"


def test_predicate_written_for_either_design_works_here(recorder: Recorder) -> None:
    """`when` and `description` callables are portable between the two middlewares."""
    agent, config = _agent(
        recorder,
        {
            "fetch_record": {
                "when": lambda req, msg: req.tool_call["args"]["record_id"] == "r-1",
                "description": lambda req, msg: (
                    f"{req.tool_call['name']} -> {len(str(msg.content))}"
                ),
            }
        },
    )
    out = agent.invoke({"messages": [("user", "go")]}, config)
    expected = f"fetch_record -> {len(RECORD)}"
    assert out["__interrupt__"][0].value["description"] == expected


def test_unlisted_tool_is_never_reviewed(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"something_else": True})
    out = agent.invoke({"messages": [("user", "go")]}, config)

    assert "__interrupt__" not in out
    assert recorder.count("fetch_record") == 1


def test_false_disables_review(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"fetch_record": False})
    out = agent.invoke({"messages": [("user", "go")]}, config)
    assert "__interrupt__" not in out


def test_rejects_pre_execution_decisions() -> None:
    with pytest.raises(ValueError, match="reviews results"):
        StateReviewMiddleware(review_on={"x": {"allowed_decisions": ["approve"]}})
    with pytest.raises(ValueError, match="empty"):
        StateReviewMiddleware(review_on={"x": {"allowed_decisions": []}})


def test_notice_can_be_switched_off(recorder: Recorder) -> None:
    agent, config = _agent(
        recorder, {"fetch_record": True}, result_edit_notice=None
    )
    agent.invoke({"messages": [("user", "go")]}, config)
    out = agent.invoke(Command(resume={"type": "edit", "content": "clean"}), config)

    assert _tool_messages(out)[0].content == "clean"


@pytest.mark.asyncio
async def test_async_path(recorder: Recorder) -> None:
    agent, config = _agent(recorder, {"fetch_record": True})
    await agent.ainvoke({"messages": [("user", "go")]}, config)
    assert recorder.count("fetch_record") == 1

    out = await agent.ainvoke(
        Command(resume={"type": "edit", "content": "phone=[redacted]"}), config
    )
    assert "9812345670" not in _tool_messages(out)[0].content
    assert recorder.count("fetch_record") == 1


def test_composes_with_the_dual_gate_before_gate(recorder: Recorder) -> None:
    """The combination worth reaching for: pre-execution approval, durable result review.

    `StateReviewMiddleware` reviews results only. Pair it with the dual-gate
    middleware's before gate and you get both sides, with the result riding in the
    checkpoint rather than a ledger.
    """
    from dual_gate_hitl import DualGateHumanInTheLoopMiddleware

    @tool
    def fetch_record(record_id: str) -> str:
        """Fetch a customer record."""
        recorder.record("fetch_record", record_id=record_id)
        return RECORD

    agent = create_agent(
        model=ScriptedModel(
            script=call_then_finish("fetch_record", {"record_id": "r-1"})
        ),
        tools=[fetch_record],
        middleware=[
            DualGateHumanInTheLoopMiddleware(interrupt_on={"fetch_record": "before"}),
            StateReviewMiddleware(review_on={"fetch_record": True}),
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "composed"}}

    first = agent.invoke({"messages": [("user", "go")]}, config)
    assert first["__interrupt__"][0].value["type"] == "tool_call_review"
    assert recorder.count("fetch_record") == 0

    second = agent.invoke(Command(resume={"type": "approve"}), config)
    assert second["__interrupt__"][0].value["type"] == "tool_result_review"
    assert recorder.count("fetch_record") == 1

    out = agent.invoke(Command(resume={"type": "edit", "content": "redacted"}), config)
    assert recorder.count("fetch_record") == 1
    assert "redacted" in _tool_messages(out)[0].content
