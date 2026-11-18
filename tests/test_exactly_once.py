"""The property the whole design exists to guarantee: a reviewed tool runs once.

LangGraph re-runs an interrupted task from the top on resume. A post-execution gate
therefore sits downstream of a side effect, and without a ledger the tool fires again.
These tests pin that down: first that the hazard is real, then that the middleware
removes it.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt

from conftest import Recorder, ScriptedModel, call_then_finish
from dual_gate_hitl import DualGateHumanInTheLoopMiddleware, MemoryLedger


def _charge_tool(recorder: Recorder):
    @tool
    def charge(amount: int) -> str:
        """Charge the customer's card."""
        recorder.record("charge", amount=amount)
        return f"charged {amount}"

    return charge


def _build(recorder: Recorder, middleware: list[AgentMiddleware], **kwargs: Any):
    agent = create_agent(
        model=ScriptedModel(script=call_then_finish("charge", {"amount": 10})),
        tools=[_charge_tool(recorder)],
        middleware=middleware,
        checkpointer=InMemorySaver(),
        **kwargs,
    )
    return agent, {"configurable": {"thread_id": "t"}}


class NaivePostGate(AgentMiddleware):
    """What you write if you have not read the interrupt docs. Kept as a control."""

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        result = handler(request)
        interrupt({"result": str(result.content)})
        return result


def test_naive_post_gate_double_executes(recorder: Recorder) -> None:
    """Control: the hazard this package exists to fix is real on this version."""
    agent, config = _build(recorder, [NaivePostGate()])
    agent.invoke({"messages": [("user", "go")]}, config)
    assert recorder.count("charge") == 1
    agent.invoke(Command(resume={"type": "accept"}), config)
    assert recorder.count("charge") == 2, "control should double-charge"


def test_post_gate_executes_exactly_once(recorder: Recorder) -> None:
    """The middleware's post-execution gate does not re-run the tool on resume."""
    agent, config = _build(
        recorder,
        [DualGateHumanInTheLoopMiddleware(interrupt_on={"charge": "after"})],
    )
    result = agent.invoke({"messages": [("user", "go")]}, config)

    assert result["__interrupt__"][0].value["type"] == "tool_result_review"
    assert recorder.count("charge") == 1

    final = agent.invoke(Command(resume={"type": "accept"}), config)
    assert recorder.count("charge") == 1, "the tool must not run a second time"
    assert final["messages"][-1].content == "all done"


def test_both_gates_two_interrupts_one_execution(recorder: Recorder) -> None:
    """Two reviews of the same call still mean one execution."""
    agent, config = _build(
        recorder,
        [DualGateHumanInTheLoopMiddleware(interrupt_on={"charge": True})],
    )
    first = agent.invoke({"messages": [("user", "go")]}, config)
    assert first["__interrupt__"][0].value["type"] == "tool_call_review"
    assert recorder.count("charge") == 0, "nothing runs before the first gate clears"

    second = agent.invoke(Command(resume={"type": "approve"}), config)
    assert second["__interrupt__"][0].value["type"] == "tool_result_review"
    assert recorder.count("charge") == 1

    agent.invoke(Command(resume={"type": "accept"}), config)
    assert recorder.count("charge") == 1


def test_sibling_tool_call_not_replayed(recorder: Recorder) -> None:
    """A gated call pausing must not re-run the ungated call beside it.

    This is the assumption that lets the ledger be keyed on the tool call id alone: the
    replay is scoped to the task that interrupted, not the whole tools node.
    """

    @tool
    def charge(amount: int) -> str:
        """Charge the customer's card."""
        recorder.record("charge", amount=amount)
        return f"charged {amount}"

    @tool
    def log_note(text: str) -> str:
        """Append a note to the audit log."""
        recorder.record("log_note", text=text)
        return f"noted {text}"

    agent = create_agent(
        model=ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "log_note", "args": {"text": "hi"}, "id": "n1"},
                        {"name": "charge", "args": {"amount": 10}, "id": "c1"},
                    ],
                ),
                AIMessage(content="all done"),
            ]
        ),
        tools=[charge, log_note],
        middleware=[DualGateHumanInTheLoopMiddleware(interrupt_on={"charge": "after"})],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "siblings"}}

    agent.invoke({"messages": [("user", "go")]}, config)
    assert recorder.count("log_note") == 1
    assert recorder.count("charge") == 1

    agent.invoke(Command(resume={"type": "accept"}), config)
    assert recorder.count("log_note") == 1, "ungated sibling must not re-run"
    assert recorder.count("charge") == 1


def test_retry_executes_once_per_attempt(recorder: Recorder) -> None:
    """A retry is a fresh execution, and each attempt is itself replay-safe."""
    agent, config = _build(
        recorder,
        [DualGateHumanInTheLoopMiddleware(interrupt_on={"charge": "after"})],
    )
    agent.invoke({"messages": [("user", "go")]}, config)
    assert recorder.count("charge") == 1

    # Retry with different arguments: one more execution, then another review.
    second = agent.invoke(
        Command(resume={"type": "retry", "args": {"amount": 99}}), config
    )
    assert recorder.count("charge") == 2
    assert recorder.executions[-1]["amount"] == 99
    assert second["__interrupt__"][0].value["attempt"] == 1

    agent.invoke(Command(resume={"type": "accept"}), config)
    assert recorder.count("charge") == 2, "accepting must not run attempt 1 again"


def test_store_backed_ledger_is_used(recorder: Recorder) -> None:
    """With a store configured, results are recorded there rather than in memory."""
    store = InMemoryStore()
    agent, config = _build(
        recorder,
        [DualGateHumanInTheLoopMiddleware(interrupt_on={"charge": "after"})],
        store=store,
    )
    agent.invoke({"messages": [("user", "go")]}, config)

    recorded = list(store.search(("dual_gate_hitl", "results")))
    assert len(recorded) == 1, "the executed result should be in the store"

    agent.invoke(Command(resume={"type": "accept"}), config)
    assert recorder.count("charge") == 1
    assert not list(store.search(("dual_gate_hitl", "results"))), "ledger cleans up"


def test_explicit_ledger_overrides_store(recorder: Recorder) -> None:
    ledger = MemoryLedger(max_entries=4)
    store = InMemoryStore()
    agent, config = _build(
        recorder,
        [
            DualGateHumanInTheLoopMiddleware(
                interrupt_on={"charge": "after"}, ledger=ledger
            )
        ],
        store=store,
    )
    agent.invoke({"messages": [("user", "go")]}, config)
    assert not list(store.search(("dual_gate_hitl", "results")))
    agent.invoke(Command(resume={"type": "accept"}), config)
    assert recorder.count("charge") == 1


@pytest.mark.asyncio
async def test_async_post_gate_executes_exactly_once(recorder: Recorder) -> None:
    """`awrap_tool_call` shares the policy, so it inherits the same guarantee."""
    agent, config = _build(
        recorder,
        [DualGateHumanInTheLoopMiddleware(interrupt_on={"charge": "after"})],
    )
    await agent.ainvoke({"messages": [("user", "go")]}, config)
    assert recorder.count("charge") == 1

    final = await agent.ainvoke(Command(resume={"type": "accept"}), config)
    assert recorder.count("charge") == 1
    assert final["messages"][-1].content == "all done"
