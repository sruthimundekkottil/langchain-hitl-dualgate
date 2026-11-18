"""Locks in the differences between the two designs.

`benchmarks/compare_designs.py` prints these as a table; this file asserts them, so a
change in LangGraph's replay behaviour or in either middleware shows up as a failure
rather than as a quietly wrong row in a README.
"""

from __future__ import annotations

from typing import Any
import json
import pytest
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from conftest import Recorder
from dual_gate_hitl import DualGateHumanInTheLoopMiddleware, StateReviewMiddleware


class ReactiveModel(BaseChatModel):
    """Asks for one tool call, then answers once a result is in the history.

    A call counter would restart in the simulated new process and ask for the tool
    again, which would look like a double execution neither design caused.
    """

    @property
    def _llm_type(self) -> str:
        return "reactive"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if any(isinstance(m, ToolMessage) for m in messages):
            message = AIMessage(content="all done")
        else:
            message = AIMessage(
                content="",
                tool_calls=[{"name": "charge", "args": {"amount": 10}, "id": "x1"}],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> ReactiveModel:
        return self


def _charge(recorder: Recorder):
    @tool
    def charge(amount: int) -> str:
        """Charge the customer's card."""
        recorder.record("charge", amount=amount)
        return f"charged {amount}"

    return charge


def _build(
    recorder: Recorder,
    engine: str,
    checkpointer: Any,
    store: Any,
) -> Any:
    """A fresh agent, as a new process would build it."""
    if engine == "state-rewrite":
        middleware: Any = StateReviewMiddleware(review_on={"charge": True})
        kwargs: dict[str, Any] = {}
    else:
        middleware = DualGateHumanInTheLoopMiddleware(interrupt_on={"charge": "after"})
        kwargs = {"store": store} if engine == "store-ledger" else {}
    return create_agent(
        model=ReactiveModel(),
        tools=[_charge(recorder)],
        middleware=[middleware],
        checkpointer=checkpointer,
        **kwargs,
    )


def _restart_run(recorder: Recorder, engine: str) -> int:
    """Pause for review, discard the process, resume, and count executions.

    The checkpointer and store stand in for durable backends. What is actually lost is
    the middleware instance, and with it any in-process ledger - a redeploy.
    """
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    config = {"configurable": {"thread_id": f"restart-{engine}"}}

    before = _build(recorder, engine, checkpointer, store)
    state = before.invoke({"messages": [("user", "go")]}, config)
    assert state["__interrupt__"], "the review should be pending"
    assert recorder.count("charge") == 1

    after = _build(recorder, engine, checkpointer, store)
    after.invoke(Command(resume={"type": "accept"}), config)
    return recorder.count("charge")


def test_in_process_ledger_loses_a_redeploy(recorder: Recorder) -> None:
    """The honest limitation of the default ledger: it does not survive the process."""
    assert _restart_run(recorder, "memory-ledger") == 2


def test_store_ledger_survives_a_redeploy(recorder: Recorder) -> None:
    assert _restart_run(recorder, "store-ledger") == 1


def test_state_rewrite_survives_a_redeploy_with_no_ledger(recorder: Recorder) -> None:
    """The reason design B exists: the result is in the checkpoint, so nothing to lose."""
    assert _restart_run(recorder, "state-rewrite") == 1


def test_neither_design_keeps_the_raw_result_off_disk(recorder: Recorder) -> None:
    """A reviewed result is persisted by both designs, just in different places.

    An earlier version of this suite checked only `channel_values["messages"]` and so
    reported that the wrap-gate design never persisted a raw result. It does: the
    interrupt payload carries the result and is checkpointed with everything else. The
    consequence matters for secrets - a password in a tool result reaches disk whichever
    design is used, so redaction has to happen before the result is stored.
    """
    found: dict[str, dict[str, bool]] = {}
    for engine in ("memory-ledger", "state-rewrite"):
        checkpointer = InMemorySaver()
        config = {"configurable": {"thread_id": f"leak-{engine}"}}
        agent = _build(recorder, engine, checkpointer, None)
        agent.invoke({"messages": [("user", "go")]}, config)

        as_message = False
        anywhere = False
        for snapshot in checkpointer.list(config):
            messages = snapshot.checkpoint.get("channel_values", {}).get("messages") or []
            if any(
                isinstance(m, ToolMessage) and "charged 10" in str(m.content)
                for m in messages
            ):
                as_message = True
            if "charged 10" in json.dumps(snapshot.checkpoint, default=str):
                anywhere = True
            if "charged 10" in json.dumps(snapshot.pending_writes, default=str):
                anywhere = True
        found[engine] = {"as_message": as_message, "anywhere": anywhere}

    # Both persist it somewhere.
    assert found["memory-ledger"]["anywhere"] is True
    assert found["state-rewrite"]["anywhere"] is True
    # The only difference is where: state rewrite puts it in the message history.
    assert found["memory-ledger"]["as_message"] is False
    assert found["state-rewrite"]["as_message"] is True


def test_sibling_call_decides_whether_retry_is_offered(recorder: Recorder) -> None:
    """Design A can retry one call in a multi-call turn; design B withholds it."""

    @tool
    def charge(amount: int) -> str:
        """Charge the customer's card."""
        recorder.record("charge", amount=amount)
        return f"charged {amount}"

    @tool
    def log_note(text: str) -> str:
        """Append to the audit log."""
        recorder.record("log_note", text=text)
        return f"noted {text}"

    class TwoCallModel(ReactiveModel):
        def _generate(self, messages: Any, stop: Any = None, **kwargs: Any) -> ChatResult:
            if any(isinstance(m, ToolMessage) for m in messages):
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="all done"))]
                )
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {"name": "log_note", "args": {"text": "a"}, "id": "n1"},
                                {"name": "charge", "args": {"amount": 10}, "id": "c1"},
                            ],
                        )
                    )
                ]
            )

    offered: dict[str, list[str]] = {}
    for engine, middleware in (
        ("wrap-gate", DualGateHumanInTheLoopMiddleware(interrupt_on={"charge": "after"})),
        ("state-rewrite", StateReviewMiddleware(review_on={"charge": True})),
    ):
        agent = create_agent(
            model=TwoCallModel(),
            tools=[charge, log_note],
            middleware=[middleware],
            checkpointer=InMemorySaver(),
        )
        state = agent.invoke(
            {"messages": [("user", "go")]}, {"configurable": {"thread_id": engine}}
        )
        offered[engine] = state["__interrupt__"][0].value["allowed_decisions"]

    assert "retry" in offered["wrap-gate"]
    assert "retry" not in offered["state-rewrite"]


def test_only_wrap_gate_offers_a_pre_execution_gate() -> None:
    """Design B reviews results only; compose it with a before-gate if you need one."""
    with pytest.raises(ValueError, match="reviews results"):
        StateReviewMiddleware(review_on={"charge": {"allowed_decisions": ["approve"]}})
