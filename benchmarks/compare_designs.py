# Head-to-head: the wrap-gate design against the state-rewrite design.

# Two ways to let a human review a tool's result:

# wrap-gate** (`DualGateHumanInTheLoopMiddleware`) pauses inside `wrap_tool_call`,
# after the tool has run, and keeps an execution ledger so the replay does not re-run it.
# state-rewrite** (`StateReviewMiddleware`) lets the tools node finish, then pauses in
# `before_model` and rewrites the `ToolMessage` in state. The result rides in the
# checkpoint, so there is nothing to make exactly-once.
 

# python benchmarks/compare_designs.py

from __future__ import annotations

import json
import warnings
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain.tools import InjectedToolCallId, tool
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from dual_gate_hitl import DualGateHumanInTheLoopMiddleware, StateReviewMiddleware

WRAP_GATE = "wrap-gate (ledger)"
WRAP_GATE_STORE = "wrap-gate (store ledger)"
STATE_REWRITE = "state-rewrite"
ENGINES = [WRAP_GATE, WRAP_GATE_STORE, STATE_REWRITE]

UNSUPPORTED = "n/a"


class ReactiveModel(BaseChatModel):
    """Asks for its tool calls, then answers once any result is in the history.

    Deliberately not a call counter: the restart scenario builds a second instance of
    the model, and a counter would start over and ask for the same tools again, which
    would look like a double execution the designs are not responsible for. Reading the
    history instead makes the model restart-safe, the way a real one is.
    """

    calls: list[dict[str, Any]]
    final: str = "done"

    @property
    def _llm_type(self) -> str:
        return "reactive"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if any(isinstance(m, ToolMessage) for m in messages):
            message = AIMessage(content=self.final)
        else:
            message = AIMessage(content="", tool_calls=self.calls)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> ReactiveModel:
        return self


# Fixtures
 

def charge_tool(log: list[str]):
    @tool
    def charge(amount: int) -> str:
        """Charge the customer's card. Not idempotent."""
        log.append(f"charge({amount})")
        return f"charged {amount}"

    return charge


def note_tool(log: list[str]):
    """A tool that returns a Command, i.e. updates agent state as well as answering."""

    @tool
    def note(text: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Record a note and update agent state."""
        log.append(f"note({text})")
        return Command( update={"messages": [ToolMessage(
            content=f"noted: {text}", tool_call_id=tool_call_id, name="note")]})

    return note


def single_call(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"name": name, "args": args, "id": "x1"}]


def two_calls() -> list[dict[str, Any]]:
    return [
        {"name": "note", "args": {"text": "audit"}, "id": "x0"},
        {"name": "charge", "args": {"amount": 10}, "id": "x1"}]


def build(
    engine: str,
    *,
    tools: list[Any],
    calls: list[dict[str, Any]],
    gated: str,
    checkpointer: Any = None,
    store: Any = None,) -> Any:
    """One agent per engine, configured as alike as the two designs allow."""
    if engine == STATE_REWRITE:
        middleware: Any = StateReviewMiddleware(review_on={gated: True})
        store = None  # this design needs no sidecar storage
    else:
        middleware = DualGateHumanInTheLoopMiddleware(interrupt_on={gated: "after"})
        if engine == WRAP_GATE_STORE and store is None:
            store = InMemoryStore()
        if engine == WRAP_GATE:
            store = None
    kwargs: dict[str, Any] = {}
    if store is not None:
        kwargs["store"] = store
    return create_agent(
        model=ReactiveModel(calls=calls),
        tools=tools,
        middleware=[middleware],
        checkpointer=checkpointer or InMemorySaver(),
        **kwargs,)


def interrupts(state: dict[str, Any]) -> list[Any]:
    return list(state.get("__interrupt__") or [])


 # Scenarios
 

def scenario_accept(engine: str) -> str:
    """Review a result and accept it. Both designs should run the tool once."""
    log: list[str] = []
    agent = build(
        engine,
        tools=[charge_tool(log)],
        calls=single_call("charge", {"amount": 10}),
        gated="charge",)
    config = {"configurable": {"thread_id": "accept"}}
    agent.invoke({"messages": [("user", "go")]}, config)
    agent.invoke(Command(resume={"type": "accept"}), config)
    return f"{log.count('charge(10)')} execution(s)"


def scenario_restart(engine: str) -> str:
    """Pause for review, discard the process, come back, and count executions.

    The checkpointer and store stand in for durable backends; what actually dies is the
    middleware instance, and with it any in-process ledger. This is a redeploy.
    """
    log: list[str] = []
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    config = {"configurable": {"thread_id": "restart"}}

    before = build(
        engine,
        tools=[charge_tool(log)],
        calls=single_call("charge", {"amount": 10}),
        gated="charge",
        checkpointer=checkpointer,
        store=store,)
    before.invoke({"messages": [("user", "go")]}, config)
    first = log.count("charge(10)")

    # New process: fresh middleware, same durable backends.
    after = build(
        engine,
        tools=[charge_tool(log)],
        calls=single_call("charge", {"amount": 10}),
        gated="charge",
        checkpointer=checkpointer,
        store=store,  )
    after.invoke(Command(resume={"type": "accept"}), config)
    total = log.count("charge(10)")
    verdict = "exactly once" if total == first else f"RE-RAN ({total} total)"
    return f"{total} execution(s), {verdict}"


def scenario_retry_single(engine: str) -> str:
    """Retry when the model turn made exactly one tool call."""
    log: list[str] = []
    agent = build(
        engine,
        tools=[charge_tool(log)],
        calls=single_call("charge", {"amount": 10}),
        gated="charge",)
    config = {"configurable": {"thread_id": "retry1"}}
    agent.invoke({"messages": [("user", "go")]}, config)
    state = agent.invoke(Command(resume={"type": "retry"}), config)
    if not interrupts(state):
        return "no second review"
    agent.invoke(Command(resume={"type": "accept"}), config)
    return f"supported, {log.count('charge(10)')} execution(s)"


def scenario_retry_with_sibling(engine: str) -> str:
    """Retry the gated call when an ungated call ran beside it in the same turn."""
    log: list[str] = []
    agent = build(
        engine,
        tools=[charge_tool(log), note_tool(log)],
        calls=two_calls(),
        gated="charge",
        store=InMemoryStore(),)
    config = {"configurable": {"thread_id": "retry2"}}
    state = agent.invoke({"messages": [("user", "go")]}, config)
    offered = interrupts(state)[0].value["allowed_decisions"]
    if "retry" not in offered:
        return "not offered (withheld by design)"
    agent.invoke(Command(resume={"type": "retry"}), config)
    agent.invoke(Command(resume={"type": "accept"}), config)
    sibling = log.count("note(audit)")
    note = "sibling re-ran" if sibling > 1 else "sibling safe"
    return f"supported, {note} ({sibling}x)"


def scenario_command_tool(engine: str) -> str:
    """Review a tool returning a `Command` rather than a plain result."""
    log: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = build(
            engine,
            tools=[note_tool(log)],
            calls=single_call("note", {"text": "audit"}),
            gated="note",)
        config = {"configurable": {"thread_id": "command"}}
        state = agent.invoke({"messages": [("user", "go")]}, config)
        if not interrupts(state):
            return "not reviewed"
        final = agent.invoke(
            Command(resume={"type": "edit", "content": "noted: [redacted]"}), config)
        warned = any("Command" in str(w.message) for w in caught)

    message = next(m for m in final["messages"] if m.type == "tool")
    if "[redacted]" not in str(message.content):
        return "edit not applied"
    if warned:
        return "reviewed, memory fallback"
    # The plain in-process ledger never claimed durability in the first place.
    return "reviewed, durable" if engine == STATE_REWRITE else "reviewed, in-memory"


def scenario_raw_in_checkpoint(engine: str) -> str:
    """Check whether an unreviewed result reaches persistent storage."""
    log: list[str] = []
    checkpointer = InMemorySaver()
    agent = build(
        engine,
        tools=[charge_tool(log)],
        calls=single_call("charge", {"amount": 10}),
        gated="charge",
        checkpointer=checkpointer,)
    config = {"configurable": {"thread_id": "leak"}}
    agent.invoke({"messages": [("user", "go")]}, config)

    # Search everything the checkpointer holds, not only the message list: an
    # interrupt payload carries the result too, and it is persisted with the rest.
    as_message = False
    anywhere = False
    for snapshot in checkpointer.list(config):
        messages = snapshot.checkpoint.get("channel_values", {}).get("messages") or []
        if any(isinstance(m, ToolMessage) and "charged 10" in str(m.content)for m in messages):
            as_message = True
        if "charged 10" in json.dumps(snapshot.checkpoint, default=str):
            anywhere = True
        if "charged 10" in json.dumps(snapshot.pending_writes, default=str):
            anywhere = True

    if not anywhere:
        return "no, never persisted raw"
    return "yes, as a message" if as_message else "yes, in the interrupt payload"

def scenario_pre_gate(engine: str) -> str:
    """Check whether the middleware can also stop a call before it runs."""
    if engine == STATE_REWRITE:
        return UNSUPPORTED
    log: list[str] = []
    agent = create_agent(
        model=ReactiveModel(calls=single_call("charge", {"amount": 10})),
        tools=[charge_tool(log)],
        middleware=[DualGateHumanInTheLoopMiddleware(interrupt_on={"charge": "before"})],
        checkpointer=InMemorySaver(),)
    config = {"configurable": {"thread_id": "pre"}}
    agent.invoke({"messages": [("user", "go")]}, config)
    blocked = log.count("charge(10)") == 0
    agent.invoke(Command(resume={"type": "reject", "message": "no"}), config)
    return "yes, call blocked" if blocked and not log else "leaked"


SCENARIOS = [
    ("accept a result", scenario_accept),
    ("process restart mid-review", scenario_restart),
    ("retry, single-call turn", scenario_retry_single),
    ("retry, call has a sibling", scenario_retry_with_sibling),
    ("tool returning a Command", scenario_command_tool),
    ("raw result in a checkpoint", scenario_raw_in_checkpoint),
    ("pre-execution gate", scenario_pre_gate),]


def main() -> None:
    """Run every scenario against every engine and print the table."""
    rows: list[tuple[str, list[str]]] = []
    for label, scenario in SCENARIOS:
        cells = []
        for engine in ENGINES:
            try:
                cells.append(scenario(engine))
            except Exception as exc:  # a raised error is itself a result here
                cells.append(f"error: {type(exc).__name__}")
        rows.append((label, cells))

    widths = [
        max(len(engine), *(len(cells[i]) for _, cells in rows)) + 2
        for i, engine in enumerate(ENGINES)]
    label_width = max(len(label) for label, _ in rows) + 2

    print()
    print("scenario".ljust(label_width) + "".join(
        e.ljust(w) for e, w in zip(ENGINES, widths, strict=True)))
    print("-" * (label_width + sum(widths)))
    for label, cells in rows:
        print(label.ljust(label_width) + "".join(
            c.ljust(w) for c, w in zip(cells, widths, strict=True)))
    print()
    print("Reading it:")
    print("  - Only the restart row separates the designs on correctness. An in-process")
    print("    ledger loses to a redeploy; a store-backed ledger and the state-rewrite")
    print("    design both hold.")
    print("  - The state-rewrite design pays for that with retry: re-running one call")
    print("    means re-running the whole tools node, so it withholds retry whenever the")
    print("    gated call has a sibling.")
    print("  - Neither design keeps an unreviewed result off disk. The wrap-gate")
    print("    design keeps it out of the message history, but the interrupt payload")
    print("    is persisted too, so a secret in a tool result is on disk either way.")
    print("    Redact before the result is stored, or gate the call before it runs.")

if __name__ == "__main__":
    main()
