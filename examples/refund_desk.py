"""A refund desk with a gate on each side of the tool call.

Three tools, three different review needs:

* `issue_refund` moves money, so a human approves the call *before* it runs.
* `fetch_customer_record` is read-only, but its output lands in the model's context,
  so a human reviews the *result* - and only when it looks like it contains PII.
* `check_order_status` is harmless and runs unattended.

Runs with no API key: the model is scripted so the review flow is the only thing moving.
Swap in a real model by replacing `ScriptedModel(...)` with
`init_chat_model("anthropic:claude-sonnet-5-5")`.

    python examples/refund_desk.py
"""

from __future__ import annotations

import re
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from dual_gate_hitl import DualGateHumanInTheLoopMiddleware

PHONE = re.compile(r"\b\d{10}\b")
LEDGER: list[str] = []  # stands in for the payments system


 # Tools
 

@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order. Moves real money."""
    LEDGER.append(f"refund {amount} on {order_id}")
    return f"Refunded {amount:.2f} on order {order_id}."


@tool
def fetch_customer_record(order_id: str) -> str:
    """Fetch the customer record attached to an order."""
    return (
        f"order={order_id} name='Asha Menon' phone=9812345670 "
        "email='asha@example.com' lifetime_value=41200")


@tool
def check_order_status(order_id: str) -> str:
    """Check whether an order has shipped."""
    return f"order={order_id} status=delivered delivered_on=2026-11-09"


 # A scripted model, so the example runs without credentials
 

class ScriptedModel(BaseChatModel):
    """Replays a fixed list of AIMessages, one per invocation."""

    script: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,) -> ChatResult:
        message = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self


SCRIPT = [
    AIMessage(
        content="",
        tool_calls=[
            {"name": "check_order_status", "args": {"order_id": "A-771"}, "id": "t1"},
            {"name": "fetch_customer_record", "args": {"order_id": "A-771"}, "id": "t2"},],
),
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "issue_refund",
                "args": {"order_id": "A-771", "amount": 1299.0}, "id": "t3", }],
                ),AIMessage(content="Refund processed and the customer has been notified."),]


 # The agent
 

def build_agent() -> Any:
    """Wire the middleware up with one gate per risk."""
    return create_agent(
        model=ScriptedModel(script=SCRIPT),
        tools=[issue_refund, fetch_customer_record, check_order_status],
        middleware=[
            DualGateHumanInTheLoopMiddleware(
                interrupt_on={
                    # Money out: approve the call, before anything happens.
                    "issue_refund": {
                        "before": {
                            "allowed_decisions": ["approve", "edit", "reject"],
                            "description": lambda req: (
                                f"Approve a refund of "
                                f"{req.tool_call['args']['amount']:.2f} on order "
                                f"{req.tool_call['args']['order_id']}?"),}},
                    # Read-only, but the output reaches the model. Review the result,
                    # and only when it looks like it carries a phone number.
                    "fetch_customer_record": {
                        "after": {
                            "allowed_decisions": ["accept", "edit", "reject"],
                            "when": lambda req, msg: bool(PHONE.search(str(msg.content))),}},
                    # check_order_status is absent, so it runs unattended.
                },)],
        checkpointer=InMemorySaver(),
        # A store makes the execution ledger durable: a result survives the process
        # dying while a reviewer is away. Swap in Postgres for production.
        store=InMemoryStore(),)


def pending(state: dict[str, Any]) -> dict[str, Any] | None:
    """The review waiting on a human, if there is one."""
    interrupts = state.get("__interrupt__")
    return interrupts[0].value if interrupts else None


def show(review: dict[str, Any]) -> None:
    """Print a review the way a console reviewer would see it."""
    gate = "BEFORE" if review["type"] == "tool_call_review" else "AFTER"
    print(f"\n  [{gate}] {review['name']}")
    print(f"  {review['description'].splitlines()[0]}")
    if gate == "AFTER":
        print(f"  result: {review['result']}")
        print(f"  attempt {review['attempt']}, {review['retries_left']} retries left")
    print(f"  choices: {', '.join(review['allowed_decisions'])}")


def main() -> None:
    agent = build_agent()
    config = {"configurable": {"thread_id": "refund-desk-1"}}

    print("=" * 78)
    print("Customer asks for a refund on order A-771")
    print("=" * 78)

    state = agent.invoke({"messages": [("user", "Refund order A-771")]}, config)

    # 1. The record came back with a phone number, so the AFTER gate opened.
    review = pending(state)
    assert review is not None
    show(review)
    print("  -> reviewer redacts the phone number and lets it through")
    state = agent.invoke(
        Command(
            resume={
                "type": "edit",
                "content": PHONE.sub("[redacted]", review["result"]),}),
        config,)

    # 2. The model then asks for the refund, so the BEFORE gate opens.
    review = pending(state)
    assert review is not None
    show(review)
    print("  -> reviewer approves")
    assert not LEDGER, "nothing should have been charged yet"
    state = agent.invoke(Command(resume={"type": "approve"}), config)

    print("\n" + "=" * 78)
    print("Final answer:", state["messages"][-1].content)
    print("Payments ledger:", LEDGER)
    print("Refunds issued:", len(LEDGER), "(exactly one, across two pauses)")

    print("\nWhat the model ended up seeing:")
    for message in state["messages"]:
        if message.type == "tool":
            print(f"  - {message.name}: {str(message.content)[:88]}")


if __name__ == "__main__":
    main()
