"""Design B: review tool results by rewriting them in agent state.

The other design in this package (`DualGateHumanInTheLoopMiddleware`) reviews a result
from inside `wrap_tool_call`, which means the pause happens downstream of the tool's side
effect and needs an execution ledger to stay exactly-once.

This one moves the pause instead of defending against it. The tools node runs to
completion and its results land in state. The review happens in `before_model`, the hook
that runs after the tools node and before the model sees anything, and applies a decision
by replacing the `ToolMessage` in state - reusing its id, which is how `add_messages`
does an in-place update.

That buys one thing the ledger cannot: the tool's result is in the **checkpoint**, so it
survives anything, including the process dying mid-review, with no sidecar storage and no
exactly-once machinery to get right. A replay of `before_model` re-reads state and
re-raises an `interrupt()` that returns the stored answer; nothing re-executes because
nothing in the hook has a side effect.

It also costs two things, both measured rather than assumed:

* **Retry is only safe when the model turn made a single tool call.** Re-running a tool
  means jumping back to the tools node, and that node re-executes *every* tool call on
  the last AI message - including ungated ones, which fire a second time, and including
  calls whose `ToolMessage` is still in state, which produces two results for one
  `tool_call_id` and a message history providers reject. So `retry` is offered only when
  the gated result is the only result in the turn.
* **The result exists in state before a human sees it.** It is replaced before the model
  is called, so the model never reads the unredacted version, but a checkpoint written
  between the tools node and the review does contain it. Where the raw value must never
  be persisted at all, gate the call before it runs.

Use this when results are what you review and durability matters. Use the dual-gate
middleware when you need a pre-execution gate, per-call retry, or review of a tool that
returns a `Command`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langgraph.types import interrupt

from .middleware import _after_schema, _as_text, _decision_type
from .types import (
    ALL_AFTER_DECISIONS,
    AfterConfig,
    AfterDecisionType,
    AfterReviewRequest,
    EditResultDecision,
    RejectResultDecision,
    RetryDecision,)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

#: State key holding retry counts per tool call id. Namespaced to avoid colliding with
#: anything the user's own state schema carries.
ATTEMPTS_KEY = "dual_gate_review_attempts"


class ReviewState(AgentState):
    """Agent state plus the retry bookkeeping this middleware needs."""

    dual_gate_review_attempts: dict[str, int]


@dataclass
class ReviewContext:
    """Stands in for a `ToolCallRequest` when a predicate runs outside the tool node.

    `when` and `description` callables written for the dual-gate middleware mostly reach
    for `request.tool_call`, so the same callables work with either design. The
    difference: `tool` is always `None` here, because the review no longer happens where
    the tool object is in scope.
    """

    tool_call: dict[str, Any]
    state: Any
    runtime: Any = None
    tool: None = field(default=None, init=False)


class StateReviewMiddleware(AgentMiddleware):
    """Review tool results in the agent loop, by rewriting them in state.

    Args:
        review_on: Per tool, `True` to review every result with every decision, `False`
            to skip, or an `AfterConfig` for `allowed_decisions`, `description`, `when`
            and `max_retries`. A tool not named here is never reviewed.
        description_prefix: Opening line of the reviewer-facing description.
        max_retries: Default cap on `retry` decisions per tool call.
        result_edit_notice: Prepended to a result a reviewer rewrote. `None` adds nothing.

    Example:
        ```python
        agent = create_agent(
            model="anthropic:claude-sonnet-5-5",
            tools=[fetch_customer_record],
            middleware=[
                StateReviewMiddleware(
                    review_on={
                        "fetch_customer_record": {
                            "allowed_decisions": ["accept", "edit", "reject"],
                            "when": lambda req, msg: "phone=" in str(msg.content),}} ) ],
            checkpointer=InMemorySaver(),)```

    """

    state_schema = ReviewState

    def __init__(
        self,
        review_on: dict[str, bool | AfterConfig],
        *,
        description_prefix: str = "Tool result requires review",
        max_retries: int = 2,
        result_edit_notice: str | None = "A human reviewer edited this tool result.", ) -> None:
        """Configure the review. Arguments are documented on the class."""
        super().__init__()
        self.review_on: dict[str, AfterConfig] = {}
        for name, value in review_on.items():
            if value is False:
                continue
            config = cast("AfterConfig", {} if value is True else dict(value))
            _validate(name, config)
            self.review_on[name] = config
        self.description_prefix = description_prefix
        self.max_retries = max_retries
        self.result_edit_notice = result_edit_notice

    @hook_config(can_jump_to=["tools"])
    # One linear pass over the turn's results; splitting it would hide the ordering.
    def before_model(  # noqa: C901
        self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:
        """Review the results of the tools node that just ran.

        Nothing in here has a side effect, which is the whole point: `before_model` is
        replayed on every resume and that replay has to be free.
        """
        messages = _messages(state)
        pending = _trailing_tool_messages(messages)
        if not pending:
            return None

        calls = _tool_calls_by_id(messages)
        attempts: dict[str, int] = dict(state.get(ATTEMPTS_KEY) or {})
        # Retry re-runs the whole tools node, so it is only safe when there is nothing
        # beside the reviewed call to re-run.
        retry_safe = len(pending) == 1
        updates: list[Any] = []

        for message in pending:
            config = self.review_on.get(message.name or "")
            if config is None:
                continue
            tool_call = calls.get(
                message.tool_call_id, {"name": message.name, "args": {}})
            context = ReviewContext(tool_call=tool_call, state=state, runtime=runtime)

            predicate = config.get("when")
            if predicate is not None and not predicate(context, message):
                continue

            attempt = attempts.get(message.tool_call_id, 0)
            budget = config.get("max_retries", self.max_retries)
            retries_left = max(budget - attempt, 0) if retry_safe else 0
            allowed = cast(
                "list[AfterDecisionType]",
                config.get("allowed_decisions") or list(ALL_AFTER_DECISIONS),)
            if retries_left == 0:
                allowed = [d for d in allowed if d != "retry"]
            if not allowed:
                continue

            decision = interrupt(
                self._payload(
                    tool_call,
                    message,
                    config,
                    allowed=allowed,
                    attempt=attempt,
                    retries_left=retries_left, ),
                response_schema=_after_schema(allowed),)
            kind = _decision_type(decision, allowed, tool_call["name"], "after")

            if kind == "accept":
                continue

            if kind == "edit":
                edit = cast("EditResultDecision", decision)
                updates.append(
                    _rewrite(
                        message,
                        content=edit.get("content", message.content),
                        status=edit.get("status", "success"),
                        notice=self.result_edit_notice,))
                continue

            if kind == "reject":
                reject = cast("RejectResultDecision", decision)
                reason = reject.get("message")
                content = (
                    f"A human reviewer withheld the result of `{tool_call['name']}`: "
                    f"{reason}"
                    if reason
                    else (
                        f"A human reviewer withheld the result of "
                        f"`{tool_call['name']}`. The tool ran, but its output was not "
                        "released. Do not assume anything about what it returned."))
                updates.append(
                    _rewrite(message, content=content, status="error", notice=None))
                continue

            # retry: drop the result and send the loop back through the tools node.
            retry = cast("RetryDecision", decision)
            attempts[message.tool_call_id] = attempt + 1
            update: dict[str, Any] = {
                "messages": [RemoveMessage(id=message.id)],
                ATTEMPTS_KEY: attempts,
                "jump_to": "tools",}
            if "args" in retry:
                # The tools node reads the call off the AI message, so a change of
                # arguments has to be written there.
                revised = _revise_tool_args(
                    messages, message.tool_call_id, retry["args"])
                if revised is not None:
                    update["messages"] = [revised, RemoveMessage(id=message.id)]
            return update

        if not updates:
            return None
        return {"messages": updates}

    async def abefore_model(
        self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:
        """Async variant. The hook does no I/O, so it shares the sync implementation."""
        return self.before_model(state, runtime)

    def _payload(  # noqa: PLR0913 - the payload needs all of this
        self,
        tool_call: dict[str, Any],
        message: ToolMessage,
        config: AfterConfig,
        *,
        allowed: list[AfterDecisionType],
        attempt: int,
        retries_left: int,) -> AfterReviewRequest:
        description = config.get("description")
        context = ReviewContext(tool_call=tool_call, state=None)
        if callable(description):
            text = description(context, message)
        elif description:
            text = description
        else:
            text = (
                f"{self.description_prefix}\n\nTool: {tool_call['name']}\n"
                f"Args: {tool_call.get('args', {})}")
        return AfterReviewRequest(
            type="tool_result_review",
            tool_call_id=message.tool_call_id,
            name=tool_call["name"],
            args=tool_call.get("args", {}),
            result=_as_text(message.content),
            status=message.status or "success",
            attempt=attempt,
            retries_left=retries_left,
            description=text,
            allowed_decisions=allowed,   )


 # Helpers
 

def _validate(name: str, config: AfterConfig) -> None:
    allowed = config.get("allowed_decisions")
    if allowed is None:
        return
    if not allowed:
        msg = f"review_on[{name!r}]['allowed_decisions'] is empty."
        raise ValueError(msg)
    unknown = [d for d in allowed if d not in ALL_AFTER_DECISIONS]
    if unknown:
        msg = (
            f"review_on[{name!r}]['allowed_decisions'] contains {unknown!r}; this "
            f"middleware reviews results, so it accepts {list(ALL_AFTER_DECISIONS)!r}.")
        raise ValueError(msg)


def _messages(state: Any) -> list[Any]:
    if isinstance(state, dict):
        return list(state.get("messages") or [])
    return list(getattr(state, "messages", []) or [])


def _trailing_tool_messages(messages: list[Any]) -> list[ToolMessage]:
    """Collect the results of the most recent tools node run, in order.

    They are the unbroken run of `ToolMessage`s at the end of the history: the tools node
    appends one per call and the model has not spoken since.
    """
    trailing: list[ToolMessage] = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            trailing.append(message)
        else:
            break
    trailing.reverse()
    return trailing


def _tool_calls_by_id(messages: list[Any]) -> dict[str, dict[str, Any]]:
    """Map tool call id to the call the model asked for, from the latest AI message."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.tool_calls:
            return {
                call["id"]: dict(call) for call in message.tool_calls if call.get("id")}
    return {}


def _rewrite(
    message: ToolMessage,
    *,
    content: Any,
    status: str,
    notice: str | None,) -> ToolMessage:
    """Build a replacement carrying `message`'s id, so `add_messages` updates it."""
    text = content
    if notice and isinstance(text, str) and notice not in text:
        text = f"{notice}\n\n{text}" if text else notice
    return ToolMessage(
        content=text,
        id=message.id,
        name=message.name,
        tool_call_id=message.tool_call_id,
        status=cast("Any", status),)


def _revise_tool_args(
    messages: list[Any], tool_call_id: str, args: dict[str, Any]) -> AIMessage | None:
    """Rewrite one call's arguments on the AI message that requested it."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.tool_calls:
            if not any(c.get("id") == tool_call_id for c in message.tool_calls):
                return None
            calls = [
                {**c, "args": args} if c.get("id") == tool_call_id else dict(c)
                for c in message.tool_calls]
            return cast(
                "AIMessage",
                message.model_copy(update={"tool_calls": calls}),)
    return None
