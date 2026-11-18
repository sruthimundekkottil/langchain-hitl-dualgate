"""A human-in-the-loop middleware with a gate on both sides of the tool call."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, Any, Union, cast

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt
from pydantic import Discriminator

from .cache import MISS, MemoryLedger, ResultLedger, StoreLedger
from .types import (
    ALL_AFTER_DECISIONS,
    ALL_BEFORE_DECISIONS,
    AcceptDecision,
    Action,
    AfterConfig,
    AfterDecisionType,
    AfterReviewRequest,
    ApproveDecision,
    BeforeConfig,
    BeforeDecisionType,
    BeforeReviewRequest,
    EditCallDecision,
    EditResultDecision,
    InterruptOnValue,
    RejectDecision,
    RejectResultDecision,
    RespondDecision,
    RetryDecision,
    ToolReviewConfig,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator

    from langchain.tools.tool_node import ToolCallRequest

ToolResult = Union[ToolMessage, "Command[Any]"]

# Steps the policy generator asks the driver to perform. Keeping these out of the policy
# is what lets one implementation serve both the sync and the async hook.
_EXECUTE = "execute"
_LEDGER_GET = "ledger_get"
_LEDGER_PUT = "ledger_put"
_LEDGER_DISCARD = "ledger_discard"


class DualGateHumanInTheLoopMiddleware(AgentMiddleware):
    """Pause for human review before a tool runs, after it runs, or both.

    LangChain's built-in `HumanInTheLoopMiddleware` gates the *decision* to call a tool.
    This one also gates the *result*, which covers a different set of risks: a tool whose
    output goes straight into the model's context (so a human should redact it), a tool
    that is cheap to run but unreliable (so a human should be able to retry it), and a
    tool whose output is only worth a human's attention some of the time (so the gate
    needs to see the result before deciding to ask).

    Both gates are raised from `wrap_tool_call`, one interrupt per tool call, so a
    reviewer answers one concrete question at a time.

    Args:
        interrupt_on: Per tool, which gates apply. `True` gates both sides with every
            decision available, `False` leaves the tool unattended, `"before"` or
            `"after"` picks one side, and a `ToolReviewConfig` gives full control.
            A tool absent from the mapping runs unattended.
        description_prefix: Opening line of the reviewer-facing description of a pending
            tool call.
        after_description_prefix: Opening line of the description of a completed call.
        max_retries: Default cap on `retry` decisions per tool call. A per-tool
            `max_retries` overrides it. Once the budget is spent, `retry` is dropped from
            the decisions offered, so a reviewer is never shown a choice that cannot run.
        edit_notice: Prepended to a result when a reviewer rewrote the arguments, so the
            model knows the call it sees is not the call that ran. `None` adds nothing.
        result_edit_notice: Prepended to a result a reviewer rewrote. `None` adds nothing.
        ledger: Where executed results are recorded so a resume does not run a tool
            twice. Defaults to the agent's LangGraph store when one is configured and an
            in-process ledger otherwise. See `dual_gate_hitl.cache`.

    Example:
        ```python
        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        from dual_gate_hitl import DualGateHumanInTheLoopMiddleware

        agent = create_agent(
            model="anthropic:claude-sonnet-5-5",
            tools=[refund, search_orders],
            middleware=[
                DualGateHumanInTheLoopMiddleware(
                    interrupt_on={
                        # money out: a human approves the call itself
                        "refund": {"before": True},
                        # read-only, but the output reaches the model: review large
                        # result sets only
                        "search_orders": {
                            "after": {
                                "allowed_decisions": ["accept", "edit", "retry"],
                                "when": lambda req, msg: len(str(msg.content)) > 2000,
                            }
                        },
                    }
                )
            ],
            checkpointer=InMemorySaver(),
        )
        ```

    """

    def __init__(  # noqa: PLR0913 - every argument here is a documented knob
        self,
        interrupt_on: dict[str, InterruptOnValue],
        *,
        description_prefix: str = "Tool execution requires approval",
        after_description_prefix: str = "Tool result requires review",
        max_retries: int = 2,
        edit_notice: str | None = "A human reviewer edited this tool call.",
        result_edit_notice: str | None = "A human reviewer edited this tool result.",
        ledger: ResultLedger | None = None,
    ) -> None:
        """Configure the gates. Arguments are documented on the class."""
        super().__init__()
        self.interrupt_on = {
            name: _normalise(name, value) for name, value in interrupt_on.items()
        }
        self.description_prefix = description_prefix
        self.after_description_prefix = after_description_prefix
        self.max_retries = max_retries
        self.edit_notice = edit_notice
        self.result_edit_notice = result_edit_notice
        self._ledger_override = ledger
        self._default_ledger = MemoryLedger()

    def _ledger_for(self, request: ToolCallRequest) -> ResultLedger:
        """Pick where executed results are recorded.

        An explicit ledger always wins. Otherwise the agent's LangGraph store is used
        when one is configured, because a store write lands immediately and survives the
        process; without a store, results are kept in memory for the life of the process.
        """
        if self._ledger_override is not None:
            return self._ledger_override
        store = getattr(request.runtime, "store", None)
        if store is not None:
            return StoreLedger(store, fallback=self._default_ledger)
        return self._default_ledger

    # ----------------------------------------------------------------------------------
    # Hooks
    # ----------------------------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolResult],
    ) -> ToolResult:
        """Run the review policy, executing the tool through `handler`."""
        config = self.interrupt_on.get(request.tool_call["name"])
        if config is None:
            return handler(request)

        ledger = self._ledger_for(request)
        policy = self._policy(request, config)
        try:
            step = next(policy)
            while True:
                kind = step[0]
                if kind == _EXECUTE:
                    sent: Any = handler(step[1])
                elif kind == _LEDGER_GET:
                    sent = ledger.get(step[1])
                elif kind == _LEDGER_PUT:
                    ledger.put(step[1], step[2])
                    sent = None
                else:
                    ledger.discard(step[1])
                    sent = None
                step = policy.send(sent)
        except StopIteration as stop:
            return cast("ToolResult", stop.value)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolResult]],
    ) -> ToolResult:
        """Async variant. The review policy itself is shared with `wrap_tool_call`."""
        config = self.interrupt_on.get(request.tool_call["name"])
        if config is None:
            return await handler(request)

        ledger = self._ledger_for(request)
        policy = self._policy(request, config)
        try:
            step = next(policy)
            while True:
                kind = step[0]
                if kind == _EXECUTE:
                    sent: Any = await handler(step[1])
                elif kind == _LEDGER_GET:
                    sent = await ledger.aget(step[1])
                elif kind == _LEDGER_PUT:
                    await ledger.aput(step[1], step[2])
                    sent = None
                else:
                    await ledger.adiscard(step[1])
                    sent = None
                step = policy.send(sent)
        except StopIteration as stop:
            return cast("ToolResult", stop.value)

    # ----------------------------------------------------------------------------------
    # The policy: one generator, driven by either hook
    # ----------------------------------------------------------------------------------

    # The policy is long because it is one linear narrative - two gates and a retry
    # loop - and splitting it would hide the ordering that makes it replay-safe.
    def _policy(  # noqa: C901, PLR0911, PLR0912, PLR0915
        self,
        request: ToolCallRequest,
        config: ToolReviewConfig,
    ) -> Generator[tuple[Any, ...], Any, ToolResult]:
        """Yield work for the driver and return the result the model should see.

        Written as a generator for one reason: every `interrupt()` in here is replayed
        on resume, so the policy must be deterministic and the only non-deterministic
        parts - running the tool, reading the ledger - must be handed out to a driver
        that can be sync or async.
        """
        # An enabled gate with no options is an empty dict, which is falsy - so these
        # have to be identity checks, not truthiness checks.
        raw_before = config.get("before")
        raw_after = config.get("after")
        before = cast(
            "BeforeConfig | None", raw_before if isinstance(raw_before, dict) else None
        )
        after = cast(
            "AfterConfig | None", raw_after if isinstance(raw_after, dict) else None
        )
        tool_call = request.tool_call

        # -- gate 1: before the tool runs ---------------------------------------------
        edited_call: Action | None = None
        if before is not None and _passes(before.get("when"), request):
            allowed = cast(
                "list[BeforeDecisionType]",
                before.get("allowed_decisions") or list(ALL_BEFORE_DECISIONS),
            )
            decision = interrupt(
                self._before_payload(request, before, allowed),
                response_schema=_before_schema(allowed),
            )
            kind = _decision_type(decision, allowed, tool_call["name"], "before")
            if kind == "edit":
                edited_call = cast("EditCallDecision", decision)["edited_action"]
                request = _apply_edit(request, edited_call)
            elif kind in ("reject", "respond"):
                return _refusal_message(
                    tool_call, cast("RejectDecision | RespondDecision", decision)
                )
            # "approve" falls through to execution.

        if after is None:
            # No second gate: execute once, straight through. Nothing is replayed after
            # this point, so the ledger is not needed.
            result = yield (_EXECUTE, request)
            return self._note_edited_call(result, request, edited_call)

        tool_call_id = tool_call.get("id")
        if not tool_call_id:
            msg = (
                f"Tool call {tool_call['name']!r} has no id, so its result cannot be "
                "tracked across a pause. A post-execution gate needs one: make sure the "
                "chat model returns tool call ids."
            )
            raise ValueError(msg)

        # -- gate 2: after the tool runs -----------------------------------------------
        budget = after.get("max_retries", self.max_retries)
        keys: list[str] = []
        attempt = 0
        while True:
            key = f"{tool_call_id}:{attempt}"
            if key not in keys:
                keys.append(key)

            # The ledger is read *before* the tool is called, so a replay reuses the
            # result instead of re-running the tool.
            recorded = yield (_LEDGER_GET, key)
            if recorded is MISS:
                result = yield (_EXECUTE, request)
                yield (_LEDGER_PUT, key, result)
            else:
                result = recorded

            result = self._note_edited_call(result, request, edited_call)
            message = _as_tool_message(result, tool_call_id)

            if message is None or not _passes_after(after.get("when"), request, message):
                yield (_LEDGER_DISCARD, keys)
                return result

            retries_left = max(budget - attempt, 0)
            allowed_after = cast(
                "list[AfterDecisionType]",
                after.get("allowed_decisions") or list(ALL_AFTER_DECISIONS),
            )
            if retries_left == 0:
                allowed_after = [d for d in allowed_after if d != "retry"]
            if not allowed_after:
                # Every decision was configured away; nothing to ask.
                yield (_LEDGER_DISCARD, keys)
                return result

            decision = interrupt(
                self._after_payload(
                    request,
                    after,
                    message,
                    allowed=allowed_after,
                    attempt=attempt,
                    retries_left=retries_left,
                ),
                response_schema=_after_schema(allowed_after),
            )
            kind = _decision_type(decision, allowed_after, tool_call["name"], "after")

            if kind == "accept":
                yield (_LEDGER_DISCARD, keys)
                return result

            if kind == "edit":
                edit = cast("EditResultDecision", decision)
                yield (_LEDGER_DISCARD, keys)
                return _swap_tool_message(
                    result,
                    tool_call_id,
                    self._note_edited_result(
                        message.model_copy(
                            update={
                                "content": edit.get("content", message.content),
                                "status": edit.get("status", "success"),
                            }
                        )
                    ),
                )

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
                        "released. Do not assume anything about what it returned."
                    )
                )
                yield (_LEDGER_DISCARD, keys)
                # Only the message is replaced: a Command's state updates are dropped
                # along with the result the reviewer rejected.
                return ToolMessage(
                    content=content,
                    name=tool_call["name"],
                    tool_call_id=tool_call_id,
                    status="error",
                )

            # retry
            retry = cast("RetryDecision", decision)
            if "args" in retry:
                request = _apply_edit(
                    request, {"name": tool_call["name"], "args": retry["args"]}
                )
            attempt += 1

    # ----------------------------------------------------------------------------------
    # Interrupt payloads
    # ----------------------------------------------------------------------------------

    def _before_payload(
        self,
        request: ToolCallRequest,
        config: BeforeConfig,
        allowed: list[BeforeDecisionType],
    ) -> BeforeReviewRequest:
        description = config.get("description")
        if callable(description):
            text = description(request)
        elif description:
            text = description
        else:
            text = (
                f"{self.description_prefix}\n\n"
                f"Tool: {request.tool_call['name']}\n"
                f"Args: {json.dumps(request.tool_call['args'], default=str)}"
            )
        return BeforeReviewRequest(
            type="tool_call_review",
            tool_call_id=request.tool_call.get("id") or "",
            name=request.tool_call["name"],
            args=request.tool_call["args"],
            description=text,
            allowed_decisions=allowed,
        )

    def _after_payload(  # noqa: PLR0913 - the payload needs all of this
        self,
        request: ToolCallRequest,
        config: AfterConfig,
        message: ToolMessage,
        *,
        allowed: list[AfterDecisionType],
        attempt: int,
        retries_left: int,
    ) -> AfterReviewRequest:
        description = config.get("description")
        if callable(description):
            text = description(request, message)
        elif description:
            text = description
        else:
            text = (
                f"{self.after_description_prefix}\n\n"
                f"Tool: {request.tool_call['name']}\n"
                f"Args: {json.dumps(request.tool_call['args'], default=str)}"
            )
        return AfterReviewRequest(
            type="tool_result_review",
            tool_call_id=request.tool_call["id"] or "",
            name=request.tool_call["name"],
            args=request.tool_call["args"],
            result=_as_text(message.content),
            status=message.status or "success",
            attempt=attempt,
            retries_left=retries_left,
            description=text,
            allowed_decisions=allowed,
        )

    # ----------------------------------------------------------------------------------
    # Notices
    # ----------------------------------------------------------------------------------

    def _note_edited_call(
        self,
        result: ToolResult,
        request: ToolCallRequest,
        edited: Action | None,
    ) -> ToolResult:
        """Tell the model which call actually ran, when a reviewer changed it."""
        if edited is None or not self.edit_notice:
            return result
        tool_call_id = request.tool_call.get("id")
        message = _as_tool_message(result, tool_call_id)
        if message is None:
            return result
        notice = (
            f"{self.edit_notice} Executed instead: {edited['name']} with arguments "
            f"{json.dumps(edited['args'], default=str)}."
        )
        return _swap_tool_message(
            result, tool_call_id, _prepend(message, notice)
        )

    def _note_edited_result(self, message: ToolMessage) -> ToolMessage:
        if not self.result_edit_notice:
            return message
        return _prepend(message, self.result_edit_notice)


#: Shorter alias, for a drop-in swap with the built-in middleware's import name.
HumanInTheLoopMiddleware = DualGateHumanInTheLoopMiddleware


# --------------------------------------------------------------------------------------
# Config normalisation
# --------------------------------------------------------------------------------------


def _normalise(name: str, value: InterruptOnValue) -> ToolReviewConfig | None:
    """Expand the shorthands into a full `ToolReviewConfig`."""
    if value is False:
        return None
    if value is True or value == "both":
        return {"before": {}, "after": {}}
    if value == "before":
        return {"before": {}}
    if value == "after":
        return {"after": {}}
    if not isinstance(value, dict):
        msg = (
            f"interrupt_on[{name!r}] must be a bool, 'before', 'after', 'both', or a "
            f"ToolReviewConfig dict; got {value!r}."
        )
        raise TypeError(msg)

    config: ToolReviewConfig = {}
    for gate in ("before", "after"):
        gate_value = value.get(gate)
        if gate_value is None or gate_value is False:
            continue
        config[gate] = {} if gate_value is True else dict(gate_value)  # type: ignore[literal-required]
    if not config:
        msg = (
            f"interrupt_on[{name!r}] enables neither gate. Use False to leave the tool "
            "unattended, or set 'before' and/or 'after'."
        )
        raise ValueError(msg)
    _validate_decisions(name, config)
    return config


def _validate_decisions(name: str, config: ToolReviewConfig) -> None:
    for gate, valid in (("before", ALL_BEFORE_DECISIONS), ("after", ALL_AFTER_DECISIONS)):
        gate_config = config.get(gate)
        if not isinstance(gate_config, dict):
            continue
        allowed = gate_config.get("allowed_decisions")
        if allowed is None:
            continue
        if not allowed:
            msg = f"interrupt_on[{name!r}]['{gate}']['allowed_decisions'] is empty."
            raise ValueError(msg)
        unknown = [d for d in allowed if d not in valid]
        if unknown:
            msg = (
                f"interrupt_on[{name!r}]['{gate}']['allowed_decisions'] contains "
                f"{unknown!r}; the {gate} gate accepts {list(valid)!r}."
            )
            raise ValueError(msg)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _passes(predicate: Any, request: ToolCallRequest) -> bool:
    return True if predicate is None else bool(predicate(request))


def _passes_after(predicate: Any, request: ToolCallRequest, message: ToolMessage) -> bool:
    return True if predicate is None else bool(predicate(request, message))


def _decision_type(
    decision: Any,
    allowed: list[Any],
    tool_name: str,
    gate: str,
) -> str:
    """Validate the reviewer's answer and return its type.

    `interrupt()` already checks the answer against `response_schema`, but a client can
    be talking to an older LangGraph, so the policy does not trust that it did.
    """
    if not isinstance(decision, dict) or "type" not in decision:
        msg = (
            f"The {gate} gate for {tool_name!r} expected a decision dict with a 'type' "
            f"key; got {decision!r}."
        )
        raise ValueError(msg)
    kind = decision["type"]
    if kind not in allowed:
        msg = (
            f"Decision {kind!r} is not allowed by the {gate} gate for {tool_name!r}. "
            f"Expected one of {allowed!r}."
        )
        raise ValueError(msg)
    if kind == "respond" and not decision.get("message"):
        msg = f"A 'respond' decision for {tool_name!r} needs a 'message'."
        raise ValueError(msg)
    if kind == "edit" and gate == "before" and "edited_action" not in decision:
        msg = f"An 'edit' decision for {tool_name!r} needs an 'edited_action'."
        raise ValueError(msg)
    return cast("str", kind)


def _apply_edit(request: ToolCallRequest, action: Action) -> ToolCallRequest:
    """Point the request at the reviewer's call, resolving a redirected tool."""
    tool_call = {**request.tool_call, "name": action["name"], "args": action["args"]}
    if action["name"] == request.tool_call["name"]:
        return request.override(tool_call=tool_call)  # type: ignore[arg-type]

    available = list(getattr(request.runtime, "tools", []) or [])
    tool = next((t for t in available if t.name == action["name"]), None)
    if tool is None:
        names = ", ".join(sorted(t.name for t in available))
        msg = (
            f"A reviewer redirected tool call {request.tool_call.get('id')!r} to "
            f"{action['name']!r}, which this agent does not have. Available: {names}."
        )
        raise ValueError(msg)
    # `tool_call['name']` and `tool` have to stay in agreement.
    return replace(request, tool_call=tool_call, tool=tool)  # type: ignore[arg-type]


def _refusal_message(
    tool_call: Any, decision: RejectDecision | RespondDecision
) -> ToolMessage:
    """Build the message the model gets in place of a call that never ran."""
    if decision["type"] == "respond":
        content = cast("RespondDecision", decision)["message"]
    elif reason := decision.get("message"):
        content = (
            f"User rejected the tool call for `{tool_call['name']}` with reason: {reason}"
        )
    else:
        content = (
            f"User rejected the tool call for `{tool_call['name']}` with id "
            f"{tool_call.get('id')}. The tool was not executed. Do not retry this tool "
            "call unless the user explicitly requests it."
        )
    return ToolMessage(
        content=content,
        name=tool_call["name"],
        tool_call_id=tool_call.get("id") or "",
        status="success" if decision["type"] == "respond" else "error",
    )


def _as_tool_message(result: ToolResult, tool_call_id: str | None) -> ToolMessage | None:
    """Find the `ToolMessage` a result carries, bare or inside a `Command`."""
    if isinstance(result, ToolMessage):
        return result
    if not isinstance(result, Command) or not isinstance(result.update, dict):
        return None
    messages = result.update.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, ToolMessage) and (
            tool_call_id is None or message.tool_call_id == tool_call_id
        ):
            return message
    return None


def _swap_tool_message(
    result: ToolResult, tool_call_id: str | None, new_message: ToolMessage
) -> ToolResult:
    """Put `new_message` where the old tool message was, keeping any state update."""
    if isinstance(result, ToolMessage):
        return new_message
    if not isinstance(result, Command) or not isinstance(result.update, dict):
        return result
    messages = result.update.get("messages")
    if not isinstance(messages, list):
        return result
    return replace(
        result,
        update={
            **result.update,
            "messages": [
                new_message
                if isinstance(m, ToolMessage)
                and (tool_call_id is None or m.tool_call_id == tool_call_id)
                else m
                for m in messages
            ],
        },
    )


def _prepend(message: ToolMessage, notice: str) -> ToolMessage:
    """Prepend `notice` to a message's content, for either content shape."""
    if isinstance(message.content, str):
        if notice in message.content:
            return message
        separator = "\n\n" if message.content else ""
        return message.model_copy(
            update={"content": f"{notice}{separator}{message.content}"}
        )
    if any(notice in str(block) for block in message.content):
        return message
    # Match the surrounding block shape; some providers reject mixed lists.
    block: Any = (
        notice
        if message.content and all(isinstance(b, str) for b in message.content)
        else {"type": "text", "text": notice}
    )
    return message.model_copy(update={"content": [block, *message.content]})


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str)


# --------------------------------------------------------------------------------------
# Response schemas, so a client can validate a decision before sending it
# --------------------------------------------------------------------------------------

_BEFORE_TYPES: dict[str, Any] = {
    "approve": ApproveDecision,
    "edit": EditCallDecision,
    "reject": RejectDecision,
    "respond": RespondDecision,
}
_AFTER_TYPES: dict[str, Any] = {
    "accept": AcceptDecision,
    "edit": EditResultDecision,
    "retry": RetryDecision,
    "reject": RejectResultDecision,
}


def _schema(types: dict[str, Any], allowed: list[str]) -> Any:
    members = tuple(types[d] for d in dict.fromkeys(allowed) if d in types)
    if not members:
        return None
    if len(members) == 1:
        # A union of one cannot carry a Discriminator.
        return members[0]
    # Check only the branch the answer's `type` names, so a bad answer produces one
    # precise error rather than one per branch.
    return Annotated[Union[members], Discriminator("type")]  # noqa: UP007


def _before_schema(allowed: list[BeforeDecisionType]) -> Any:
    return _schema(_BEFORE_TYPES, list(allowed))


def _after_schema(allowed: list[AfterDecisionType]) -> Any:
    return _schema(_AFTER_TYPES, list(allowed))
