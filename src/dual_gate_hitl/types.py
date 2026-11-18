"""Decision and configuration types for the dual-gate HITL middleware.

Two vocabularies live here:

* **Pre-execution decisions** (`approve`, `edit`, `reject`, `respond`) - deliberately
  identical to LangChain's built-in ``HumanInTheLoopMiddleware`` so existing review
  UIs keep working.
* **Post-execution decisions** (`accept`, `edit`, `retry`, `reject`) - new. These act
  on a result that already exists, so the verbs differ: you cannot "approve" a thing
  that has already happened, you can only accept it, rewrite it, run it again, or
  withhold it from the model.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Protocol,
    Required,
    TypedDict,)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.messages import ToolMessage

 # Pre-execution decisions (mirror the built-in middleware)
 
BeforeDecisionType = Literal["approve", "edit", "reject", "respond"]


class Action(TypedDict):
    """A tool call as a reviewer sees and edits it."""

    name: str
    args: dict[str, Any]


class ApproveDecision(TypedDict):
    """Run the tool with the arguments the model chose."""

    type: Literal["approve"]


class EditCallDecision(TypedDict):
    """Run the tool, but with arguments the reviewer rewrote."""

    type: Literal["edit"]
    edited_action: Action


class RejectDecision(TypedDict, total=False):
    """Do not run the tool. The model is told it was rejected."""

    type: Required[Literal["reject"]]
    message: str


class RespondDecision(TypedDict):
    """Do not run the tool. The reviewer answers in the tool's place."""

    type: Literal["respond"]
    message: str


BeforeDecision = Annotated[
    ApproveDecision | EditCallDecision | RejectDecision | RespondDecision,
    "discriminated on `type`",]

# Post-execution decisions (the part the built-in middleware has no equivalent for)
 
AfterDecisionType = Literal["accept", "edit", "retry", "reject"]


class AcceptDecision(TypedDict):
    """Pass the tool's result to the model unchanged."""

    type: Literal["accept"]


class EditResultDecision(TypedDict, total=False):
    """Pass a result the reviewer rewrote - redaction, truncation, correction."""

    type: Required[Literal["edit"]]
    content: str
    status: Literal["success", "error"]


class RetryDecision(TypedDict, total=False):
    """Discard this result and run the tool again, optionally with new arguments.

    The new result goes through the post-execution gate again, up to `max_retries`.
    """

    type: Required[Literal["retry"]]
    args: dict[str, Any]
    message: str


class RejectResultDecision(TypedDict, total=False):
    """Withhold the result. The model is told the result was not released."""

    type: Required[Literal["reject"]]
    message: str


AfterDecision = Annotated[
    AcceptDecision | EditResultDecision | RetryDecision | RejectResultDecision,
    "discriminated on `type`",]

ALL_BEFORE_DECISIONS: tuple[BeforeDecisionType, ...] = (
    "approve",
    "edit",
    "reject",
    "respond",)
ALL_AFTER_DECISIONS: tuple[AfterDecisionType, ...] = (
    "accept",
    "edit",
    "retry",
    "reject",)

# Interrupt payloads
 

class BeforeReviewRequest(TypedDict):
    """What a client receives when a tool call is waiting for approval."""

    type: Literal["tool_call_review"]
    tool_call_id: str
    name: str
    args: dict[str, Any]
    description: str
    allowed_decisions: list[BeforeDecisionType]


class AfterReviewRequest(TypedDict):
    """What a client receives when a tool *result* is waiting for review."""

    type: Literal["tool_result_review"]
    tool_call_id: str
    name: str
    args: dict[str, Any]
    result: str
    status: str
    attempt: int
    retries_left: int
    description: str
    allowed_decisions: list[AfterDecisionType]


# Per-tool configuration
 

class BeforeDescriptionFactory(Protocol):
    """Builds the reviewer-facing description of a pending tool call."""

    def __call__(self, request: ToolCallRequest) -> str:
        """Describe the call a reviewer is being asked to approve."""
        ...


class AfterDescriptionFactory(Protocol):
    """Builds the reviewer-facing description of a completed tool call."""

    def __call__(self, request: ToolCallRequest, result: ToolMessage) -> str:
        """Describe the call and result a reviewer is being asked to review."""
        ...


class BeforeConfig(TypedDict, total=False):
    """How a tool call is gated before it runs."""

    allowed_decisions: list[BeforeDecisionType]
    description: str | BeforeDescriptionFactory
    when: Callable[[ToolCallRequest], bool]


class AfterConfig(TypedDict, total=False):
    """How a tool result is gated after the tool runs.

    `when` is the feature with no counterpart in the built-in middleware: it is handed
    the result as well as the call, so a tool can run unattended and escalate to a
    human only when its *output* warrants it.
    """

    allowed_decisions: list[AfterDecisionType]
    description: str | AfterDescriptionFactory
    when: Callable[[ToolCallRequest, ToolMessage], bool]
    max_retries: int


class ToolReviewConfig(TypedDict, total=False):
    """Per-tool gating. Omit a gate, or set it to False, to skip it."""

    before: bool | BeforeConfig
    after: bool | AfterConfig


#: What `interrupt_on` accepts per tool.
#:
#: * ``True``  - both gates, every decision
#: * ``False`` - no gates (the tool runs unattended)
#: * ``"before"`` / ``"after"`` - that one gate only, every decision
#: * ``ToolReviewConfig`` - full control
InterruptOnValue = bool | Literal["before", "after", "both"] | ToolReviewConfig
