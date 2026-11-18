"""Human-in-the-loop middleware for LangChain agents, with a gate on both sides.

LangChain's built-in `HumanInTheLoopMiddleware` pauses before a tool runs. This package
adds a review after it runs, which is harder than it sounds: LangGraph resumes an
interrupted task by re-running it from the top, so a pause placed after a tool sits
downstream of its side effect.

Two answers to that, both implemented and benchmarked against each other in
`benchmarks/compare_designs.py`:

    from dual_gate_hitl import DualGateHumanInTheLoopMiddleware  # wrap-gate + ledger
    from dual_gate_hitl import StateReviewMiddleware             # state rewrite

`DualGateHumanInTheLoopMiddleware` keeps the pause inside `wrap_tool_call` and defends
against the replay with an execution ledger. It carries a pre-execution gate too, can
retry a single call in a multi-call turn, and never lets an unreviewed result reach a
checkpoint.

`StateReviewMiddleware` moves the pause into `before_model` and rewrites the result in
agent state, so the replay is free and the result rides in the checkpoint. It reviews
results only, and withholds retry when the reviewed call has a sibling.

The two compose: use the first one's before gate alongside the second.
"""

from .cache import MISS, MemoryLedger, ResultLedger, StoreLedger
from .middleware import DualGateHumanInTheLoopMiddleware, HumanInTheLoopMiddleware
from .state_gate import ReviewContext, ReviewState, StateReviewMiddleware
from .types import (
    ALL_AFTER_DECISIONS,
    ALL_BEFORE_DECISIONS,
    AcceptDecision,
    Action,
    AfterConfig,
    AfterDecision,
    AfterDecisionType,
    AfterReviewRequest,
    ApproveDecision,
    BeforeConfig,
    BeforeDecision,
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

__all__ = [
    "ALL_AFTER_DECISIONS",
    "ALL_BEFORE_DECISIONS",
    "MISS",
    "AcceptDecision",
    "Action",
    "AfterConfig",
    "AfterDecision",
    "AfterDecisionType",
    "AfterReviewRequest",
    "ApproveDecision",
    "BeforeConfig",
    "BeforeDecision",
    "BeforeDecisionType",
    "BeforeReviewRequest",
    "DualGateHumanInTheLoopMiddleware",
    "EditCallDecision",
    "EditResultDecision",
    "HumanInTheLoopMiddleware",
    "InterruptOnValue",
    "MemoryLedger",
    "RejectDecision",
    "RejectResultDecision",
    "RespondDecision",
    "ResultLedger",
    "RetryDecision",
    "ReviewContext",
    "ReviewState",
    "StateReviewMiddleware",
    "StoreLedger",
    "ToolReviewConfig",
]

__version__ = "0.1.0"
