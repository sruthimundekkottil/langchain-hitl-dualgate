# dual-gate-hitl

Human-in-the-loop middleware for LangChain agents that reviews a tool call on both
sides — before it runs and after it returns — built two different ways, and benchmarked
against each other.

LangChain's built-in `HumanInTheLoopMiddleware` pauses only before a tool runs. But in certain cases, an option for the human to check its response after the model calls a tool before the model reads it is also needed.
For eg: If a ecord comes back with a customer's phone number in it , post- execution gate can redact before the model sees it. Or when  SQL query returns 40k rows that will blow the context window, it can truncate or reject.
 Adding a review after it runs turns out to be a genuinely different engineering problem,because LangGraph resumes an interrupted task by re-running it from the top. This repo contains both solutions to that problem, a measured comparison, and tests that pin thetradeoff down.

```bash
python -m pytest -q                        # 54 tests
python benchmarks/compare_designs.py       # the comparison table below
python examples/refund_desk.py             # end-to-end demo, no API key
```

---

## What the pre-execution gate misses
Pre-execution gate covers one class of risk: the agent is about to do something it
should not. It does not cover the other class: the tool already ran, and its output is
about to become part of the model's context and the user's answer.

## The replay problem
From LangGraph's own docs: when you resume after `interrupt()`, **the task restarts from
the beginning**. `interrupt()` itself is replay-safe — second time through it returns the
stored answer instead of pausing — but everything before it runs again.

A pre-execution gate is therefore trivially safe: nothing has happened yet. A
post-execution gate sits *downstream of a side effect*, so the naive version charges the
card twice:

```python
# DON'T: measured on langchain 1.5.0 / langgraph 1.2.22, this charges twice
def wrap_tool_call(self, request, handler):
    result = handler(request)              # money leaves the account
    interrupt({"result": result.content})  # pause  -> on resume, re-runs from here up
    return result
```

That control case is kept as a passing test — `test_naive_post_gate_double_executes` —
so the hazard stays visible and provable.

## Two designs

There are two ways out, and they are not equivalent.

### A. Wrap-gate + execution ledger — `DualGateHumanInTheLoopMiddleware`

Keep the pause where the tool is, and **defend against the replay**. The result is
written to a ledger the moment the tool returns, and the ledger is read *before* the tool
is called. On replay the result is already there, so the tool is not called again.

```
first pass:   ledger miss -> run tool -> write ledger -> interrupt (pause)
on resume:    ledger hit  -> skip tool -> interrupt returns the human's decision
```

Two measured facts make one entry per `(tool_call_id, attempt)` sufficient: the replay is
scoped to the task that paused (a sibling tool call in the same turn is not re-run), and
the entry must live *outside* the checkpoint, because a paused super-step never commits.
A LangGraph store write lands immediately, which is exactly what the replay needs — hence
`StoreLedger` (durable) and `MemoryLedger` (bounded, in-process fallback).

This design also carries the pre-execution gate, so one middleware does both sides.

### B. State rewrite — `StateReviewMiddleware`

**Move the pause instead of defending against it.** Let the tools node finish and its
results land in state, then review in `before_model` — the hook that runs after the tools
node and before the model sees anything — and apply the decision by replacing the
`ToolMessage` in state, reusing its id so `add_messages` updates in place.

Nothing in that hook has a side effect, so the replay is free. No ledger, no
exactly-once machinery, and the result rides in the **checkpoint**, so it survives
anything the checkpointer survives.

## Measured comparison

Output of `benchmarks/compare_designs.py` — every cell is an executed scenario, not an
opinion:

```
scenario                    wrap-gate (ledger)                wrap-gate (store ledger)       state-rewrite
accept a result             1 execution(s)                    1 execution(s)                 1 execution(s)
process restart mid-review  2 execution(s), RE-RAN (2 total)  1 execution(s), exactly once   1 execution(s), exactly once
retry, single-call turn     supported, 2 execution(s)         supported, 2 execution(s)      supported, 2 execution(s)
retry, call has a sibling   supported, sibling safe (1x)      supported, sibling safe (1x)   not offered (withheld by design)
tool returning a Command    reviewed, in-memory               reviewed, memory fallback      reviewed, durable
raw result in a checkpoint  yes, in the interrupt payload     yes, in the interrupt payload  yes, as a message
pre-execution gate          yes, call blocked                 yes, call blocked              n/a
```

The restart row simulates a redeploy: the checkpointer and store stand in for durable
backends, and what actually dies is the middleware instance — and with it any in-process
ledger. Three findings worth stating plainly:

1. **The default in-process ledger loses a redeploy.** It re-runs the tool. Configure a
   store, or use design B.
2. **Design B cannot retry a call that has a sibling.** Re-running one call means jumping
   back to the tools node, and that node re-executes *every* call on the last AI message
   — firing ungated tools a second time and leaving two results for one `tool_call_id`,
   a message history providers reject. Measured, then designed around: `retry` is simply
   not offered when the gated result is not alone.
3. 3. **Neither design keeps an unreviewed result off disk.** Design B persists it as a
   message; design A keeps it out of the message history, but its interrupt payload
   carries the result and is checkpointed with everything else. The practical
   consequence: a credential in a tool result reaches disk either way, so **redaction
   has to happen before the result is stored**, not at review time.

These are asserted, not just printed: see `tests/test_design_comparison.py`.

## Which to use

| If you need | Use |
| --- | --- |
| A pre-execution gate (with or without result review) | **A**, `DualGateHumanInTheLoopMiddleware` |
| Retry of one call in a multi-call turn | **A** |
| The raw result never written to disk | **A** || Review of a tool that returns a `Command`, durably | **B**, `StateReviewMiddleware` |
| Durability with no store to configure | **B** |
| Both sides, durable results | **A**'s before gate + **B**, composed |

That last row is a supported combination and has a test:

```python
middleware=[
    DualGateHumanInTheLoopMiddleware(interrupt_on={"fetch_record": "before"}),
    StateReviewMiddleware(review_on={"fetch_record": True}),]
```

## Install

```bash
pip install -e ".[dev]"     # needs langchain>=1.5, langgraph>=1.2
```

## Usage

Both middlewares speak the same decision vocabulary, and `when` / `description`
callables are portable between them.

### Design A — both gates in one middleware

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from dual_gate_hitl import DualGateHumanInTheLoopMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-5-5",
    tools=[issue_refund, fetch_customer_record, check_order_status],
    middleware=[
        DualGateHumanInTheLoopMiddleware(
            interrupt_on={
                "issue_refund": {
                    "before": {"allowed_decisions": ["approve", "edit", "reject"]}},
                "fetch_customer_record": {
                    "after": {
                        "allowed_decisions": ["accept", "edit", "reject"],
                        # Only bother a human when the output warrants it.
                        "when": lambda req, msg: "phone=" in str(msg.content),} }, })],
                # check_order_status is absent, so it runs unattended.
           
    checkpointer=InMemorySaver(),   # required: a pause needs somewhere to persist
    store=InMemoryStore(),  )        # recommended: makes the ledger durable


config = {"configurable": {"thread_id": "desk-1"}}
state = agent.invoke({"messages": [("user", "Refund order A-771")]}, config)

review = state["__interrupt__"][0].value
# {'type': 'tool_result_review', 'name': 'fetch_customer_record',
#  'result': "...phone=9812345670...", 'attempt': 0, 'retries_left': 2,
#  'allowed_decisions': ['accept', 'edit', 'reject'], ...}

state = agent.invoke(
    Command(resume={"type": "edit", "content": redact(review["result"])}), config)
```

### Design B — results only, no ledger

```python
from dual_gate_hitl import StateReviewMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-5-5",
    tools=[fetch_customer_record],
    middleware=[
        StateReviewMiddleware(
            review_on={
                "fetch_customer_record": {
                    "allowed_decisions": ["accept", "edit", "reject"],
                    "when": lambda req, msg: "phone=" in str(msg.content),}})],
    checkpointer=InMemorySaver(),  )  # the result lives here; no store needed

```

Run `python examples/refund_desk.py` for design A end to end, no API key needed.

### Configuring the gates

Design A's `interrupt_on`, per tool:

| Value | Meaning |
| --- | --- |
| `True` | both gates, every decision |
| `False` | unattended (same as leaving the tool out) |
| `"before"` / `"after"` | that gate only, every decision |
| `{"before": ..., "after": ...}` | full control; omit a key to skip that gate |

Design B's `review_on` takes `True`, `False`, or the same `after` config. Each gate
accepts `allowed_decisions`, a `description` (string or callable), and a `when`
predicate; the after gate also takes `max_retries`.

### Decisions

**Before the tool runs** — identical to the built-in middleware, so an existing review UI
keeps working:

| Decision | Effect |
| --- | --- |
| `{"type": "approve"}` | run it as the model asked |
| `{"type": "edit", "edited_action": {"name": ..., "args": {...}}}` | run the reviewer's version; `name` may point at a **different tool**, which is how you downgrade `send_email` to `draft_email` |
| `{"type": "reject", "message": ...}` | skip it; the model is told it was rejected |
| `{"type": "respond", "message": ...}` | skip it; the reviewer answers in the tool's place |

**After the tool runs** — a different vocabulary, because you cannot "approve" something
that has already happened:

| Decision | Effect |
| --- | --- |
| `{"type": "accept"}` | pass the result through unchanged |
| `{"type": "edit", "content": ..., "status": ...}` | pass a rewritten result — redaction, truncation, correction |
| `{"type": "retry", "args": {...}}` | discard the result, run the tool again (optionally with new args), review again |
| `{"type": "reject", "message": ...}` | withhold the result; the model is told the output was not released |

Once the retry budget is spent — or, in design B, whenever the call has a sibling —
`retry` is dropped from `allowed_decisions` in the payload, so a reviewer is never
offered a choice that cannot run.

Both designs raise **one interrupt per tool call** and attach a `response_schema`, so a
client can validate a decision before sending it. The decision is re-validated anyway, so
an older LangGraph or a hand-rolled client cannot push a decision a tool never allowed.

## How it works

**Design A** does everything in `wrap_tool_call` / `awrap_tool_call`, the hook that wraps
an individual tool execution and may call the handler zero times (short-circuit), once, or
many times (retry) — the shape a two-sided gate needs.

```
wrap_tool_call
  ├── gate 1: interrupt  ─── reject/respond ──> ToolMessage, tool never runs
  │                      └── edit ──> rewrite args (may redirect to another tool)
  ├── ledger read ──> hit? reuse : run tool, write ledger
  ├── gate 2: when(request, result)? interrupt
  │     ├── accept ──> result
  │     ├── edit   ──> result with rewritten content
  │     ├── reject ──> error ToolMessage, result withheld
  │     └── retry  ──> attempt += 1, loop
  └── ledger cleanup
```

The review policy is one generator (`_policy`) that yields work — *run the tool*, *read
the ledger* — to a driver. The sync hook drives it with blocking calls, the async hook
with awaits. One copy of the policy, so the two paths cannot drift, and the generator
makes the ordering that keeps it replay-safe readable top to bottom.

**Design B** runs in `before_model`, decorated `@hook_config(can_jump_to=["tools"])`:

```
before_model
  ├── find the trailing ToolMessages (the tools node that just ran)
  ├── per gated result: when(call, result)? interrupt
  │     ├── accept ──> no state update
  │     ├── edit   ──> ToolMessage with the same id, replaced in place
  │     ├── reject ──> same id, error content
  │     └── retry  ──> RemoveMessage + {"jump_to": "tools"}   (single-call turns only)
  └── return one {"messages": [...]} update
```

Both handle tools that return a `Command`: design A reviews the `ToolMessage` inside the
command's update and swaps it back in, leaving the state update intact; design B sees it
in state like any other result.

## Tests

```bash
python -m pytest -q      # 54 tests
ruff check .
```

- `tests/test_exactly_once.py` — the property design A exists to guarantee, asserted
  under a post gate, under both gates, across a retry, in async, and with a store-backed
  ledger. Keeps the naive double-executing version as a control, so the suite
  demonstrates both the bug and its absence.
- `tests/test_design_comparison.py` — the tradeoff itself: which design survives a
  redeploy, which keeps the raw result off disk, which can retry beside a sibling.
- `tests/test_before_gate.py`, `test_after_gate.py`, `test_state_gate.py` — per-decision
  behaviour for both designs.

## Limits

- **Neither design can un-send an email.** A post-execution gate reviews the result, so
  the side effect has already happened. It controls what the model and user see, and
  whether the call is tried again — not whether it happened. For effects that must never
  happen without sign-off, use a before gate.
- **Design A's default ledger does not survive the process.** Configure a store in
  production. A tool returning a `Command` cannot be stored durably at all (it is not
  JSON-serialisable), so those results fall back to memory and warn once.
- **Design B persists the raw result** as a message, and cannot offer a pre-execution
  gate or retry beside a sibling.
- **A review is not a redaction boundary.** Both designs persist the unreviewed result
  (design A inside the interrupt payload), and tracing captures it as well. If tool
  output can contain a credential, scrub it before it is stored rather than relying on a
  reviewer to catch it.
- **Pinned to measured behaviour.** Both designs rest on how langgraph 1.2.22 scopes a
  replay and re-runs a tools node. If either changes, a named test fails — which is why
  those tests exist.
- **No batched mode.** The built-in middleware can raise one interrupt per model turn,
  answered with an ordered list. These are always per-call, which is the mode LangChain
  now recommends and the only mode in which a result can be reviewed at all.

## Layout

```
src/dual_gate_hitl/
  middleware.py   design A: two gates, retry loop, shared sync/async policy
  cache.py        design A: the execution ledger, in-memory and store-backed
  state_gate.py   design B: result review by state rewrite in before_model
  types.py        decision and config types, shared by both designs
tests/            54 tests, including the double-execution control
benchmarks/       compare_designs.py - the table above, regenerated
examples/         refund_desk.py - runnable with no API key
```

## License

MIT
