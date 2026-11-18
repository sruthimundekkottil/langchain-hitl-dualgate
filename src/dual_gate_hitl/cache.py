"""Exactly-once execution ledger.

Why this file exists
--------------------
LangGraph implements a pause by raising an exception and, on resume, **re-running the
interrupted task from the top**. ``interrupt()`` itself is replay-safe: the second time
through, it returns the stored answer instead of pausing again. Anything *else* that ran
before the interrupt simply runs again.

A pre-execution gate is therefore trivially safe - nothing has happened yet. A
post-execution gate is not: the tool must run before a human can look at its result, so
on resume the tool runs a second time. Measured on langchain 1.5.0 / langgraph 1.2.22, a
naive post-execution gate charges a card twice.

The fix is a ledger, written the moment a tool returns and read before the tool is
called. On replay the result is already there, so the tool is not called again. Two
things make this safe to rely on:

* the replay is scoped to the one task that interrupted - sibling tool calls in the same
  model turn are *not* re-run, so one entry per ``(tool_call_id, attempt)`` is enough;
* the entry is written *outside* the graph checkpoint, so it is visible to the replay.

`StoreLedger` is the durable implementation: it writes through LangGraph's
:class:`~langgraph.store.base.BaseStore`, so a result survives the process dying while a
reviewer is away from their desk. `MemoryLedger` is the fallback when no store is
configured; it is correct as long as the resume reaches the same process.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Protocol, Union, runtime_checkable

from langchain_core.load import dumpd, load
from langchain_core.messages import ToolMessage

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore
    from langgraph.types import Command

#: Sentinel for "nothing recorded", so that a legitimately `None` result is not confused
#: with a miss.
MISS = object()

ToolResult = Union[ToolMessage, "Command[Any]"]

_NAMESPACE = ("dual_gate_hitl", "results")


@runtime_checkable
class ResultLedger(Protocol):
    """Records what a tool returned, so a replay does not run it again."""

    def get(self, key: str) -> Any:
        """Return the recorded result for `key`, or `MISS`."""
        ...

    def put(self, key: str, result: ToolResult) -> None:
        """Record `result` under `key`."""
        ...

    def discard(self, keys: list[str]) -> None:
        """Forget `keys`. Called once a tool call is finally resolved."""
        ...

    async def aget(self, key: str) -> Any:
        """Async `get`."""
        ...

    async def aput(self, key: str, result: ToolResult) -> None:
        """Async `put`."""
        ...

    async def adiscard(self, keys: list[str]) -> None:
        """Async `discard`."""
        ...


class MemoryLedger:
    """In-process ledger with a bounded size.

    Correct whenever the resume lands in the same process that ran the tool - the usual
    case for a single API server, a notebook, or `langgraph dev`. If the process can die
    while a review is pending, use `StoreLedger` instead.

    The bound matters: a thread abandoned mid-review never reaches the `discard` call, so
    without a cap the ledger would grow for as long as the process lives.
    """

    def __init__(self, max_entries: int = 1024) -> None:
        self._entries: OrderedDict[str, ToolResult] = OrderedDict()
        self._max_entries = max_entries

    def get(self, key: str) -> Any:
        if key not in self._entries:
            return MISS
        self._entries.move_to_end(key)  # keep the live review warm under the LRU bound
        return self._entries[key]

    def put(self, key: str, result: ToolResult) -> None:
        self._entries[key] = result
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def discard(self, keys: list[str]) -> None:
        for key in keys:
            self._entries.pop(key, None)

    async def aget(self, key: str) -> Any:
        return self.get(key)

    async def aput(self, key: str, result: ToolResult) -> None:
        self.put(key, result)

    async def adiscard(self, keys: list[str]) -> None:
        self.discard(keys)


class StoreLedger:
    """Durable ledger backed by a LangGraph store.

    A store write lands immediately rather than at the end of a super-step, which is
    exactly what a replay needs. Pair it with a persistent store (Postgres, Redis) and a
    result survives a deploy, a crash, or a reviewer who answers the next morning.

    `ToolMessage` results are serialised with LangChain's own serialiser. A tool that
    returns a `Command` is not JSON-serialisable in general, so those results fall back
    to an in-process ledger and a warning says so once.
    """

    def __init__(self, store: BaseStore, *, fallback: MemoryLedger | None = None) -> None:
        self._store = store
        self._fallback = fallback or MemoryLedger()
        self._warned = False

    # -- serialisation ----------------------------------------------------------------

    def _encode(self, result: ToolResult) -> dict[str, Any] | None:
        """Return a storable payload, or None if this result cannot be stored."""
        if isinstance(result, ToolMessage):
            return {"kind": "tool_message", "payload": dumpd(result)}
        if not self._warned:
            warnings.warn(
                "dual_gate_hitl: a tool returned a Command, which cannot be stored "
                "durably. Its result is kept in memory for the duration of the review, "
                "so a process restart while the review is pending will re-run the tool. "
                "Make such tools idempotent, or gate them before execution only.",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warned = True
        return None

    @staticmethod
    def _decode(value: dict[str, Any]) -> ToolResult:
        # Only messages are ever written here, so the deserialiser is restricted to
        # messages. A store can hold data this process did not write; narrowing what it
        # is allowed to reconstruct keeps a poisoned entry from instantiating anything
        # else.
        with warnings.catch_warnings():
            # `load` is flagged beta upstream. That is a note for the maintainers of
            # this package, not something to print on every resume.
            warnings.simplefilter("ignore")
            return load(value["payload"], allowed_objects="messages")

    # -- sync -------------------------------------------------------------------------

    def get(self, key: str) -> Any:
        item = self._store.get(_NAMESPACE, key)
        if item is None:
            return self._fallback.get(key)
        return self._decode(dict(item.value))

    def put(self, key: str, result: ToolResult) -> None:
        encoded = self._encode(result)
        if encoded is None:
            self._fallback.put(key, result)
            return
        self._store.put(_NAMESPACE, key, encoded)

    def discard(self, keys: list[str]) -> None:
        self._fallback.discard(keys)
        for key in keys:
            self._store.delete(_NAMESPACE, key)

    # -- async ------------------------------------------------------------------------

    async def aget(self, key: str) -> Any:
        item = await self._store.aget(_NAMESPACE, key)
        if item is None:
            return await self._fallback.aget(key)
        return self._decode(dict(item.value))

    async def aput(self, key: str, result: ToolResult) -> None:
        encoded = self._encode(result)
        if encoded is None:
            await self._fallback.aput(key, result)
            return
        await self._store.aput(_NAMESPACE, key, encoded)

    async def adiscard(self, keys: list[str]) -> None:
        await self._fallback.adiscard(keys)
        for key in keys:
            await self._store.adelete(_NAMESPACE, key)
