"""
Caller-owned follow scopes for read-your-writes routing.

A :class:`FollowScope` is opened explicitly by the caller. While it is active,
slots whose writes succeeded inside the scope are pinned to the primary that
accepted the write, so subsequent reads of those slots are sent to that primary
instead of a replica. Bindings live on the scope object, never on the client,
and scopes can nest: an inner scope sees the outer bindings, bindings added
inside it disappear when it exits, and the outer bindings remain.

The activation stack is kept in a :class:`contextvars.ContextVar`. It is
task-local under asyncio (tasks copy the context at creation) and thread-local
for regular threads, so two sibling scopes in different threads/tasks never
interact. The same scope object may not be entered twice, and under asyncio it
may not be entered concurrently by two tasks.

This module only owns the binding state and the pure routing helpers. The
clients call into it from their existing node-selection paths; nothing here
opens sockets or knows how to talk to Redis.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from typing import Any, Callable, Optional

from redis.exceptions import FollowScopeError

# Commands that neither create a binding nor change their target node because
# of one: blocking pop/move commands, pub/sub control commands and MONITOR.
# PubSub and MONITOR do not go through the slot routing path at all in the
# first place, and the blocking commands are listed explicitly so they keep
# their current primary target without a binding.
FOLLOW_SCOPE_IGNORED_COMMANDS = frozenset(
    {
        "BLPOP",
        "BRPOP",
        "BLMOVE",
        "BLMPOP",
        "BZPOPMIN",
        "BZPOPMAX",
        "BZMPOP",
        "BRPOPLPUSH",
        "SUBSCRIBE",
        "UNSUBSCRIBE",
        "PSUBSCRIBE",
        "PUNSUBSCRIBE",
        "SSUBSCRIBE",
        "SUNSUBSCRIBE",
        "MONITOR",
        # Transaction control commands modify no data and never drive routing.
        "WATCH",
        "UNWATCH",
        "MULTI",
        "EXEC",
        "DISCARD",
    }
)

# Script runners that may modify data. Every key declared in KEYS binds its
# slot, not just the first one.
WRITE_SCRIPT_COMMANDS = frozenset({"EVAL", "EVALSHA"})
# Read-only script runners create no bindings but obey the existing ones.
READ_ONLY_SCRIPT_COMMANDS = frozenset({"EVAL_RO", "EVALSHA_RO"})

_MISSING = object()

_current_scope: "contextvars.ContextVar[Optional[FollowScope]]" = (
    contextvars.ContextVar("redis_follow_scope", default=None)
)


def normalize_command(args: tuple | list) -> str:
    """Normalize a raw command's name to upper case, joining two-word commands."""
    command = str(args[0]).upper()
    if len(args) >= 2 and f"{args[0]} {args[1]}".upper() in {
        "CLIENT TRACKING",
        "HIMPORT SET",
    }:
        return f"{args[0]} {args[1]}".upper()
    return command


def is_follow_ignored(command: str) -> bool:
    """True for commands that must neither bind nor follow a binding."""
    return str(command).upper() in FOLLOW_SCOPE_IGNORED_COMMANDS


def write_command_slots(
    command: str,
    args: tuple | list,
    keyslot: Callable[[Any], int],
    determine_slot: Callable[..., Optional[int]],
) -> Optional[list[int]]:
    """
    Return the slots a successful write command must bind.

    ``None`` (or an empty list) means the command creates no bindings. For
    ``EVAL``/``EVALSHA`` every key in the KEYS section is considered - cluster
    routing requires them to share a slot, but the binding records each key so
    the rule does not silently depend on the first key only. Every other
    command binds the single slot the existing routing logic resolves.
    """
    command = str(command).upper()
    if is_follow_ignored(command):
        return None

    if command in WRITE_SCRIPT_COMMANDS:
        # EVAL/EVALSHA "<script>" numkeys key [key ...] arg [arg ...]
        if len(args) <= 2:
            return None
        try:
            num_keys = int(args[2])
        except (TypeError, ValueError):
            return None
        if num_keys <= 0:
            return None
        return [keyslot(key) for key in args[3 : 3 + num_keys]]

    try:
        slot = determine_slot(*args)
    except Exception:
        # Commands whose slot the client cannot resolve (keyless commands,
        # two-word specials handled elsewhere) never produce a binding.
        return None
    if slot is None:
        return None
    return [slot]


class FollowScope:
    """
    A caller-owned scope in which written slots follow the accepting primary.

    Use ``with client.follow_scope()`` (sync) or
    ``async with client.follow_scope()`` (async). The scope is single-use:
    leaving it and entering again raises, and under asyncio two concurrent
    tasks cannot enter the same scope object.
    """

    def __init__(self) -> None:
        # slot -> primary node name; recorded on this layer only
        self._bindings: dict[int, str] = {}
        # sentinel service -> master address recorded by a write
        self._sentinel_pins: dict[str, Any] = {}
        self._parent: Optional[FollowScope] = None
        self._active = False
        self._closed = False
        self._owner: Any = None
        self._token: Optional[contextvars.Token] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ state

    @classmethod
    def current(cls) -> Optional["FollowScope"]:
        """The innermost active scope in this thread/task, or ``None``."""
        return _current_scope.get()

    @staticmethod
    def _owner_id() -> Any:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is not None:
            return ("task", task)
        return ("thread", threading.get_ident())

    def _enter(self) -> "FollowScope":
        """Single locked check-and-activate transition for both protocols."""
        with self._lock:
            if self._closed:
                raise FollowScopeError(
                    "A FollowScope cannot be entered again after it has exited"
                )
            owner = self._owner_id()
            if self._active:
                if self._owner is not None and owner != self._owner:
                    kind, _ = self._owner
                    if kind == "task":
                        # Concurrent entry from a different, running task.
                        raise FollowScopeError(
                            "A FollowScope cannot be entered by two concurrent "
                            "asyncio tasks"
                        )
                    raise FollowScopeError(
                        "A FollowScope cannot be entered concurrently from "
                        "multiple threads"
                    )
                raise FollowScopeError(
                    "A FollowScope cannot be entered while it is already active"
                )
            self._owner = owner
            self._parent = _current_scope.get()
            self._token = _current_scope.set(self)
            self._active = True
        return self

    def __enter__(self) -> "FollowScope":
        if self._owner_id()[0] == "task":
            raise FollowScopeError(
                "Use 'async with' to enter a FollowScope from inside a coroutine"
            )
        return self._enter()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._leave()

    async def __aenter__(self) -> "FollowScope":
        return self._enter()

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self._leave()

    def _leave(self) -> None:
        with self._lock:
            token, self._token = self._token, None
            self._active = False
            self._closed = True
            self._parent = None
        if token is not None:
            try:
                _current_scope.reset(token)
            except (ValueError, RuntimeError):
                # The context moved (another task/thread took over) or the token
                # was already reset; leaving must never mask user exceptions.
                _current_scope.set(None)

    # ----------------------------------------------------------- slot bindings

    def lookup_binding(self, slot: int) -> Optional[str]:
        """Primary node name bound to ``slot`` in this scope chain, if any."""
        scope: Optional[FollowScope] = self
        while scope is not None:
            name = scope._bindings.get(slot)
            if name is not None:
                return name
            scope = scope._parent
        return None

    def _owning_layer(self, slot: int) -> Optional["FollowScope"]:
        scope: Optional[FollowScope] = self
        while scope is not None:
            if slot in scope._bindings:
                return scope
            scope = scope._parent
        return None

    def bind(self, slot: int, node_name: str) -> None:
        """Bind ``slot`` on this (innermost) layer."""
        with self._lock:
            self._bindings[slot] = node_name

    def rebind(self, slot: int, node_name: str) -> None:
        """Update whichever layer of the chain currently owns ``slot``."""
        with self._lock:
            layer = self._owning_layer(slot) or self
            layer._bindings[slot] = node_name

    def snapshot_slots(self, slots) -> dict[int, Any]:
        return {slot: self.lookup_binding(slot) or _MISSING for slot in slots}

    def restore_slots(self, snapshot: dict[int, Any]) -> None:
        """Restore the chain to a previous snapshot taken by this scope."""
        with self._lock:
            for slot, previous in snapshot.items():
                layer = self._owning_layer(slot)
                if layer is None:
                    continue
                if previous is _MISSING:
                    layer._bindings.pop(slot, None)
                else:
                    layer._bindings[slot] = previous

    # -------------------------------------------------------- sentinel pinning

    def sentinel_pinned(self, service_name: str) -> Any:
        scope: Optional[FollowScope] = self
        while scope is not None:
            pin = scope._sentinel_pins.get(service_name)
            if pin is not None:
                return pin
            scope = scope._parent
        return None

    def pin_sentinel_master(self, service_name: str, address: Any) -> None:
        with self._lock:
            self._sentinel_pins[service_name] = address

    def rebase_sentinel_master(self, service_name: str, address: Any) -> None:
        with self._lock:
            scope: Optional[FollowScope] = self
            target = self
            while scope is not None:
                if service_name in scope._sentinel_pins:
                    target = scope
                    break
                scope = scope._parent
            target._sentinel_pins[service_name] = address

    # -------------------------------------------------------------- introspect

    @property
    def bindings(self) -> dict[int, str]:
        """This layer's own bindings (excluding outer scopes)."""
        return dict(self._bindings)

    @property
    def is_active(self) -> bool:
        return self._active


def select_cluster_node(scope: FollowScope, slot: int, nodes_manager: Any) -> Any:
    """
    Resolve the node a bound slot must be sent to.

    Returns the cluster node, or ``None`` when the slot is not bound in this
    scope chain. A binding always points at a primary: the current primary of
    the slot cache is authoritative, so a MOVED/failover that already repaired
    the cache rebases the binding onto the new primary and reads never fall
    back to a replica.
    """
    bound_name = scope.lookup_binding(slot)
    if bound_name is None:
        return None
    primary = nodes_manager.get_node_from_slot(slot, False)
    if primary.name != bound_name:
        scope.rebind(slot, primary.name)
    return primary
