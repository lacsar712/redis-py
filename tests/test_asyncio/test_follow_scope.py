"""
Async tests for the caller-owned follow scope (read-your-writes pinning).

Uses the deterministic mocked cluster client; no running Redis required.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from redis import FollowScopeError
from redis.asyncio import Redis
from redis.asyncio.cluster import RedisCluster
from redis.asyncio.sentinel import SentinelConnectionPool
from redis.exceptions import (
    ConnectionError,
    CrossSlotTransactionError,
    MovedError,
    ResponseError,
)
from redis.follow_scope import FollowScope

from .test_cluster import (
    default_host,
    default_port,
    get_mocked_redis_client,
)


async def _fake_determine_slot(self, command, *args):
    """Deterministic, socket-free slot resolution for the commands under test."""
    name = command.upper()
    if name in ("EVAL", "EVALSHA", "EVAL_RO", "EVALSHA_RO"):
        # after command: <script/shasum> numkeys key [key ...]
        num_keys = int(args[1])
        keys = list(args[2 : 2 + num_keys])
    else:
        keys = [args[0]] if args else []
    if not keys:
        return 0
    return self.keyslot(keys[0])


@pytest.fixture(autouse=True)
def _local_slots():
    with patch.object(RedisCluster, "_determine_slot", _fake_determine_slot):
        yield


async def make_client(**kwargs):
    kwargs.setdefault("host", default_host)
    kwargs.setdefault("port", default_port)
    kwargs.setdefault("read_from_replicas", True)
    return await get_mocked_redis_client(**kwargs)


def keys_for_distinct_slots(rc, count=2):
    keys, slots, i = [], set(), 0
    while len(keys) < count:
        name = f"k-{i}"
        slot = rc.keyslot(name)
        if slot not in slots:
            keys.append((name, slot))
            slots.add(slot)
        i += 1
    return keys


def install_router(rc, routed):
    async def fake_execute(node, *args, **kwargs):
        routed.append((node.name, args[0]))
        return b"v" if args[0] == "GET" else True

    rc._execute_command = fake_execute
    return fake_execute


class TestAsyncScopeCore:
    async def test_scope_is_single_use(self):
        scope = FollowScope()
        async with scope:
            pass
        with pytest.raises(FollowScopeError):
            async with scope:
                pass

    async def test_two_concurrent_tasks_cannot_enter_same_scope(self):
        scope = FollowScope()
        entered = asyncio.Event()

        async def outer():
            async with scope:
                entered.set()
                await asyncio.sleep(0.1)

        async def inner():
            await entered.wait()
            with pytest.raises(FollowScopeError):
                async with scope:
                    pass

        await asyncio.gather(asyncio.create_task(outer()), asyncio.create_task(inner()))

    async def test_nesting_and_sibling_independence(self):
        async with FollowScope() as outer:
            outer.bind(7, "a")
            async with FollowScope() as inner:
                assert inner.lookup_binding(7) == "a"
                inner.bind(8, "b")
            assert outer.lookup_binding(8) is None
            assert outer.lookup_binding(7) == "a"
        async with FollowScope() as second:
            assert second.lookup_binding(7) is None


class TestAsyncClusterFollowScope:
    async def test_write_then_read_pins_slot(self):
        rc = await make_client()
        name, slot = keys_for_distinct_slots(rc, 1)[0]
        primary = rc.nodes_manager.get_node_from_slot(slot, False)
        replica = rc.nodes_manager.slots_cache[slot][1]
        routed = []
        install_router(rc, routed)

        async with rc.follow_scope():
            await rc.set(name, "v")
            await rc.get(name)

        assert routed == [(primary.name, "SET"), (primary.name, "GET")]
        assert replica.name not in [n for n, _ in routed]
        await rc.aclose()

    async def test_unbound_slot_keeps_replica_policy(self):
        rc = await make_client()
        (bound, _), (other, _) = keys_for_distinct_slots(rc, 2)
        routed = []
        install_router(rc, routed)

        with patch.object(
            type(rc.nodes_manager),
            "get_node_from_slot",
            autospec=True,
            wraps=type(rc.nodes_manager).get_node_from_slot,
        ) as get_node:
            async with rc.follow_scope():
                await rc.set(bound, "v")
                await rc.get(bound)
                assert get_node.call_args[0][2] is False
                get_node.reset_mock()
                await rc.get(other)
                assert get_node.call_args[0][2] is True
        await rc.aclose()

    async def test_eval_ro_obeying_existing_binding(self):
        rc = await make_client()
        name, slot = keys_for_distinct_slots(rc, 1)[0]
        primary = rc.nodes_manager.get_node_from_slot(slot, False)
        routed = []
        install_router(rc, routed)

        async with rc.follow_scope() as scope:
            await rc.execute_command("EVAL_RO", "return 1", 1, name)
            assert scope.lookup_binding(slot) is None
            await rc.set(name, "v")
            await rc.execute_command("EVAL_RO", "return 1", 1, name)
        assert routed[1:] == [(primary.name, "SET"), (primary.name, "EVAL_RO")]
        await rc.aclose()

    # -------------------------------------------------------------- pipelines

    def _patch_pipeline_nodes(self, rc, responder):
        from redis.asyncio.cluster import ClusterNode as _Node

        async def execute_pipeline(self, commands):
            responder(self, commands)
            # mirrors the real return value: whether the node saw an error
            return any(isinstance(c.result, Exception) for c in commands)

        return patch.object(_Node, "execute_pipeline", execute_pipeline)

    async def test_pipeline_groups_write_and_later_read_on_primary(self):
        rc = await make_client()
        name, slot = keys_for_distinct_slots(rc, 1)[0]
        primary = rc.nodes_manager.get_node_from_slot(slot, False)
        replica = rc.nodes_manager.slots_cache[slot][1]
        grouped = {}

        def responder(node, commands):
            grouped.setdefault(node.name, []).extend(c.args[0] for c in commands)
            for c in commands:
                c.result = True if c.args[0] == "SET" else b"v"

        async with rc.follow_scope():
            with self._patch_pipeline_nodes(rc, responder):
                result = await rc.pipeline().set(name, "v").get(name).execute()

        assert result == [True, b"v"]
        assert grouped.get(primary.name) == ["SET", "GET"]
        assert replica.name not in grouped
        await rc.aclose()

    async def test_pipeline_unbound_read_keeps_replica_policy(self):
        rc = await make_client()
        (bound, bound_slot), (other, other_slot) = keys_for_distinct_slots(rc, 2)

        def responder(node, commands):
            for c in commands:
                c.result = True if c.args[0] == "SET" else b"v"

        with patch.object(
            type(rc.nodes_manager),
            "get_node_from_slot",
            autospec=True,
            wraps=type(rc.nodes_manager).get_node_from_slot,
        ) as get_node:
            async with rc.follow_scope():
                with self._patch_pipeline_nodes(rc, responder):
                    await (
                        rc.pipeline()
                        .set(bound, "v")
                        .get(other)
                        .execute(raise_on_error=False)
                    )
        # The written slot is forced primary; the other slot keeps the
        # replica-eligible selection.
        calls = {call.args[1]: call.args[2] for call in get_node.call_args_list}
        assert calls[bound_slot] is False
        assert calls[other_slot] is True
        await rc.aclose()

    async def test_pipeline_failed_write_rolls_back(self):
        rc = await make_client()
        name, slot = keys_for_distinct_slots(rc, 1)[0]

        def responder(node, commands):
            for c in commands:
                c.result = ResponseError("nope") if c.args[0] == "SET" else b"v"

        async with rc.follow_scope() as scope:
            with self._patch_pipeline_nodes(rc, responder):
                with pytest.raises(ResponseError):
                    await rc.pipeline().set(name, "v").execute()
            assert scope.lookup_binding(slot) is None
        await rc.aclose()

    async def test_pipeline_moved_retries_failed_command_and_rebinds(self):
        rc = await make_client()
        name, slot = keys_for_distinct_slots(rc, 1)[0]
        primary = rc.nodes_manager.get_node_from_slot(slot, False)
        new_primary = rc.nodes_manager.slots_cache[slot][1]
        packed = {}

        def responder(node, commands):
            packed.setdefault(node.name, []).extend(c.args[0] for c in commands)
            for c in commands:
                if c.args[0] == "GET":
                    c.result = MovedError(
                        f"{slot} {new_primary.host}:{new_primary.port}"
                    )
                else:
                    c.result = True

        retried = []

        async def fake_exec(node, *args, **kwargs):
            retried.append((node.name, args[0]))
            # emulate the real MOVED repair, then serve on the promoted node
            if node.name == primary.name:
                await rc.nodes_manager.move_slot(
                    MovedError(f"{slot} {new_primary.host}:{new_primary.port}")
                )
                rebound = rc.nodes_manager.get_node_from_slot(slot, False)
                FollowScope.current().rebind(slot, rebound.name)
            return b"v"

        rc._execute_command = fake_exec

        async with rc.follow_scope() as scope:
            with self._patch_pipeline_nodes(rc, responder):
                result = await rc.pipeline().set(name, "v").get(name).execute()

        assert result == [True, b"v"]
        # SET stayed in the single packed batch; only GET was retried
        assert packed[primary.name] == ["SET", "GET"]
        assert retried == [(primary.name, "GET")]
        assert scope.lookup_binding(slot) == new_primary.name
        await rc.aclose()

    # ----------------------------------------------------------- transactions

    async def test_cross_slot_transaction_still_rejected(self):
        rc = await make_client()
        async with rc.follow_scope():
            pipe = rc.pipeline(transaction=True)
            pipe.set("{a}", 1)
            pipe.set("{b}", 2)
            with pytest.raises(CrossSlotTransactionError):
                await pipe.execute()
        await rc.aclose()

    # --------------------------------------------------------------- sentinel

    def _make_conn(self, address, command_value):
        conn = MagicMock()
        conn.host, conn.port = address
        conn.disconnect = AsyncMock()
        current = {"cmd": None}

        async def send_command(*a, **k):
            current["cmd"] = a[0] if a else None

        async def read_response(*a, **k):
            return b"OK" if current["cmd"] in ("SET", "EVALSHA") else command_value

        async def run_retry(fn, cb, with_failure_count=False):
            return await fn()

        conn.send_command.side_effect = send_command
        conn.read_response.side_effect = read_response
        conn.retry.call_with_retry = run_retry
        return conn

    async def _sentinel_clients(self):
        master_address = ("10.0.0.1", 6379)
        manager = MagicMock()
        manager.discover_master = AsyncMock(return_value=master_address)
        manager.discover_slaves = AsyncMock(return_value=[("10.0.0.2", 6379)])

        master_pool = SentinelConnectionPool("svc", manager, is_master=True)
        replica_pool = SentinelConnectionPool("svc", manager, is_master=False)
        master_conn = self._make_conn(master_address, b"v")
        replica_conn = self._make_conn(("10.0.0.2", 6379), b"v")
        master_pool.get_connection = AsyncMock(return_value=master_conn)
        master_pool.release = AsyncMock()
        replica_pool.get_connection = AsyncMock(return_value=replica_conn)
        replica_pool.release = AsyncMock()
        master = Redis.from_pool(master_pool)
        replica = Redis.from_pool(replica_pool)
        return manager, master, master_conn, replica

    async def test_sentinel_pin_and_replica_refusal(self):
        manager, master, _, replica = await self._sentinel_clients()
        async with master.follow_scope():
            assert await master.set("k", "v")
            assert await master.get("k") == b"v"
            with pytest.raises(FollowScopeError):
                await replica.get("k")
        await master.aclose()
        await replica.aclose()

    async def test_sentinel_failover_before_confirmation_fails(self):
        manager, master, old_conn, _ = await self._sentinel_clients()
        new_address = ("10.0.0.3", 6379)
        from redis.asyncio.sentinel import MasterNotFoundError

        async with master.follow_scope():
            assert await master.set("k", "v")
            manager.discover_master = AsyncMock(
                side_effect=MasterNotFoundError("no master")
            )
            with pytest.raises(ConnectionError):
                await master.get("k")

            manager.discover_master = AsyncMock(return_value=new_address)
            with pytest.raises(ConnectionError):
                await master.get("k")
            assert old_conn.disconnect.called
        await master.aclose()
