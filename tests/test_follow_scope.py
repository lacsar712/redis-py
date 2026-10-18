"""
Tests for the caller-owned follow scope (read-your-writes node pinning).

These tests use the deterministic mocked cluster client, so they need no
running Redis deployment.
"""

import threading

import pytest
from unittest.mock import MagicMock, Mock, patch

from redis import FollowScope, FollowScopeError, Redis
from redis.cluster import RedisCluster
from redis.exceptions import (
    ConnectionError,
    CrossSlotTransactionError,
    MovedError,
    ResponseError,
)
from redis.follow_scope import (
    FOLLOW_SCOPE_IGNORED_COMMANDS,
    write_command_slots,
)
from redis.sentinel import MasterNotFoundError, SentinelConnectionPool

from tests.test_cluster import (
    default_host,
    default_port,
    get_mocked_redis_client,
)


def _fake_command_keys(self, *args):
    command = str(args[0]).upper()
    if command in ("EVAL", "EVALSHA", "EVAL_RO", "EVALSHA_RO"):
        # <script> numkeys key [key ...] ...
        return list(args[3 : 3 + int(args[2])])
    return [args[1]]


@pytest.fixture(autouse=True)
def _local_command_keys():
    # ClusterPipeline is a fresh object per pipeline(), so the key-extraction
    # shortcut has to live on the class to cover command planning too.
    with patch.object(RedisCluster, "_get_command_keys", _fake_command_keys):
        yield


def make_client(**kwargs):
    kwargs.setdefault("host", default_host)
    kwargs.setdefault("port", default_port)
    kwargs.setdefault("read_from_replicas", True)
    rc = get_mocked_redis_client(**kwargs)
    # The mocked COMMAND reply only carries "get"; answer key extraction
    # locally so no COMMAND GETKEYS round-trip reaches a socket.
    rc._get_command_keys = lambda *args: _fake_command_keys(rc, *args)
    return rc


def keys_for_distinct_slots(rc, count=2):
    keys = []
    slots = set()
    i = 0
    while len(keys) < count:
        name = f"k-{i}"
        slot = rc.keyslot(name)
        if slot not in slots:
            keys.append((name, slot))
            slots.add(slot)
        i += 1
    return keys


class TestFollowScopeCore:
    def test_no_active_scope_outside_context(self):
        assert FollowScope.current() is None
        scope = FollowScope()
        with scope:
            assert FollowScope.current() is scope
        assert FollowScope.current() is None

    def test_scope_is_single_use(self):
        scope = FollowScope()
        with scope:
            pass
        with pytest.raises(FollowScopeError):
            with scope:
                pass

    def test_scope_cannot_be_nested_into_itself(self):
        scope = FollowScope()
        with scope:
            with pytest.raises(FollowScopeError):
                with scope:
                    pass

    def test_nested_scopes_see_outer_bindings(self):
        with FollowScope() as outer:
            outer.bind(1, "a")
            with FollowScope() as inner:
                assert inner.lookup_binding(1) == "a"
                inner.bind(2, "b")
                assert inner.lookup_binding(2) == "b"
            # inner bindings disappear on exit, outer bindings remain
            assert outer.lookup_binding(1) == "a"
            assert outer.lookup_binding(2) is None

    def test_inner_binding_same_slot_is_dropped_on_exit(self):
        with FollowScope() as outer:
            outer.bind(1, "outer-node")
            with FollowScope() as inner:
                inner.bind(1, "inner-node")
                assert inner.lookup_binding(1) == "inner-node"
            assert outer.lookup_binding(1) == "outer-node"
            assert outer.bindings == {1: "outer-node"}

    def test_sibling_scopes_do_not_interact(self):
        with FollowScope() as first:
            first.bind(1, "a")
        with FollowScope() as second:
            assert second.lookup_binding(1) is None

    def test_concurrent_thread_cannot_enter_same_scope(self):
        scope = FollowScope()
        entered = threading.Event()
        errors = []

        def worker():
            try:
                with scope:
                    entered.set()
                    entered.wait(0.2)
            except FollowScopeError as exc:
                errors.append(exc)

        with scope:
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
        assert errors, "a second thread must not enter an active scope"

    def test_write_slots_for_eval_covers_every_key(self):
        rc = make_client()
        keys = ("{tag}a", "{tag}b", "{tag}c")
        args = ("EVAL", "redis.call('set',KEYS[1],1)", 3, *keys)
        slots = write_command_slots("EVAL", args, rc.keyslot, rc.determine_slot)
        assert slots == [rc.keyslot(k) for k in keys]

    def test_write_slots_for_zero_key_eval_is_none(self):
        rc = make_client()
        args = ("EVAL", "return 1", 0)
        assert write_command_slots("EVAL", args, rc.keyslot, rc.determine_slot) is None

    def test_ignored_commands(self):
        assert "BLPOP" in FOLLOW_SCOPE_IGNORED_COMMANDS
        assert "MONITOR" in FOLLOW_SCOPE_IGNORED_COMMANDS


def install_router(rc, routed):
    """Route single commands through a recorder without real sockets."""

    def fake_execute(node, *args, **kwargs):
        routed.append((node.name, args[0]))
        return b"v" if args[0] == "GET" else True

    rc._execute_command = fake_execute
    return fake_execute


class TestSyncClusterFollowScope:
    def test_outside_scope_reads_keep_replica_policy(self):
        rc = make_client()
        (name, slot) = keys_for_distinct_slots(rc, 1)[0]
        rc._execute_command = lambda node, *a, **k: b"v"
        with patch.object(
            type(rc.nodes_manager),
            "get_node_from_slot",
            autospec=True,
            wraps=type(rc.nodes_manager).get_node_from_slot,
        ) as get_node:
            rc.get(name)
        assert get_node.call_args[0][1] == slot
        assert get_node.call_args[0][2] is True  # replica eligible

    def test_write_then_read_pins_slot_to_primary(self):
        rc = make_client()
        name, slot = keys_for_distinct_slots(rc, 1)[0]
        primary = rc.nodes_manager.get_node_from_slot(slot, False)
        replica = rc.nodes_manager.slots_cache[slot][1]
        routed = []
        install_router(rc, routed)

        with rc.follow_scope():
            rc.set(name, "v")
            rc.get(name)

        assert routed == [(primary.name, "SET"), (primary.name, "GET")]
        assert replica.name not in [node for node, _ in routed]

    def test_unbound_slot_keeps_replica_routing_in_scope(self):
        rc = make_client()
        (bound_key, bound_slot), (other_key, other_slot) = keys_for_distinct_slots(
            rc, 2
        )
        install_router(rc, [])

        with patch.object(
            type(rc.nodes_manager),
            "get_node_from_slot",
            autospec=True,
            wraps=type(rc.nodes_manager).get_node_from_slot,
        ) as get_node:
            with rc.follow_scope():
                rc.set(bound_key, "v")
                rc.get(bound_key)
                assert get_node.call_args[0][2] is False  # pinned -> primary
                get_node.reset_mock()
                rc.get(other_key)
                assert get_node.call_args[0][2] is True  # unbound -> replica policy

    def test_two_scopes_are_independent(self):
        rc = make_client()
        name, slot = keys_for_distinct_slots(rc, 1)[0]
        routed = []
        install_router(rc, routed)

        with rc.follow_scope():
            rc.set(name, "v")
            rc.get(name)
        with rc.follow_scope():
            rc.get(name)

        replica = rc.nodes_manager.slots_cache[slot][1]
        # The second scope's read is back on the normal primary/replica policy.
        assert routed[-1][0] in {
            rc.nodes_manager.get_node_from_slot(slot, False).name,
            replica.name,
        }

    def test_nested_scope_follows_outer_and_keeps_outer_binding(self):
        rc = make_client()
        name, slot = keys_for_distinct_slots(rc, 1)[0]
        primary = rc.nodes_manager.get_node_from_slot(slot, False)
        seen = []

        def fake_execute(node, *args, **kwargs):
            seen.append((node.name, args[0]))
            return b"v" if args[0] == "GET" else True

        rc._execute_command = fake_execute

        with rc.follow_scope() as outer:
            rc.set(name, "outer")
            with rc.follow_scope():
                rc.get(name)  # follows the outer binding
                rc.set(name, "inner")
            assert outer.lookup_binding(slot) == primary.name

        assert all(node == primary.name for node, _ in seen)

    def test_eval_write_binds_every_key_slot(self):
        rc = make_client()
        keys = ("{tag}a", "{tag}b")
        script = "redis.call('set',KEYS[1],ARGV[1])"
        rc._execute_command = lambda node, *a, **k: True
        with rc.follow_scope() as scope:
            rc.eval(script, len(keys), *keys)
            slot = rc.keyslot("{tag}a")
            assert scope.lookup_binding(slot) == (
                rc.nodes_manager.get_node_from_slot(slot, False).name
            )

    def test_eval_ro_creates_no_binding_but_follows_existing(self):
        rc = make_client()
        name, slot = keys_for_distinct_slots(rc, 1)[0]
        primary = rc.nodes_manager.get_node_from_slot(slot, False)
        seen = []

        def fake_execute(node, *args, **kwargs):
            seen.append((node.name, args[0]))
            return "r"

        rc._execute_command = fake_execute
        with rc.follow_scope() as scope:
            with patch.object(
                type(rc.nodes_manager),
                "get_node_from_slot",
                autospec=True,
                wraps=type(rc.nodes_manager).get_node_from_slot,
            ) as get_node:
                rc.execute_command("EVAL_RO", "return 1", 1, name)
                # unbound read-only script keeps the replica policy
                assert get_node.call_args[0][2] is True
            assert scope.lookup_binding(slot) is None
            rc.set(name, "v")
            rc.execute_command("EVAL_RO", "return 1", 1, name)

        assert seen[1:] == [(primary.name, "SET"), (primary.name, "EVAL_RO")]

    def test_blpop_creates_no_binding(self):
        rc = make_client()
        name, slot = keys_for_distinct_slots(rc, 1)[0]
        rc._execute_command = lambda node, *a, **k: None
        with rc.follow_scope() as scope:
            rc.execute_command("BLPOP", name, 1)
        assert scope.lookup_binding(slot) is None

    def test_failed_write_does_not_bind(self):
        rc = make_client()
        name, slot = keys_for_distinct_slots(rc, 1)[0]

        def fail(node, *args, **kwargs):
            raise ResponseError("boom")

        rc._execute_command = fail
        with rc.follow_scope() as scope:
            with pytest.raises(ResponseError):
                rc.set(name, "v")
            assert scope.lookup_binding(slot) is None

    # -------------------------------------------------------------- pipelines

    def _mock_all_connections(self, rc):
        for node in rc.get_nodes():
            conn = Mock()
            conn.host = node.host
            conn.port = node.port
            conn.read_response.return_value = b"OK"
            node.redis_connection.connection = conn
        return rc

    def test_pipeline_write_constrains_later_read_single_packed_write(self):
        rc = self._mock_all_connections(make_client())
        name, slot = keys_for_distinct_slots(rc, 1)[0]
        primary = rc.nodes_manager.get_node_from_slot(slot, False)
        replica = rc.nodes_manager.slots_cache[slot][1]

        # The node receives SET then GET, in that order.
        primary.redis_connection.connection.read_response.side_effect = [
            b"OK",
            b"v",
        ]
        replica.redis_connection.connection.read_response.return_value = b"OK"

        with rc.follow_scope():
            result = rc.pipeline().set(name, "v").get(name).execute()

        assert result == [True, b"v"]
        # One packed write per node; the replica never received anything.
        assert primary.redis_connection.connection.send_packed_command.call_count == 1
        assert replica.redis_connection.connection.send_packed_command.call_count == 0

    def test_pipeline_unbound_read_still_uses_replica_policy(self):
        rc = self._mock_all_connections(make_client())
        (bound_key, _), (other_key, other_slot) = keys_for_distinct_slots(rc, 2)
        unbound_node = rc.nodes_manager.get_node_from_slot(other_slot, True)

        with rc.follow_scope():
            rc.pipeline().set(bound_key, "v").get(other_key).execute(
                raise_on_error=False
            )

        assert unbound_node.redis_connection.connection.send_packed_command.called

    def test_pipeline_failed_write_rolls_binding_back(self):
        rc = self._mock_all_connections(make_client())
        name, slot = keys_for_distinct_slots(rc, 1)[0]
        primary = rc.nodes_manager.get_node_from_slot(slot, False)
        primary.redis_connection.connection.read_response.side_effect = ResponseError(
            "nope"
        )

        with rc.follow_scope() as scope:
            with pytest.raises(ResponseError):
                rc.pipeline().set(name, "v").execute()
            assert scope.lookup_binding(slot) is None

            # the rolled-back slot is once again eligible for replica routing
            primary.redis_connection.connection.read_response.side_effect = None
            rc._execute_command = lambda node, *a, **k: b"v"
            with patch.object(
                type(rc.nodes_manager),
                "get_node_from_slot",
                autospec=True,
                wraps=type(rc.nodes_manager).get_node_from_slot,
            ) as get_node:
                rc.get(name)
                assert get_node.call_args[0][2] is True

    def test_pipeline_moved_resends_only_failed_group_and_rebinds(self):
        rc = self._mock_all_connections(make_client())
        name, slot = keys_for_distinct_slots(rc, 1)[0]
        # ensure the key lives on a shard that actually has a replica
        if len(rc.nodes_manager.slots_cache[slot]) < 2:
            pytest.skip("mocked topology always provides a replica for slot 0")
        primary = rc.nodes_manager.get_node_from_slot(slot, False)
        new_primary = rc.nodes_manager.slots_cache[slot][1]  # promoted replica

        old_conn = primary.redis_connection.connection
        new_conn = new_primary.redis_connection.connection

        # packed read: SET -> OK, GET -> MOVED; the individual retry sees MOVED
        # once more while the slot table is repaired.
        moved = MovedError(f"{slot} {new_primary.host}:{new_primary.port}")
        old_conn.read_response.side_effect = [b"OK", moved, moved]
        new_conn.read_response.side_effect = [b"v"]

        with rc.follow_scope() as scope:
            result = rc.pipeline().set(name, "v").get(name).execute()

        assert result == [True, b"v"]
        # SET was packed once and never executed again.
        assert old_conn.send_packed_command.call_count == 1
        # binding follows the promoted primary
        assert scope.lookup_binding(slot) == new_primary.name

    # ----------------------------------------------------------- transactions

    def test_cross_slot_transaction_still_rejected_in_scope(self):
        rc = make_client()
        with rc.follow_scope():
            pipe = rc.pipeline(transaction=True)
            pipe.set("{a}", 1)
            pipe.set("{b}", 2)
            with pytest.raises(CrossSlotTransactionError):
                pipe.execute()

    def test_transaction_success_binds_slot(self):
        rc = self._mock_all_connections(make_client())
        name, slot = keys_for_distinct_slots(rc, 1)[0]
        primary = rc.nodes_manager.get_node_from_slot(slot, False)
        conn = primary.redis_connection.connection
        # MULTI -> OK, SET queued -> QUEUED, GET queued -> QUEUED,
        # EXEC -> [OK, v]
        conn.read_response.side_effect = [
            b"OK",
            b"QUEUED",
            b"QUEUED",
            [b"OK", b"v"],
        ]
        with rc.follow_scope() as scope:
            result = rc.pipeline(transaction=True).set(name, 1).get(name).execute()
        # cluster transactions only transform commands in the cluster
        # response callback table; SET is not one of them.
        assert result == [b"OK", b"v"]
        assert scope.lookup_binding(slot) == primary.name

    # -------------------------------------------------------------------- lock

    def test_lock_write_and_read_follow_bound_primary(self):
        rc = make_client()
        name, slot = keys_for_distinct_slots(rc, 1)[0]
        primary = rc.nodes_manager.get_node_from_slot(slot, False)
        routed = []

        def fake_execute(node, *args, **kwargs):
            routed.append((node.name, args[0]))
            command = args[0]
            if command == "SET":
                return b"OK"
            if command == "GET":
                return b"token"
            return 1  # EVALSHA release

        rc._execute_command = fake_execute
        with rc.follow_scope():
            lock = rc.lock(name, blocking=False)
            assert lock.acquire() is True
            assert lock.locked() is True
            lock.release()

        assert all(node == primary.name for node, _ in routed)
        assert {"SET", "GET", "EVALSHA"}.issubset({cmd for _, cmd in routed})

    # --------------------------------------------------------------- sentinel

    def _sentinel_clients(self):
        master_address = ("10.0.0.1", 6379)
        manager = Mock()
        manager.discover_master.side_effect = lambda service: master_address
        manager.discover_slaves.side_effect = lambda service: [("10.0.0.2", 6379)]

        master_pool = SentinelConnectionPool("svc", manager, is_master=True)
        replica_pool = SentinelConnectionPool("svc", manager, is_master=False)

        def make_conn(address):
            conn = MagicMock()
            conn.host, conn.port = address
            conn.retry.call_with_retry.side_effect = (
                lambda fn, cb, with_failure_count=False: fn()
            )
            current = {"cmd": None}

            def send_command(*a, **k):
                current["cmd"] = a[0] if a else None

            def read_response(*a, **k):
                return b"OK" if current["cmd"] in ("SET", "EVALSHA") else b"v"

            conn.send_command.side_effect = send_command
            conn.read_response.side_effect = read_response
            return conn

        master_conn = make_conn(master_address)
        replica_conn = make_conn(("10.0.0.2", 6379))

        master_pool.get_connection = lambda *a, **k: master_conn
        master_pool.release = lambda conn: None
        replica_pool.get_connection = lambda *a, **k: replica_conn
        replica_pool.release = lambda conn: None

        master = Redis.from_pool(master_pool)
        replica = Redis.from_pool(replica_pool)
        return manager, master, master_conn, replica

    def test_sentinel_write_pins_master_and_replica_read_fails(self):
        manager, master, _, replica = self._sentinel_clients()
        with master.follow_scope():
            assert master.set("k", "v")
            assert master.get("k") == b"v"
            assert manager.discover_master.called
            with pytest.raises(FollowScopeError):
                replica.get("k")

    def test_sentinel_failover_read_fails_before_confirmation_then_new_master(self):
        manager, master, old_conn, _ = self._sentinel_clients()
        new_address = ("10.0.0.3", 6379)

        with master.follow_scope():
            assert master.set("k", "v")

            # Failover happened but Sentinel has not confirmed a new master.
            manager.discover_master.side_effect = MasterNotFoundError("no master")
            with pytest.raises(ConnectionError):
                master.get("k")

            # Confirmation arrives but the borrowed connection is still stale:
            # fail once (never to a replica), then serve from the new master.
            manager.discover_master.side_effect = lambda service: new_address
            with pytest.raises(ConnectionError):
                master.get("k")
            assert old_conn.disconnect.called

            new_conn = MagicMock()
            new_conn.host, new_conn.port = new_address
            new_conn.retry.call_with_retry.side_effect = (
                lambda fn, cb, with_failure_count=False: fn()
            )
            new_conn.read_response.side_effect = lambda *a, **k: b"v"
            master.connection_pool.get_connection = lambda *a, **k: new_conn
            assert master.get("k") == b"v"

    def test_sentinel_read_without_prior_write_unchanged(self):
        manager, master, _, replica = self._sentinel_clients()
        with master.follow_scope():
            # no write -> no pin -> replica pool keeps working as before
            assert replica.get("k") == b"v"

    def test_standalone_client_scope_is_neutral(self):
        client = Redis(host="127.0.0.1", port=6399)
        conn = MagicMock()
        conn.retry.call_with_retry.side_effect = (
            lambda fn, cb, with_failure_count=False: fn()
        )
        conn.host, conn.port = "127.0.0.1", 6399
        conn.read_response.side_effect = lambda *a, **k: b"OK"
        client.connection_pool.get_connection = lambda *a, **k: conn
        client.connection_pool.release = lambda c: None
        with client.follow_scope():
            client.set("k", "v")
            client.get("k")
        assert conn.send_command.call_count == 2
