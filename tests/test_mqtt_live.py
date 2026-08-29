"""
tests/test_mqtt_live.py — the task backlog (not in this public copy) T3.4, the last named gap: "mqtt_client.py's
'replay with a fake broker' — the existing test_fault_injection.py coverage
stubs client.publish/client.subscribe directly and manually toggles
up.connected, which tests the offline-queue LOGIC thoroughly but never
exercises MqttUplink.start()'s real connect_async/loop_start path or an
actual TCP handshake."

This file closes that gap using `tools/fake_mqtt_broker.py`, a minimal
dependency-free MQTT 3.1.1 broker built for exactly this (see that file's
own docstring for why a hand-rolled broker was chosen over installing
`amqtt`, which this run confirmed IS installable in this sandbox). Every
test below drives `MqttUplink` through its REAL public API
(`start()`/`stop()`/`publish_telemetry()`/`publish_anomaly()`) against a
REAL listening socket — no callback is ever invoked by hand, and
`up.connected` is read, never set, by these tests. That is the difference
from `test_fault_injection.py` §4, which is complementary, not redundant:
that file proves the offline-queue LOGIC is correct in isolation; this file
proves the REAL paho state machine (TCP connect, CONNACK parsing, automatic
reconnect with backoff) actually drives that logic the way production code
path does.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "tools", ROOT / "firmware"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fake_mqtt_broker import FakeBroker      # noqa: E402
from mqtt_client import MqttUplink            # noqa: E402


def _wait_until(predicate, timeout=5.0, interval=0.02):
    """Poll `predicate()` until it is truthy or `timeout` elapses. Returns
    whether it succeeded — tests assert on the return value so a timeout
    fails with the test's own message, not a bare `AssertionError` from
    inside this helper."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _cfg(port, keepalive=None):
    m = {"host": "127.0.0.1", "port": port, "tls": False, "api_key": "",
         "base_topic": "devices"}
    if keepalive is not None:
        m["keepalive"] = keepalive
    return {"device": {"id": "dev-live-test"}, "mqtt": m}


@pytest.fixture()
def broker():
    b = FakeBroker()
    b.start()
    yield b
    b.stop()


# ============================================================================
# Real connect: TCP handshake, CONNACK, retained status, cmd subscription
# ============================================================================

def test_real_start_connects_and_publishes_retained_status(broker):
    up = MqttUplink(_cfg(broker.port))
    up.start()
    try:
        ok = _wait_until(lambda: up.connected is True)
        assert ok, "MqttUplink.start() never reached connected=True against a real broker"
        assert broker.connect_count == 1

        got_status = _wait_until(
            lambda: any(t == "devices/dev-live-test/status" for t, *_ in broker.received))
        assert got_status, "no retained status message reached the real broker on connect"
        topic, payload, qos, retain = next(
            r for r in broker.received if r[0] == "devices/dev-live-test/status")
        assert json.loads(payload) == {"online": True}
        assert qos == 1 and retain is True

        subscribed = _wait_until(
            lambda: "devices/dev-live-test/cmd" in broker.subscriptions)
        assert subscribed, "MqttUplink never subscribed to its own cmd topic"
    finally:
        up.stop()


def test_real_stop_lets_the_connection_go(broker):
    up = MqttUplink(_cfg(broker.port))
    up.start()
    assert _wait_until(lambda: up.connected is True)
    up.stop()
    # loop_stop() blocks until the network thread exits, so this is
    # synchronous, not a race — no polling needed for the local flag.
    assert up.connected is False


# ============================================================================
# Real publish: telemetry (QoS 0) and anomaly (QoS 1) actually reach the wire
# ============================================================================

def test_real_publish_telemetry_and_anomaly_reach_the_broker(broker):
    up = MqttUplink(_cfg(broker.port))
    up.start()
    try:
        assert _wait_until(lambda: up.connected is True)

        up.publish_telemetry({"window": 7, "score": 1.2})
        up.publish_anomaly({"episode": "e1", "score": 40.0})

        got_telemetry = _wait_until(
            lambda: any(t == "devices/dev-live-test/telemetry" for t, *_ in broker.received))
        got_anomaly = _wait_until(
            lambda: any(t == "devices/dev-live-test/anomaly" for t, *_ in broker.received))
        assert got_telemetry, "telemetry never reached the real broker while connected"
        assert got_anomaly, "anomaly never reached the real broker while connected"

        tele = next(r for r in broker.received if r[0] == "devices/dev-live-test/telemetry")
        anom = next(r for r in broker.received if r[0] == "devices/dev-live-test/anomaly")
        assert json.loads(tele[1]) == {"window": 7, "score": 1.2}
        assert tele[2] == 0, "telemetry must be QoS 0 per mqtt_client.py's own docstring"
        assert json.loads(anom[1]) == {"episode": "e1", "score": 40.0}
        assert anom[2] == 1, "anomaly must be QoS 1 (must-arrive) per the module docstring"
    finally:
        up.stop()


# ============================================================================
# Real downlink: a broker-pushed command reaches on_command via real on_message
# ============================================================================

def test_real_downlink_command_invokes_on_command(broker):
    received_cmds = []
    up = MqttUplink(_cfg(broker.port), on_command=received_cmds.append)
    up.start()
    try:
        assert _wait_until(lambda: up.connected is True)
        assert _wait_until(lambda: "devices/dev-live-test/cmd" in broker.subscriptions)

        broker.push("devices/dev-live-test/cmd",
                    json.dumps({"cmd": "start_learning"}).encode(), qos=1)

        got = _wait_until(lambda: len(received_cmds) == 1)
        assert got, "a real broker-pushed command never reached on_command"
        assert received_cmds[0] == {"cmd": "start_learning"}
    finally:
        up.stop()


# ============================================================================
# The T3.4 headline case: offline queueing + REAL automatic reconnect drain
# ============================================================================

def test_offline_queue_drains_on_real_automatic_reconnect(broker):
    """This is the scenario test_fault_injection.py's own comment names as
    untested: not the queue LOGIC (already covered), but whether paho's real
    background thread — on a connection it drops and reconnects entirely on
    its own, with no test code calling _on_connect/_on_disconnect — actually
    drains the queue the same way production does. `up.connected` is only
    ever read here, driven purely by real callbacks fired from a real
    socket."""
    up = MqttUplink(_cfg(broker.port))
    up.client.reconnect_delay_set(min_delay=1, max_delay=1)   # fast, bounded test time
    up.start()
    try:
        assert _wait_until(lambda: up.connected is True), "initial real connect failed"

        # Sever the TCP connection without touching the listener — models a
        # network blip / broker restart, the case the offline queue exists
        # for. Do NOT call _on_disconnect by hand: the whole point of this
        # test is that paho's own detection drives up.connected.
        broker.kick()
        assert _wait_until(lambda: up.connected is False, timeout=6.0), (
            "real disconnect was never detected by the client's own callback")

        # Genuinely offline now (per the client's own state, not an assumption):
        # queue three anomalies through the real public API.
        for i in range(3):
            up.publish_anomaly({"episode": i})
        assert len(up._queue) == 3, "anomalies must queue locally while really offline"

        # Let paho's real background thread reconnect on its own schedule.
        reconnected = _wait_until(lambda: up.connected is True, timeout=8.0)
        assert reconnected, "paho's automatic reconnect never re-established the connection"
        assert broker.connect_count == 2, (
            f"expected exactly one real reconnect, saw {broker.connect_count} connects")

        drained = _wait_until(lambda: len(up._queue) == 0, timeout=3.0)
        assert drained, "queue did not drain after a real reconnect"

        arrived = _wait_until(
            lambda: sum(1 for t, *_ in broker.received
                        if t == "devices/dev-live-test/anomaly") >= 3,
            timeout=3.0)
        assert arrived, "queued anomalies never actually reached the broker after reconnect"

        episodes = sorted(
            json.loads(p)["episode"] for t, p, *_ in broker.received
            if t == "devices/dev-live-test/anomaly")
        assert episodes == [0, 1, 2], "all three queued anomalies must survive the reconnect"
    finally:
        up.stop()


# ============================================================================
# Refused connection: a non-zero CONNACK return code must not look connected
# ============================================================================

def test_refused_connack_does_not_mark_the_client_connected(broker):
    broker.refuse_next = True
    up = MqttUplink(_cfg(broker.port))
    up.start()
    try:
        # Give the (failing) handshake a real chance, then assert it stayed
        # false — this is a negative wait, so bound it rather than loop
        # forever on a predicate that is never going to become true.
        time.sleep(1.0)
        assert up.connected is False, (
            "a refused CONNACK (rc != 0) must not be treated as connected")
    finally:
        up.stop()
