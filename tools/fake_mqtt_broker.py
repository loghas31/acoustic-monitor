"""
Minimal single-client MQTT 3.1.1 broker, built directly on raw sockets.

Why this exists (the task backlog (not in this public copy) T3.4): every MQTT test up to this point (see
`tests/test_fault_injection.py` §4 "Broker unreachable for days") stubs
`MqttUplink.client.publish`/`.subscribe` directly and calls `_on_connect`/
`_on_disconnect` by hand. That exercises the offline-queue LOGIC thoroughly
but never touches `MqttUplink.start()`'s real code path: a real
`connect_async()` + `loop_start()`, a real TCP handshake, paho's own
CONNECT/CONNACK wire encoding, or paho's built-in automatic-reconnect state
machine (`loop_forever`'s `_reconnect_wait` backoff, confirmed by reading
paho 2.1.0's own source: `loop_start` -> `_thread_main` -> `loop_forever`,
which calls `self.reconnect()` after `_reconnect_wait()` whenever the
connection drops and `reconnect_on_failure` is left at its default `True`).

Two ways existed to close that gap (the task backlog (not in this public copy) names both): install a real
broker package, or hand-roll just enough of the wire protocol. `amqtt` was
confirmed installable in this sandbox's pip proxy (`pip install amqtt`
succeeds, 0.12.0) but pulls in ~10 extra transitive dependencies (asyncio
web/auth/CLI tooling this project has no other use for) for a test-only
need. This repo is a teaching codebase that otherwise keeps its dependency
footprint deliberately small (see CONTRIBUTING.md, and `ml/evaluate.py`
already avoiding heavier alternatives elsewhere) — a ~250-line, dependency-
free, single-file broker that speaks exactly the subset of MQTT 3.1.1
`MqttUplink` actually uses is more in keeping with that, and is easier for
Logan to read and trust six months from now than a third-party async broker
he's never had to debug. That is a judgement call, recorded here rather than
silently made.

Deliberately NOT a spec-complete broker. It implements only:
  CONNECT  -> CONNACK              (OASIS MQTT v3.1.1 §3.1/§3.2)
  SUBSCRIBE -> SUBACK              (§3.8/§3.9)
  UNSUBSCRIBE -> UNSUBACK          (§3.10/§3.11)
  PUBLISH (QoS 0 and QoS 1)        (§3.3), with PUBACK for QoS 1 (§3.4)
  PINGREQ -> PINGRESP              (§3.12/§3.13)
  DISCONNECT                       (§3.14)
No retained-message store, no QoS 2, no auth checking (beyond an optional
"refuse the next CONNECT" test hook), no MQTT 5. That is exactly the set
`firmware/mqtt_client.py`'s `MqttUplink` sends and expects; anything else a
real broker does is out of scope for what this file is for.
"""

from __future__ import annotations

import socket
import struct
import threading

# Fixed-header packet types, OASIS MQTT v3.1.1 §2.2.1, Table 2.1.
CONNECT = 0x10
CONNACK = 0x20
PUBLISH_TYPE = 0x30          # top nibble; low nibble carries DUP/QoS/RETAIN
PUBACK = 0x40
SUBSCRIBE_TYPE = 0x80
SUBACK = 0x90
UNSUBSCRIBE_TYPE = 0xA0
UNSUBACK = 0xB0
PINGREQ = 0xC0
PINGRESP = 0xD0
DISCONNECT = 0xE0


def _encode_remaining_length(n: int) -> bytes:
    """§2.2.3 variable-length-integer encoding, 7 bits/byte + continuation."""
    out = bytearray()
    while True:
        byte = n % 128
        n //= 128
        if n > 0:
            byte |= 0x80
        out.append(byte)
        if n == 0:
            return bytes(out)


def _read_remaining_length(recv_exact) -> int:
    multiplier = 1
    value = 0
    while True:
        b = recv_exact(1)[0]
        value += (b & 0x7F) * multiplier
        if not (b & 0x80):
            return value
        multiplier *= 128


def _encode_string(s: str) -> bytes:
    """§1.5.3 UTF-8 encoded string: 2-byte big-endian length prefix + bytes."""
    b = s.encode("utf-8")
    return struct.pack("!H", len(b)) + b


def _decode_string(buf: bytes, i: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("!H", buf, i)
    i += 2
    return buf[i:i + n].decode("utf-8"), i + n


class FakeBroker:
    """A real (if minimal) MQTT broker listening on 127.0.0.1:<ephemeral>.

    One accepted connection is tracked at a time, which is all a single
    `MqttUplink` ever needs. `stop()` tears the whole listener down (models
    "broker gone for good" / process exit); `kick()` only drops the current
    client connection while the listener keeps accepting (models a network
    blip or broker restart the client's own auto-reconnect must recover
    from) — the two are deliberately different methods because
    `MqttUplink`'s behaviour under each should be, and is, different.
    """

    def __init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]

        self.received: list[tuple[str, bytes, int, bool]] = []  # (topic, payload, qos, retain)
        self.subscriptions: list[str] = []
        self.connect_count = 0

        self.refuse_next = False        # next CONNECT gets a non-zero return code
        self.drop_after_connack = False  # close the socket right after handshake

        self._stop_flag = threading.Event()
        self._accept_thread: threading.Thread | None = None
        self._conn: socket.socket | None = None
        self._conn_lock = threading.Lock()

    def start(self) -> None:
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        try:
            self._sock.close()
        except OSError:
            pass
        self.kick()

    def kick(self) -> None:
        with self._conn_lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            # MEASURED BUG, not assumed: plain `conn.close()` here does NOT
            # send a TCP FIN to the client while `_serve`'s own thread is
            # blocked in a `recv()` call on this same fd -- reproduced with
            # a 10-line isolated repro (two threads, one blocked in recv(),
            # the other closing the fd from outside: the peer's socket never
            # becomes select()-readable, even after several seconds).
            # `shutdown(SHUT_RDWR)` first forces that blocked recv() to
            # return AND reliably delivers the FIN; `close()` alone does
            # neither in this environment. Without this, `kick()` would
            # silently fail to simulate a real disconnect and every
            # reconnect test built on it would hang or time out.
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass                    # already half-closed / peer gone — fine
            try:
                conn.close()
            except OSError:
                pass

    def push(self, topic: str, payload: bytes, qos: int = 1) -> None:
        """Server -> client PUBLISH — used to simulate a downlink command
        arriving on the device's `cmd` topic."""
        with self._conn_lock:
            conn = self._conn
        if conn is None:
            raise RuntimeError("fake broker: no client connected to push to")
        pkt_id = 1
        body = _encode_string(topic)
        if qos:
            body += struct.pack("!H", pkt_id)
        body += payload
        header = bytes([PUBLISH_TYPE | (qos << 1)]) + _encode_remaining_length(len(body))
        conn.sendall(header + body)

    # -- internals ---------------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                conn, _addr = self._sock.accept()
            except OSError:
                return                                    # listener closed
            self.connect_count += 1
            with self._conn_lock:
                self._conn = conn
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        def recv_exact(n: int) -> bytes:
            buf = b""
            while len(buf) < n:
                chunk = conn.recv(n - len(buf))
                if not chunk:
                    raise ConnectionError("fake broker: client closed")
                buf += chunk
            return buf

        try:
            while not self._stop_flag.is_set():
                header_byte = recv_exact(1)[0]
                pkt_type = header_byte & 0xF0
                remaining = _read_remaining_length(recv_exact)
                body = recv_exact(remaining) if remaining else b""

                if pkt_type == CONNECT:
                    rc = 5 if self.refuse_next else 0       # 5 = not authorised, §3.2.2.3
                    self.refuse_next = False
                    conn.sendall(bytes([CONNACK, 0x02, 0x00, rc]))
                    if self.drop_after_connack:
                        self.drop_after_connack = False
                        return
                elif pkt_type == PUBLISH_TYPE:
                    qos = (header_byte >> 1) & 0x03
                    retain = bool(header_byte & 0x01)
                    topic, i = _decode_string(body, 0)
                    pkt_id = None
                    if qos > 0:
                        (pkt_id,) = struct.unpack_from("!H", body, i)
                        i += 2
                    payload = body[i:]
                    self.received.append((topic, payload, qos, retain))
                    if qos == 1:
                        conn.sendall(bytes([PUBACK, 0x02]) + struct.pack("!H", pkt_id))
                elif pkt_type == SUBSCRIBE_TYPE:
                    (pkt_id,) = struct.unpack_from("!H", body, 0)
                    i = 2
                    granted = bytearray()
                    while i < len(body):
                        topic, i = _decode_string(body, i)
                        requested_qos = body[i]
                        i += 1
                        self.subscriptions.append(topic)
                        granted.append(requested_qos)
                    resp = struct.pack("!H", pkt_id) + bytes(granted)
                    conn.sendall(bytes([SUBACK]) + _encode_remaining_length(len(resp)) + resp)
                elif pkt_type == UNSUBSCRIBE_TYPE:
                    (pkt_id,) = struct.unpack_from("!H", body, 0)
                    conn.sendall(bytes([UNSUBACK, 0x02]) + struct.pack("!H", pkt_id))
                elif header_byte == PINGREQ:
                    conn.sendall(bytes([PINGRESP, 0x00]))
                elif header_byte == DISCONNECT:
                    return
                # else: unrecognised packet type — ignore rather than crash
                # the broker thread; a real broker would fault the client,
                # but nothing MqttUplink sends should ever hit this branch.
        except (ConnectionError, OSError):
            pass
        finally:
            with self._conn_lock:
                if self._conn is conn:
                    self._conn = None
