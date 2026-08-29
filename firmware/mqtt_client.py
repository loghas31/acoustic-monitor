"""
MQTT uplink. Telemetry + anomaly events up, commands down.

Topics (device_id = cfg.device.id):
    devices/{id}/telemetry   one JSON per window (QoS 0 — losing one is fine)
    devices/{id}/anomaly     anomaly events       (QoS 1 — must arrive)
    devices/{id}/status      retained online/offline via LWT
    devices/{id}/cmd         downlink: {"cmd": "start_learning" | "set_threshold", ...}

Offline behaviour (non-negotiable in the spec): if the broker is unreachable,
anomaly events queue in RAM (bounded deque) and replay on reconnect; telemetry
is dropped (it's a time series — stale points have little value and the cloud
interpolates). Local alerting never depends on this module.
"""

from __future__ import annotations

import collections
import json
import logging
import ssl

log = logging.getLogger(__name__)


class MqttUplink:
    def __init__(self, cfg: dict, on_command=None):
        import paho.mqtt.client as mqtt
        m = cfg["mqtt"]
        self.device_id = cfg["device"]["id"]
        self.base = m.get("base_topic", "devices")
        self.on_command = on_command
        self._queue = collections.deque(maxlen=500)     # ~ a day of anomalies, bounded
        self.connected = False

        # paho 1.x/2.x compat: 2.x requires an explicit callback API version;
        # VERSION1 keeps the (client, userdata, ...) signatures used below.
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,
                                      client_id=self.device_id, clean_session=False)
        except AttributeError:
            self.client = mqtt.Client(client_id=self.device_id, clean_session=False)
        # API key doubles as the MQTT password — one credential per device,
        # revocable server-side without touching the unit.
        self.client.username_pw_set(self.device_id, m.get("api_key") or None)
        if m.get("tls", True):
            ctx = ssl.create_default_context(cafile=m.get("ca_cert") or None)
            self.client.tls_set_context(ctx)
        self.client.will_set(f"{self.base}/{self.device_id}/status",
                             json.dumps({"online": False}), qos=1, retain=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.host, self.port = m["host"], int(m.get("port", 8883))

    def start(self) -> None:
        try:
            self.client.connect_async(self.host, self.port, keepalive=60)
            self.client.loop_start()                    # paho's own thread; survives drops
        except Exception as e:                          # noqa: BLE001 — never kill the loop
            log.warning("mqtt connect failed (%s); running offline", e)

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()

    # -- callbacks -----------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc):
        self.connected = rc == 0
        if not self.connected:
            log.warning("mqtt connect rc=%s", rc)
            return
        client.publish(f"{self.base}/{self.device_id}/status",
                       json.dumps({"online": True}), qos=1, retain=True)
        client.subscribe(f"{self.base}/{self.device_id}/cmd", qos=1)
        while self._queue:                              # replay queued anomalies
            topic, payload = self._queue.popleft()
            client.publish(topic, payload, qos=1)
        log.info("mqtt connected, queue drained")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        log.warning("mqtt disconnected rc=%s — queuing anomalies locally", rc)

    def _on_message(self, client, userdata, msg):
        try:
            cmd = json.loads(msg.payload)
        except json.JSONDecodeError:
            log.warning("bad cmd payload: %r", msg.payload[:100])
            return
        if self.on_command:
            self.on_command(cmd)

    # -- publishing ----------------------------------------------------------

    def publish_telemetry(self, payload: dict) -> None:
        if self.connected:
            self.client.publish(f"{self.base}/{self.device_id}/telemetry",
                                json.dumps(payload), qos=0)

    def publish_anomaly(self, payload: dict) -> None:
        topic = f"{self.base}/{self.device_id}/anomaly"
        data = json.dumps(payload)
        if self.connected:
            self.client.publish(topic, data, qos=1)
        else:
            # T4.3: a `deque(maxlen=...)` silently drops from the OPPOSITE
            # end once full — appending here evicts the OLDEST queued
            # anomaly with no error and no log line. 500 anomaly events is a
            # lot (this product alerts at most a handful of times per unit
            # per year after the persistence gate), but a multi-day outage
            # is exactly the scenario T4.3 asks about, and a silent drop is
            # the one outcome the whole audit exists to catch. Log once per
            # drop rather than fail: losing the single oldest queued event
            # during a multi-day broker outage is the documented trade-off
            # (offline anomalies queue in bounded RAM), it just should never
            # be a *quiet* one.
            if len(self._queue) == self._queue.maxlen:
                log.warning("mqtt offline queue full (%d) — dropping the "
                           "oldest queued anomaly to make room for this one",
                           self._queue.maxlen)
            self._queue.append((topic, data))


class NullUplink:
    """Stand-in when running with --no-mqtt (tests, fully local demo)."""
    connected = False
    def start(self): pass
    def stop(self): pass
    def publish_telemetry(self, payload): pass
    def publish_anomaly(self, payload): pass
