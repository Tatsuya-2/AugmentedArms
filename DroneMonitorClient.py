"""WebSocket client for sending raw BCI signals to drone_monitor.

Drone monitor (port 9090) expects external BCI senders to:
  1. Connect to ws://<host>:9090
  2. Send a JSON registration message: {"type": "register", "role": "bci_sender"}
  3. Send raw text signals ("1"/"2"/"3") as plain text WebSocket messages

This module provides a thread-based best-effort client that auto-reconnects
in the background, so the main BCI/Pygame loop is never blocked by network I/O.

Reference: dji-drone/isaac_ros_ws/src/drone_monitor/tools/bci_signal_sender.py
"""

import json
import queue
import socket
import threading
import time

try:
    import websocket  # provided by the `websocket-client` package
    from websocket import WebSocketException
except ImportError as e:
    raise ImportError(
        "DroneMonitorClient requires the 'websocket-client' package. "
        "Install with: pip3 install websocket-client"
    ) from e


_RECONNECT_BACKOFF_SEC = 2.0
_CONNECT_TIMEOUT_SEC = 5.0
_QUEUE_MAX_SIZE = 32
_SEND_QUEUE_POLL_SEC = 0.5


class DroneMonitorClient:
    """Best-effort WebSocket sender for raw BCI signals to drone_monitor.

    Usage:
        client = DroneMonitorClient(host="192.168.1.10", port=9090)
        client.start()
        ...
        client.send_signal("1")   # non-blocking, fire-and-forget
        ...
        client.stop()             # optional; thread is a daemon
    """

    def __init__(self, host: str, port: int = 9090):
        self._uri = f"ws://{host}:{port}"
        self._send_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX_SIZE)
        self._stop_event = threading.Event()
        self._connected = False
        self._thread = threading.Thread(
            target=self._run_loop,
            name="DroneMonitorClient",
            daemon=True,
        )

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def uri(self) -> str:
        return self._uri

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def send_signal(self, signal: str) -> None:
        """Enqueue a raw text signal for delivery. Non-blocking.

        Drops the signal (with a warning) if the queue is full.
        """
        try:
            self._send_queue.put_nowait(signal)
        except queue.Full:
            print(f"[DroneMonitorClient] send queue full, dropping signal: {signal!r}")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            ws = None
            try:
                print(f"[DroneMonitorClient] connecting to {self._uri} ...")
                ws = websocket.create_connection(
                    self._uri, timeout=_CONNECT_TIMEOUT_SEC
                )
                ws.send(json.dumps({"type": "register", "role": "bci_sender"}))
                self._connected = True
                print(f"[DroneMonitorClient] connected and registered as bci_sender")

                while not self._stop_event.is_set():
                    try:
                        signal = self._send_queue.get(timeout=_SEND_QUEUE_POLL_SEC)
                    except queue.Empty:
                        continue
                    ws.send(signal)
                    print(f"[DroneMonitorClient] sent signal: {signal!r}")
            except (WebSocketException, ConnectionError, OSError, socket.timeout) as e:
                if not self._stop_event.is_set():
                    print(
                        f"[DroneMonitorClient] connection error: {e}; "
                        f"reconnecting in {_RECONNECT_BACKOFF_SEC}s"
                    )
            finally:
                self._connected = False
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass

            if self._stop_event.is_set():
                break
            time.sleep(_RECONNECT_BACKOFF_SEC)
