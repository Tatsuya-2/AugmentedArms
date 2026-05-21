"""WebSocket client for streaming raw EEG samples to a viewer / streaming PC.

The streaming PC is expected to run a WebSocket server (see
``streaming_pc/eeg_viewer.py``) that accepts:

  1. A connection from this client.
  2. A JSON registration message describing the stream:
        {"type": "register", "role": "eeg_sender",
         "user_id": "...", "sample_rate_hz": 250,
         "channels": 8, "units": "uV"}
  3. JSON batch frames thereafter:
        {"type": "eeg_batch",
         "user_id": "...", "sample_rate_hz": 250, "channels": 8,
         "seq_start": 12345,
         "t_start": 1716271234.567, "t_end": 1716271234.663,
         "samples": [[c1, c2, ..., c8], ...],
         "labels":  ["", "", ...]}

Design mirrors ``DroneMonitorClient`` (best-effort, daemon thread,
non-blocking, auto-reconnect) so the BrainFlow loop in ``BCIBoard.stream``
is never blocked by network I/O. The caller pushes per-sample dicts onto
``input_queue``; this client batches them and forwards as JSON frames.
"""

import json
import queue
import socket
import threading
import time

import numpy as np

try:
    import websocket  # provided by the `websocket-client` package
    from websocket import WebSocketException
except ImportError as e:
    raise ImportError(
        "EEGStreamClient requires the 'websocket-client' package. "
        "Install with: pip3 install websocket-client"
    ) from e


_RECONNECT_BACKOFF_SEC = 2.0
_CONNECT_TIMEOUT_SEC = 5.0
_QUEUE_MAX_SIZE = 2000  # ~8 s of raw samples at 250 Hz
_DEFAULT_BATCH_SIZE = 25  # 25 samples @ 250 Hz = 100 ms per batch
_DEFAULT_FLUSH_INTERVAL_MS = 150  # safety flush so startup isn't starved


class EEGStreamClient:
    """Best-effort WebSocket sender for raw EEG samples to a streaming PC.

    Usage:
        client = EEGStreamClient(
            host="192.168.1.50", port=9091,
            user_id=userID, sample_rate_hz=250, channels=8)
        client.start()
        # In the BCIBoard stream worker:
        client.input_queue.put_nowait(sample_dict)
        ...
        client.stop()  # optional; thread is a daemon
    """

    def __init__(
        self,
        host: str,
        port: int = 9091,
        *,
        user_id: str = "",
        sample_rate_hz: int = 250,
        channels: int = 8,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        flush_interval_ms: int = _DEFAULT_FLUSH_INTERVAL_MS,
        units: str = "uV",
    ):
        self._uri = f"ws://{host}:{port}"
        self._user_id = str(user_id)
        self._sample_rate_hz = int(sample_rate_hz)
        self._channels = int(channels)
        self._batch_size = max(1, int(batch_size))
        self._flush_interval_sec = max(0.01, flush_interval_ms / 1000.0)
        self._units = units

        # Public: producer (BCIBoard stream worker) puts per-sample dicts here.
        # Each sample dict matches what BCIBoard builds:
        #   {"timestamp": float, "eeg": np.ndarray (channels,),
        #    "label": str|None, "seq": int}
        self.input_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX_SIZE)

        self._stop_event = threading.Event()
        self._connected = False
        self._thread = threading.Thread(
            target=self._run_loop,
            name="EEGStreamClient",
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

    def _drain_batch(self):
        """Block up to flush_interval_sec waiting for the first sample, then
        non-blockingly pull up to batch_size-1 more. Returns the list of
        sample dicts (possibly empty if the wait timed out).
        """
        batch = []
        try:
            first = self.input_queue.get(timeout=self._flush_interval_sec)
        except queue.Empty:
            return batch
        batch.append(first)

        deadline = time.monotonic() + self._flush_interval_sec
        while len(batch) < self._batch_size and time.monotonic() < deadline:
            try:
                batch.append(self.input_queue.get_nowait())
            except queue.Empty:
                # Brief sleep to coalesce arrivals without busy-looping.
                time.sleep(0.001)
        return batch

    def _build_batch_frame(self, batch):
        first = batch[0]
        last = batch[-1]
        samples = []
        labels = []
        for s in batch:
            eeg = s.get("eeg")
            if isinstance(eeg, np.ndarray):
                samples.append(eeg.tolist())
            else:
                samples.append(list(eeg) if eeg is not None else [])
            label = s.get("label")
            labels.append("" if label is None else str(label))
        return {
            "type": "eeg_batch",
            "user_id": self._user_id,
            "sample_rate_hz": self._sample_rate_hz,
            "channels": self._channels,
            "seq_start": int(first.get("seq", 0)),
            "t_start": float(first.get("timestamp", 0.0)),
            "t_end": float(last.get("timestamp", 0.0)),
            "samples": samples,
            "labels": labels,
        }

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            ws = None
            try:
                print(f"[EEGStreamClient] connecting to {self._uri} ...")
                ws = websocket.create_connection(
                    self._uri, timeout=_CONNECT_TIMEOUT_SEC
                )
                ws.send(json.dumps({
                    "type": "register",
                    "role": "eeg_sender",
                    "user_id": self._user_id,
                    "sample_rate_hz": self._sample_rate_hz,
                    "channels": self._channels,
                    "units": self._units,
                }))
                self._connected = True
                print(
                    f"[EEGStreamClient] connected and registered as eeg_sender "
                    f"(user={self._user_id}, {self._sample_rate_hz} Hz, "
                    f"{self._channels} ch)"
                )

                while not self._stop_event.is_set():
                    batch = self._drain_batch()
                    if not batch:
                        continue
                    frame = self._build_batch_frame(batch)
                    ws.send(json.dumps(frame))
            except (WebSocketException, ConnectionError, OSError, socket.timeout) as e:
                if not self._stop_event.is_set():
                    print(
                        f"[EEGStreamClient] connection error: {e}; "
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
