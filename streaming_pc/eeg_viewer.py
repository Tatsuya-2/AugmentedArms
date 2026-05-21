"""Streaming-PC side: WebSocket server + live 8-channel EEG waveform viewer.

Run on the machine that should display real-time waveforms. The B2J PC
connects here as a WebSocket client (see ``AugmentedArms/EEGStreamClient.py``).

Usage:
    python3 eeg_viewer.py --host 0.0.0.0 --port 9091

Then on the B2J PC:
    python3 B2J-User.py --eeg-stream-host <this-pc-ip> --eeg-stream-port 9091

Protocol:
  1. Client connects and sends a JSON registration message:
       {"type":"register","role":"eeg_sender","user_id":...,
        "sample_rate_hz":250,"channels":8,"units":"uV"}
  2. Client streams JSON batch frames:
       {"type":"eeg_batch","seq_start":...,"t_start":...,"t_end":...,
        "samples":[[c1..cN], ...], "labels":["",...]}

Design:
  - ``websockets.serve`` runs on an asyncio loop in a background thread.
  - Incoming batches are converted to per-channel float arrays and pushed
    onto a thread-safe ``queue.Queue`` (drop-oldest on overflow).
  - The main thread runs Qt; a ``QTimer`` (~30 Hz) drains the queue,
    appends into a NumPy ring buffer (10 s × Fs), and updates PlotCurveItems.

Only one active sender at a time is assumed (typical lab setup). If a second
sender connects, both will write into the same buffer; the seq_start log
makes that case easy to spot.
"""

import argparse
import asyncio
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime

import numpy as np

try:
    import websockets
except ImportError as e:
    raise SystemExit(
        "Missing dependency: websockets. Install with: pip3 install websockets"
    ) from e

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError as e:
    raise SystemExit(
        "Missing dependency: pyqtgraph (and PyQt5). Install with: "
        "pip3 install pyqtgraph PyQt5"
    ) from e


# Match AugmentedArms/ABMI_Utils.py CABLE_COLORS_RGB
CHANNEL_COLORS = [
    (120, 120, 120),
    (129, 37, 186),
    (19, 26, 120),
    (24, 168, 101),
    (196, 187, 16),
    (219, 120, 13),
    (196, 0, 0),
    (107, 54, 29),
]


class EEGWebSocketServer:
    """Run ``websockets.serve`` on a dedicated asyncio thread and push
    decoded EEG batches into a thread-safe queue for the Qt viewer.
    """

    def __init__(self, host, port, batch_queue, status_callback=None,
                 record_dir=None):
        self._host = host
        self._port = port
        self._batch_queue = batch_queue
        self._status_callback = status_callback
        self._record_dir = record_dir
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        self._loop = None
        self._thread = threading.Thread(
            target=self._run, name="EEGWebSocketServer", daemon=True
        )

    def start(self):
        self._thread.start()

    def _set_status(self, msg):
        if self._status_callback is not None:
            self._status_callback(msg)
        print(f"[EEGViewer] {msg}")

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve_forever())

    async def _serve_forever(self):
        async with websockets.serve(self._handle_client, self._host, self._port):
            self._set_status(f"listening on ws://{self._host}:{self._port}")
            await asyncio.Future()  # run forever

    async def _handle_client(self, websocket):
        peer = "?"
        try:
            peer = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        except Exception:
            pass
        self._set_status(f"client connected ({peer})")
        record_file = None
        try:
            async for raw in websocket:
                record_file = self._handle_message(raw, peer, record_file)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if record_file is not None:
                try:
                    record_file.close()
                except Exception:
                    pass
            self._set_status(f"client disconnected ({peer})")

    def _open_record_file(self, user_id):
        if not self._record_dir:
            return None
        safe_user = "".join(c if c.isalnum() or c in "-_" else "_"
                            for c in str(user_id)) or "unknown"
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(self._record_dir, f"eeg_{safe_user}_{stamp}.jsonl")
        try:
            f = open(path, "w", buffering=1)  # line-buffered
        except OSError as e:
            self._set_status(f"failed to open record file {path}: {e}")
            return None
        self._set_status(f"recording to {path}")
        return f

    def _handle_message(self, raw, peer, record_file):
        try:
            msg = json.loads(raw)
        except (TypeError, ValueError):
            return record_file
        msg_type = msg.get("type")
        if msg_type == "register":
            sr = msg.get("sample_rate_hz")
            ch = msg.get("channels")
            self._set_status(
                f"registered eeg_sender from {peer} "
                f"(user={msg.get('user_id','?')}, "
                f"{sr} Hz, {ch} ch, units={msg.get('units','?')})"
            )
            if record_file is None:
                record_file = self._open_record_file(msg.get("user_id", "unknown"))
            if record_file is not None:
                try:
                    record_file.write(raw if isinstance(raw, str) else raw.decode("utf-8"))
                    record_file.write("\n")
                except OSError:
                    pass
            return record_file
        if msg_type != "eeg_batch":
            return record_file
        if record_file is not None:
            try:
                record_file.write(raw if isinstance(raw, str) else raw.decode("utf-8"))
                record_file.write("\n")
            except OSError:
                pass
        try:
            samples = np.asarray(msg["samples"], dtype=np.float32)
        except (KeyError, ValueError):
            return record_file
        if samples.ndim != 2:
            return record_file
        batch = {
            "samples": samples,  # shape: (n_samples, n_channels)
            "seq_start": int(msg.get("seq_start", 0)),
            "t_start": float(msg.get("t_start", 0.0)),
            "t_end": float(msg.get("t_end", 0.0)),
        }
        try:
            self._batch_queue.put_nowait(batch)
        except queue.Full:
            try:
                self._batch_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._batch_queue.put_nowait(batch)
            except queue.Full:
                pass
        return record_file


class EEGViewer(QtWidgets.QMainWindow):
    """8-channel stacked rolling waveform display."""

    def __init__(self, sample_rate_hz=250, channels=8, window_sec=10.0):
        super().__init__()
        self._sample_rate_hz = sample_rate_hz
        self._channels = channels
        self._buffer_len = int(sample_rate_hz * window_sec)
        self._ring = np.zeros((channels, self._buffer_len), dtype=np.float32)
        self._t_axis = np.linspace(-window_sec, 0.0, self._buffer_len, dtype=np.float32)
        self._batch_queue: queue.Queue = queue.Queue(maxsize=200)
        self._last_seq = None
        self._missed_seqs = 0

        self.setWindowTitle("EEG Viewer (B2J)")
        self.resize(1100, 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._status_label = QtWidgets.QLabel("waiting for connection...")
        self._status_label.setStyleSheet("color: #ccc; padding: 4px;")
        layout.addWidget(self._status_label)

        self._plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self._plot_widget, stretch=1)

        self._plots = []
        self._curves = []
        for ch in range(channels):
            p = self._plot_widget.addPlot(row=ch, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel("left", f"Ch{ch+1}", units="uV")
            p.setYRange(-200, 200)
            if ch < channels - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            p.setMouseEnabled(x=False, y=True)
            color = CHANNEL_COLORS[ch % len(CHANNEL_COLORS)]
            curve = p.plot(self._t_axis, self._ring[ch], pen=pg.mkPen(color=color, width=1))
            self._plots.append(p)
            self._curves.append(curve)
        # link x axes so panning/zoom stays in sync
        for p in self._plots[1:]:
            p.setXLink(self._plots[0])
        self._plots[-1].setLabel("bottom", "time", units="s")

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)  # ~30 Hz redraw

    @property
    def batch_queue(self):
        return self._batch_queue

    def set_status(self, text):
        # Called from the network thread; QLabel.setText is thread-safe enough
        # for this read-only display, but route through a queued slot to be safe.
        QtCore.QMetaObject.invokeMethod(
            self._status_label, "setText", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, text)
        )

    def _refresh(self):
        drained = 0
        while True:
            try:
                batch = self._batch_queue.get_nowait()
            except queue.Empty:
                break
            self._ingest(batch)
            drained += 1
            if drained > 50:
                break
        if drained == 0:
            return
        for ch in range(self._channels):
            self._curves[ch].setData(self._t_axis, self._ring[ch])

    def _ingest(self, batch):
        samples = batch["samples"]  # (n_samples, n_channels)
        n_samples, n_channels = samples.shape
        if n_channels < self._channels:
            return
        if n_samples >= self._buffer_len:
            self._ring[:] = samples[-self._buffer_len:, : self._channels].T
        else:
            self._ring = np.roll(self._ring, -n_samples, axis=1)
            self._ring[:, -n_samples:] = samples[:, : self._channels].T

        seq_start = batch["seq_start"]
        if self._last_seq is not None:
            expected = self._last_seq + 1
            if seq_start != expected:
                self._missed_seqs += max(0, seq_start - expected)
        self._last_seq = seq_start + n_samples - 1


def main():
    parser = argparse.ArgumentParser(
        description="Streaming PC EEG viewer (WebSocket server)."
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=9091,
                        help="Bind port (default: 9091).")
    parser.add_argument("--sample-rate-hz", type=int, default=250,
                        help="Sample rate for the ring buffer (default: 250).")
    parser.add_argument("--channels", type=int, default=8,
                        help="Channel count (default: 8).")
    parser.add_argument("--window-sec", type=float, default=10.0,
                        help="Ring-buffer length in seconds (default: 10).")
    parser.add_argument("--record-dir", default=None,
                        help="When set, every connected session is recorded "
                             "to a JSONL file in this directory "
                             "(one file per connection). Filename pattern: "
                             "eeg_<user_id>_<YYYY-MM-DD_HH-MM-SS>.jsonl")
    args = parser.parse_args()

    pg.setConfigOption("background", "k")
    pg.setConfigOption("foreground", "w")

    app = QtWidgets.QApplication(sys.argv)
    viewer = EEGViewer(
        sample_rate_hz=args.sample_rate_hz,
        channels=args.channels,
        window_sec=args.window_sec,
    )
    server = EEGWebSocketServer(
        args.host, args.port, viewer.batch_queue,
        status_callback=viewer.set_status,
        record_dir=args.record_dir,
    )
    server.start()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
