"""Hardware mocks for B2J-User.py (enabled by ``--mock-hardware``).

This module replaces the BCI board, piezo sensor, and M5 serial link with
in-process fakes so ``B2J-User.py`` can run with **no real hardware** — useful
for developing/testing the state machine and the drone_monitor forwarding path
on a machine that has neither the BCI dongle, the piezo GPIO, nor the M5 serial
device.

Separation of concerns:
- Production launches (``python3 B2J-User.py`` with no flag) NEVER import this
  module. It is only imported from B2J-User.py's top-level bootstrap when
  ``--mock-hardware`` is present in ``sys.argv``.
- ``pygame`` is intentionally NOT mocked here: with ``--mock-hardware`` you
  still get a real window and trigger predictions with the real keyboard
  ('b' key).

``install()`` must be called BEFORE B2J-User.py imports ``ABMI_Utils`` /
``ALS_Utils`` / ``serial`` so the fakes are already in ``sys.modules``.
"""
import os
import sys
import time
import types

YELLOW = "\033[93m"
RESET = "\033[0m"


def _log(msg):
	print(f"{YELLOW}[MOCK-HW]{RESET} {msg}")


# --------------------------------------------------------------------------
# Fake BCI board (stands in for ABMI_Utils.BCIBoard)
# --------------------------------------------------------------------------
class _FakeBoard:
	def __init__(self, *args, **kwargs):
		self.connected = True
		self._last_data_frame = "TESTFRAME"

	def connect(self):
		self.connected = True
		_log("board.connect() (mock)")

	def stream(self):
		_log("board.stream() (mock)")

	def stop_stream(self):
		_log("board.stop_stream() (mock)")


# --------------------------------------------------------------------------
# Fake training-sequence thread: reports alive for a short window so the
# "recording" state lasts a realistic beat instead of finishing instantly.
# --------------------------------------------------------------------------
def _record_duration_s():
	# Test knob: how long the mock "recording" state lasts. Lets paced
	# real-drone tests space out the forwarded 1/2/3 signals. Default 0.8s.
	try:
		return float(os.environ.get("B2J_MOCK_RECORD_SEC", "0.8"))
	except (TypeError, ValueError):
		return 0.8


class _FakeSequenceThread:
	def __init__(self, duration_s=None):
		if duration_s is None:
			duration_s = _record_duration_s()
		self._end = time.time() + duration_s

	def is_alive(self):
		return time.time() < self._end


# --------------------------------------------------------------------------
# Prediction stub: cycles 1 -> 2 -> 3 on successive calls so every branch
# (left/center/right) is exercised across runs.
# --------------------------------------------------------------------------
_PREDICT_CYCLE = [1, 2, 3]
_predict_idx = {"n": 0}


def _useModelToPredict(latest_test_file, model_path):
	i = _predict_idx["n"]
	choice = _PREDICT_CYCLE[i % len(_PREDICT_CYCLE)]
	_predict_idx["n"] += 1
	_log(f"useModelToPredict() -> {choice} (mock, cycling 1->2->3)")
	return choice


def _build_abmi():
	m = types.ModuleType("ABMI_Utils")
	m.BCIBoard = lambda *a, **k: _FakeBoard()
	m.getUserID = lambda *a, **k: "testuser"
	m.set_latency_timer = lambda *a, **k: None
	m.play_single_sound = lambda *a, **k: None
	m.startSingleTrainingSequence = lambda *a, **k: (_FakeSequenceThread(), object())
	m.useModelToPredict = _useModelToPredict
	m.deleteTestingFiles = lambda *a, **k: None
	return m


# --------------------------------------------------------------------------
# Fake piezo: always returns False. The trigger comes from the keyboard 'b'
# key handled by the real pygame event loop in B2J-User.py.
# --------------------------------------------------------------------------
class _FakePiezo:
	def __init__(self, *args, **kwargs):
		pass

	def was_pressed(self):
		return False


def _build_als():
	m = types.ModuleType("ALS_Utils")
	m.PiezoSensor = _FakePiezo
	return m


# --------------------------------------------------------------------------
# Fake serial (M5): a no-op context manager so sendToM5() works without a
# real device. comports() returns one fake port so pick_m5_port() succeeds
# and the app does not spam "No serial port selected".
# --------------------------------------------------------------------------
class _SerialException(Exception):
	pass


class _FakeSerial:
	def __init__(self, port, baud=None, timeout=None, *args, **kwargs):
		self._port = port

	def __enter__(self):
		return self

	def __exit__(self, *exc):
		return False

	def write(self, data):
		try:
			text = data.decode("utf-8", "replace").strip()
		except Exception:
			text = repr(data)
		_log(f"M5 write (mock): {text}")

	def flush(self):
		pass


class _FakePort:
	device = "MOCK-M5"
	description = "Mock M5 (test-mode)"
	manufacturer = "test"
	product = "mock"


def _build_serial():
	serial_mod = types.ModuleType("serial")
	serial_mod.Serial = _FakeSerial
	serial_mod.SerialException = _SerialException

	tools = types.ModuleType("serial.tools")
	list_ports = types.ModuleType("serial.tools.list_ports")
	list_ports.comports = lambda: [_FakePort()]
	tools.list_ports = list_ports
	serial_mod.tools = tools
	return serial_mod, tools, list_ports


def install():
	"""Inject fake hardware modules into ``sys.modules``. Idempotent."""
	existing = sys.modules.get("ABMI_Utils")
	if existing is not None and getattr(existing, "_B2J_MOCK_HW", False):
		return

	abmi = _build_abmi()
	abmi._B2J_MOCK_HW = True
	als = _build_als()
	als._B2J_MOCK_HW = True
	serial_mod, tools, list_ports = _build_serial()

	sys.modules["ABMI_Utils"] = abmi
	sys.modules["ALS_Utils"] = als
	sys.modules["serial"] = serial_mod
	sys.modules["serial.tools"] = tools
	sys.modules["serial.tools.list_ports"] = list_ports

	_log("hardware mocks installed (BCI board, piezo, M5 serial)")
