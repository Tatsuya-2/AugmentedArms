"""
ABMI-Utils.py
Utility functions for BMI-Trainer project.
Core code written and designed by Mikito Ogino
All copyright and intellectual property belongs to Mikito Ogino
"""

# -*- coding: utf-8 -*-
import time
import numpy as np
import serial
import serial.tools.list_ports
import time
import threading
import os
import csv
import random
import pygame
import shutil
import queue
import datetime
import joblib
import glob
import subprocess
import pandas as pd

from pathlib import Path
from scipy.signal import iirfilter, filtfilt
from pyOpenBCI import OpenBCICyton
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from ftplib import FTP, error_perm, all_errors
from scipy.signal import firwin, lfilter
from scipy.signal import filtfilt
from sklearn.preprocessing import StandardScaler

# --- Constants ---
SERIES_R = 2200.0			# Series resistance (2.2 kΩ)
I_DRIVE = 6.0e-9			# Lead-off drive current (6 nA default)
MEAS_SEC = 6.0				# Measurement duration in seconds
BAND = (5, 50)				# Bandpass filter range (Hz)
ISI = 0.35 					# Inter-Stimulus Interval in seconds
SOUND_LENGTH = .15			# Duration of Audio

# Hand-coded color table for cable impedance display
# Each color is defined as RGB tuple (0-255 range)
CABLE_COLORS_RGB = [
	(120, 120, 120),  # Channel 1: Gray
	(129, 37, 186),    # Channel 2: Purple  
	(19, 26, 120),      # Channel 3: Blue
	(24, 168, 101),      # Channel 4: Green
	(196, 187, 16),    # Channel 5: Yellow
	(219, 120, 13),    # Channel 6: Orange
	(196, 0, 0),      # Channel 7: Red
	(107, 54, 29)     # Channel 8: Brown
]

# Legacy string color names for backward compatibility
CABLE_COLORS = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']

class SingleTrainingSequenceError(Exception):
	"""Raised when a single training sequence fails to start."""

class BCIBoard:
	def __init__(self, port="/dev/ttyUSB0"):
		self.port = port
		self.board = None
		self.board_id = BoardIds.CYTON_BOARD.value
		self.fs = BoardShim.get_sampling_rate(self.board_id)
		self.channels = list(range(1, 9))  # default 8 channels
		self.connected = False
		self.streaming = False
		self._last_data_time = None
		self._stream_thread = None
		self._last_data_frame = None
  
		#Recording Vars
		self.recording = False
		self.stimulus_sound = 0
		self.sequence_id = 0
		self._record_thread = None
		self._record_stop_event = None
		self._record_file_path = None
		self._sample_queue = queue.Queue(maxsize=1000) #Queue holds up to 4 seconds of data
		self.minimum_recorded_rows = 5800 #20seconds x250, + 500for baseline x2

		self.STREAM_CFG_DEFAULT = dict(gain=6, input_type=0, bias=1, srb2=1, srb1=0)  # reasonable "normal" preset
		self.IMP_CFG = dict(gain=0, input_type=0, bias=1, srb2=0, srb1=0)            # GUI-like impedance preset

        # Track what this app last set per channel so we can revert on toggle-off
		self._ch_cfg = [self.STREAM_CFG_DEFAULT.copy() for _ in range(8)]
		self._ch_last_cfg = [None for _ in range(8)]


	def connect(self):
		BoardShim.disable_board_logger()
		params = BrainFlowInputParams()
		params.serial_port = self.port
		self.board = BoardShim(self.board_id, params)

		try:
			self.board.prepare_session()
			self.board.start_stream()
			print(f"Connected to OpenBCI board at {self.port}")
			self.connected = True
			self._last_data_time = None
			self.streaming = False
			return True

		except Exception as e:
			print(f"Failed to connect to OpenBCI board at {self.port}: {e}")
			try:
				self.board.release_session()
			except Exception:
				pass
			self.board = None
			self.connected = False
			self.streaming = False
			self._last_data_time = None
			return False

	def disconnect(self):
		if self.board:
			try:
				self.board.stop_stream()
				self.board.release_session()
				print(f"[BCIBoard] Disconnected from board at {self.port}")
			except Exception:
				pass
			self.board = None
		self.connected = False
		self.streaming = False
		self._last_data_time = None
		self._stream_thread = None

	def _refresh_stream_state(self, stale_timeout=1.5, probe=False):
		if not self.streaming:
			return False

		if not self.board:
			self.streaming = False
			self.connected = False
			self._last_data_time = None
			return False

		if probe:
			try:
				data = self.board.get_current_board_data(1)
				if data.size > 0:
					self._last_data_time = time.time()
			except BrainFlowError as e:
				print(f"[BCIBoard] Streaming probe failed: {e}")
				self.streaming = False
				self.connected = False
				self._last_data_time = None
				return False
			except Exception as e:
				print(f"[BCIBoard] Unexpected streaming probe error: {e}")
				self.streaming = False
				self.connected = False
				self._last_data_time = None
				return False

		if self._last_data_time is None:
			self._last_data_time = time.time()

		if time.time() - self._last_data_time > stale_timeout:
			print(f"[BCIBoard] No data received for {stale_timeout} s; marking board disconnected.")
			self.streaming = False
			self.connected = False
			self._last_data_time = None
			return False

		return True

	def stream(self, callback=None):
		if not self.board or not self.connected:
			print("[BCIBoard] Cannot start streaming: board not connected")
			return False

		if self.streaming:
			print("[BCIBoard] Streaming already in progress")
			return True

		self.streaming = True
		self._last_data_time = None

		eeg_idxs = BoardShim.get_eeg_channels(self.board_id)
		try:
			ts_idx = BoardShim.get_timestamp_channel(self.board_id)
		except Exception:
			ts_idx = None

		def _worker():
			stale_timeout = 1.5
			while self.streaming and self.board:
				if not self._refresh_stream_state(stale_timeout=stale_timeout):
					break

				try:
					data = self.board.get_board_data()
				except BrainFlowError as e:
					print(f"[BCIBoard] Stream error: {e}")
					self.connected = False
					self.streaming = False
					self._last_data_time = None
					break
				except Exception as e:
					print(f"[BCIBoard] Unexpected stream error: {e}")
					self.connected = False
					self.streaming = False
					self._last_data_time = None
					break

				if data.size > 0:
					self._last_data_time = time.time()
					num_samples = data.shape[1]
					for i in range(num_samples):
						timestamp = float(data[ts_idx, i]) if ts_idx is not None else time.time()
						eeg_values = np.array(data[eeg_idxs, i], copy=True)
						sample = {
							"timestamp": timestamp,
							"eeg": eeg_values,
							"label": self.stimulus_sound,
							"seq": self.sequence_id
						}
						if self.recording and self._record_stop_event and not self._record_stop_event.is_set():
							try:
								self._sample_queue.put_nowait(sample)
							except queue.Full:
								try:
									self._sample_queue.get_nowait()
								except queue.Empty:
									pass
								try:
									self._sample_queue.put_nowait(sample)
								except queue.Full:
									pass
				else:
					time.sleep(0.002)
					continue

				time.sleep(0.001)

			self.streaming = False
			if not self.connected:
				self._last_data_time = None
			self._stream_thread = None

		self._stream_thread = threading.Thread(target=_worker, daemon=True)
		self._stream_thread.start()
		print(f"[BCIBoard] Started streaming from board at {self.port}")
		return True

	def stop_stream(self, wait=True, clear_last_time=True):
		"""Stop the worker thread and optionally clear timing state while staying connected."""
		self.streaming = False
		thread = self._stream_thread
		if thread and thread.is_alive():
			if wait:
				thread.join(timeout=1.0)
		self._stream_thread = None
		if clear_last_time:
			self._last_data_time = None

	def start_recording(self, folder_path, filename=None):
		"""Start a recording worker that logs incoming EEG samples to CSV."""
		if not folder_path:
			raise ValueError('folder_path is required')

		if self.recording:
			print('[BCIBoard] Recording already in progress')
			return False

		folder = Path(folder_path).expanduser().resolve()
		folder.mkdir(parents=True, exist_ok=True)

		if filename:
			file_path = folder / filename
		else:
			timestamp_label = time.strftime('%Y%m%d_%H%M%S')
			file_path = folder / f'eeg_data_{timestamp_label}.csv'

		header = ['Timestamp', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8', 'Label', 'Seq']

		stop_event = threading.Event()
		self._record_stop_event = stop_event
		self._record_file_path = str(file_path)
		self._sample_queue = queue.Queue(maxsize=1000)
		sample_queue = self._sample_queue

		def _record_worker():
			rows_written = 0
			should_delete = False
			try:
				with file_path.open('w', newline='') as csvfile:
					writer = csv.writer(csvfile)
					writer.writerow(header)
					while not stop_event.is_set() or not sample_queue.empty():
						if not self.connected or not self.streaming:
							should_delete = True
							stop_event.set()
							while not sample_queue.empty():
								try:
									sample_queue.get_nowait()
								except queue.Empty:
									break
							break

						try:
							sample = sample_queue.get(timeout=0.05)
						except queue.Empty:
							continue

						timestamp = sample.get('timestamp')
						if timestamp is None:
							timestamp = time.time()

						eeg_values = sample.get('eeg')
						if eeg_values is None:
							continue

						row = [float(timestamp)]
						eeg_array = np.asarray(eeg_values).flatten()
						row.extend(float(val) for val in eeg_array[:8])

						label = sample.get('label', '')
						seq = sample.get('seq', '')
						row.append(label if label is not None else '')
						row.append(seq if seq is not None else '')

						writer.writerow(row)
						self._last_data_frame = row #Hold it locally to be read
						rows_written += 1
						if rows_written % 50 == 0:
							csvfile.flush()
					csvfile.flush()
			except Exception as e:
				print(f'[BCIBoard] Recording error: {e}')
				should_delete = True
			finally:
				self.recording = False
				self._record_thread = None
				self._record_stop_event = None
				min_rows = getattr(self, "minimum_recorded_rows", None)
				if not should_delete and isinstance(min_rows, int) and min_rows > 0:
					if rows_written < min_rows:
						print(f"[BCIBoard] Recording discarded: only {rows_written} rows (minimum {min_rows}).")
						should_delete = True
				if self._record_file_path and should_delete:
					try:
						Path(self._record_file_path).unlink()
						print(f"[BCIBoard] Discarded incomplete recording: {self._record_file_path}")
					except Exception:
						pass
				self._record_file_path = None
				self._sample_queue = queue.Queue(maxsize=1000)

		self.recording = True
		thread = threading.Thread(target=_record_worker, daemon=True)
		self._record_thread = thread
		thread.start()
		print(f"[BCIBoard] Recording started: {file_path}")
		return str(file_path)

	def stop_recording(self, wait=True):
		"""Signal the recording worker to stop and close resources."""
		stop_event = self._record_stop_event
		if stop_event:
			stop_event.set()

		self.recording = False

		thread = self._record_thread
		if thread and thread.is_alive():
			if wait:
				thread.join(timeout=2.0)

		self._record_thread = None
		self._record_stop_event = None
		self._record_file_path = None
		self._sample_queue = queue.Queue(maxsize=1000)
		print('[BCIBoard] Recording stopped')

	def check_impedance(self, channels=None):
		channels = channels or self.channels
		results = []

		if not self.board:
			print("Board not connected. Cannot check impedance.")
			return results

		print("Starting impedance check...")

		for ch in channels:
			color = CABLE_COLORS[ch - 1] if 1 <= ch <= len(CABLE_COLORS) else "black"
			print(f"Measuring CH{ch} ({color})...")

			try:
				self.board.stop_stream()
				reset_to_defaults(self.board)
				self._ch_cfg, self._ch_last_cfg = change_leadoff(self.board, ch, True, self._ch_cfg, self._ch_last_cfg, self.STREAM_CFG_DEFAULT, self.IMP_CFG)  # enable lead-off
				self.board.start_stream()
				self.board.get_board_data()  # clear buffer
				time.sleep(MEAS_SEC + 0.2)
				data = self.board.get_board_data()

				row = BoardShim.get_eeg_channels(self.board_id)[ch - 1]
				x_uV = data[row, :]
				x_bp = bandpass_apply(x_uV, self.fs)
				uVrms = take_recent_1s(x_bp, self.fs)
				z_kohm = calc_impedance_from_vrms(uVrms) / 1000.0
				results.append((ch, z_kohm))
				print(f"CH{ch} ({color}): {z_kohm:.2f} kΩ")
				self.board.stop_stream()
				self._ch_cfg, self._ch_last_cfg = change_leadoff(self.board, ch, False, self._ch_cfg, self._ch_last_cfg, self.STREAM_CFG_DEFAULT, self.IMP_CFG)  # disable lead-off
				self.board.start_stream()

			except Exception as e:
				results.append((ch, float('nan')))
				print(f"Failed to measure CH{ch}: {e}")

		print("Impedance check complete.")
		return results

class CloudConnection:
	def __init__(self, host="ftp.yourdomain.com", user="yourusername", password="yourpassword", timeout=10):
		self.host = host
		self.user = user
		self.password = password
		self.timeout = timeout
		self.ftp = None
		
	def connect(self):
		try:
			self.ftp = FTP(self.host, timeout=self.timeout)
			self.ftp.login(self.user, self.password)
		except Exception as e:
			raise Exception(f"Failed to connect to cloud: {e}")
		return True
	
	def folder_exists(self, user_id):
		if self.ftp is None:
			self.connect()
		folder_path = ("/home/File-EXTERNAL/B2J" + user_id)
		try:
			self.ftp.cwd(folder_path)
			return True
		except error_perm:
			return False
		except Exception as e:
			print(f"Failed to check folder: {e}")
			return False
	
	def create_user_folder(self, user_id):
		if self.ftp is None:
			self.connect()
		try:
			self.ftp.mkd("/home/File-EXTERNAL/B2J" + user_id)
		except error_perm as e:
			raise Exception(f"Failed to create user folder: {e}")
		except Exception as e:
			raise Exception(f"Failed to create user folder: {e}")
		return True
	
	def count_files_in_folder(self, user_id):
		try:
			self.ftp.cwd(f"/home/File-EXTERNAL/B2J{user_id}/")
			return len(self.ftp.nlst())
		except Exception as e:
			raise Exception(f"Failed to count files in user folder: {e}")
		return 0
	
	def upload_all_files(self, remote_root, user_id, local_root="BMI Trainer Data"):
		if self.ftp is None:
			self.connect()

		base_path = Path(local_root)
		if not base_path.exists():
			raise FileNotFoundError(f"Local directory '{local_root}' does not exist.")

		session_dirs = [
			p for p in base_path.iterdir()
			if p.is_dir() and p.name.startswith(user_id)
		]
		
		if not session_dirs:
			print(f"No session folders found for user {user_id} in {local_root}")
			return 0
		
		remote_base = f"/home/File-EXTERNAL/B2J{remote_root}"
		try:
			self.ftp.cwd(remote_base)
		except error_perm:
			print(f"User folder {remote_base} does not exist, creating...")
			self.create_user_folder(user_id)
			self.ftp.cwd(remote_base)
		except Exception as e:
			print(f"Failed to change directory to {remote_base}: {e}")
			return 0

		try:
			existing_remote_files = set(self.ftp.nlst())
		except Exception:
			existing_remote_files = set()

		files_uploaded = 0
		for session_dir in session_dirs:
			for file_path in session_dir.rglob("*"):
				if not file_path.is_file():
					continue
				flattened = "__".join([session_dir.name] + list(file_path.relative_to(session_dir).parts))
				if flattened in existing_remote_files:
					print(f"Skipping existing file on cloud: {flattened}")
					continue
				try:
					with open(file_path, "rb") as fh:
						self.ftp.storbinary(f"STOR {flattened}", fh)
					files_uploaded += 1
					existing_remote_files.add(flattened)
				except Exception as err:
					print(f"Failed to upload {file_path}: {err}")
		return files_uploaded

	def download_all_files(self, remote_root, local_root):
		if self.ftp is None:
			self.connect()

		remote_base = f"/home/File-EXTERNAL/B2J{remote_root}"
		try:
			self.ftp.cwd(remote_base)
		except Exception as e:
			raise Exception(f"Failed to access remote folder '{remote_root}': {e}")

		local_path = Path(local_root)
		local_path.mkdir(parents=True, exist_ok=True)

		files_downloaded = 0
		try:
			for filename in self.ftp.nlst():
				local_file = local_path / filename
				with open(local_file, "wb") as f:
					self.ftp.retrbinary(f"RETR {filename}", f.write)
				files_downloaded += 1
		except Exception as e:
			raise Exception(f"Failed to download files from '{remote_root}': {e}")

		return files_downloaded

def build_channel_settings_cmd(ch: int, gain: int, input_type: int, bias: int, srb2: int, srb1: int, power_down: int = 0) -> str:
	"""Build Cyton channel settings command: x(CH, POWER_DOWN, GAIN, INPUT, BIAS, SRB2, SRB1)X"""
	return f"x{ch}{power_down}{gain}{input_type}{bias}{srb2}{srb1}X"

def build_impedance_cmd(ch: int, active: bool, is_n: bool) -> str:
	"""Build Cyton impedance/lead-off command: z(CH, PCHAN, NCHAN)Z"""
	p = "0"
	n = "0"
	if active:
		if is_n:
			n = "1"
		else:
			p = "1"
	return f"z{ch}{p}{n}Z"

def reset_to_defaults(board: BoardShim):
	"""
	Reset all channels to Cyton default configuration.
	"""
	try:
		board.config_board("d")
		time.sleep(0.1)
		return True, "OK"
	except Exception as e:
		return False, f"ERR: {e}"

def calc_impedance_from_vrms(vrms_uV):
	"""
	Calculate impedance from Vrms (in microvolts).
	Z = (sqrt(2) * Vrms) / I_drive - R_series
	"""
	Vrms_V = float(vrms_uV) * 1e-6
	Z = (np.sqrt(2.0) * Vrms_V) / I_DRIVE
	Z -= SERIES_R
	return max(Z, 0.0)

def take_recent_1s(x_uV, fs):
	"""
	Take the most recent 1 second of data and compute RMS.
	"""
	n = int(fs * 1.0)
	seg = x_uV[-n:]
	return float(np.sqrt(np.mean(seg ** 2)))

def bandpass_apply(x, fs):
	"""
	Apply a 4th-order Butterworth bandpass filter.
	"""
	b, a = iirfilter(4, [BAND[0]/(fs/2.0), BAND[1]/(fs/2.0)], btype='band', ftype='butter')
	return filtfilt(b, a, x)

def change_leadoff(board, ch, is_on, _ch_cfg=None, _ch_last_cfg=None, STREAM_CFG_DEFAULT=None, IMP_CFG=None):
	ch_idx = ch - 1  # zero-based index
	is_n = True

	if is_on:
		_ch_last_cfg[ch_idx] = _ch_cfg[ch_idx].copy()
		_ch_cfg[ch_idx] = IMP_CFG.copy()
	else:
		if _ch_last_cfg[ch_idx] is not None:
			_ch_cfg[ch_idx] = _ch_last_cfg[ch_idx].copy()
			_ch_last_cfg[ch_idx] = None
		else:
			_ch_cfg[ch_idx] = STREAM_CFG_DEFAULT.copy()

	cfg = _ch_cfg[ch_idx]
	x_cmd = build_channel_settings_cmd(
		ch=ch,
		gain=cfg["gain"],
		input_type=cfg["input_type"],
		bias=cfg["bias"],
		srb2=cfg["srb2"],
		srb1=cfg["srb1"],
		power_down=0
	)
	z_cmd = build_impedance_cmd(ch=ch, active=is_on, is_n=is_n)
	cmd = x_cmd + z_cmd

	try:
		resp = board.config_board(cmd)
		print(f"Ch{ch} Cmd: {cmd} | Resp: {resp}")
	except UnicodeDecodeError:
		print(f"Ch{ch} Cmd: {cmd} | (Success but response decode error)")

	time.sleep(0.1)

	return _ch_cfg, _ch_last_cfg

def set_latency_timer(value=1, retries=3, delay=0.2):
    # /sys/bus/usb-serial/devices/ttyUSB*/latency_timer を探す
    paths = glob.glob("/sys/bus/usb-serial/devices/ttyUSB*/latency_timer")
    if not paths:
        print("⚠️ USB-serial device not found yet.")
        return False

    all_ok = True
    for path in paths:
        print(f"Setting latency_timer at {path} to {value}")
        try:
            # echo value > /sys/... の Python 版（sudo を使って書き込み）
            subprocess.run(["sudo", "tee", path], input=(str(value) + "\n").encode(), check=True, capture_output=True)
        except Exception as e:
            print(f"❌ Error writing: {e}")
            all_ok = False
            continue

        # 設定が反映されたか確認（リトライあり）
        ok = False
        for attempt in range(1, retries + 1):
            try:
                with open(path, "r") as f:
                    current = f.read().strip()
                if current == str(value):
                    ok = True
                    break
                else:
                    print(f"⚠️ Verification attempt {attempt}: got '{current}' (expected '{value}')")
            except Exception as e:
                print(f"❌ Error reading (attempt {attempt}): {e}")
            time.sleep(delay)

        if ok:
            print(f"✅ Verified latency_timer at {path} == {value}")
        else:
            print(f"❌ Failed to verify latency_timer at {path} (expected {value})")
            all_ok = False

    return all_ok

# ---- Mark implemented functions ---- #

def play_single_sound(filepath="beep_left.wav", block=False):
	"""
	Play a .wav file. By default this is asynchronous; pass block=True to wait until completion.
	"""
	if not os.path.exists(filepath):
		print(f"[Sound] File not found: {filepath}")
		return

	def _play_audio():
		sound = pygame.mixer.Sound(filepath)
		channel = sound.play()
		if channel is not None:
			while channel.get_busy():
				time.sleep(0.01)
		else:
			time.sleep(sound.get_length())

	try:
		if block:
			_play_audio()
		else:
			threading.Thread(target=_play_audio, daemon=True).start()
	except Exception as e:
		print(f"[Sound] Failed to play sound: {e}")

def createSessionFolder(user_id, timestamp, base_path="BMI Trainer Data/"):
	'''Create a session folder named '<user_id>-YYYY-MM-DD-HH-MM' inside base_path.'''
	if not isinstance(user_id, str):
		user_id = str(user_id)
	if len(user_id) != 9 or not user_id.isdigit():
		raise ValueError('user_id must be a 9-digit string')

	if hasattr(timestamp, 'strftime'):
		timestamp_str = timestamp.strftime('%Y-%m-%d-%H-%M')
	else:
		timestamp_str = str(timestamp)

	folder_name = f"{user_id}-{timestamp_str}"
	base_dir = os.path.abspath(base_path)
	os.makedirs(base_dir, exist_ok=True)
	full_path = os.path.join(base_dir, folder_name)
	if not os.path.isdir(full_path):
		os.makedirs(full_path)
	return full_path

def createModelFolder(base_path="Model/"):
	base_dir = os.path.abspath(base_path)
	if not os.path.isdir(base_dir):
		os.makedirs(base_dir, exist_ok=True)
	return base_dir

def createTestingFolder(base_path="Testing/"):
	base_dir = os.path.abspath(base_path)
	if not os.path.isdir(base_dir):
		os.makedirs(base_dir, exist_ok=True)
	return base_dir

def deleteEmptyFolders(base_path="BMI Trainer Data/"):
	"""Remove subdirectories under base_path that contain no files."""
	base_dir = Path(base_path).expanduser().resolve()
	if not base_dir.is_dir():
		return

	for entry in base_dir.iterdir():
		if not entry.is_dir():
			continue
		try:
			contains_files = any(child.is_file() for child in entry.rglob('*'))
			if not contains_files:
				shutil.rmtree(entry)
		except Exception:
			pass

def startSingleTrainingSequence(board, user_id, timestamp, lcr_value, base_path):
	"""
	Start a single training sequence on a background thread.

	Returns a tuple of (csv_path, worker_thread, cancel_event).
	"""
	if board is None:
		raise ValueError('board is required')

	if not getattr(board, 'connected', False):
		raise RuntimeError('BCIBoard is not connected')

	if not getattr(board, 'streaming', False):
		raise RuntimeError('BCIBoard is not streaming')

	if not isinstance(user_id, str):
		user_id = str(user_id)
	if len(user_id) != 9 or not user_id.isdigit():
		raise ValueError('user_id must be a 9-digit string')

	if lcr_value not in (0, 1, 2, 3, 4):
		raise ValueError('lcr_value must be 0 (testing), 1 (left), 2 (center), 3 (right), 4 (live)')

	direction_map = {0: "testing", 1: "left", 2: "center", 3: "right", 4: "live"}
	direction = direction_map[lcr_value]

	base_dir = Path(base_path).expanduser().resolve()
	base_dir.mkdir(parents=True, exist_ok=True)

	if hasattr(timestamp, 'strftime'):
		timestamp_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
	else:
		timestamp_str = str(timestamp)

	filename = f"{user_id}-{timestamp_str}-{lcr_value}.csv"

	instruction_path = f"Sounds/instruction_{direction}.wav"
	beep_path = f"Sounds/beep_{direction}.wav"
	instruction_starting = "Sounds/instruction_starting.wav"
 
	stimulus_sound, sequence_id = generateSequence()
	cancel_event = threading.Event()

	def _sequence_worker():
		try:
			
			if cancel_event.is_set():
				return

			play_single_sound(instruction_path, block=True) #First Instruction
			time.sleep(ISI)

			if cancel_event.is_set():
				return

			if lcr_value == 0 or lcr_value == 4: #Testing Mode< one of each beep.
				play_single_sound("Sounds/beep_left.wav", block=True)
				time.sleep(.5)
				play_single_sound("Sounds/beep_center.wav", block=True)
				time.sleep(.5)
				play_single_sound("Sounds/beep_right.wav", block=True)
				time.sleep(.5)
				
			else:
				for _ in range(3): #3 Beeps
					play_single_sound(beep_path, block=True)
					time.sleep(ISI)

			if cancel_event.is_set():
				return
   
			if lcr_value == 4:
					instruction_starting = "Sounds/begin.mp3"
			else:
				instruction_starting = "Sounds/instruction_starting.wav"
			
			play_single_sound(instruction_starting, block=True) #Say Starting

			if cancel_event.is_set():
				return
   
			#Start Recording
			board.stimulus_sound = 0
			board.sequence_id = 0
			board.start_recording(base_path, filename=filename)
			time.sleep(2)
   
			start_time = time.time() + 0.1  # 少し余裕を持って開始
			for stim_idx, (sound, id) in enumerate(zip(stimulus_sound, sequence_id)):
				
				#Wait until the scheduled time only to play sound and update
				scheduled_time = start_time + stim_idx * (ISI + SOUND_LENGTH)
				while time.time() < scheduled_time:
					time.sleep(0.0005)
			  
				if cancel_event.is_set():
					break
 
				board.stimulus_sound = sound
				board.sequence_id = id
	
				if sound == 1:
					play_single_sound("Sounds/beep_left.wav", block=True)
				elif sound == 2:
					play_single_sound("Sounds/beep_center.wav", block=True)
				elif sound == 3:
					play_single_sound("Sounds/beep_right.wav", block=True)
				elif sound == 4:
					play_single_sound("Sounds/beep_silent.wav", block=True)
  
			if cancel_event.is_set():
				board.stop_recording()
				return
	
			board.stimulus_sound = 0
			board.sequence_id = 0
			time.sleep(2)

			board.stop_recording()
			
		except Exception as exc:
			print(f"[BCIBoard] Training sequence error: {exc}")
			raise SingleTrainingSequenceError(str(exc)) from exc
		finally:
			board.stimulus_sound = 0
			board.sequence_id = 0

	sequence_thread = threading.Thread(target=_sequence_worker, daemon=True)
	sequence_thread.start()

	return sequence_thread, cancel_event

def getUserID(filepath="BMI Trainer Data/UserID.txt"):
	"""
	Check if the given file exists. If not, create it and save a 9-digit UserID.
	Returns the UserID (as a string).
	"""
	if not os.path.exists(filepath):
		user_id = str(random.randint(100000000, 999999999))
		with open(filepath, "w") as f:
			f.write(user_id)
		print(f"[UserID] Created new file '{filepath}' with ID {user_id}")
	else:
		with open(filepath, "r") as f:
			user_id = f.read().strip()
		if not user_id.isdigit() or len(user_id) != 9:
			user_id = str(random.randint(100000000, 999999999))
			with open(filepath, "w") as f:
				f.write(user_id)
			print(f"[UserID] Invalid file content, generated new ID {user_id}")
		else:
			print(f"[UserID] Found existing ID {user_id}")
	
	return user_id

def generateSequence():
	"""
	Generates non-repeating order of sounds to play (Left,Center,Right,Silent), and corresponding sequence id
	"""
	
	stimulation_sequence = []
	sequence_ids = []
	prev_stim = None
	
	#Modifiable
	numSequences=10
	playableIDs = [1,2,3,4]
	
	for set_idx in range(numSequences):
		stim_order = random.sample(playableIDs, len(playableIDs))
		while prev_stim is not None and stim_order[0] == prev_stim:
			stim_order = random.sample(playableIDs, len(playableIDs))
		prev_stim = stim_order[-1]
		stimulation_sequence.extend(stim_order)
		sequence_ids.extend([set_idx + 1] * len(stim_order))
		
	return stimulation_sequence, sequence_ids

def countRecordings(user_id, base_folder="BMI Trainer Data/"):
	"""Count left/center/right recordings for a user across session folders."""
	if not isinstance(user_id, str):
		user_id = str(user_id)
	if len(user_id) != 9 or not user_id.isdigit():
		raise ValueError("user_id must be a 9-digit string")

	base_dir = Path(base_folder).expanduser().resolve()
	if not base_dir.exists():
		return 0, 0, 0

	left = center = right = 0

	for session_dir in base_dir.iterdir():
		if not session_dir.is_dir():
			continue
		for csv_path in session_dir.glob(f"{user_id}-*.csv"):
			name = csv_path.name.rstrip().lower()
			if name.endswith('1.csv'):
				left += 1
			elif name.endswith('2.csv'):
				center += 1
			elif name.endswith('3.csv'):
				right += 1

	return left, center, right

def chooseNewLCRValue(counts, acceptable_span=5, max_per_class=333):
	"""
	Choose the next L/C/R value (1,2,3) to keep counts balanced.

	counts must be an iterable of three non-negative integers representing (left, center, right).
	If the spread between the minimum and maximum counts exceeds acceptable_span, the lowest count is chosen.
	Otherwise a random choice is made among classes that are still below max_per_class.
	"""
	try:
		left, center, right = counts
	except Exception as exc:
		raise ValueError("counts must be an iterable with three entries (left, center, right)") from exc

	for value in (left, center, right):
		if not isinstance(value, int) or value < 0:
			raise ValueError("All count values must be non-negative integers")

	class_counts = [left, center, right]

	eligible_indices = [idx for idx, count in enumerate(class_counts) if count < max_per_class]
	if not eligible_indices:
		raise ValueError("All class counts have reached the maximum allowed recordings")

	min_count = min(class_counts[idx] for idx in eligible_indices)
	max_count = max(class_counts[idx] for idx in eligible_indices)

	if max_count - min_count > acceptable_span:
		target_indices = [idx for idx in eligible_indices if class_counts[idx] == min_count]
	else:
		target_indices = eligible_indices

	chosen_index = random.choice(target_indices)
	return chosen_index + 1  # map 0->1 (left), 1->2 (center), 2->3 (right)

def deleteMostRecent(base_path="BMI Trainer Data/"):
	"""Delete the most recent recording (by timestamp) across all session folders."""
	base_dir = Path(base_path).expanduser().resolve()
	if not base_dir.exists():
		print(f"[deleteMostRecent] Base path not found: {base_dir}")
		return False

	most_recent_path = None
	most_recent_time = None

	for session_dir in base_dir.iterdir():
		if not session_dir.is_dir():
			continue
		for csv_path in session_dir.glob('*.csv'):
			name = csv_path.name
			parts = name.split('-')
			if len(parts) < 8:
				continue
			try:
				timestamp_str = '-'.join(parts[1:7])
				tstamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S')
			except Exception:
				continue

			if most_recent_time is None or tstamp > most_recent_time:
				most_recent_time = tstamp
				most_recent_path = csv_path

	if most_recent_path is None:
		print('[deleteMostRecent] No recordings found.')
		return False

	try:
		most_recent_path.unlink()
		print(f"[deleteMostRecent] Deleted most recent recording: {most_recent_path}")
		return True
	except Exception as exc:
		print(f"[deleteMostRecent] Failed to delete {most_recent_path}: {exc}")
		return False

def deleteTestingFiles(base_path="Testing/"):
	"""Delete all files under the testing directory."""
	base_dir = Path(base_path).expanduser().resolve()
	if not base_dir.exists():
		print(f"[deleteTestingFiles] Testing path not found: {base_dir}")
		return False

	deleted = False
	for item in base_dir.glob("*"):
		try:
			if item.is_file():
				item.unlink()
				deleted = True
			elif item.is_dir():
				shutil.rmtree(item)
				deleted = True
		except Exception as exc:
			print(f"[deleteTestingFiles] Failed to delete {item}: {exc}")
	return deleted

def labelTestingFile(test_file, session_folder, lcr_value):
	"""
	Rename a testing file by replacing the trailing class label with the provided lcr_value
	and move it into the given session folder.
	"""
	if lcr_value not in (1, 2, 3):
		raise ValueError("lcr_value must be 1 (Left), 2 (Center), or 3 (Right)")

	test_path = Path(test_file).expanduser().resolve()
	if not test_path.exists() or not test_path.is_file():
		raise FileNotFoundError(f"Testing file not found: {test_path}")

	session_dir = Path(session_folder).expanduser().resolve()
	session_dir.mkdir(parents=True, exist_ok=True)

	name_parts = test_path.stem.split('-')
	if not name_parts:
		raise ValueError(f"Unexpected testing filename format: {test_path.name}")

	name_parts[-1] = str(lcr_value)
	new_name = '-'.join(name_parts) + test_path.suffix
	target_path = session_dir / new_name

	shutil.move(str(test_path), str(target_path))
	return target_path

#Ogino Model Functions

def useModelToPredict(test_file_path, model_folder="Model/"):
	"""
	Use the model to predict the label of the given file.
	"""
 
	model_path = os.path.join(model_folder, "model.pkl")
 
	# モデルとスケーラーをロード
	svm = joblib.load(model_path)
	
	# 特徴量を抽出
	features = process_file(test_file_path)
		
	# 予測を実行
	predictions = svm.predict(features)
	return int(predictions[0])

def process_file(file_path, use_mean_features=True):
	# データ処理と特徴量抽出を行う関数
	original_sampling_rate = 250
	numtaps = 11
	
	# CSVファイルを読み込む
	data = pd.read_csv(file_path).values
	
	# 刺激ラベルの列を取得
	stimulus_labels = data[:, 9]
	
	# NaNをゼロに置き換え（または適切な値に置き換え）
	stimulus_labels = np.nan_to_num(stimulus_labels, nan=0)
	
	# 刺激オンセットのインデックスを取得
	stimulus_onsets = np.where(np.diff(stimulus_labels) != 0)[0] + 1
	stimulus_onsets = stimulus_onsets[stimulus_labels[stimulus_onsets] != 0]
	
	# 刺激ラベルの0を削除し、連続するものを1つにする
	processed_stimulus_labels = stimulus_labels[stimulus_labels != 0]
	processed_stimulus_labels = np.concatenate(([processed_stimulus_labels[0]], processed_stimulus_labels[1:][processed_stimulus_labels[1:] != processed_stimulus_labels[:-1]]))
	
	# 脳波データ（ch5列目のみ）を取得
	# バンドパスフィルタの設計
	nyquist_rate = original_sampling_rate / 2
	low_cutoff = 0.5/ nyquist_rate
	high_cutoff = 30 / nyquist_rate
	b = firwin(numtaps, [low_cutoff, high_cutoff], pass_zero=False)
	
	# EEGデータを取得（多極対応）
	eeg_data = data[:, [i for i, col in enumerate(pd.read_csv(file_path, nrows=0).columns) if "Ch" in col]]*-1
				
	eeg_data = StandardScaler().fit_transform(eeg_data)

	# フィルタリング方法を選択
	use_filter = "filtfilt"  # Options: "filtfilt", "lfilter", or None

	if use_filter == "filtfilt":
		# フィルタを適用 (filtfiltで遅延をなくす)
		eeg_data = filtfilt(b, 1.0, eeg_data, axis=0)
	elif use_filter == "lfilter":
		# フィルタを適用 (lfilter)
		eeg_data = lfilter(b, 1.0, eeg_data, axis=0)
	elif use_filter is None:
		# フィルタを適用しない
		pass
		
	# エポック毎のERPを計算
	erp_epochs, pure_erp_epochs = compute_erp(eeg_data, stimulus_onsets)
	
	features = []
	if use_mean_features:
		# 刺激ラベルが1と3のエポックを分ける
		erp_label_1 = erp_epochs[stimulus_labels[stimulus_onsets] == 1]
		erp_label_2 = erp_epochs[stimulus_labels[stimulus_onsets] == 2]
		erp_label_3 = erp_epochs[stimulus_labels[stimulus_onsets] == 3]
		
		# 平均を計算
		mean_erp_label_1 = np.mean(erp_label_1, axis=0)
		mean_erp_label_2 = np.mean(erp_label_2, axis=0)
		mean_erp_label_3 = np.mean(erp_label_3, axis=0)
		
		# 差を計算して特徴量に追加
		if(0):
			diff_erp = (mean_erp_label_1 - mean_erp_label_3).flatten()
		
		# 各音に対する特徴量を統合
		if(1):
			diff_erp = np.concatenate((mean_erp_label_1.flatten(), mean_erp_label_2.flatten(), mean_erp_label_3.flatten()))

		# 刺激ラベルを特徴量に追加
		if(0):
			diff_erp = np.concatenate((diff_erp, processed_stimulus_labels[:len(diff_erp)]))

		features.append(diff_erp)

	else:
		# 刺激ラベルが1と3のエポックを分ける
		erp_label_1 = erp_epochs[stimulus_labels[stimulus_onsets] == 1]
		erp_label_3 = erp_epochs[stimulus_labels[stimulus_onsets] == 3]
		
		# 各エポックの電極平均を計算して特徴量に追加
		for epoch_1, epoch_3 in zip(erp_label_1, erp_label_3):
			mean_epoch_1 = np.mean(epoch_1, axis=1)  # 電極平均
			mean_epoch_3 = np.mean(epoch_3, axis=1)  # 電極平均
			concatenated_erp = np.concatenate((
				mean_epoch_1.flatten(), 
				mean_epoch_3.flatten()
			))
			features.append(concatenated_erp)
	
	return np.vstack(features)

def compute_erp(data, stimulus_onsets):
	window_size = 250
	downsampling_rate = 25
	diff_downsampling_rate = 25
	fir_delay = 0
	
	downsampling_factor = window_size // downsampling_rate
	diff_downsampling_factor = window_size // diff_downsampling_rate
	erp = []
	features = []
	for onset in stimulus_onsets:
		if onset + window_size <= data.shape[0]:
			# ベースライン区間を計算
			baseline_start = max(0, onset - 5 - fir_delay)
			baseline_end = onset - fir_delay
			baseline = data[baseline_start:baseline_end].mean(axis=0)
			segment = data[onset - fir_delay:onset + window_size - fir_delay] - baseline
			# # 区間を平均してダウンサンプリング
			downsampled_segment = segment.reshape(-1, downsampling_factor, segment.shape[1]).mean(axis=1)
			
			# diff_downsampled_segment = segment.reshape(-1, diff_downsampling_factor, segment.shape[1]).mean(axis=1)
			
			if False:  # 変化率特徴量を計算する場合はTrueに変更
				# 変化率特徴量を計算
				rate_of_change = np.diff(diff_downsampled_segment, axis=0)

				# 刺激後1秒間のデータを取得
				combined_features = np.concatenate((downsampled_segment, rate_of_change), axis=0)
				features.append(combined_features)
			else:
				features.append(downsampled_segment)
	
	for onset in stimulus_onsets:
		if onset + window_size <= data.shape[0]:
			# ベースライン区間を計算
			baseline_start = max(0, onset - 5 - fir_delay)
			baseline_end = onset - fir_delay
			baseline = data[baseline_start:baseline_end].mean(axis=0)
			segment = data[onset - fir_delay:onset + window_size - fir_delay] - baseline
			# # 区間を平均してダウンサンプリング
			downsampled_segment = segment.reshape(-1, downsampling_factor, segment.shape[1]).mean(axis=1)
			
			# diff_downsampled_segment = segment.reshape(-1, diff_downsampling_factor, segment.shape[1]).mean(axis=1)
			
			if False:  # 変化率特徴量を計算する場合はTrueに変更
				# 変化率特徴量を計算
				rate_of_change = np.diff(diff_downsampled_segment, axis=0)

				# 刺激後1秒間のデータを取得
				combined_features = np.concatenate((downsampled_segment, rate_of_change), axis=0)
				erp.append(combined_features)
			else:
				erp.append(downsampled_segment)
				
	return np.array(features), np.array(erp)
