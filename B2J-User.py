import ABMI_Utils
import ALS_Utils
import argparse
import glob
import serial
import pygame
import os
import numpy as np
import queue
from queue import Empty
from datetime import datetime
from serial.tools import list_ports
from multiprocessing import Process, Queue
import time
import random

from DroneMonitorClient import DroneMonitorClient

#---------------------------------------------
# ---- OGINO IMPLEMENTTED----
#---------------------------------------------

#念の為以下のコマンドで実行を推奨
# sudo nice -n -10 python main.py

# ANSIカラー（ターミナル前提）
RED   = "\033[91m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RESET = "\033[0m"

ESP_VID = 0x1a86 
esp_ser = None #ESP Now Serial Port

def pick_m5_port():
	ports = list(list_ports.comports())
	for p in ports:
		text = f"{p.description} {p.manufacturer} {p.product}".lower()

		if "ftdi" in text:
			continue

		return p.device

m5_port = pick_m5_port()
baud = 115200
MAX_ESP_PAYLOAD_BYTES = 240

# Font Path
notoFont = "/home/b2j/Desktop/AugmentedArms/Font/NotoSansJP-Bold.otf"

# Colors
white = (217,217,217)
blue = (23,100,255)
dark_blue = (21,19,186)
cool_blue = (74,198,255)
light_blue = (54,106,217)
warning_orange = (214,162,17)
soft_red = (250,61,55)
light_red = (255,143,133)
red = (200,0,0)
black = (0,0,0)
light_green = (82,255,128)
light_yellow = (220,200,80)
green = (0,200,0)
soft_green = (152,250,132)
mint_green = (54,217,62)
dark_green = (2,140,0)
dark_yellow = (201,185,0)

#Pygame vars
render_w, render_h = 480, 320
screen = None
clock = None
title_font = None
state_font = None

board = ABMI_Utils.BCIBoard(port="/dev/ttyUSB0")
sequence_thread = None
cancel_event = None

testing_path = "Testing/"
model_path = "Model/"
latest_test_file = ""

sound_map = {
	1: "Sounds/prediction_left.wav",
	2: "Sounds/prediction_center.wav",
	3: "Sounds/prediction_right.wav"
}

prediction_choice = None

userID = ABMI_Utils.getUserID()

piezo = ALS_Utils.PiezoSensor(pin = 21, cooldown=1)
state = "idle"
drone_monitor_client = None
title_surface = None
state_surface = None
last_text_draw = 0
text_draw_interval_ms = 200

def readRecentLatestTestFile(file_path):
	if not file_path or not os.path.exists(file_path):
		return None

	try:
		with open(file_path, "r") as csv_file:
			rows = [line.strip() for line in csv_file if line.strip()]
	except OSError:
		return None

	if len(rows) <= 1:
		return None

	return rows[-1]

def sendToM5(port: str, baud: int, message: str):
	
	if not message:
		return

	message = message + '\n'

	try:
		with serial.Serial(port, baud, timeout=1) as ser:
			ser.write(message.encode('utf-8'))
			ser.flush()
	except serial.SerialException as e:
		print(f"Serial error: {e}")

def _send_single_command(ch, val):
	command = f"Ch{ch}-{val}\n"
	sendToM5(m5_port, baud, f"S,{command}/n")

def send_led_command(ch, val): 
    """LEDコントローラーへコマンドを送信"""
    global esp_ser
    
    # チャンネル9が選択されている場合は、1〜8全てに同じコマンドを送る
    if ch == 9:
        print(f"Sending to ALL Channels -> Pattern: {val}{RESET}")
        for i in range(1, 9):
            _send_single_command(i, val)
            time.sleep(0.01) 
    else:
        print(f"Sending to Ch: {ch} -> Pattern: {val}{RESET}")
        _send_single_command(ch, val)

def send_led_flicker():
    channels = list(range(1, 9))  # 1〜9
    random.shuffle(channels)

    for rnd_ch in channels:
        send_led_command(rnd_ch, 4)
        time.sleep(0.1)

def send_led_left():
    for ch in [1, 4, 7]:
        send_led_command(ch, 3)
    time.sleep(1)
    for ch in [1, 4, 7]:
        send_led_command(ch, 1)

def send_led_center():
    for ch in [2, 5]:
        send_led_command(ch, 3)
    time.sleep(1)
    for ch in [2, 5]:
        send_led_command(ch, 1)

def send_led_right():
    for ch in [3, 6, 8]:
        send_led_command(ch, 3)
    time.sleep(1)
    for ch in [3, 6, 8]:
        send_led_command(ch, 1)

def send_led_all_off():
    send_led_command(9, 0)  # 全てのチャンネルにオフコマンドを送る

def handleIdle():

	global board, state, latest_test_file, sequence_thread, cancel_event

	#Send ESP Data
	#recent = str(board.board.get_board_data())
	#sendToM5(m5_port, baud, recent)

	#Check for Piezo Press
	if piezo.was_pressed():
		state = "recording"
		ABMI_Utils.play_single_sound("Sounds/Click.mp3", block=True)

		timestamp = datetime.now()
		lcr_choice = 4
		if hasattr(timestamp, 'strftime'):
			timestamp_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
		else:
			timestamp_str = str(timestamp)
		latest_test_file = os.path.join("Testing", f"{userID}-{timestamp_str}-{lcr_choice}.csv")

		sequence_thread, cancel_event = ABMI_Utils.startSingleTrainingSequence(board, userID, timestamp, lcr_choice, testing_path)

		send_led_all_off()

	return

def handleRecording():

	global state, sequence_thread, latest_test_file

	#msg = readRecentLatestTestFile(latest_test_file)
	#if msg is None:
	#	return
	recent = str(board._last_data_frame)
	sendToM5(m5_port, baud, f"E,{recent}/n")
	#sendToM5(m5_port, baud, msg)
	#print(msg)

	if not sequence_thread.is_alive():
		state = "predicting"

	return

def handlePredicting():

	global state, prediction_choice, sound_map, latest_test_file, model_path

	try:
		prediction_choice = ABMI_Utils.useModelToPredict(latest_test_file, model_path)
		if prediction_choice==1:
			send_led_left()
		elif prediction_choice==2:
			send_led_center()
		elif prediction_choice==3:
			send_led_right()

		if prediction_choice == -1:
			ABMI_Utils.play_single_sound("Sounds/prediction_unknown.wav")
			state = "idle"
		else:
			ABMI_Utils.play_single_sound(sound_map.get(prediction_choice, "Sounds/prediction_unknown.wav"))
			state = "triggering"
	except Exception as err:
		print(f"Model prediction failed: {err}")
		ABMI_Utils.play_single_sound("Sounds/prediction_unknown.wav")
		state = "idle"

	return

def handleTriggering():

	global state, sound_map, testing_path, drone_monitor_client

	#Ambient activity is basically predicting:
	#send a single message that includes the trigger keyword and a direction

	#Check for Piezo Press, this single event is triggering
	if piezo.was_pressed():
		ABMI_Utils.play_single_sound(sound_map.get(prediction_choice, "Sounds/prediction_unknown.wav"))
		#send Trigger
		sendToM5(m5_port, baud, f"T,{prediction_choice}/n")
		# Forward the confirmed choice to drone_monitor (raw "1"/"2"/"3" over WebSocket).
		# Send is best-effort and non-blocking; failures are logged by the client.
		if drone_monitor_client is not None and prediction_choice in (1, 2, 3):
			drone_monitor_client.send_signal(str(prediction_choice))
		ABMI_Utils.deleteTestingFiles(testing_path)
		state = "idle"
		send_led_all_off()

	return

def draw_status_text(force=False):
	global title_surface, state_surface, last_text_draw, title_font, state_font, state

	if screen is None:
		return

	now = pygame.time.get_ticks()
	if not force and (now - last_text_draw) < text_draw_interval_ms:
		return

	if title_font is None:
		title_font = pygame.font.Font(notoFont, 48)
		title_surface = title_font.render("B2J User", True, white)

	if state_font is None:
		state_font = pygame.font.Font(notoFont, 36)

	state_surface = state_font.render(state.upper(), True, white)

	screen.fill(black)

	title_rect = title_surface.get_rect(center=(render_w // 2, render_h // 2 - title_surface.get_height()))
	state_rect = state_surface.get_rect(center=(render_w // 2, render_h // 2 + state_surface.get_height() // 2))

	screen.blit(title_surface, title_rect)
	screen.blit(state_surface, state_rect)

	pygame.display.flip()
	last_text_draw = now

if __name__ == '__main__':

	# CLI args for drone_monitor WebSocket connection.
	# Default behavior is M5-only (no WebSocket); pass --bmi-drone-host to
	# enable forwarding confirmed predictions to a drone_monitor instance.
	parser = argparse.ArgumentParser(
		description="B2J BCI user runtime. Default behavior matches the "
		            "original M5-only setup. Pass --bmi-drone-host to also "
		            "forward confirmed predictions to a drone_monitor via WebSocket."
	)
	parser.add_argument("--bmi-drone-host", default=None,
	                    help="drone_monitor host/IP. When omitted, WebSocket "
	                         "forwarding is disabled (M5 serial only).")
	parser.add_argument("--bmi-drone-port", type=int, default=9090,
	                    help="drone_monitor WebSocket port (default: 9090). "
	                         "Only used when --bmi-drone-host is set.")
	args = parser.parse_args()

	#Window at top left
	os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

	#Start Pygame
	pygame.init()
	pygame.mixer.init()
	screen = pygame.display.set_mode((render_w, render_h), pygame.NOFRAME)
	pygame.display.set_caption("BMI Trainer")

	# Start drone_monitor WebSocket client only when --bmi-drone-host is given.
	# Default (no flag) preserves the original M5-only behavior.
	if args.bmi_drone_host:
		drone_monitor_client = DroneMonitorClient(args.bmi_drone_host, args.bmi_drone_port)
		drone_monitor_client.start()

	#Start Latency Timer
	ABMI_Utils.set_latency_timer(1)
	
	#Connect to BCI
	board.connect()
	board.stream()

	screen.fill(black)
	draw_status_text(force=True)

	send_led_flicker()

	if not board.connected:
		quit()

	#Main Loop
	while True:
		draw_status_text()

		if state == "idle":

			handleIdle()

		if state == "recording":

			handleRecording()

		if state == "predicting":
			
			handlePredicting()

		if state == "triggering":

			handleTriggering()
		
