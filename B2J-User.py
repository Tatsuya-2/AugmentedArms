import ABMI_Utils
import ALS_Utils
import argparse
import glob
import serial
import pygame
import os
import time
import random
from datetime import datetime
from serial.tools import list_ports

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

baud = 115200
MAX_ESP_PAYLOAD_BYTES = 240

def pick_m5_port():
    ports = list(list_ports.comports())

    for p in ports:
        text = f"{p.description} {p.manufacturer} {p.product}".lower()

        if "ftdi" in text:
            continue

        return p.device

    return None

m5_port = pick_m5_port()

if m5_port is None:
    print("[ERROR] No M5 serial port found")
else:
    print(f"[OK] M5 port found: {m5_port}")

notoFont = "/home/b2j/Desktop/AugmentedArms/Font/NotoSansJP-Bold.otf"

white = (217, 217, 217)
black = (0, 0, 0)

render_w, render_h = 480, 320
screen = None
clock = None
title_font = None
state_font = None

board = ABMI_Utils.BCIBoard(port="/dev/bci_dongle")
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
keyboard_pressed = False

userID = ABMI_Utils.getUserID()

piezo = ALS_Utils.PiezoSensor(pin=21, cooldown=1)

state = "idle"
drone_monitor_client = None
title_surface = None
state_surface = None
last_text_draw = 0
text_draw_interval_ms = 200


def was_trigger_pressed():
    global keyboard_pressed

    if piezo.was_pressed() or keyboard_pressed:
        keyboard_pressed = False
        return True

    return False


def sendToM5(port: str, baud: int, message: str):
    if not port:
        print("[ERROR] No serial port selected")
        return

    if not message:
        return

    message = message.strip() + "\n"

    try:
        with serial.Serial(port, baud, timeout=1) as ser:
            ser.write(message.encode("utf-8"))
            ser.flush()
            print(f"[TX M5] {message.strip()}")

    except serial.SerialException as e:
        print(f"[ERROR] Serial error: {e}")


def _send_single_command(ch, val):
    sendToM5(m5_port, baud, f"S,Ch{ch}-{val}")


def send_led_command(ch, val):
    if ch == 9:
        print(f"{GREEN}Sending to ALL Channels -> Pattern: {val}{RESET}")

        for i in range(1, 9):
            _send_single_command(i, val)
            time.sleep(0.01)

    else:
        print(f"{GREEN}Sending to Ch: {ch} -> Pattern: {val}{RESET}")
        _send_single_command(ch, val)


def send_led_flicker():
    channels = list(range(1, 9))
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
    send_led_command(9, 0)


def handleIdle():
    global board, state, latest_test_file, sequence_thread, cancel_event

    if was_trigger_pressed():
        state = "recording"
        ABMI_Utils.play_single_sound("Sounds/Click.mp3", block=True)

        timestamp = datetime.now()
        lcr_choice = 4
        timestamp_str = timestamp.strftime("%Y-%m-%d-%H-%M-%S")

        latest_test_file = os.path.join(
            "Testing",
            f"{userID}-{timestamp_str}-{lcr_choice}.csv"
        )

        sequence_thread, cancel_event = ABMI_Utils.startSingleTrainingSequence(
            board,
            userID,
            timestamp,
            lcr_choice,
            testing_path
        )

        send_led_all_off()


def handleRecording():
    global state, sequence_thread, latest_test_file

    recent = str(board._last_data_frame)

    sendToM5(m5_port, baud, f"E,{recent}")

    if not sequence_thread.is_alive():
        state = "predicting"


def handlePredicting():
    global state, prediction_choice, sound_map, latest_test_file, model_path

    try:
        prediction_choice = ABMI_Utils.useModelToPredict(
            latest_test_file,
            model_path
        )

        if prediction_choice == 1:
            send_led_left()
        elif prediction_choice == 2:
            send_led_center()
        elif prediction_choice == 3:
            send_led_right()

        if prediction_choice == -1:
            ABMI_Utils.play_single_sound("Sounds/prediction_unknown.wav")
            state = "idle"
        else:
            ABMI_Utils.play_single_sound(
                sound_map.get(prediction_choice, "Sounds/prediction_unknown.wav")
            )
            state = "triggering"

    except Exception as err:
        print(f"[ERROR] Model prediction failed: {err}")
        ABMI_Utils.play_single_sound("Sounds/prediction_unknown.wav")
        state = "idle"


def handleTriggering():
    global state, prediction_choice, testing_path, drone_monitor_client

    if was_trigger_pressed():
        ABMI_Utils.play_single_sound(
            sound_map.get(prediction_choice, "Sounds/prediction_unknown.wav")
        )

        sendToM5(m5_port, baud, f"T,{prediction_choice}")
		# Forward the confirmed choice to drone_monitor (raw "1"/"2"/"3" over WebSocket).
		# Send is best-effort and non-blocking; failures are logged by the client.
		if drone_monitor_client is not None and prediction_choice in (1, 2, 3):
			drone_monitor_client.send_signal(str(prediction_choice))

        ABMI_Utils.deleteTestingFiles(testing_path)

        state = "idle"
        send_led_all_off()


def draw_status_text(force=False):
    global title_surface, state_surface
    global last_text_draw, title_font, state_font, state

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

    title_rect = title_surface.get_rect(
        center=(render_w // 2, render_h // 2 - title_surface.get_height())
    )

    state_rect = state_surface.get_rect(
        center=(render_w // 2, render_h // 2 + state_surface.get_height() // 2)
    )

    screen.blit(title_surface, title_rect)
    screen.blit(state_surface, state_rect)

    pygame.display.flip()
    last_text_draw = now


if __name__ == "__main__":
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

	#Start Latency Timer
	ABMI_Utils.set_latency_timer(1)

	#Connect to BCI
	board.connect()
	board.stream()

    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"

    pygame.init()
    pygame.mixer.init()

    screen = pygame.display.set_mode((render_w, render_h), pygame.NOFRAME)
    pygame.display.set_caption("BMI Trainer")

	# Start drone_monitor WebSocket client only when --bmi-drone-host is given.
	# Default (no flag) preserves the original M5-only behavior.
	if args.bmi_drone_host:
		drone_monitor_client = DroneMonitorClient(args.bmi_drone_host, args.bmi_drone_port)
		drone_monitor_client.start()

    ABMI_Utils.set_latency_timer(1)

    board.connect()
    board.stream()

    screen.fill(black)
    draw_status_text(force=True)

    send_led_flicker()

    if not board.connected:
        quit()

    running = True

    while running:

        for event in pygame.event.get():

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_b:
                    keyboard_pressed = True
                    print("B pressed -> simulated piezo press")

            elif event.type == pygame.QUIT:
                running = False

        draw_status_text()

        if state == "idle":
            handleIdle()

        elif state == "recording":
            handleRecording()

        elif state == "predicting":
            handlePredicting()

        elif state == "triggering":
            handleTriggering()

    send_led_all_off()

    try:
        board.stop_stream()
    except:
        pass

    pygame.quit()
    quit()
