Augmented Arms application on Rasp Pi 5 for Muto Arm animation recording and playback, using audio-tap trigger system (Piezo sensor)

BMI Trainer application on Rasp Pi 5 for ALS Patients (or anyone really) to record brainwave activity via OpenBCI board, and train a local model for audio-stimuli based selection activity.

TODO:
  -Remove local recordings after upload?
  -Japanese Language switch for Trainer Scenes

## B2J-User.py — drone forwarding & developer mode

`B2J-User.py` forwards each confirmed prediction (1/2/3) to the drone_monitor
over WebSocket via `DroneMonitorClient`.

Options:
  --bmi-drone-host HOST   drone_monitor host/IP. Empty by default = forwarding
                          DISABLED. Pass a host (e.g. 192.168.12.10 = Jetson)
                          to enable forwarding.
  --bmi-drone-port PORT   drone_monitor WebSocket port (default: 9090).
  --mock-hardware         Developer/test mode: run with NO real hardware.

### --mock-hardware (no BCI / piezo / M5 required)

Replaces the BCI board, piezo sensor, and M5 serial link with in-process mocks
(see `B2J_mocks.py`), so the full state machine and the drone-forwarding path can
be exercised on any machine. Production launches (without the flag) are unaffected
and never import `B2J_mocks`.

  python3 B2J-User.py --mock-hardware

Behavior in this mode:
  - A red `[MOCK HARDWARE]` banner is printed at startup.
  - Trigger with the keyboard 'b' key (instead of the piezo); ESC quits.
  - Predictions cycle 1 -> 2 -> 3 on each recognition.
  - Forwarding is OFF by default. To also verify forwarding to the drone, pass
    a host explicitly:
      python3 B2J-User.py --mock-hardware --bmi-drone-host 192.168.12.10
  - Note: a real display is still required (pygame window is not mocked).
