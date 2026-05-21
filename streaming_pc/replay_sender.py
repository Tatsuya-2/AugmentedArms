"""Replay a recorded EEG JSONL file back into a viewer as a fake live sender.

Pairs with ``eeg_viewer.py`` (which records sessions when ``--record-dir`` is
set). The viewer is the WebSocket server; this script connects to it as a
client and re-sends frames at their original pace (optionally accelerated
or looped).

Usage:
    # In one terminal — start the viewer (recording optional this time):
    python3 eeg_viewer.py --port 9091

    # In another terminal — replay a recorded session:
    python3 replay_sender.py --file eeg_recordings/eeg_123_2026-05-21_10-22-15.jsonl \\
        --host 127.0.0.1 --port 9091 --speed 1.0

Flags:
    --speed N     speed multiplier (default 1.0; 2.0 = 2x faster, 0.5 = half)
    --loop        replay the file end-to-end repeatedly
    --no-register skip the recorded registration line (useful if it is absent
                  or already invalid)
"""

import argparse
import json
import sys
import time

try:
    import websocket  # provided by the `websocket-client` package
    from websocket import WebSocketException
except ImportError as e:
    raise SystemExit(
        "Missing dependency: websocket-client. "
        "Install with: pip3 install websocket-client"
    ) from e


def _iter_messages(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line), line
            except ValueError:
                print(f"[replay_sender] skipping malformed line {line_no}",
                      file=sys.stderr)


def replay_once(ws, path, speed, skip_register):
    prev_t_start = None
    wall_start = time.monotonic()
    stream_origin = None
    sent_batches = 0
    sent_registers = 0
    for line_no, msg, raw in _iter_messages(path):
        msg_type = msg.get("type")
        if msg_type == "register":
            if skip_register:
                continue
            ws.send(raw)
            sent_registers += 1
            continue
        if msg_type != "eeg_batch":
            continue
        t_start = msg.get("t_start")
        if isinstance(t_start, (int, float)):
            if prev_t_start is None:
                stream_origin = t_start
                # No sleep before the very first batch; subsequent ones pace
                # against the original t_start deltas, scaled by speed.
            else:
                # target wallclock for this batch since wall_start
                target_dt = (t_start - stream_origin) / max(speed, 1e-6)
                now_dt = time.monotonic() - wall_start
                delay = target_dt - now_dt
                if delay > 0:
                    time.sleep(delay)
            prev_t_start = t_start
        ws.send(raw)
        sent_batches += 1
    return sent_registers, sent_batches


def main():
    parser = argparse.ArgumentParser(
        description="Replay a recorded EEG JSONL file to a live viewer."
    )
    parser.add_argument("--file", required=True,
                        help="Path to the JSONL recording.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Viewer host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=9091,
                        help="Viewer port (default: 9091).")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speed multiplier (default: 1.0).")
    parser.add_argument("--loop", action="store_true",
                        help="Replay the file repeatedly until Ctrl-C.")
    parser.add_argument("--no-register", action="store_true",
                        help="Skip the recorded registration line.")
    parser.add_argument("--synthetic-register", default=None,
                        help="If the recording has no register message, send "
                             "this JSON string as the registration. Useful "
                             "for older or hand-crafted recordings.")
    args = parser.parse_args()

    if args.speed <= 0:
        raise SystemExit("--speed must be positive")

    uri = f"ws://{args.host}:{args.port}"
    print(f"[replay_sender] connecting to {uri} ...")
    ws = websocket.create_connection(uri, timeout=5.0)
    print(f"[replay_sender] connected. Replaying {args.file} at {args.speed}x"
          f"{' (loop)' if args.loop else ''}")

    if args.synthetic_register:
        ws.send(args.synthetic_register)
        print("[replay_sender] sent synthetic registration")

    try:
        loop_idx = 0
        while True:
            t0 = time.monotonic()
            try:
                regs, batches = replay_once(
                    ws, args.file, args.speed,
                    skip_register=args.no_register or (loop_idx > 0),
                )
            except FileNotFoundError:
                print(f"[replay_sender] file not found: {args.file}",
                      file=sys.stderr)
                return 1
            dt = time.monotonic() - t0
            print(f"[replay_sender] iteration {loop_idx}: "
                  f"sent {regs} register, {batches} batches in {dt:.1f}s")
            loop_idx += 1
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\n[replay_sender] interrupted")
    except (WebSocketException, ConnectionError, OSError) as e:
        print(f"[replay_sender] connection error: {e}", file=sys.stderr)
        return 2
    finally:
        try:
            ws.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
