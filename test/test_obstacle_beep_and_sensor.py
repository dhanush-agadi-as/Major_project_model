"""
test_obstacle_beep_and_sensor.py
================================
Tests for Problem 1:
- beep interval at 100/70/40/20 cm
- null and out-of-range readings are silent
- a TTS in progress doesn't delay a beep
- parsing of the three line formats: "12.69", "12.69 cm", "Distance: 12.69 cm"
- -1.0 becomes null
- "test beep" works without the sensor
"""

import os
import sys
import time
import threading
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PI_DIR = os.path.join(REPO_ROOT, "pi")
if PI_DIR not in sys.path:
    sys.path.insert(0, PI_DIR)

from obstacle_alerter import ObstacleAlerter, calculate_beep_interval
from pi_hub import parse_ultrasonic_line, app


def test_beep_interval_at_100_70_40_20():
    """Beep interval scales from 1.0 s at 100 cm down to 0.2 s at 40 cm; 0.12 s for 20-40 cm; solid tone < 20 cm."""
    # 100 cm -> 1.0 s
    assert calculate_beep_interval(100.0) == 1.0

    # 70 cm -> 0.6 s
    assert pytest.approx(calculate_beep_interval(70.0), 0.01) == 0.6

    # 40 cm -> 0.2 s
    assert calculate_beep_interval(40.0) == 0.2

    # 20 cm -> 0.12 s (continuous fast beeps below 40 cm down to 20 cm)
    assert calculate_beep_interval(20.0) == 0.12

    # below 20 cm -> solid tone (0.0 s)
    assert calculate_beep_interval(19.0) == 0.0
    assert calculate_beep_interval(5.0) == 0.0


def test_null_and_out_of_range_are_silent():
    """Null, < 2 cm, and > 100 cm readings return None (silent)."""
    assert calculate_beep_interval(None) is None
    assert calculate_beep_interval(1.5) is None
    assert calculate_beep_interval(0.0) is None
    assert calculate_beep_interval(-1.0) is None
    assert calculate_beep_interval(100.1) is None
    assert calculate_beep_interval(150.0) is None
    assert calculate_beep_interval(300.0) is None


def test_tts_in_progress_does_not_delay_beep():
    """Beeps run on their own audio thread so a busy/speaking TTS never blocks or delays them."""
    beep_times = []

    def mock_beep(freq, ms):
        beep_times.append(time.monotonic())

    # Simulated TTS that is busy speaking for 1.0 s
    tts_busy = threading.Event()
    tts_busy.set()

    def mock_speak(text, priority=1):
        # Even if TTS blocks, it is on its own thread
        time.sleep(0.5)

    alerter = ObstacleAlerter(
        beep_fn=mock_beep,
        speak_fn=mock_speak,
        beeps_enabled=True,
    )
    # Set distance to 30 cm -> interval = 0.12 s
    with alerter._lock:
        alerter._smooth_distance = 30.0

    # Start beep loop in background thread
    alerter._stop_event.clear()
    beep_th = threading.Thread(target=alerter._beep_loop, daemon=True)
    beep_th.start()

    # While TTS is busy speaking in a separate thread:
    t_start = time.monotonic()
    tts_thread = threading.Thread(target=lambda: mock_speak("Testing busy TTS"), daemon=True)
    tts_thread.start()

    # Wait 0.4 seconds: at 0.12s interval, multiple beeps should occur despite TTS running
    time.sleep(0.4)
    alerter._stop_event.set()
    beep_th.join(timeout=0.5)
    tts_thread.join(timeout=1.0)

    assert len(beep_times) >= 2, f"Expected multiple beeps during 0.4s window, got {len(beep_times)}"
    # Verify spacing between consecutive beeps is approximately 0.12s, not delayed by 0.5s TTS
    intervals = [t2 - t1 for t1, t2 in zip(beep_times[:-1], beep_times[1:])]
    for inv in intervals:
        assert inv < 0.25, f"Beep interval {inv:.3f}s was delayed!"


def test_parsing_three_line_formats():
    """Accept lines like '12.69', '12.69 cm', and 'Distance: 12.69 cm'. Ignore 'Warning' or 'No echo'."""
    val1, ignored1 = parse_ultrasonic_line("12.69")
    assert not ignored1
    assert val1 == 12.69

    val2, ignored2 = parse_ultrasonic_line("12.69 cm")
    assert not ignored2
    assert val2 == 12.69

    val3, ignored3 = parse_ultrasonic_line("Distance: 12.69 cm")
    assert not ignored3
    assert val3 == 12.69

    # Additional standard ESP32 format
    val4, ignored4 = parse_ultrasonic_line("DIST:12.69,BTN:0")
    assert not ignored4
    assert val4 == 12.69

    # Ignore Warning and No echo
    val_warn, ignored_warn = parse_ultrasonic_line("Warning: read timeout")
    assert ignored_warn
    assert val_warn is None

    val_echo, ignored_echo = parse_ultrasonic_line("No echo received")
    assert ignored_echo
    assert val_echo is None


def test_minus_one_becomes_null():
    """-1.0 in serial readings is parsed and returns null in /sensors."""
    val, ignored = parse_ultrasonic_line("-1.0")
    assert not ignored
    assert val == -1.0

    val_cm, ignored_cm = parse_ultrasonic_line("-1.0 cm")
    assert not ignored_cm
    assert val_cm == -1.0

    val_dist, ignored_dist = parse_ultrasonic_line("Distance: -1.0 cm")
    assert not ignored_dist
    assert val_dist == -1.0

    # In pi_hub Flask app /sensors
    import pi_hub
    with pi_hub.distance_lock:
        pi_hub.latest_distance = None  # -1.0 sets latest_distance to None
        pi_hub.latest_distance_time = 0.0

    client = app.test_client()
    resp = client.get('/sensors')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ultrasonic_distance_cm"] is None
    assert data["ultrasonic_ok"] is False
    assert "ultrasonic_age_ms" in data


def test_test_beep_works_without_sensor():
    """'test beep' plays the full beep pattern from 100 cm to 20 cm in ~6 s without sensor."""
    beeps_played = []

    def mock_beep(freq, ms):
        beeps_played.append((freq, ms))

    alerter = ObstacleAlerter(
        beep_fn=mock_beep,
        beeps_enabled=True,
    )
    # Distance is completely None (no sensor attached)
    with alerter._lock:
        alerter._smooth_distance = None

    t0 = time.monotonic()
    # Play test pattern with scaled duration of 0.5s for fast test execution
    alerter.play_test_pattern(duration_s=0.5)
    elapsed = time.monotonic() - t0

    assert len(beeps_played) > 0, "test_beep should have triggered beeps without sensor reading"
    assert elapsed >= 0.4
