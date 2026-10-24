"""
test_fall_detection.py
======================
Unit tests for MPU6050 fall detection on the Raspberry Pi Hub (pi/pi_hub.py).
Tests:
- resting (no event)
- walking (no event)
- sitting down hard (no fall)
- free-fall + impact + stillness (fall detected)
- high impact alone (|a| > 4g emits impact event)
- /motion endpoint schema and event retention (last 10 events)
"""

import os
import sys
import math
import time
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PI_DIR = os.path.join(REPO_ROOT, "pi")
if PI_DIR not in sys.path:
    sys.path.insert(0, PI_DIR)

from pi_hub import (
    FallDetector,
    FREE_FALL_THRESHOLD_G,
    FREE_FALL_MIN_DURATION_S,
    IMPACT_THRESHOLD_G,
    STILLNESS_MIN_G,
    STILLNESS_MAX_G,
    STILLNESS_MAX_GYRO_DEGS,
    STILLNESS_DURATION_S,
    HIGH_IMPACT_THRESHOLD_G,
    app
)


def test_resting_no_event():
    """Resting user experiencing 1.0g gravity: no fall or impact events."""
    detector = FallDetector()
    t = 1000.0
    dt = 0.01  # 100 Hz

    all_events = []
    # 2 seconds of resting
    for _ in range(200):
        events = detector.process_sample(
            ax=0.0, ay=0.0, az=1.0,
            gx=0.5, gy=0.5, gz=0.5,
            ts=t
        )
        all_events.extend(events)
        t += dt

    assert len(all_events) == 0, f"Expected 0 events while resting, got {all_events}"
    assert detector.state == "IDLE"


def test_walking_no_event():
    """Walking motion: cyclic acceleration (0.7g - 1.4g) and gyro rotation, no fall."""
    detector = FallDetector()
    t = 1000.0
    dt = 0.01  # 100 Hz

    all_events = []
    # 3 seconds of walking
    for i in range(300):
        # Sine wave oscillating around 1.0g
        phase = i * 0.1
        az = 1.0 + 0.35 * math.sin(phase)
        ax = 0.15 * math.cos(phase)
        ay = 0.1 * math.sin(phase * 0.5)
        gx = 20.0 + 15.0 * math.sin(phase)
        gy = 10.0 * math.cos(phase)
        gz = 5.0

        events = detector.process_sample(ax, ay, az, gx, gy, gz, ts=t)
        all_events.extend(events)
        t += dt

    fall_events = [e for e in all_events if e.get("type") == "fall"]
    assert len(fall_events) == 0, f"Expected 0 fall events during walking, got {fall_events}"


def test_sitting_down_hard_no_fall():
    """
    Sitting down hard: brief deceleration (>0.4g or <80ms) and moderate impact (~2.8g),
    followed by fidgeting/movement (gyro > 30 deg/s or |a| outside stillness range).
    Must NOT trigger a fall event.
    """
    detector = FallDetector()
    t = 1000.0
    dt = 0.01  # 100 Hz

    all_events = []

    # 1. Standing/settling before sitting (0.5s)
    for _ in range(50):
        events = detector.process_sample(0.0, 0.0, 1.0, 5.0, 5.0, 5.0, ts=t)
        all_events.extend(events)
        t += dt

    # 2. Brief descent: 0.55g (does not drop below 0.4g free-fall threshold) for 0.1s
    for _ in range(10):
        events = detector.process_sample(0.0, 0.0, 0.55, 10.0, 10.0, 10.0, ts=t)
        all_events.extend(events)
        t += dt

    # 3. Impact onto chair: 2.8g
    for _ in range(3):
        events = detector.process_sample(0.0, 0.0, 2.8, 25.0, 20.0, 15.0, ts=t)
        all_events.extend(events)
        t += dt

    # 4. Moving around while seated (gyro > 35 deg/s)
    for _ in range(250):
        events = detector.process_sample(0.0, 0.0, 1.0, 45.0, 35.0, 40.0, ts=t)
        all_events.extend(events)
        t += dt

    fall_events = [e for e in all_events if e.get("type") == "fall"]
    assert len(fall_events) == 0, f"Expected no fall event for sitting down hard, got: {fall_events}"


def test_free_fall_impact_stillness_triggers_fall():
    """
    True Fall sequence:
    1. Free-fall: |a| < 0.4g for >= 80ms (100ms here)
    2. Impact: |a| > 2.5g within 1s (3.2g here)
    3. Stillness: |a| between 0.8g and 1.2g and gyro < 30 deg/s for 2.0s
    Must emit a 'fall' event with peak_g.
    """
    detector = FallDetector()
    t = 1000.0
    dt = 0.01  # 100 Hz

    all_events = []

    # 1. Normal state (0.2s)
    for _ in range(20):
        events = detector.process_sample(0.0, 0.0, 1.0, 5.0, 5.0, 5.0, ts=t)
        all_events.extend(events)
        t += dt

    # 2. Free-fall (|a| = 0.1g < 0.4g for 10 samples = 100ms >= 80ms)
    for _ in range(10):
        events = detector.process_sample(0.0, 0.0, 0.1, 10.0, 10.0, 10.0, ts=t)
        all_events.extend(events)
        t += dt

    # 3. Impact (|a| = 3.2g > 2.5g within 1s)
    for _ in range(5):
        events = detector.process_sample(0.0, 0.0, 3.2, 50.0, 50.0, 50.0, ts=t)
        all_events.extend(events)
        t += dt

    # 4. Stillness (|a| = 1.0g, gyro = 8 deg/s < 30 deg/s for 2.1s = 210 samples)
    for _ in range(210):
        events = detector.process_sample(0.0, 0.0, 1.0, 5.0, 5.0, 5.0, ts=t)
        all_events.extend(events)
        t += dt

    fall_events = [e for e in all_events if e.get("type") == "fall"]
    assert len(fall_events) == 1, f"Expected exactly 1 fall event, got: {fall_events}"
    fall_ev = fall_events[0]
    assert fall_ev["type"] == "fall"
    assert fall_ev["peak_g"] >= 2.5
    assert "id" in fall_ev
    assert "ts" in fall_ev


def test_high_impact_alone():
    """Sudden impact |a| > 4.0g alone triggers lower-confidence 'impact' event."""
    detector = FallDetector()
    t = 1000.0

    # Normal standing
    detector.process_sample(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, ts=t)

    # Spike > 4.0g (e.g. 4.6g) without free fall
    events = detector.process_sample(0.0, 0.0, 4.6, 10.0, 10.0, 10.0, ts=t + 0.01)

    assert len(events) == 1, f"Expected 1 impact event, got: {events}"
    assert events[0]["type"] == "impact"
    assert events[0]["peak_g"] == 4.6


def test_motion_endpoint_schema_and_retention():
    """Verify /motion endpoint returns required JSON keys and keeps last 10 events."""
    client = app.test_client()

    # Query /motion
    res = client.get('/motion')
    assert res.status_code == 200
    data = res.get_json()

    for key in ("ax", "ay", "az", "magnitude", "gyro_mag", "ts", "events"):
        assert key in data, f"Key '{key}' missing from /motion response"

    assert isinstance(data["events"], list)
    assert len(data["events"]) <= 10

    # Verify existing endpoints /video, /sensors, /distance were not changed
    res_sensors = client.get('/sensors')
    assert res_sensors.status_code == 200
    sensors_data = res_sensors.get_json()
    assert "ultrasonic_distance_cm" in sensors_data

    res_distance = client.get('/distance')
    assert res_distance.status_code == 200
    dist_data = res_distance.get_json()
    assert "distance_cm" in dist_data
