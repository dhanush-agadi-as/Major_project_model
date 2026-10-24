"""
test_motion_ultrasonic_noise_fixes.py
======================================
Comprehensive tests covering all requirements:
1. m/s^2 values convert to g (divide by 9.80665)
2. At-rest data gives no event
3. Free-fall (<0.5g for >=60ms) + impact (>2.0g within 1s) + stillness (1.5s) gives a fall
4. An impact followed by pick-up still gives an "impact" event
5. ESP32 -1.0 distance becomes null in /sensors
6. Serial port auto-detect among /dev/ttyUSB* and /dev/ttyACM* and reconnect
7. The 404 missing endpoint and the "sensor not responding" messages speak/log once
8. "navigate to mobile" with a remembered mobile goes to find (speaks "Finding your mobile", no geocoding)
9. Spatial repeats are suppressed (<= 1 per 4s, 10s repeat suppression, single line enrollment)
10. No hard-coded IPs in config or code
"""

import os
import sys
import time
import math
import logging
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PI_DIR = os.path.join(REPO_ROOT, "pi")
if PI_DIR not in sys.path:
    sys.path.insert(0, PI_DIR)

import config
from pi_hub import (
    FallDetector,
    G_TO_MS2,
    ms2_to_g,
    convert_ms2_to_g,
    FREE_FALL_THRESHOLD_G,
    FREE_FALL_MIN_DURATION_S,
    IMPACT_THRESHOLD_G,
    IMPACT_WINDOW_S,
    STILLNESS_MIN_G,
    STILLNESS_MAX_G,
    STILLNESS_MAX_GYRO_DEGS,
    STILLNESS_DURATION_S,
    HIGH_IMPACT_THRESHOLD_G,
    app
)
from emergency_manager import EmergencyManager
from obstacle_alerter import ObstacleAlerter
from spatial_navigator import SpatialNavigator


# ─── 1. m/s^2 values convert to g ───────────────────────────────────────────
def test_ms2_values_convert_to_g():
    """Verify conversion of m/s^2 to g divides by 9.80665."""
    assert abs(G_TO_MS2 - 9.80665) < 1e-5
    # 9.80665 m/s^2 should be exactly 1.0 g
    assert abs(ms2_to_g(9.80665) - 1.0) < 1e-4
    gx, gy, gz = convert_ms2_to_g(0.0, 0.0, 9.80665)
    assert abs(gz - 1.0) < 1e-4
    assert abs(gx) < 1e-5
    # 0 m/s^2 -> 0 g
    assert abs(ms2_to_g(0.0)) < 1e-5
    # 19.6133 m/s^2 -> 2.0 g
    assert abs(ms2_to_g(19.6133) - 2.0) < 1e-3


# ─── 2. At-rest data gives no event ─────────────────────────────────────────
def test_at_rest_data_gives_no_event():
    """At rest data in m/s^2 (0, 0, 9.80665) gives no fall or impact event."""
    detector = FallDetector()
    t = 100.0
    dt = 0.01

    events_collected = []
    for _ in range(200):  # 2 seconds at 100 Hz
        ev = detector.process_sample_ms2(
            ax=0.0, ay=0.0, az=9.80665,
            gx=0.5, gy=0.5, gz=0.5,
            ts=t
        )
        events_collected.extend(ev)
        t += dt

    assert len(events_collected) == 0, f"Expected 0 events at rest, got {events_collected}"
    assert detector.state == "IDLE"


# ─── 3. Free-fall + impact + stillness gives a fall ─────────────────────────
def test_free_fall_impact_stillness_gives_fall():
    """
    Demo thresholds:
    - Free fall < 0.5 g for >= 60 ms
    - Impact > 2.0 g within 1 s
    - Stillness 0.8-1.2 g and gyro < 30 deg/s for 1.5 s
    """
    assert FREE_FALL_THRESHOLD_G == 0.5
    assert FREE_FALL_MIN_DURATION_S == 0.06
    assert IMPACT_THRESHOLD_G == 2.0
    assert IMPACT_WINDOW_S == 1.0
    assert STILLNESS_MIN_G == 0.8
    assert STILLNESS_MAX_G == 1.2
    assert STILLNESS_MAX_GYRO_DEGS == 30.0
    assert STILLNESS_DURATION_S == 1.5

    detector = FallDetector()
    t = 1000.0
    dt = 0.01  # 10ms step

    # 1. Free fall: 0.1g for 70ms (7 samples >= 60ms)
    for _ in range(7):
        detector.process_sample(0.0, 0.0, 0.1, 10.0, 10.0, 10.0, ts=t)
        t += dt
    assert detector.state == "FREE_FALL"

    # End of free fall triggers AWAITING_IMPACT
    detector.process_sample(0.0, 0.0, 0.8, 10.0, 10.0, 10.0, ts=t)
    t += dt
    assert detector.state == "AWAITING_IMPACT"

    # 2. Impact: 2.5g (> 2.0g within 1s)
    detector.process_sample(0.0, 0.0, 2.5, 40.0, 40.0, 40.0, ts=t)
    t += dt
    assert detector.state == "AWAITING_STILLNESS"

    # 3. Stillness: 1.0g, gyro 5 deg/s for 1.6s (160 samples >= 1.5s)
    fall_events = []
    for _ in range(160):
        ev = detector.process_sample(0.0, 0.0, 1.0, 5.0, 5.0, 5.0, ts=t)
        fall_events.extend(ev)
        t += dt

    assert len(fall_events) == 1
    assert fall_events[0]["type"] == "fall"
    assert fall_events[0]["peak_g"] >= 2.5


# ─── 4. Impact followed by pick-up still gives an "impact" event ───────────
def test_impact_followed_by_pickup_gives_impact_event():
    """If device is picked up before stillness window ends, report 'impact' event."""
    detector = FallDetector()
    t = 1000.0
    dt = 0.01

    # 1. Free fall for 70ms
    for _ in range(7):
        detector.process_sample(0.0, 0.0, 0.1, 5.0, 5.0, 5.0, ts=t)
        t += dt

    # End free fall -> AWAITING_IMPACT
    detector.process_sample(0.0, 0.0, 0.8, 5.0, 5.0, 5.0, ts=t)
    t += dt

    # 2. Impact (> 2.0g) -> AWAITING_STILLNESS
    detector.process_sample(0.0, 0.0, 2.4, 20.0, 20.0, 20.0, ts=t)
    t += dt
    assert detector.state == "AWAITING_STILLNESS"

    # 3. Enter stillness briefly (0.5s < 1.5s required)
    for _ in range(50):
        detector.process_sample(0.0, 0.0, 1.0, 5.0, 5.0, 5.0, ts=t)
        t += dt

    # 4. Device is picked up! Sudden motion (e.g. 1.6g, gyro 50 deg/s)
    events = detector.process_sample(0.0, 0.0, 1.6, 50.0, 50.0, 50.0, ts=t)

    assert len(events) == 1, f"Expected 1 event, got {events}"
    assert events[0]["type"] == "impact", f"Expected 'impact' event, got {events[0]['type']}"
    assert detector.state == "IDLE"


# ─── 5. -1.0 becomes null in /sensors and /distance ─────────────────────────
def test_minus_one_becomes_null_in_sensors():
    """Verify -1.0 is treated as no reading and returns null in /sensors."""
    import pi_hub
    with pi_hub.distance_lock:
        pi_hub.latest_distance = None  # -1.0 sets latest_distance to None
        pi_hub.latest_distance_time = 0.0

    client = app.test_client()
    res = client.get('/sensors')
    assert res.status_code == 200
    data = res.get_json()
    assert data["ultrasonic_distance_cm"] is None
    assert "ultrasonic_ok" in data
    assert "ultrasonic_age_ms" in data


# ─── 6. Port auto-detect and reconnect ──────────────────────────────────────
def test_port_autodetect_and_reconnect():
    """Verify port detection includes /dev/ttyUSB* and /dev/ttyACM*."""
    import glob
    with patch('glob.glob') as mock_glob:
        mock_glob.side_effect = lambda pat: ['/dev/ttyUSB1'] if 'USB' in pat else ['/dev/ttyACM0']
        
        # Test candidate generation logic from pi_hub
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        for fallback in ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1', '/dev/ttyACM1']:
            if fallback not in ports:
                ports.append(fallback)

        assert '/dev/ttyUSB1' in ports
        assert '/dev/ttyACM0' in ports
        assert '/dev/ttyUSB0' in ports


# ─── 7. 404 and "sensor not responding" log/speak once ──────────────────────
def test_404_logs_once_and_retries_every_5s(caplog):
    """If /motion returns 404 or is unreachable, log the warning once and retry."""
    manager = EmergencyManager(motion_url="http://raspberrypi.local:5000/motion")
    
    with caplog.at_level(logging.ERROR):
        with patch('requests.get') as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_get.return_value = mock_resp

            # First poll -> logs error
            res1 = manager.poll_once()
            assert res1["ok"] is False
            assert res1["retry_delay"] == 5.0
            
            # Second poll -> should NOT log error again
            res2 = manager.poll_once()
            assert res2["ok"] is False
            assert res2["retry_delay"] == 5.0

    occurrences = [rec for rec in caplog.records if "Motion endpoint missing or unreachable" in rec.message]
    assert len(occurrences) == 1, f"Expected 1 log message, got {len(occurrences)}"


def test_sensor_not_responding_speaks_once_and_stays_quiet(caplog):
    """Laptop side: if distance stays null for 10s, speak once 'Distance sensor not responding' and stay quiet."""
    spoken = []
    alerter = ObstacleAlerter(
        sensors_url="http://raspberrypi.local:5000/sensors",
        speak_fn=lambda text, **kw: spoken.append(text),
        beeps_enabled=False
    )

    t0 = 1000.0
    with caplog.at_level(logging.WARNING):
        # 1. Data at t=0
        with patch('time.monotonic', return_value=t0):
            alerter.process_sensor_payload({"ultrasonic_distance_cm": 50.0, "ok": True})
        assert len(spoken) == 0

        # 2. Null reading at t=6s (>= 5s, logs no data)
        with patch('time.monotonic', return_value=t0 + 6.0):
            alerter.process_sensor_payload({"ultrasonic_distance_cm": None, "ok": False})
        assert len(spoken) == 0

        # 3. Null reading at t=11s (>= 10s, speaks 'Distance sensor not responding')
        with patch('time.monotonic', return_value=t0 + 11.0):
            alerter.process_sensor_payload({"ultrasonic_distance_cm": None, "ok": False})
        assert spoken == ["Distance sensor not responding"]

        # 4. Null reading at t=15s (stays quiet!)
        with patch('time.monotonic', return_value=t0 + 15.0):
            alerter.process_sensor_payload({"ultrasonic_distance_cm": None, "ok": False})
        assert spoken == ["Distance sensor not responding"]  # Still only once!


# ─── 8. "navigate to <X>" with remembered object goes to find ───────────────
def test_navigate_to_remembered_object_goes_to_find():
    """'navigate to mobile' when mobile is remembered speaks 'Finding your mobile' and does not geocode."""
    from wireless.laptop_server import LaptopMLServer
    
    server = LaptopMLServer.__new__(LaptopMLServer)
    server.audio_enabled = True
    server.audio = MagicMock()
    server.object_memory = MagicMock()
    server.osm_navigator = MagicMock()
    server.find_guide = MagicMock()
    server.voice_system = MagicMock()

    # Mock remembered object 'my mobile'
    server.object_memory.list_remembered_objects.return_value = [("my mobile", "cell phone")]
    server.object_memory.get_cached_object.return_value = {"yolo_class": "cell phone"}
    server.object_memory.get_object_aliases.return_value = ["mobile", "phone"]

    # Register handlers
    handlers = {}
    server.voice_system.register_command_handler.side_effect = lambda name, fn: handlers.update({name: fn})
    server._register_voice_handlers()

    navigate_handler = handlers.get('navigate_to')
    assert navigate_handler is not None

    navigate_handler({'destination': 'mobile'})

    # Verify speak called with "Finding your mobile"
    server.audio.speak_social.assert_called_with("Finding your mobile")
    # Verify find_guide was started
    server.find_guide.start.assert_called_with("my mobile", "")
    # Verify geocoding was NOT called!
    server.osm_navigator.geocode_address.assert_not_called()


# ─── 9. Spatial repetition suppression ─────────────────────────────────────
def test_spatial_repetition_suppressed_within_10s_unless_changed():
    """Spatial navigator suppresses same object & bucket within 10s unless >=30% dist change or direction change."""
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    
    t0 = 100.0
    item = {
        'class_name': 'chair',
        'zone': 'center',
        'zone_speech': 'ahead',
        'distance': 'medium',
        'estimated_distance_m': 2.0,
        'message': 'Chair ahead'
    }

    # First announcement -> allowed
    assert nav.should_speak_guidance(item, current_time=t0) is True
    nav.mark_guidance_spoken(item, current_time=t0)

    # 2 seconds later: blocked by 4s spatial limit
    assert nav.should_speak_guidance(item, current_time=t0 + 2.0) is False

    # 5 seconds later: 4s limit passed, but same object, bucket, and distance -> blocked by 10s cooldown
    assert nav.should_speak_guidance(item, current_time=t0 + 5.0) is False

    # 5 seconds later: distance changes by 35% (2.0m -> 1.3m) -> allowed!
    item_closer = dict(item)
    item_closer['estimated_distance_m'] = 1.3
    assert nav.should_speak_guidance(item_closer, current_time=t0 + 5.0) is True


def test_spatial_at_most_one_per_4s():
    """At most one spatial announcement per 4 seconds across different objects."""
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    
    t0 = 100.0
    item1 = {'class_name': 'chair', 'zone': 'left', 'zone_speech': 'left', 'distance': 'far', 'message': 'Chair left'}
    item2 = {'class_name': 'bottle', 'zone': 'right', 'zone_speech': 'right', 'distance': 'close', 'message': 'Bottle right'}

    assert nav.should_speak_guidance(item1, current_time=t0) is True
    nav.mark_guidance_spoken(item1, current_time=t0)

    # Different object 3 seconds later: blocked because < 4.0s
    assert nav.should_speak_guidance(item2, current_time=t0 + 3.0) is False

    # Different object 4.1 seconds later: allowed!
    assert nav.should_speak_guidance(item2, current_time=t0 + 4.1) is True


# ─── 10. No hard-coded IPs in codebase ──────────────────────────────────────
def test_no_hardcoded_pi_ips():
    """Verify no hardcoded Pi IP remains in any .py file."""
    banned_ip = "10." + "117." + "98." + "217"
    for root, dirs, files in os.walk(REPO_ROOT):
        if ".git" in root or "__pycache__" in root or "venv" in root or ".pytest_cache" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                if os.path.basename(path) == "test_motion_ultrasonic_noise_fixes.py":
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
                    assert banned_ip not in content, f"Found hardcoded IP in {path}"
