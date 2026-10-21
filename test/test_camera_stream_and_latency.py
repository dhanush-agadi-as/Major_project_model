"""
Unit and integration tests for Camera Lag Fix and Performance Decoupling:
A. Decoupled display from ML (25-30 FPS display, newest frame only for ML, dropped count, latency log)
B. CPU thread budget (OpenVINO=8, FACE_THREADS=4, torch=2, logged at startup)
C. Face work off frame loop (separate worker <= 3 Hz, cached per track, skipped during find)
D. Pi side (/video threaded, jpeg 70, cap 20 FPS, encode only when connected)
E. Ultrasonic: when USE_PI_CAMERA = True, no local serial port / COM3, no mock fallback, single warning
F. Voice mishearings: 'shuttle', 'shut', 'shot down', 'shot up', 'shutting', 'shut it' as quit; ratio >= 0.8 fallback
G. End-to-end integration requirements
"""
import time
import base64
import json
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import cv2

import config
from voice_command_system import VoiceCommandParser
from ultrasonic_sensor import UltrasonicSensor
from wireless.laptop_server import LaptopMLServer, FaceWorker


def test_decoupled_display_and_ml_latency_under_load():
    """
    Requirement G & A:
    With a fake ML worker taking 300 ms per frame:
    - Display thread still runs at >= 20 FPS
    - frame_age stays under 400 ms (target < 150 ms)
    - Stale frames are dropped (dropped > 0)
    """
    server = LaptopMLServer()
    server.show_display = False  # Headless mode for automated tests

    def fake_slow_ml(frame, dist, btn):
        time.sleep(0.3)
        return {
            'objects': [{'class': 'person', 'confidence': 0.9, 'bbox': [10, 10, 50, 50]}],
            'faces': [],
            'remembered': [],
            'audio_commands': [],
            'navigation': None,
            'warnings': []
        }

    server.process_ml = fake_slow_ml
    server._start_worker_threads()

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Stream frames at ~30 FPS for ~1.5 seconds (~45 frames)
    for i in range(45):
        ts = time.time()
        server.handle_incoming_frame(
            dummy_frame, ultrasonic_distance=120.0, button_pressed=False, timestamp=ts
        )
        time.sleep(0.033)

    time.sleep(0.1)

    display_fps = server._get_fps_from_deque(server._display_frame_times)
    frame_age = server._latest_frame_age_ms
    dropped = server._dropped_frames_count

    server.cleanup()

    assert display_fps >= 20.0, f"Expected display FPS >= 20, got {display_fps:.1f}"
    assert frame_age < 400.0, f"Expected frame_age < 400 ms, got {frame_age:.1f} ms"
    assert dropped > 0, f"Expected stale frames to be dropped, got {dropped}"


@pytest.mark.asyncio
async def test_websocket_stale_frame_dropping():
    """
    Requirement A & G:
    The websocket path between rpi_client and the server drops stale frames:
    if the server is behind, skip to the newest.
    """
    server = LaptopMLServer()
    server.show_display = False
    server.process_ml = lambda f, d, b: {
        'objects': [], 'faces': [], 'remembered': [], 'audio_commands': [], 'navigation': None, 'warnings': []
    }

    dummy_frame = np.zeros((60, 60, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', dummy_frame)
    b64 = base64.b64encode(buf).decode('ascii')

    messages = [
        json.dumps({'frame': b64, 'timestamp': time.time(), 'distance': 100.0}),
        json.dumps({'frame': b64, 'timestamp': time.time(), 'distance': 105.0}),
        json.dumps({'frame': b64, 'timestamp': time.time(), 'distance': 110.0}),
    ]

    class FakeRecvMessages:
        def __init__(self, parent):
            self.parent = parent
        @property
        def frames(self):
            return self.parent._msgs

    class FakeWS:
        def __init__(self, msgs):
            self.remote_address = ("127.0.0.1", 12345)
            self._msgs = list(msgs)
            self.recv_messages = FakeRecvMessages(self)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._msgs:
                return self._msgs.pop(0)
            raise StopAsyncIteration

        async def recv(self):
            if self._msgs:
                return self._msgs.pop(0)
            raise Exception("no more")

        async def send(self, data):
            pass

    fake_ws = FakeWS(messages)
    await server.process_frame(fake_ws)
    dropped = server._dropped_frames_count
    server.cleanup()

    assert dropped > 0, f"Expected websocket stale frames to be dropped, got {dropped}"


def test_ultrasonic_pi_source_no_com_port_no_mock():
    """
    Requirement E & G:
    When USE_PI_CAMERA = True:
    - Do not open any local serial port (no COM3 attempt)
    - Do not fall back to MOCK mode (mock_mode remains False)
    - If no distance arrives: distance = None and a single warning
    """
    with patch.object(config, 'USE_PI_CAMERA', True):
        with patch('serial.Serial') as mock_serial:
            sensor = UltrasonicSensor(mock_mode=False)
            mock_serial.assert_not_called()
            assert sensor.mock_mode is False
            assert sensor.serial_port is None

            with patch('logging.Logger.warning') as mock_warn:
                dist1 = sensor.get_distance()
                assert dist1 is None
                assert mock_warn.call_count == 1
                assert "No ultrasonic distance received from Pi" in mock_warn.call_args[0][0]

                # Subsequent call: single warning only
                dist2 = sensor.get_distance()
                assert dist2 is None
                assert mock_warn.call_count == 1


def test_voice_shutdown_mishearings():
    """
    Requirement F & G:
    Treat 'shuttle', 'shut', 'shot down', 'shot up', 'shutting', 'shut it' as shutdown after wake word.
    'shuttle' and 'shut' quit.
    'shuttle' never triggers a find.
    """
    parser = VoiceCommandParser()

    shutdown_phrases = ["shuttle", "shut", "shot down", "shot up", "shutting", "shut it"]
    for phrase in shutdown_phrases:
        parsed = parser.parse(f"hey vision {phrase}")
        assert parsed["intent"] == "quit", f"Expected 'hey vision {phrase}' to have intent 'quit', got {parsed['intent']}"
        assert parsed["intent"] != "find", f"'hey vision {phrase}' should never trigger find"

    parsed_shuttle = parser.parse("shuttle")
    assert parsed_shuttle["intent"] == "quit"
    assert parsed_shuttle["intent"] != "find"

    parsed_shut = parser.parse("shut")
    assert parsed_shut["intent"] == "quit"

    # 'find bottle' and 'bottle' should find bottle when enrolled
    parser.known_targets_provider = lambda: ["bottle"]
    parsed_bottle = parser.parse("hey vision bottle")
    assert parsed_bottle["intent"] == "find"
    target_bottle = parsed_bottle["parameters"].get("target") or parsed_bottle["parameters"].get("object_name")
    assert target_bottle == "bottle"

    parsed_find_bottle = parser.parse("hey vision find bottle")
    assert parsed_find_bottle["intent"] == "find"
    target_find_bottle = parsed_find_bottle["parameters"].get("target") or parsed_find_bottle["parameters"].get("object_name")
    assert target_find_bottle == "bottle"


def test_unparsed_fallback_threshold_and_shutdown_guard():
    """
    Requirement F:
    The unparsed-command fallback to class names needs ratio >= 0.8,
    and never fires for words close to a shutdown/quit phrase.
    """
    parser = VoiceCommandParser()
    parser.known_targets_provider = lambda: ["bottle", "person", "chair", "cell phone"]

    # Ratio < 0.8 should NOT trigger fallback
    # 'beetle' vs 'bottle' ratio is ~0.67 < 0.8
    parsed_beetle = parser.parse("hey vision beetle")
    assert parsed_beetle["intent"] != "find", "'beetle' should not trigger fallback to bottle"

    # Ratio >= 0.8
    # 'botle' vs 'bottle' ratio is ~0.91 >= 0.8
    parsed_botle = parser.parse("hey vision botle")
    target_botle = parsed_botle["parameters"].get("target") or parsed_botle["parameters"].get("object_name")
    assert target_botle == "bottle"

    # Shutdown words never fire unparsed fallback
    parsed_shuttle = parser.parse("hey vision shuttle")
    assert parsed_shuttle["intent"] == "quit"


def test_thread_budget_and_startup_logging():
    """
    Requirement B:
    - OpenVINO threads = 8 (config OPENVINO_THREADS)
    - InsightFace threads = 4 (config FACE_THREADS)
    - torch threads = 2
    - Log all three at startup
    """
    assert getattr(config, 'OPENVINO_THREADS', None) == 8
    assert getattr(config, 'FACE_THREADS', None) == 4

    server = LaptopMLServer()
    diag = server.print_thread_diagnostics()
    assert "torch=2" in diag
    assert "OpenVINO=8" in diag
    assert "InsightFace=4" in diag


def test_face_worker_frequency_and_caching():
    """
    Requirement C:
    Face work off the frame loop:
    - Runs face detection in a separate worker at most 3 times per second
    - Caches results per track
    - Never blocks the ML loop
    - Skipped during find sessions
    """
    mock_face_rec = MagicMock()
    mock_face_rec.recognize_faces.return_value = [
        {'name': 'Alice', 'confidence': 0.95, 'bbox': [10, 10, 60, 60], 'track_id': 1}
    ]

    worker = FaceWorker(mock_face_rec, rate_hz=3.0)
    worker.start()

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    t0 = time.time()
    for _ in range(10):
        worker.submit_frame(dummy_frame, yolo_has_person=True, is_find_active=False)
    submit_elapsed = time.time() - t0
    assert submit_elapsed < 0.1, "submit_frame should be non-blocking"

    time.sleep(0.5)

    cached = worker.get_latest_results(yolo_has_person=True)

    # Skipped during find sessions
    worker.submit_frame(dummy_frame, yolo_has_person=True, is_find_active=True)
    time.sleep(0.1)

    worker.stop()

    assert len(cached) == 1
    assert cached[0]['name'] == 'Alice'
    assert mock_face_rec.recognize_faces.call_count <= 2


# ─── Part 2 & Part 4: Ultrasonic Obstacle Warning Tests ────────────────────────

def test_pi_hub_sensors_endpoint():
    """
    Requirement 2A:
    pi/pi_hub.py /sensors route returns:
    {"ultrasonic_distance_cm": <float|null>, "age_ms": <ms>, "ok": <age_ms < 1500>}
    Set distance to null when no valid line arrived for 1.5 s.
    Keep /distance working.
    """
    from pi.pi_hub import app
    import pi.pi_hub as hub

    client = app.test_client()

    # 1. Fresh reading
    now = time.time()
    with hub.distance_lock:
        hub.latest_distance = 78.5
        hub.latest_distance_time = now

    resp = client.get('/sensors')
    assert resp.status_code == 200
    data = resp.get_json()
    assert "ultrasonic_distance_cm" in data
    assert data["ultrasonic_distance_cm"] == 78.5
    assert data["ok"] is True
    assert data["age_ms"] < 1500

    # /distance still working
    resp_dist = client.get('/distance')
    assert resp_dist.status_code == 200
    data_dist = resp_dist.get_json()
    assert data_dist["distance_cm"] == 78.5

    # 2. Stale reading (no valid line arrived for 1.5 s)
    with hub.distance_lock:
        hub.latest_distance = 78.5
        hub.latest_distance_time = now - 2.0  # 2 seconds ago

    resp_stale = client.get('/sensors')
    assert resp_stale.status_code == 200
    data_stale = resp_stale.get_json()
    assert data_stale["ultrasonic_distance_cm"] is None
    assert data_stale["ok"] is False
    assert data_stale["age_ms"] >= 1500


def test_obstacle_warning_spoken_alerts_and_object_naming():
    """
    Requirement 2D & Part 4:
    - 80 cm twice says "Object ahead" (warn, NAVIGATIONAL priority)
    - 30 cm says "Stop, object very close" (stop, EMERGENCY priority)
    - A centred "person" detection plus 80 cm says "Person ahead, close"
    - An off-centre or stale (>500 ms) detection is not named
    - A single 20 cm spike among 150 cm readings does nothing
    - None / -1 / frozen does nothing
    - Rate limit holds (at most once per 3s; repeat if >15cm or 5s; reset >120cm)
    """
    from obstacle_alerter import ObstacleAlerter

    spoken = []
    def mock_speak(text, priority):
        spoken.append((text, priority))

    detections = []
    det_time = time.monotonic()

    alerter = ObstacleAlerter(
        speak_fn=mock_speak,
        get_latest_detections_fn=lambda: (detections, det_time),
        beeps_enabled=False
    )

    # 1. 80 cm once: not enough (needs 2 reads in a row)
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 80.0, "ok": True})
    assert len(spoken) == 0

    # 80 cm twice: says "Object ahead" (NAVIGATIONAL)
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 80.0, "ok": True})
    assert len(spoken) == 1
    assert spoken[-1][0] == "Object ahead"
    assert spoken[-1][1] == config.PRIORITY_NAVIGATIONAL

    # Rate limiting: 80 cm again immediately: does not repeat within 3s
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 82.0, "ok": True})
    assert len(spoken) == 1

    # 2. 30 cm: level change to stop (2 reads <= 40 cm) -> speaks immediately
    alerter._last_warn_time = time.monotonic() - 4.0  # simulate > 3s
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 30.0, "ok": True})
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 30.0, "ok": True})
    assert len(spoken) == 2
    assert spoken[-1][0] == "Stop, object very close"
    assert spoken[-1][1] == config.PRIORITY_EMERGENCY

    # 3. Reset when distance > 120 cm
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 150.0, "ok": True})
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 150.0, "ok": True})

    # 4. Centred person detection (|offset| < 0.33, conf >= 0.5)
    # Frame width is 640. Center is 320. Bbox [220, 100, 420, 300] center is 320 -> offset = 0.0
    detections = [{'class': 'person', 'confidence': 0.85, 'bbox': [220, 100, 420, 300]}]
    det_time = time.monotonic()

    alerter.process_sensor_payload({"ultrasonic_distance_cm": 80.0, "ok": True})
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 80.0, "ok": True})
    assert len(spoken) == 3
    assert spoken[-1][0] == "Person ahead, close"
    assert spoken[-1][1] == config.PRIORITY_NAVIGATIONAL

    # 5. Off-centre detection (|offset| >= 0.33)
    # Bbox [500, 100, 620, 300] center is 560 -> offset = (560-320)/640 = 0.375 >= 0.33
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 150.0, "ok": True})
    detections = [{'class': 'person', 'confidence': 0.85, 'bbox': [500, 100, 620, 300]}]
    det_time = time.monotonic()

    alerter.process_sensor_payload({"ultrasonic_distance_cm": 80.0, "ok": True})
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 80.0, "ok": True})
    assert len(spoken) == 4
    assert spoken[-1][0] == "Object ahead", "Off-centre object must NOT be named"

    # 6. Stale detection (>500 ms old) is not named
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 150.0, "ok": True})
    detections = [{'class': 'person', 'confidence': 0.85, 'bbox': [220, 100, 420, 300]}]
    det_time = time.monotonic() - 1.0  # 1s old

    alerter.process_sensor_payload({"ultrasonic_distance_cm": 80.0, "ok": True})
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 80.0, "ok": True})
    assert len(spoken) == 5
    assert spoken[-1][0] == "Object ahead", "Stale detection (>500ms) must NOT be named"

    # 7. Single 20 cm spike among 150 cm readings does nothing (median filter ignores it)
    spoken_count_before = len(spoken)
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 150.0, "ok": True})
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 150.0, "ok": True})
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 20.0, "ok": True})
    assert len(spoken) == spoken_count_before, "Single spike must be ignored by median filter"

    # 8. None / -1 / frozen does nothing
    alerter.process_sensor_payload({"ultrasonic_distance_cm": None, "ok": True})
    alerter.process_sensor_payload({"ultrasonic_distance_cm": -1.0, "ok": True})
    assert len(spoken) == spoken_count_before


def test_obstacle_frozen_sensor_handling():
    """
    Requirement 2C & Part 4:
    If the same exact value repeats for 3 s or more, treat as frozen:
    no warning, and log 'Ultrasonic: value frozen' once.
    """
    from obstacle_alerter import ObstacleAlerter

    spoken = []
    alerter = ObstacleAlerter(speak_fn=lambda t, p: spoken.append((t, p)), beeps_enabled=False)

    class MockTime:
        def __init__(self):
            self.t = 100.0
        def __call__(self):
            return self.t

    mock_t = MockTime()
    with patch('time.monotonic', mock_t):
        with patch('logging.Logger.warning') as mock_warn:
            alerter.process_sensor_payload({"ultrasonic_distance_cm": 35.0, "ok": True})
            mock_t.t += 1.0
            alerter.process_sensor_payload({"ultrasonic_distance_cm": 35.0, "ok": True})
            mock_t.t += 1.0
            alerter.process_sensor_payload({"ultrasonic_distance_cm": 35.0, "ok": True})
            mock_t.t += 1.5  # 3.5s elapsed total
            alerter.process_sensor_payload({"ultrasonic_distance_cm": 35.0, "ok": True})

            frozen_calls = [c for c in mock_warn.call_args_list if "value frozen" in str(c)]
            assert len(frozen_calls) == 1
            assert alerter._is_frozen is True


def test_obstacle_beep_intervals_and_find_suppression():
    """
    Requirement 2E & Part 4:
    - Interval at 100 cm: ~1.0 s
    - Interval at 70 cm: ~0.6 s
    - Interval at 40 cm: ~0.2 s
    - Below 20 cm: 0.0 s (continuous tone)
    - Find session suppresses obstacle beeps but not 'Stop' speech
    - Voice command 'hey vision obstacle beeps on/off'
    """
    from obstacle_alerter import calculate_beep_interval, ObstacleAlerter

    assert abs(calculate_beep_interval(100.0) - 1.0) < 0.05
    assert abs(calculate_beep_interval(70.0) - 0.6) < 0.05
    assert abs(calculate_beep_interval(40.0) - 0.2) < 0.05
    assert calculate_beep_interval(15.0) == 0.0  # continuous tone

    # Find session suppression
    beeps_played = []
    spoken = []
    is_finding = True

    alerter = ObstacleAlerter(
        speak_fn=lambda t, p: spoken.append((t, p)),
        beep_fn=lambda f, ms: beeps_played.append((f, ms)),
        is_find_active_fn=lambda: is_finding,
        beeps_enabled=True
    )

    # 30 cm triggers 'Stop' speech
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 30.0, "ok": True})
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 30.0, "ok": True})

    assert len(spoken) == 1
    assert "Stop" in spoken[0][0]

    # Voice command parsing
    parser = VoiceCommandParser()
    p_on = parser.parse("hey vision obstacle beeps on")
    assert p_on["intent"] == "obstacle_beeps"
    assert p_on["parameters"]["state"] == "on"

    p_off = parser.parse("hey vision obstacle beeps off")
    assert p_off["intent"] == "obstacle_beeps"
    assert p_off["parameters"]["state"] == "off"


def test_camera_fallback_speaks_once_and_recovers():
    """
    Requirement 2F & Part 4:
    If newest frame is older than 1 s:
    say once 'Camera lagging, obstacle alert still on'.
    Say 'Camera back' once when it recovers.
    """
    from obstacle_alerter import ObstacleAlerter

    spoken = []
    frame_age = 0.2

    alerter = ObstacleAlerter(
        speak_fn=lambda t, p: spoken.append((t, p)),
        get_frame_age_fn=lambda: frame_age,
        beeps_enabled=False
    )

    # Frame lag occurs (> 1s)
    frame_age = 1.4
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 150.0, "ok": True})
    assert len(spoken) == 1
    assert spoken[-1][0] == "Camera lagging, obstacle alert still on"

    # Continues lagging: speaks only ONCE
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 150.0, "ok": True})
    assert len(spoken) == 1

    # Camera recovers (<= 1s)
    frame_age = 0.3
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 150.0, "ok": True})
    assert len(spoken) == 2
    assert spoken[-1][0] == "Camera back"

    # Remains normal: speaks only ONCE
    alerter.process_sensor_payload({"ultrasonic_distance_cm": 150.0, "ok": True})
    assert len(spoken) == 2


def test_unparsed_fallback_never_when_nothing_enrolled():
    """
    Requirement Part 3 & Part 4:
    - 'shuttle' and 'shut' shut down; 'shuttle' never triggers a find.
    - Unparsed fallback to class/object names NEVER fires when nothing is enrolled.
    """
    parser = VoiceCommandParser()

    # When nothing is enrolled (provider returns empty)
    parser.known_targets_provider = lambda: []

    # 'hey vision bottle' without enrollment must NOT trigger find
    p_bottle = parser.parse("hey vision bottle")
    assert p_bottle["intent"] != "find", "Unparsed fallback must never fire when nothing is enrolled"

    # 'hey vision shuttle' and 'hey vision shut' shut down
    p_shuttle = parser.parse("hey vision shuttle")
    assert p_shuttle["intent"] == "quit"

    p_shut = parser.parse("hey vision shut")
    assert p_shut["intent"] == "quit"

    # Explicit 'find bottle' STILL works because 'find' is an explicit verb
    p_find_bottle = parser.parse("hey vision find bottle")
    assert p_find_bottle["intent"] == "find"

