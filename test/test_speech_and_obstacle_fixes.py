"""
Tests for duplicate speech prevention, obstacle warning fixes, and spatial wording.

Covers:
1. When CLIENT_SPEAKS is False, client never calls speak.
2. A None distance or mock ultrasonic mode never generates fake obstacle warnings,
   while real sensor behavior remains unchanged.
3. Spoken messages never contain vertical zone words ("bottom", "top", "middle"),
   and "Person close bottom" becomes "Person close ahead".
4. rpi_client navigation logging is throttled (only logs when message changes or 10s elapsed).
"""

import os
import sys
import time
import re
from unittest.mock import MagicMock
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from spatial_navigator import SpatialNavigator
from obstacle_fusion import ObstacleFusion
from ultrasonic_sensor import UltrasonicSensor
from wireless.rpi_client import RaspberryPiClient
from wireless.laptop_server import LaptopMLServer


# ============================================================================
# Requirement 1 & 5: CLIENT_SPEAKS = False -> client never calls speak
# ============================================================================

def test_client_speaks_false_never_calls_speak(monkeypatch):
    """With CLIENT_SPEAKS False, rpi_client never calls its speak function."""
    monkeypatch.setattr(config, "CLIENT_SPEAKS", False)
    client = RaspberryPiClient(client_speaks=False)
    client.speak = MagicMock()

    result = {'navigation': 'Person close ahead'}
    client.process_results(result)
    client.process_results(result)
    client.process_results({'navigation': 'Move left'})

    assert client.speak.call_count == 0


def test_client_speaks_false_blocks_direct_speak(monkeypatch):
    """With CLIENT_SPEAKS False, calling speak directly does not queue audio."""
    monkeypatch.setattr(config, "CLIENT_SPEAKS", False)
    client = RaspberryPiClient(client_speaks=False)

    client.speak("Test message")
    assert client.audio_queue.empty() is True


def test_client_speaks_true_calls_speak(monkeypatch):
    """With CLIENT_SPEAKS True, rpi_client calls its speak function on navigation guidance."""
    monkeypatch.setattr(config, "CLIENT_SPEAKS", True)
    client = RaspberryPiClient(client_speaks=True)
    client.speak = MagicMock()

    result = {'navigation': 'Person close ahead'}
    client.process_results(result)

    assert client.speak.call_count == 1
    client.speak.assert_called_with('Person close ahead')


# ============================================================================
# Requirement 2 & 5: None distance and mock mode never generate obstacle warnings
# ============================================================================

def test_rpi_client_sends_none_when_esp32_missing():
    """When ESP32 serial is not connected, rpi_client sets ultrasonic_distance = None."""
    client = RaspberryPiClient()
    client.esp32_serial = None
    client.read_esp32_sensors()

    assert client.ultrasonic_distance is None


def _create_test_server():
    server = LaptopMLServer(port=8089)
    server.audio_enabled = False

    class EmptyDetector:
        def detect(self, frame):
            return []
    server.object_detector = EmptyDetector()

    class EmptyTracker:
        def update_tracking(self, detections):
            return []
    server.item_tracker = EmptyTracker()

    class EmptyMemory:
        def recognize_object(self, frame, bbox, yolo_class):
            return None
    server.object_memory = EmptyMemory()

    class EmptyFaceRecognizer:
        def recognize_faces(self, frame):
            return []
    server.face_recognizer = EmptyFaceRecognizer()

    server.spatial_navigator = SpatialNavigator(frame_width=640, frame_height=480)
    server.find_guide = None
    return server


def test_none_distance_never_produces_obstacle_warning_in_laptop_server():
    """When ultrasonic_distance is None, laptop_server does not generate 'Obstacle N cm ahead'."""
    server = _create_test_server()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = server.process_ml(dummy_frame, ultrasonic_distance=None, button_pressed=False)

    assert "Obstacle" not in result['navigation']
    assert result['navigation'] == ""


def test_mock_mode_never_produces_obstacle_warning_in_laptop_server(monkeypatch):
    """When ULTRASONIC_MOCK_MODE is True on server, mock distance does not generate 'Obstacle N cm ahead'."""
    monkeypatch.setattr(config, "ULTRASONIC_MOCK_MODE", True)
    server = _create_test_server()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Even if client or mock sent 25 cm, in mock mode it must never generate obstacle warning
    result = server.process_ml(dummy_frame, ultrasonic_distance=25.0, button_pressed=False)

    assert "Obstacle" not in result['navigation']
    assert result['navigation'] == ""


def test_real_sensor_generates_obstacle_warning_when_not_mock(monkeypatch):
    """When ULTRASONIC_MOCK_MODE is False and real sensor reading < 50cm arrives, warning is generated."""
    monkeypatch.setattr(config, "ULTRASONIC_MOCK_MODE", False)
    server = _create_test_server()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = server.process_ml(dummy_frame, ultrasonic_distance=35.0, button_pressed=False)

    assert result['navigation'] == "Obstacle 35 cm ahead"


def test_obstacle_fusion_none_distance_uses_camera_only():
    """In ObstacleFusion, None distance treats reading as absent and uses camera estimation only."""
    fusion = ObstacleFusion(ultrasonic_sensor=None)
    # A small distant object in center of frame (area ratio < 0.05 -> est distance 4.0m)
    objects = [{
        'bbox': (300, 220, 340, 260),
        'class_name': 'chair',
        'confidence': 0.85
    }]

    enhanced = fusion.get_object_with_distance(objects, ultrasonic_distance=None)
    assert enhanced[0]['distance_source'] == 'camera_estimate'
    assert enhanced[0]['precise_distance'] >= 2.0

    warnings = fusion.generate_obstacle_warnings(objects, ultrasonic_distance=None)
    assert len(warnings) == 0  # 4m distance is not critical (<1.5m), so no warning


def test_obstacle_fusion_mock_mode_never_generates_warnings(monkeypatch):
    """In ObstacleFusion, mock ultrasonic sensor never generates fake obstacle warnings."""
    monkeypatch.setattr(config, "ULTRASONIC_MOCK_MODE", True)
    sensor = UltrasonicSensor(mock_mode=True)
    fusion = ObstacleFusion(ultrasonic_sensor=sensor)

    # Distant object
    objects = [{
        'bbox': (300, 220, 340, 260),
        'class_name': 'person',
        'confidence': 0.90
    }]

    warnings = fusion.generate_obstacle_warnings(objects)
    assert len(warnings) == 0


def test_ultrasonic_sensor_mock_mode_never_reports_obstacle_close():
    """UltrasonicSensor in mock mode never reports obstacle close."""
    sensor = UltrasonicSensor(mock_mode=True)
    assert sensor.is_obstacle_close() is False
    status = sensor.get_obstacle_status()
    assert status['warning_level'] == 'clear'
    assert status['below_threshold'] is False


# ============================================================================
# Requirement 3 & 5: Spoken messages never contain "bottom", "top" or "middle"
# ============================================================================

def test_person_close_bottom_becomes_person_close_ahead():
    """'Person close bottom' becomes 'Person close ahead'."""
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    msg = nav._create_guidance_message(class_name='person', zone='bottom', distance='close')
    assert msg == "Person close ahead"


def test_spoken_messages_never_contain_vertical_words():
    """All 9 zones and distance variations produce messages without 'bottom', 'top', or 'middle'."""
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    all_zones = [
        'top-left', 'top', 'top-right',
        'left', 'center', 'middle', 'right',
        'bottom-left', 'bottom', 'bottom-right'
    ]
    distances = ['very-close', 'close', 'medium', 'far']

    for zone in all_zones:
        for dist in distances:
            msg = nav._create_guidance_message('person', zone, dist)
            assert not re.search(r'\b(bottom|top|middle)\b', msg, re.IGNORECASE), (
                f"Forbidden vertical zone word found in message '{msg}' for zone '{zone}' dist '{dist}'"
            )
            # Must contain one of horizontal words
            assert any(h in msg.lower() for h in ['left', 'ahead', 'right']), (
                f"No horizontal direction found in message '{msg}'"
            )


def test_generate_guidance_all_zones_free_of_vertical_words():
    """Full generate_guidance output across multiple bboxes contains no vertical words."""
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    detections = [
        {'bbox': (50, 400, 150, 470), 'class_name': 'chair', 'confidence': 0.9},   # bottom-left
        {'bbox': (300, 400, 400, 470), 'class_name': 'person', 'confidence': 0.9},  # bottom
        {'bbox': (500, 400, 600, 470), 'class_name': 'couch', 'confidence': 0.9},   # bottom-right
        {'bbox': (300, 50, 400, 150), 'class_name': 'person', 'confidence': 0.9},   # top
    ]

    guidance = nav.generate_guidance(detections)
    assert len(guidance) > 0
    for g in guidance:
        msg = g['message']
        match = re.search(r'\b(bottom|top|middle)\b', msg, re.IGNORECASE)
        assert match is None, f"Forbidden vertical word '{match.group(0)}' in guidance message: '{g['message']}'"


# ============================================================================
# Requirement 4: rpi_client logging cooldown
# ============================================================================

def test_rpi_client_logging_throttled(monkeypatch):
    """rpi_client only logs 'Navigation: <msg>' when message changes or 10s pass."""
    client = RaspberryPiClient()
    logged_messages = []

    # Intercept logger.info
    import logging
    target_logger = logging.getLogger("wireless.rpi_client")
    original_info = target_logger.info

    def mock_info(msg, *args, **kwargs):
        logged_messages.append(msg)
        original_info(msg, *args, **kwargs)

    monkeypatch.setattr(target_logger, "info", mock_info)

    # Frame 1 at t=100: new message -> should log
    monkeypatch.setattr(time, "time", lambda: 100.0)
    client.process_results({'navigation': 'Move left'})
    assert len(logged_messages) == 1
    assert "Navigation: Move left" in logged_messages[0]

    # Frame 2 at t=101: same message within 10s -> should NOT log
    monkeypatch.setattr(time, "time", lambda: 101.0)
    client.process_results({'navigation': 'Move left'})
    assert len(logged_messages) == 1

    # Frame 3 at t=109.9: same message still < 10s -> should NOT log
    monkeypatch.setattr(time, "time", lambda: 109.9)
    client.process_results({'navigation': 'Move left'})
    assert len(logged_messages) == 1

    # Frame 4 at t=110.1: 10s passed -> should log again
    monkeypatch.setattr(time, "time", lambda: 110.1)
    client.process_results({'navigation': 'Move left'})
    assert len(logged_messages) == 2

    # Frame 5 at t=111.0: message changed -> should log immediately
    monkeypatch.setattr(time, "time", lambda: 111.0)
    client.process_results({'navigation': 'Move right'})
    assert len(logged_messages) == 3
    assert "Navigation: Move right" in logged_messages[2]
