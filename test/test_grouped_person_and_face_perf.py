"""
Test Grouped Person Announcements, Face Recognition Performance, Torch Threads, and Voice Parsing
Verifies:
1. 2 people produce one combined announcement and no repeats within 6 s;
   Never speak separate "Person ahead", "Person left", and "Move left" within the same 6 s;
   During active find session, skip all person messages unless closer than 50 cm.
2. Face recognition cost: IoU tracker preserves identity, recognises a given track
   at most once per second (and once at first sight), only 2 largest faces per frame,
   skips recognition entirely when frame time > 150 ms, resumes when recovered.
3. Torch threads: torch.set_num_threads(2), torch.set_num_interop_threads(1),
   startup log shows torch=2, OpenVINO config uses LATENCY and num_streams=1.
4. Voice parse: strips filler words so "do you remember a person", "remember this person",
   and "remember me" all start the face flow.
"""

import sys
import os
import io
import time
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from spatial_navigator import SpatialNavigator
from face_recognizer import FaceRecognizer
from voice_command_system import VoiceCommandParser, VoiceCommandSystem, strip_filler_words
from wireless.laptop_server import LaptopMLServer, print_thread_diagnostics, _OV_CONFIG


class DummyAudio:
    """Mock audio feedback for testing announcements."""
    def __init__(self):
        self.messages = []
        self.social_messages = []
        self.nav_messages = []
        self.emergency_messages = []

    def speak_social(self, text, priority=None):
        self.messages.append(text)
        self.social_messages.append(text)

    def speak_navigational(self, text, priority=None):
        self.messages.append(text)
        self.nav_messages.append(text)

    def speak_emergency(self, text, priority=None):
        self.messages.append(text)
        self.emergency_messages.append(text)

    def speak_command_reply(self, text, priority=None):
        self.messages.append(text)


def _setup_mock_server():
    """Create a LaptopMLServer instance with mocked components for integration testing."""
    server = LaptopMLServer()
    server.audio = DummyAudio()
    server.audio_enabled = True
    server.current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    server.object_memory = MagicMock()
    server.object_memory.list_remembered_objects = MagicMock(return_value=[])
    server.object_memory.get_object_aliases = MagicMock(return_value=[])
    server.object_memory.get_object_location = MagicMock(return_value=None)
    server.item_tracker = MagicMock()
    server.item_tracker.query_item_location = MagicMock(return_value=None)
    server.item_tracker.update_tracking = MagicMock(return_value=[])
    server.object_detector = MagicMock()
    server.object_detector.detect = MagicMock(return_value=[])
    server.face_recognizer = MagicMock()
    server.face_recognizer.recognize_faces = MagicMock(return_value=[])
    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server.voice_system.open_dictation_window = MagicMock()
    server.voice_system.open_yes_no_window = MagicMock()
    server._register_voice_handlers()
    return server


# ─── 1. Grouped Person Announcements (Requirement 1) ──────────────────────────

def test_two_people_produce_one_combined_announcement_and_no_repeats_within_6s():
    """2 people produce one combined announcement and no repeats within 6 s."""
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    now = 1000.0

    # Two people in view, both ahead
    detections = [
        {'class': 'person', 'bbox': [200, 50, 320, 400], 'confidence': 0.9},
        {'class': 'person', 'bbox': [320, 60, 440, 410], 'confidence': 0.85}
    ]

    guidance = nav.generate_guidance(detections, current_time=now)
    person_items = [g for g in guidance if g.get('type') == 'person']

    # Exactly 1 combined person announcement
    assert len(person_items) == 1
    combined_msg = person_items[0]['message']
    assert combined_msg == "2 people ahead"

    # Should speak on first occurrence
    assert nav.should_speak_guidance(person_items[0], current_time=now) is True
    nav.mark_guidance_spoken(person_items[0], current_time=now)

    # Within 6 s (e.g. at 3 s later): NO repeats
    guidance_3s = nav.generate_guidance(detections, current_time=now + 3.0)
    person_items_3s = [g for g in guidance_3s if g.get('type') == 'person']
    assert len(person_items_3s) == 1
    assert nav.should_speak_guidance(person_items_3s[0], current_time=now + 3.0) is False

    # Never speak separate "Move left" within the same 6 s
    nav_item = {'type': 'navigation', 'action': 'Move left', 'message': 'Move left'}
    assert nav.should_speak_guidance(nav_item, current_time=now + 3.0) is False

    # After 6.5 s, count changes to 1 person close left -> should speak
    detections_single = [
        {'class': 'person', 'bbox': [10, 50, 190, 460], 'confidence': 0.95}
    ]
    guidance_6s = nav.generate_guidance(detections_single, current_time=now + 6.5)
    person_items_6s = [g for g in guidance_6s if g.get('type') == 'person']
    assert len(person_items_6s) == 1
    assert person_items_6s[0]['message'] == "Person close left"
    assert nav.should_speak_guidance(person_items_6s[0], current_time=now + 6.5) is True


def test_person_announcements_skipped_during_find_session_unless_closer_than_50cm():
    """During active find session, skip all person messages unless closer than 50 cm (< 0.5 m)."""
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    now = 500.0

    detections = [
        {'class': 'person', 'bbox': [200, 50, 320, 400], 'confidence': 0.9}
    ]
    guidance = nav.generate_guidance(detections, current_time=now)
    person_item = [g for g in guidance if g.get('type') == 'person'][0]

    # Outside find session: should speak
    assert nav.should_speak_guidance(person_item, current_time=now, is_find_active=False) is True

    # During find session with obstacle > 50 cm (e.g. 1.2 m): SKIPPED
    assert nav.should_speak_guidance(
        person_item, current_time=now, is_find_active=True, ultrasonic_distance=1.2
    ) is False

    # During find session with obstacle closer than 50 cm (e.g. 0.4 m): ALLOWED
    assert nav.should_speak_guidance(
        person_item, current_time=now, is_find_active=True, ultrasonic_distance=0.4
    ) is True


# ─── 2. Face Recognition Cost & IoU Tracker (Requirement 2) ───────────────────

def test_face_recognition_iou_tracking_and_once_per_second_rate():
    """IoU tracker keeps identity; recognises at most once per second and once at first sight."""
    recognizer = FaceRecognizer()
    recognizer.insightface_available = True

    match_log = []
    def mock_match(emb):
        match_log.append((emb[0], time.time()))
        return f"Person_{int(emb[0])}", 0.35, 0.25

    recognizer._match_face = mock_match
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # First sight at t = 10.0s: Face 1 and Face 2
    recognizer.detect_faces = MagicMock(return_value=[
        {'bbox': [50, 50, 150, 150], 'embedding': np.array([1.0]), 'confidence': 0.92},
        {'bbox': [300, 100, 420, 220], 'embedding': np.array([2.0]), 'confidence': 0.88}
    ])
    res_t0 = recognizer.recognize_faces(dummy_frame, current_time=10.0)
    assert len(res_t0) == 2
    assert {r['name'] for r in res_t0} == {"Person_1", "Person_2"}
    assert len(match_log) == 2  # recognized once at first sight

    # Frame at t = 10.4s (< 1s later): faces moved slightly (IoU matches)
    match_log.clear()
    recognizer.detect_faces = MagicMock(return_value=[
        {'bbox': [55, 52, 155, 152], 'embedding': np.array([1.0]), 'confidence': 0.92},
        {'bbox': [305, 102, 425, 222], 'embedding': np.array([2.0]), 'confidence': 0.88}
    ])
    res_t1 = recognizer.recognize_faces(dummy_frame, current_time=10.4)
    assert len(res_t1) == 2
    assert {r['name'] for r in res_t1} == {"Person_1", "Person_2"}
    assert len(match_log) == 0, "Must not run _match_face within 1 second of last recognition"

    # Frame at t = 11.2s (> 1s later): recognition runs again
    match_log.clear()
    res_t2 = recognizer.recognize_faces(dummy_frame, current_time=11.2)
    assert len(res_t2) == 2
    assert len(match_log) == 2, "Must run _match_face after 1 second has elapsed"


def test_face_recognition_only_2_largest_faces_per_frame():
    """Only the 2 largest faces per frame are recognized; smaller ones are dropped."""
    recognizer = FaceRecognizer()
    recognizer.insightface_available = True
    recognizer._match_face = lambda emb: (f"Face_{int(emb[0])}", 0.3, 0.2)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 4 faces in frame with different sizes:
    # Face 1: area = 80 * 80 = 6,400
    # Face 2: area = 200 * 200 = 40,000 (largest)
    # Face 3: area = 50 * 50 = 2,500
    # Face 4: area = 150 * 150 = 22,500 (2nd largest)
    recognizer.detect_faces = MagicMock(return_value=[
        {'bbox': [10, 10, 90, 90], 'embedding': np.array([1.0]), 'confidence': 0.9},
        {'bbox': [100, 100, 300, 300], 'embedding': np.array([2.0]), 'confidence': 0.9},
        {'bbox': [320, 320, 370, 370], 'embedding': np.array([3.0]), 'confidence': 0.9},
        {'bbox': [400, 100, 550, 250], 'embedding': np.array([4.0]), 'confidence': 0.9},
    ])

    results = recognizer.recognize_faces(dummy_frame, current_time=1.0)
    assert len(results) == 2
    names = {r['name'] for r in results}
    assert names == {"Face_2", "Face_4"}, f"Expected 2 largest faces, got {names}"


def test_face_recognition_skips_when_frame_time_above_150ms_and_resumes():
    """Skip recognition entirely when frame time is > 150 ms, resume when recovered."""
    recognizer = FaceRecognizer()
    recognizer.insightface_available = True
    recognizer.detect_faces = MagicMock(return_value=[
        {'bbox': [100, 100, 200, 200], 'embedding': np.array([1.0]), 'confidence': 0.9}
    ])
    recognizer._match_face = MagicMock(return_value=("Alice", 0.3, 0.2))
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame time = 180 ms (> 150 ms): skip recognition entirely
    res_high = recognizer.recognize_faces(dummy_frame, current_time=1.0, frame_time_ms=180.0)
    assert res_high == []
    recognizer.detect_faces.assert_not_called()

    # Frame time recovers to 85 ms (<= 150 ms): resumes
    res_normal = recognizer.recognize_faces(dummy_frame, current_time=2.0, frame_time_ms=85.0)
    assert len(res_normal) == 1
    assert res_normal[0]['name'] == "Alice"


# ─── 3. Torch Threads & OpenVINO Runtime (Requirement 3) ───────────────────────

def test_torch_threads_and_openvino_config_log(capsys):
    """Torch threads configured to 2 and interop to 1; startup log displays torch=2."""
    assert torch.get_num_threads() == 2
    assert torch.get_num_interop_threads() == 1

    # Verify OpenVINO config dictionary
    assert _OV_CONFIG.get("PERFORMANCE_HINT") == "LATENCY"
    assert _OV_CONFIG.get("NUM_STREAMS") == 1 or _OV_CONFIG.get("NUM_STREAMS") == "1"

    # Verify print_thread_diagnostics outputs torch=2 and OpenVINO config
    print_thread_diagnostics()
    captured = capsys.readouterr()
    assert "torch=2" in captured.out
    assert "OpenVINO" in captured.out
    assert "LATENCY" in captured.out


# ─── 4. Voice Parse Filler Stripping & Face Flow (Requirement 4) ───────────────

@pytest.mark.parametrize("phrase", [
    "do you remember a person",
    "can you remember a person",
    "please remember a person",
    "remember this person",
    "remember that person",
    "remember me",
    "do you remember me",
    "can you remember my face",
])
def test_voice_parser_filler_words_trigger_enroll_face(phrase):
    """Stripping filler words allows variations to start enroll_face intent."""
    parser = VoiceCommandParser()
    parsed = parser.parse(phrase)
    assert parsed is not None
    assert parsed["intent"] == "enroll_face", f"Phrase '{phrase}' parsed as {parsed}"


def test_do_you_remember_a_person_starts_face_flow_on_server():
    """'do you remember a person' starts the face flow on the server."""
    server = _setup_mock_server()
    parser = VoiceCommandParser()

    command_text = "do you remember a person"
    parsed = parser.parse(command_text)
    assert parsed["intent"] == "enroll_face"

    handler = server.voice_system.command_callbacks[parsed["intent"]]
    handler(parsed["parameters"])

    # Face flow prompts for the person's name
    assert any("Who is this? Say their name" in msg for msg in server.audio.messages)
    # Opens 8 s dictation window
    server.voice_system.open_dictation_window.assert_called_once()
    kwargs = server.voice_system.open_dictation_window.call_args[1]
    timeout = kwargs.get("timeout") or kwargs.get("timeout_seconds")
    assert timeout == 8.0
