"""
Test suite for find session reliability and clean shutdown (Parts 1-5).
Tests:
1. With one remembered phone, a 0.65 match is high confidence and no question is asked;
   with two remembered phones it still asks.
2. A 1 s gap in detections during a find session produces no "can't see it" message.
3. Face recognition is skipped while a find session is active and restored after it ends.
4. stop() during an active beep session stops the beep thread within 1 s.
5. The shutdown handler calls find_guide.stop().
6. Beeps stop after arrival.
"""

import sys
import os
import time
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from find_guidance import FindGuide
from object_memory import ObjectMemory
from wireless.laptop_server import LaptopMLServer
from voice_command_system import VoiceCommandSystem


class DummyAudio:
    """Mock audio feedback capturing spoken messages."""
    def __init__(self):
        self.messages = []
        self.social_messages = []
        self.status_messages = []
        self.emergency_messages = []
        self.nav_messages = []
        self.command_replies = []
        self.is_speaking = False
        self.is_playing = False
        self.last_speech_end_time = 0.0

    def speak(self, text, priority=3, is_command_reply=False):
        self.messages.append((priority, text))
        if is_command_reply:
            self.command_replies.append(text)

    def speak_social(self, text, is_command_reply=False):
        self.messages.append((config.PRIORITY_SOCIAL, text))
        self.social_messages.append(text)
        if is_command_reply:
            self.command_replies.append(text)

    def speak_status(self, text, is_command_reply=False):
        self.messages.append((config.PRIORITY_STATUS, text))
        self.status_messages.append(text)
        if is_command_reply:
            self.command_replies.append(text)

    def speak_command_reply(self, text, priority=config.PRIORITY_SOCIAL):
        self.messages.append((priority, text))
        self.command_replies.append(text)
        self.social_messages.append(text)

    def speak_navigational(self, text, is_command_reply=False):
        self.messages.append((config.PRIORITY_NAVIGATIONAL, text))
        self.nav_messages.append(text)
        if is_command_reply:
            self.command_replies.append(text)

    def speak_emergency(self, text):
        self.messages.append((config.PRIORITY_EMERGENCY, text))
        self.emergency_messages.append(text)

    def is_tts_active(self):
        return False

    def flush_non_emergency(self):
        pass

    def stop(self):
        pass


# ─────────────────────────────────────────────────────────────
# 1. Match Tier Rule: 1 remembered object vs 2 remembered objects
# ─────────────────────────────────────────────────────────────

def test_single_remembered_phone_065_is_high_confidence():
    """
    With exactly one remembered phone, a 0.65 similarity match is high confidence
    (>= SINGLE_OBJECT_MATCH_MIN = 0.60) and no confirmation question is asked.
    """
    mock_db = MagicMock()
    # Exactly one remembered object of class 'cell phone'
    single_phone = {
        'id': 1,
        'custom_name': 'phone',
        'yolo_class': 'cell phone',
        'descriptors': np.ones((10, 32), dtype=np.uint8)
    }
    mock_db.get_remembered_objects_by_class.return_value = [single_phone]

    memory = ObjectMemory(db=mock_db)
    memory.matcher = MagicMock()
    memory.matcher.extract_features.return_value = {
        'descriptors': np.ones((10, 32), dtype=np.uint8),
        'feature_type': 'orb'
    }
    memory.matcher.find_best_match.return_value = (1, 'phone', 0.65)

    fake_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = memory.recognize_object(fake_img, (100, 100, 200, 200), 'cell phone')

    assert result is not None
    custom_name, confidence, tier = result
    assert custom_name == 'phone'
    assert confidence == 0.65
    assert tier == 'high', "With 1 remembered phone, 0.65 match must be 'high' tier"
    assert memory.pending_confirmation is None, "High confidence match must NOT ask confirmation"


def test_two_remembered_phones_065_asks_question():
    """
    With two remembered phones of the same class, a 0.65 similarity match is ambiguous
    (< OBJECT_MATCH_HIGH_CONFIDENCE = 0.75) and asks confirmation.
    """
    mock_db = MagicMock()
    # Two competing remembered objects of class 'cell phone'
    phone1 = {
        'id': 1,
        'custom_name': 'work phone',
        'yolo_class': 'cell phone',
        'descriptors': np.ones((10, 32), dtype=np.uint8)
    }
    phone2 = {
        'id': 2,
        'custom_name': 'personal phone',
        'yolo_class': 'cell phone',
        'descriptors': np.ones((10, 32), dtype=np.uint8)
    }
    mock_db.get_remembered_objects_by_class.return_value = [phone1, phone2]

    memory = ObjectMemory(db=mock_db)
    memory.matcher = MagicMock()
    memory.matcher.extract_features.return_value = {
        'descriptors': np.ones((10, 32), dtype=np.uint8),
        'feature_type': 'orb'
    }
    memory.matcher.find_best_match.return_value = (1, 'work phone', 0.65)

    fake_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = memory.recognize_object(fake_img, (100, 100, 200, 200), 'cell phone')

    assert result is not None
    custom_name, confidence, tier = result
    assert custom_name == 'work phone'
    assert confidence == 0.65
    assert tier == 'ambiguous', "With 2 remembered phones, 0.65 match must be 'ambiguous' tier"
    assert memory.pending_confirmation is not None, "Ambiguous match MUST set pending confirmation to ask"
    assert memory.pending_confirmation['custom_name'] == 'work phone'


# ─────────────────────────────────────────────────────────────
# 2. Hysteresis: 1s gap produces no "can't see it" message
# ─────────────────────────────────────────────────────────────

def test_1s_gap_in_detections_produces_no_cant_see_it():
    """
    During a find session, a 1 s gap in detections keeps the last known position
    (1.5 s hysteresis) and produces NO 'can't see it' message.
    """
    spoken = []
    guide = FindGuide(
        speak=lambda msg: spoken.append(msg),
        beep=lambda f, ms: None,
        frame_w=640,
        frame_h=480
    )

    guide.start("phone", "Last seen on desk")
    spoken.clear()

    # Frame 1 at t=0: detection visible
    det = [{
        "custom_name": "phone",
        "class": "cell phone",
        "confidence": 0.85,
        "bbox": (280, 180, 360, 300)
    }]
    state1 = guide.update(det, ultrasonic_cm=150.0)
    assert state1["visible"] is True

    # Fast forward clock by 1.0 s using monkeypatch
    initial_monotonic = time.monotonic()
    with patch("time.monotonic", side_effect=lambda: initial_monotonic + 1.0):
        # Frame 2 with 1.0s gap (empty detections)
        state2 = guide.update([], ultrasonic_cm=150.0)
        assert state2["visible"] is True, "Hysteresis (1.5s) must maintain target visible during 1s gap"
        assert not any("can't see it" in msg.lower() for msg in spoken), (
            f"Expected no 'can't see it' message during 1s gap, got: {spoken}"
        )

    guide.stop()


# ─────────────────────────────────────────────────────────────
# 3. Face recognition skipped during find and restored after
# ─────────────────────────────────────────────────────────────

def test_face_recognition_skipped_during_find_and_restored():
    """
    Face recognition is completely skipped while find session is active,
    and restored once the find session stops.
    """
    server = LaptopMLServer()
    server.audio = DummyAudio()
    server.audio_enabled = True

    # Mocks
    server.object_detector = MagicMock()
    server.object_detector.detect.return_value = []
    server.face_recognizer = MagicMock()
    server.face_recognizer.recognize_faces.return_value = []
    server.item_tracker = MagicMock()
    server.item_tracker.update.return_value = []
    server.object_memory = MagicMock()
    server.spatial_navigator = MagicMock()
    server.spatial_navigator.update_obstacles.return_value = {'zones': {}, 'guidance': []}
    server.spatial_navigator.get_clear_direction.return_value = 'center'
    server.navigation_engine = MagicMock()
    server.obstacle_fusion = MagicMock()
    server.obstacle_fusion.update.return_value = {'guidance': [], 'nearest_obstacle': None}
    server.ultrasonic = MagicMock()
    server.ultrasonic.get_distance.return_value = None
    server.gps_tracker = MagicMock()
    server.find_guide = FindGuide(
        speak=lambda msg: server.audio.speak_navigational(msg),
        beep=lambda f, ms: None
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Normal state: find session inactive
    assert server.find_guide.is_active is False
    server.process_ml(frame, ultrasonic_distance=None, button_pressed=False)
    assert server.face_recognizer.recognize_faces.called is True
    server.face_recognizer.recognize_faces.reset_mock()

    # 2. Find session active: face recognition MUST be skipped
    server.find_guide.start("phone", "")
    assert server.find_guide.is_active is True
    server.process_ml(frame, ultrasonic_distance=None, button_pressed=False)
    assert server.face_recognizer.recognize_faces.called is False, (
        "Face recognition must be skipped while find session is active"
    )
    # Also verify YOLO ran at imgsz=480 with lowered confidence
    args, kwargs = server.object_detector.detect.call_args
    assert kwargs.get("imgsz") == 480
    assert kwargs.get("class_conf_overrides") == {"cell phone": 0.15}

    # 3. Find session ends: face recognition MUST be restored
    server.find_guide.stop()
    assert server.find_guide.is_active is False
    server.object_detector.detect.reset_mock()

    server.process_ml(frame, ultrasonic_distance=None, button_pressed=False)
    assert server.face_recognizer.recognize_faces.called is True, (
        "Face recognition must be restored after find session ends"
    )
    # Verify YOLO restored normal settings
    args, kwargs = server.object_detector.detect.call_args
    assert kwargs.get("imgsz") is None
    assert kwargs.get("class_conf_overrides") is None

    server.cleanup()


# ─────────────────────────────────────────────────────────────
# 4. stop() stops beep thread within 1 second
# ─────────────────────────────────────────────────────────────

def test_stop_stops_beep_thread_within_1s():
    """
    Calling FindGuide.stop() during an active beep session terminates the
    beep thread within 1.0 second.
    """
    beep_calls = []
    guide = FindGuide(
        speak=lambda msg: None,
        beep=lambda f, ms: beep_calls.append((f, ms))
    )

    guide.start("phone", "")
    assert guide._beep_thread is not None
    assert guide._beep_thread.is_alive()

    t0 = time.time()
    guide.stop()
    elapsed = time.time() - t0

    assert elapsed < 1.0, f"stop() took {elapsed:.3f}s, expected < 1.0s"
    assert guide._beep_thread is None or not guide._beep_thread.is_alive(), (
        "Beep thread must not be alive after stop()"
    )


# ─────────────────────────────────────────────────────────────
# 5. Shutdown handler calls find_guide.stop()
# ─────────────────────────────────────────────────────────────

def test_shutdown_handler_calls_find_guide_stop():
    """
    The server shutdown path (both _shutdown and cleanup) calls find_guide.stop().
    """
    server = LaptopMLServer()
    server._suppress_force_exit = True
    server.audio = MagicMock()
    server.find_guide = MagicMock()
    server.voice_system = MagicMock()
    server.gps_tracker = MagicMock()
    server.ultrasonic = MagicMock()
    server.server = MagicMock()
    server.db = MagicMock()

    # Trigger _shutdown()
    server._shutdown()

    # Verify find_guide.stop was called first, then other components
    server.find_guide.stop.assert_called_once()
    server.voice_system.stop.assert_called_once()
    server.gps_tracker.stop.assert_called_once()
    server.ultrasonic.stop.assert_called_once()
    server.audio.stop.assert_called_once()
    server.server.close.assert_called_once()


# ─────────────────────────────────────────────────────────────
# 6. Beeps stop after arrival
# ─────────────────────────────────────────────────────────────

def test_beeps_stop_after_arrival(monkeypatch):
    """
    When the user arrives at the target (distance <= 45 cm),
    FindGuide stops the session and stops proximity beeps.
    """
    monkeypatch.setattr(config, "ULTRASONIC_MOCK_MODE", False)

    spoken = []
    guide = FindGuide(
        speak=lambda msg: spoken.append(msg),
        beep=lambda f, ms: None,
        frame_w=640,
        frame_h=480
    )

    guide.start("phone", "")
    assert guide.is_active is True
    assert guide._beep_thread is not None
    assert guide._beep_thread.is_alive()

    # Close detection: bbox size 440px -> pinhole distance = 550 * 15 / 440 = 18.75 cm <= 45 cm
    det = [{
        "custom_name": "phone",
        "class": "cell phone",
        "confidence": 0.95,
        "bbox": (100, 20, 540, 460)
    }]
    result = guide.update(det, ultrasonic_cm=20.0)

    assert result.get("arrived") is True
    assert result.get("visible") is True
    assert guide.is_active is False, "Guide session must be inactive after arrival"
    assert guide._beep_thread is None or not guide._beep_thread.is_alive(), (
        "Beep thread must be stopped on arrival"
    )
    assert any("reach out" in msg.lower() or "right in front of you" in msg.lower() for msg in spoken)


# ─────────────────────────────────────────────────────────────
# 7. Additional find-guidance, voice, and alias tests
# ─────────────────────────────────────────────────────────────

def test_no_beep_when_visible_is_false():
    """
    FindGuide must NOT beep at all while the object is not visible
    (after 1.5 s hysteresis). Silence until it is seen again.
    """
    beeps = []
    guide = FindGuide(
        speak=lambda msg: None,
        beep=lambda f, ms: beeps.append((f, ms)),
        frame_w=640,
        frame_h=480
    )

    # 1. Start guide - target is not yet visible
    guide.start("phone", "")
    assert guide.is_active is True
    assert guide._is_target_visible is False

    # Wait 0.15s - should be completely silent
    time.sleep(0.15)
    assert len(beeps) == 0, f"Expected no beeps before target is visible, got {beeps}"

    # 2. Target becomes visible
    det = [{
        "custom_name": "phone",
        "class": "cell phone",
        "confidence": 0.85,
        "bbox": (280, 180, 360, 300)
    }]
    state = guide.update(det, ultrasonic_cm=150.0)
    assert state["visible"] is True
    assert guide._is_target_visible is True

    # Allow beep worker to run briefly
    time.sleep(0.1)
    # Target now lost beyond hysteresis (simulate 2.0s later)
    now = time.monotonic()
    with patch("time.monotonic", side_effect=lambda: now + 2.0):
        state_lost = guide.update([], ultrasonic_cm=150.0)
        assert state_lost["visible"] is False
        assert guide._is_target_visible is False

    # Clear beeps and check that no new beeps occur while not visible
    beeps.clear()
    time.sleep(0.15)
    assert len(beeps) == 0, f"Expected no beeps while target not visible, got {beeps}"

    guide.stop()


def test_beep_interval_shorter_and_length_longer_as_closer():
    """
    Beep interval gets shorter and length longer as distance bucket gets closer:
      far (> 200 cm):           60 ms beep every 1.2 s, 800 Hz
      mid (100-200 cm):         90 ms beep every 0.7 s, 1000 Hz
      near (60-100 cm):        140 ms beep every 0.4 s, 1200 Hz
      within arm's reach (<60): 220 ms beep every 0.25 s, 1400 Hz
      arrival:                 600 ms tone
    """
    guide = FindGuide(speak=lambda msg: None, beep=lambda f, ms: None)

    # Far bucket (> 200 cm)
    params_far = guide.get_beep_params(250.0)
    int_far = params_far["interval_s"]
    dur_far = params_far["duration_ms"]
    freq_far = params_far["freq_hz"]

    # Mid bucket (100-200 cm)
    params_mid = guide.get_beep_params(150.0)
    int_mid = params_mid["interval_s"]
    dur_mid = params_mid["duration_ms"]
    freq_mid = params_mid["freq_hz"]

    # Near bucket (60-100 cm)
    params_near = guide.get_beep_params(80.0)
    int_near = params_near["interval_s"]
    dur_near = params_near["duration_ms"]
    freq_near = params_near["freq_hz"]

    # Within arm's reach (< 60 cm)
    params_reach = guide.get_beep_params(50.0)
    int_reach = params_reach["interval_s"]
    dur_reach = params_reach["duration_ms"]
    freq_reach = params_reach["freq_hz"]

    # Arrival
    profile = getattr(config, 'FIND_BEEP_PROFILE', {})
    arr_profile = profile.get("arrival", {})
    dur_arr = arr_profile.get("duration_ms", 600)
    freq_arr = arr_profile.get("freq_hz", 1400)

    # Exact expected values from config / requirements
    assert int_far == 1.2 and dur_far == 60 and freq_far == 800
    assert int_mid == 0.7 and dur_mid == 90 and freq_mid == 1000
    assert int_near == 0.4 and dur_near == 140 and freq_near == 1200
    assert int_reach == 0.25 and dur_reach == 220 and freq_reach == 1400
    assert dur_arr == 600

    # Interval gets strictly shorter as we get closer
    assert int_far > int_mid > int_near > int_reach

    # Length (duration) gets strictly longer as we get closer
    assert dur_far < dur_mid < dur_near < dur_reach < dur_arr

    # Pitch rises as we get closer
    assert freq_far < freq_mid < freq_near < freq_reach


def test_remember_cell_stores_cell_phone_with_alias():
    """
    'remember cell' stores under class name 'cell phone' with alias 'cell'.
    'find my mobile', 'find my phone' and 'find cell phone' all find it.
    """
    server = LaptopMLServer()
    server.audio = DummyAudio()
    server.audio_enabled = True
    server.object_memory = MagicMock()
    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server._register_voice_handlers()

    server.current_objects = [
        {"class_name": "cell phone", "bbox": [100, 100, 200, 300], "confidence": 0.88}
    ]

    from voice_command_system import VoiceCommandParser
    parser = VoiceCommandParser()
    parsed = parser.parse("remember cell")
    assert parsed["intent"] == "remember"

    handler = server.voice_system.command_callbacks["remember"]
    handler(parsed["parameters"])

    assert server.enrollment_active is True
    assert server.enrollment_target_obj["custom_name"] == "cell phone"
    assert server.enrollment_target_obj["yolo_class"] == "cell phone"
    assert "cell" in server.enrollment_target_obj.get("aliases", [])
    server.object_memory.start_enrollment.assert_called_once_with(
        "cell phone", "cell phone", aliases=["cell"]
    )

    # Verify matching works for 'cell', 'mobile', 'phone', and 'cell phone'
    server.object_memory.list_remembered_objects.return_value = [
        ("cell phone", "cell phone")
    ]
    server.object_memory.get_object_aliases.return_value = ["cell"]

    assert server._match_remembered("cell") == "cell phone"
    assert server._match_remembered("mobile") == "cell phone"
    assert server._match_remembered("phone") == "cell phone"
    assert server._match_remembered("cell phone") == "cell phone"


def test_single_remembered_phone_055_asks_no_question():
    """
    With exactly one remembered phone, a 0.55 match (>= SINGLE_OBJECT_MATCH_MIN = 0.50)
    is treated as high confidence and asks no confirmation question.
    """
    mock_db = MagicMock()
    single_phone = {
        'id': 1,
        'custom_name': 'phone',
        'yolo_class': 'cell phone',
        'descriptors': np.ones((10, 32), dtype=np.uint8)
    }
    mock_db.get_remembered_objects_by_class.return_value = [single_phone]

    memory = ObjectMemory(db=mock_db)
    memory.matcher = MagicMock()
    memory.matcher.extract_features.return_value = {
        'descriptors': np.ones((10, 32), dtype=np.uint8),
        'feature_type': 'orb'
    }
    memory.matcher.find_best_match.return_value = (1, 'phone', 0.55)

    fake_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = memory.recognize_object(fake_img, (100, 100, 200, 200), 'cell phone')

    assert result is not None
    custom_name, confidence, tier = result
    assert custom_name == 'phone'
    assert confidence == 0.55
    assert tier == 'high', "With 1 remembered phone, 0.55 match must be 'high' tier"
    assert memory.pending_confirmation is None, "High confidence match must NOT ask confirmation"


def test_audio_not_dropped_when_is_speaking_false():
    """
    Audio chunk is not dropped when is_speaking is False (and not within 0.25s mute).
    Audio is dropped only while TTS is actually speaking or in post-speech mute.
    """
    dummy_audio = DummyAudio()
    dummy_audio.is_speaking = False
    dummy_audio.is_playing = False
    dummy_audio.last_speech_end_time = 0.0

    vcs = VoiceCommandSystem(audio_feedback=dummy_audio)
    vcs.enabled = True
    vcs.running = True

    mock_recognizer = MagicMock()
    mock_recognizer.recognize_google.return_value = "hello"
    mock_audio = MagicMock()

    with patch.object(config, "STT_ENGINE", "google"):
        # Case 1: is_speaking is False -> audio is NOT dropped, recognize is called
        vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)
        assert mock_recognizer.recognize_google.called is True, "Audio must not be dropped when TTS is inactive"
        mock_recognizer.recognize_google.reset_mock()

        # Case 2: is_speaking is True -> audio IS dropped, recognize is NOT called
        dummy_audio.is_speaking = True
        vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)
        assert mock_recognizer.recognize_google.called is False, "Audio must be dropped while TTS is speaking"

        # Case 3: is_speaking goes back to False, but within 0.25s mute window -> dropped
        dummy_audio.is_speaking = False
        dummy_audio.last_speech_end_time = time.time()
        vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)
        assert mock_recognizer.recognize_google.called is False, "Audio must be dropped during post-speech mute"

        # Case 4: after 0.25s mute expires -> audio is NOT dropped
        dummy_audio.last_speech_end_time = time.time() - 0.30
        vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)
        assert mock_recognizer.recognize_google.called is True, "Audio must not be dropped after post-speech mute ends"

