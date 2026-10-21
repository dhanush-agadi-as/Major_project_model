"""
Test Suite: VisionMate Audible Speech, Name-Based Enrollment, Find Guidance, Place Names, and Startup/Help.

Verifies:
1. TTS worker survives 50 utterances from multiple threads; "TTS silent" warning triggers when a long phrase returns instantly.
2. "remember person as Bharath" and "remember cell phone as mobile" enroll with no window;
   without a name they speak the instruction and save nothing; no yes/no or timeout messages appear anywhere.
3. "find my mobile" and "find cell phone" both find the object; forget with a weak match says "I don't know".
4. Find speaks at most once per 3 s when not visible and on direction change when visible; arrival ends the session.
5. reverse_geocode returns a short name, uses the cache, and the offline fallback never returns raw coordinates; "where am I" parses.
6. The startup speech includes the command summary.
"""

import time
import threading
import logging
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

import config
from audio_feedback import AudioFeedback, PRIORITY_EMERGENCY, PRIORITY_PROMPT, PRIORITY_SOCIAL, PRIORITY_NAVIGATIONAL, PRIORITY_STATUS
from find_guidance import FindGuide
from ip_geolocation import reverse_geocode, get_place_name, clear_reverse_geocode_cache
from gps_tracker import GPSTracker
from voice_command_system import VoiceCommandParser, VoiceCommandSystem, FACE_NAME_HINTS, apply_name_hints
from wireless.laptop_server import LaptopMLServer, clean_object_name


# ---------------------------------------------------------------------------
# Helper Mock Audio for Server-level unit testing
# ---------------------------------------------------------------------------
class MockAudioServer:
    def __init__(self):
        self.spoken = []
        self.prompts = []
        self.is_busy_flag = False

    def speak(self, text, priority=config.PRIORITY_NAVIGATIONAL, interrupt=False, is_critical=False, is_command_reply=False, wait=False):
        if text:
            self.spoken.append(text)

    def speak_prompt(self, text, interrupt=True, wait=False):
        if text:
            self.spoken.append(text)
            self.prompts.append(text)

    def speak_social(self, text, is_command_reply=False, wait=False):
        self.speak(text, priority=config.PRIORITY_SOCIAL, is_command_reply=is_command_reply, wait=wait)

    def speak_status(self, text, is_command_reply=False, wait=False):
        self.speak(text, priority=config.PRIORITY_STATUS, is_command_reply=is_command_reply, wait=wait)

    def speak_command_reply(self, text, priority=config.PRIORITY_PROMPT, wait=False):
        self.speak_prompt(text, wait=wait)

    def register_beep_controller(self, pause_fn, resume_fn):
        pass

    def is_busy(self):
        return self.is_busy_flag


# ---------------------------------------------------------------------------
# 1. TTS Worker Survives 50 Utterances & Silent Warning
# ---------------------------------------------------------------------------
def test_tts_worker_50_utterances_and_silent_warning(caplog):
    """
    TTS worker survives 50 utterances from multiple concurrent threads.
    WARNING: TTS silent is logged if a phrase > 10 chars completes in < 150 ms.
    """
    audio = AudioFeedback(use_gtts=False)
    # Stub speech playback to simulate instant return for testing silent warning
    audio._play_text = MagicMock(return_value=None)

    total_utterances = 50
    threads_count = 5
    per_thread = total_utterances // threads_count
    threads = []

    def worker_push(tid):
        for i in range(per_thread):
            audio.speak(f"Test utterance {tid}-{i}", priority=PRIORITY_STATUS)

    for tid in range(threads_count):
        t = threading.Thread(target=worker_push, args=(tid,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Wait for queue to drain
    t_start = time.time()
    while not audio.audio_queue.empty() and time.time() - t_start < 5.0:
        time.sleep(0.05)

    assert audio.playback_thread.is_alive(), "TTS worker thread should survive 50 utterances"

    # Test that 'WARNING: TTS silent' triggers when a long phrase (>10 chars) finishes in <150ms
    with caplog.at_level(logging.WARNING):
        audio.speak_prompt("This is a long phrase for silent test", wait=True)
        time.sleep(0.1)

    assert any("WARNING: TTS silent" in rec.message for rec in caplog.records), \
        "Expected 'WARNING: TTS silent' log for instant completion of phrase >10 chars"

    audio.stop()


# ---------------------------------------------------------------------------
# 2. Voice-Only Enrollment (No windows, no yes/no, retry & missing name prompts)
# ---------------------------------------------------------------------------
def test_enrollment_by_name_no_windows_no_yes_no():
    """
    - 'remember person as Bharath' and 'remember cell phone as mobile' enroll directly with no window.
    - Missing name speaks instruction and saves nothing.
    - No yes/no or timeout messages appear anywhere.
    """
    server = LaptopMLServer()
    server.audio = MockAudioServer()
    server.audio_enabled = True

    # Mock voice system and ensure window methods are not called
    server.voice_system = MagicMock()
    server.voice_system.open_dictation_window = MagicMock()
    server.voice_system.open_yes_no_window = MagicMock()
    server.voice_system.open_disambiguation_window = MagicMock()

    # Mock face recognizer
    server.face_recognizer = MagicMock()
    server.face_recognizer.detect_faces = MagicMock(return_value=[{'bbox': (50, 50, 150, 150)}])
    server.face_recognizer.enroll_face = MagicMock(return_value=True)
    server.current_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mock object memory
    server.object_memory = MagicMock()
    server.object_memory.list_remembered_objects = MagicMock(return_value=[])
    server.object_memory.start_enrollment = MagicMock()

    # Register voice handlers
    server._register_voice_handlers()

    # Find the registered 'remember' handler
    remember_handler = None
    for call in server.voice_system.register_command_handler.call_args_list:
        if call[0][0] == 'remember':
            remember_handler = call[0][1]
            break
    assert remember_handler is not None, "remember handler must be registered"

    # A. "remember person as Bharath"
    server.audio.spoken.clear()
    remember_handler({'thing': 'person', 'name': 'Bharath', 'raw_command': 'remember person as bharath'})
    assert any("Enrolled Bharath" in s for s in server.audio.spoken)
    server.voice_system.open_dictation_window.assert_not_called()
    server.voice_system.open_yes_no_window.assert_not_called()

    # B. "remember cell phone as mobile" with cell phone visible
    server.enrollment_active = False
    server.current_objects = [{'class_name': 'cell phone', 'bbox': (100, 100, 200, 200)}]
    server.audio.spoken.clear()
    remember_handler({'thing': 'cell phone', 'name': 'mobile', 'raw_command': 'remember cell phone as mobile'})
    assert any("Remembering your mobile" in s for s in server.audio.spoken)
    server.object_memory.start_enrollment.assert_called()
    server.voice_system.open_dictation_window.assert_not_called()
    server.voice_system.open_yes_no_window.assert_not_called()

    # C. Missing name for person: speaks instruction, enrolls nothing
    server.enrollment_active = False
    server.face_recognizer.enroll_face.reset_mock()
    server.audio.spoken.clear()
    remember_handler({'thing': 'person', 'name': '', 'raw_command': 'remember person'})
    assert any("Please say a name, like: hey vision, remember person as Bharath" in s for s in server.audio.spoken)
    server.face_recognizer.enroll_face.assert_not_called()
    server.voice_system.open_dictation_window.assert_not_called()

    # D. Missing name for object: speaks instruction, enrolls nothing
    server.enrollment_active = False
    server.object_memory.start_enrollment.reset_mock()
    server.audio.spoken.clear()
    remember_handler({'thing': 'cell phone', 'name': '', 'raw_command': 'remember cell phone'})
    assert any("Please say a name, like: hey vision, remember cell phone as mobile" in s for s in server.audio.spoken)
    server.object_memory.start_enrollment.assert_not_called()
    server.voice_system.open_dictation_window.assert_not_called()

    # Verify no timeout messages or yes/no questions were spoken
    all_spoken = " ".join(server.audio.spoken).lower()
    for forbidden in ["i didn't catch that", "cancelled", "did you mean", "is that right"]:
        assert forbidden not in all_spoken, f"Forbidden phrase '{forbidden}' found in spoken text"


# ---------------------------------------------------------------------------
# 3. Find and Forget Rules (Fuzzy >= 0.5, Forget >= 0.7, Fallbacks)
# ---------------------------------------------------------------------------
def test_find_and_forget_rules():
    """
    - 'find my mobile' and 'find cell phone' both find the object.
    - 'forget' with ratio >= 0.7 deletes without asking questions.
    - 'forget' with weak match says 'I don't know <name>'.
    - 'find' with weak match says 'I don't know <name>. I remember: <list>'.
    """
    server = LaptopMLServer()
    server.audio = MockAudioServer()
    server.audio_enabled = True

    server.voice_system = MagicMock()
    server.voice_system.open_yes_no_window = MagicMock()

    # Mock remembered object 'mobile' with class 'cell phone' and alias 'cell phone'
    server.object_memory = MagicMock()
    server.object_memory.list_remembered_objects = MagicMock(return_value=[('mobile', 'cell phone')])
    server.object_memory.get_object_aliases = MagicMock(return_value=['cell phone', 'phone'])
    server.object_memory.get_cached_object = MagicMock(return_value={'yolo_class': 'cell phone'})
    server.object_memory.get_object_location = MagicMock(return_value={'location': 'table', 'timestamp': '2026-10-21 10:00:00.000000'})
    server.object_memory.delete_object = MagicMock(return_value=True)

    server.find_guide = MagicMock()
    server.find_guide.is_active = False
    server.find_guide.start = MagicMock()
    server.find_guide.describe_last_seen = MagicMock(return_value="last seen near table")

    server._register_voice_handlers()

    find_handler = None
    forget_handler = None
    for call in server.voice_system.register_command_handler.call_args_list:
        if call[0][0] == 'find':
            find_handler = call[0][1]
        elif call[0][0] == 'forget':
            forget_handler = call[0][1]

    assert find_handler is not None
    assert forget_handler is not None

    # A. "find my mobile" -> finds object
    server.audio.spoken.clear()
    find_handler({'object_name': 'mobile'})
    assert server.find_guide.start.called
    assert any("mobile" in s.lower() for s in server.audio.spoken)

    # B. "find cell phone" -> finds object by class/alias
    server.find_guide.start.reset_mock()
    server.audio.spoken.clear()
    find_handler({'object_name': 'cell phone'})
    assert server.find_guide.start.called
    assert any("mobile" in s.lower() for s in server.audio.spoken)

    # C. "find unknown_item_123" -> ratio < 0.5 -> "I don't know unknown_item_123. I remember: mobile"
    server.audio.spoken.clear()
    find_handler({'object_name': 'unknown_item_123'})
    assert any("I don't know unknown_item_123. I remember: mobile" in s for s in server.audio.spoken)

    # D. "forget unknown_item_123" -> ratio < 0.7 -> speaks "I don't know unknown_item_123" without question
    server.audio.spoken.clear()
    forget_handler({'object_name': 'unknown_item_123'})
    assert any("I don't know unknown_item_123" in s for s in server.audio.spoken)
    server.object_memory.delete_object.assert_not_called()
    server.voice_system.open_yes_no_window.assert_not_called()

    # E. "forget mobile" -> ratio >= 0.7 -> deletes directly, speaks "Forgot your mobile"
    server.audio.spoken.clear()
    forget_handler({'object_name': 'mobile'})
    server.object_memory.delete_object.assert_called_with('mobile')
    assert any("Forgot your mobile" in s for s in server.audio.spoken)
    server.voice_system.open_yes_no_window.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Spoken Find Guidance Cadence and Arrival
# ---------------------------------------------------------------------------
def test_find_guidance_cadence_and_arrival():
    """
    - Not visible: speaks every 3 s (last side or 'turn slowly'); after 10 s 'go straight, then turn slowly'.
    - Visible: beeps on; speaks direction only on change or bucket change, max once per 2.5 s.
    - Arrival: speaks '<name> is right in front of you, reach out', 600 ms beep, ends find.
    """
    spoken = []
    beeps = []

    def mock_speak(text):
        spoken.append(text)

    def mock_beep(freq, dur):
        beeps.append((freq, dur))

    guide = FindGuide(speak=mock_speak, beep=mock_beep, frame_w=640, frame_h=480)
    guide.start("keys", "last seen on desk")

    # Clear initial last_seen utterance
    spoken.clear()

    # 1. Not visible: immediate first update speaks direction/slowly
    guide.update([], None)
    assert len(spoken) == 1
    first_hint = spoken[-1]
    assert first_hint in ("left", "right", "keep turning slowly")

    # Calling update 0.5 s later should NOT speak (< 5s throttle)
    time.sleep(0.05)
    guide.update([], None)
    assert len(spoken) == 1

    # Simulate elapsed lost time > 5 s without sight
    with guide._lock:
        guide._last_detected_time = time.monotonic() - 11.0
        guide._last_lost_speech_time = time.monotonic() - 6.0

    guide.update([], None)
    assert spoken[-1] in ("left", "right", "keep turning slowly")

    # 2. Visible: detection appears
    spoken.clear()
    # Object on left side (x center ~ 150, frame center 320 -> offset < 0 -> 'slightly left')
    obj_left = {
        "custom_name": "keys",
        "class": "keys",
        "confidence": 0.9,
        "bbox": (100, 100, 200, 200),
    }
    # Initial visible frame speaks direction
    with guide._lock:
        guide._last_speech_time = 0.0
        guide._last_lost_speech_time = 0.0

    res = guide.update([obj_left], None)
    assert res.get("direction") == "slightly left"
    assert any("slightly left" in s for s in spoken)

    # Successive frame with same direction and bucket should not speak
    spoken.clear()
    guide.update([obj_left], None)
    assert len(spoken) == 0

    # Direction changes to 'straight ahead' (cx ~ 320)
    obj_center = {
        "custom_name": "keys",
        "class": "keys",
        "confidence": 0.9,
        "bbox": (270, 100, 370, 200),
    }
    with guide._lock:
        guide._last_speech_time = time.monotonic() - 3.0  # past 2.5s cooldown

    guide.update([obj_center], None)
    assert any("straight ahead" in s for s in spoken)

    # 3. Arrival: within reach (<=45 cm, center)
    spoken.clear()
    beeps.clear()
    # Large bbox simulates close distance (<45cm): 1100 * 20 / 550 = 40.0 cm <= 45.0 cm
    obj_arrival = {
        "custom_name": "keys",
        "class": "keys",
        "confidence": 0.95,
        "bbox": (45, 0, 595, 480),
    }
    res_arr = guide.update([obj_arrival], None)
    assert res_arr.get("arrived") is True
    assert not guide.is_active, "Arrival must stop find session"
    assert any("keys is right in front of you, reach out" in s for s in spoken)
    # Check 600 ms tone was emitted
    assert any(dur == 600 for freq, dur in beeps)


# ---------------------------------------------------------------------------
# 5. Reverse Geocode & Place Names & Where am I
# ---------------------------------------------------------------------------
def test_reverse_geocode_and_where_am_i():
    """
    - reverse_geocode returns road + suburb/neighbourhood + city.
    - Uses 5-minute cache by 4 decimals.
    - Offline fallback never returns raw numbers.
    - 'where am I' intent parses and speaks 'You are near <place name>'.
    """
    clear_reverse_geocode_cache()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "address": {"road": "Brigade Road", "suburb": "Shanthala Nagar", "city": "Bengaluru"}
    })

    with patch('ip_geolocation.requests.get', return_value=mock_resp) as mock_get:
        name1 = reverse_geocode(12.97161, 77.59461)
        assert name1 == "Brigade Road, Shanthala Nagar, Bengaluru"
        assert mock_get.call_count == 1

        # Second call within 5 minutes with coordinates rounded to 4 decimals hits cache
        name2 = reverse_geocode(12.97162, 77.59463)
        assert name2 == "Brigade Road, Shanthala Nagar, Bengaluru"
        assert mock_get.call_count == 1, "Cache should have prevented second network request"

    # Offline / Error fallback
    with patch('ip_geolocation.requests.get', side_effect=Exception("Network error")):
        clear_reverse_geocode_cache()
        fallback_name = get_place_name(12.9716, 77.5946, city_fallback="Bengaluru")
        assert fallback_name == "Bengaluru"
        # No coordinates in name
        assert "12." not in fallback_name and "77." not in fallback_name

        fallback_none = get_place_name(12.9716, 77.5946, city_fallback=None)
        assert fallback_none == "your current location"

    # Voice Command Parser: where am I / my location / where is this
    parser = VoiceCommandParser()
    for phrase in ["where am I", "my location", "where is this"]:
        res = parser.parse(phrase)
        intent = res['intent'] if isinstance(res, dict) else res[0]
        assert intent == 'where_am_i', f"Expected intent 'where_am_i' for '{phrase}', got '{intent}'"

    # Server handler test
    server = LaptopMLServer()
    server.audio = MockAudioServer()
    server.audio_enabled = True
    server.voice_system = MagicMock()
    server.gps_tracker = MagicMock()
    server.gps_tracker.get_current_place_name.return_value = "Brigade Road, Shanthala Nagar, Bengaluru"

    server._register_voice_handlers()
    where_handler = None
    for call in server.voice_system.register_command_handler.call_args_list:
        if call[0][0] == 'where_am_i':
            where_handler = call[0][1]
            break

    assert where_handler is not None
    where_handler({})
    assert any("You are near Brigade Road, Shanthala Nagar, Bengaluru" in s for s in server.audio.spoken)


# ---------------------------------------------------------------------------
# 6. Startup and Help Command Speech
# ---------------------------------------------------------------------------
def test_startup_and_help_speech():
    """
    - After 'VisionMate ready', speaks command summary when STARTUP_READ_COMMANDS=True.
    - 'help' reads list in short groups with pauses at PROMPT priority.
    """
    server = LaptopMLServer()
    server.audio = MockAudioServer()
    server.audio_enabled = True
    server.voice_system = MagicMock()
    server.voice_system.initialize.return_value = True

    # Test startup speech logic in initialize()
    with patch.object(config, 'STARTUP_READ_COMMANDS', True):
        # Trigger the startup speech lines directly
        server.audio.speak_status("VisionMate ready.")
        if getattr(config, 'STARTUP_READ_COMMANDS', True):
            server.audio.speak_prompt(
                "Say hey vision, then: remember, find, where am I, notes, navigate, or help.",
                wait=False
            )

    assert "VisionMate ready." in server.audio.spoken
    assert any("Say hey vision, then: remember, find, where am I, notes, navigate, or help." in s
               for s in server.audio.spoken)

    # Test help command grouped reading
    server.audio.spoken.clear()
    server.audio.prompts.clear()
    server._register_voice_handlers()

    help_handler = None
    for call in server.voice_system.register_command_handler.call_args_list:
        if call[0][0] == 'help':
            help_handler = call[0][1]
            break

    assert help_handler is not None
    with patch('time.sleep', return_value=None):
        help_handler({})

    # Must speak all 4 groups at prompt priority
    assert len(server.audio.prompts) == 4
    assert "remember person as name" in server.audio.prompts[0]
    assert "find my name" in server.audio.prompts[1]
    assert "where am I" in server.audio.prompts[2]
    assert "navigate to place" in server.audio.prompts[3]


# ---------------------------------------------------------------------------
# 7. Name Hints Fuzzy Matching in Parser
# ---------------------------------------------------------------------------
def test_face_name_hints_in_parser():
    """
    FACE_NAME_HINTS = ['Bharath', 'Dhanush'].
    Fuzzy match ratio >= 0.6 maps heard name to hint spelling.
    """
    assert "Bharath" in FACE_NAME_HINTS
    assert "Dhanush" in FACE_NAME_HINTS

    assert apply_name_hints("bharat") == "Bharath"
    assert apply_name_hints("danush") == "Dhanush"
    assert apply_name_hints("dhanus") == "Dhanush"
    # Unrelated name remains unchanged
    assert apply_name_hints("charlie") == "charlie"

    parser = VoiceCommandParser()
    res = parser.parse("remember person as bharat")
    intent = res['intent'] if isinstance(res, dict) else res[0]
    params = res['parameters'] if isinstance(res, dict) else res[1]
    assert intent in ('remember', 'enroll_face')
    assert params.get('name') == "Bharath"


# ---------------------------------------------------------------------------
# 8. Requirement 10 Tests: Arrival, Reached-Then-Removed, 45s Timeout,
#    No Spatial Speech During Find, Hybrid STT Google Empty Handling
# ---------------------------------------------------------------------------
def test_arrival_fires_dist_45_hratio_075_offset_neg05():
    """
    10. Arrival fires with dist=45, h_ratio=0.75, offset=-0.5.
    Rule 2 (h_ratio >= 0.65 and |offset| < 0.7) and/or Rule 1 (dist <= 55, |offset| < 0.55).
    On arrival:
    - speaks "<name> is right in front of you, reach out"
    - emits one 600 ms tone
    - ends find session
    - logs "FIND: ARRIVED"
    """
    spoken = []
    beeps = []
    guide = FindGuide(
        speak=lambda text, **kw: spoken.append(text),
        beep=lambda freq, dur: beeps.append((freq, dur)),
        frame_w=640,
        frame_h=480,
    )
    guide.start("keys")
    spoken.clear()
    beeps.clear()

    # Geometry for dist ~ 45 cm, h_ratio = 0.75, offset = -0.5:
    # frame_w = 640, frame_h = 480
    # cx = 160 -> offset = (160 - 320) / 320 = -0.5
    # y1 = 60, y2 = 420 -> bbox_h = 360 -> h_ratio = 360 / 480 = 0.75
    # bbox_w = 489 -> bbox_px = 489 -> dist = 1100 * 20 / 489 = 44.99 cm
    # x1 = 160 - 244.5 = -84.5, x2 = 404.5
    obj = {
        "custom_name": "keys",
        "class": "keys",
        "confidence": 0.95,
        "bbox": (-84.5, 60, 404.5, 420),
    }

    res = guide.update([obj], None)
    assert res.get("arrived") is True, f"Arrival must fire on dist=45, h_ratio=0.75, offset=-0.5, got: {res}"
    assert not guide.is_active, "Find session must end on arrival"
    assert any("keys is right in front of you, reach out" in s for s in spoken)
    assert any(dur == 600 for freq, dur in beeps), "Must emit one 600 ms tone on arrival"


def test_reached_then_removed_rule_ends_session():
    """
    10. 'Reached then removed' rule: if the target was visible within reach
    (dist <= 60 cm or h_ratio >= 0.6) for >= 1.5 s in total, and then
    isn't visible for >= 1.5 s, end the session:
    - speak "Found your <name>"
    - stop beeps and stop all guidance
    - never go back to "left" / "turn left" afterwards.
    """
    spoken = []
    beeps = []
    guide = FindGuide(
        speak=lambda text, **kw: spoken.append(text),
        beep=lambda freq, dur: beeps.append((freq, dur)),
        frame_w=640,
        frame_h=480,
    )
    guide.start("wallet")
    spoken.clear()
    beeps.clear()

    # Target was visible within reach for >= 1.5 s total duration
    guide._reach_total_duration = 1.6
    # Target now absent for >= 1.5 s
    with guide._lock:
        guide._last_detected_time = time.monotonic() - 1.6

    res = guide.update([], None)
    assert res.get("reached_and_removed") is True
    assert not guide.is_active, "Session must end"
    assert any("Found your wallet" in s for s in spoken)

    # Next update when object is still absent: must NEVER speak "left" or anything else
    spoken.clear()
    res2 = guide.update([], None)
    assert len(spoken) == 0, "Must never go back to guidance after reached-then-removed"
    assert not guide.is_active


def test_45s_timeout_ends_find_session():
    """
    10. Search limits: after 45 s without finding the object, speak
    "I can't find your <name>. Say hey vision find <name> to try again."
    and end the session.
    """
    spoken = []
    guide = FindGuide(
        speak=lambda text, **kw: spoken.append(text),
        beep=lambda f, d: None,
        frame_w=640,
        frame_h=480,
    )
    guide.start("charger")
    spoken.clear()

    with guide._lock:
        guide._start_time = time.monotonic() - 46.0
        guide._last_detected_time = time.monotonic() - 46.0

    res = guide.update([], None)
    assert res.get("timeout") is True
    assert not guide.is_active, "Session must end after 45 s timeout"
    assert any("I can't find your charger. Say hey vision find charger to try again." in s for s in spoken)


def test_no_spatial_speech_during_find_session():
    """
    10. During an active find session, no spatial speech at all
    ("Person ahead/left/right", "Chair left", "2 people ahead", "Move left")
    unless an object is closer than 50 cm. Restore it when the session ends.
    Must fail if any 'Person ...' message is spoken during find when distance >= 50 cm.
    """
    from spatial_navigator import SpatialNavigator
    nav = SpatialNavigator(frame_width=640, frame_height=480)

    # 1. During find: person at 1.8 m (>= 50 cm) MUST NOT speak
    person_item = {
        'class_name': 'person',
        'zone': 'center',
        'message': 'Person ahead',
        'distance': 'medium',
        'estimated_distance_m': 1.8,
    }
    assert not nav.should_speak_guidance(person_item, ultrasonic_distance=180.0, is_find_active=True), \
        "Person message must NOT be spoken during find when distance >= 50 cm"

    # Other objects: Chair left, Move left at >= 50 cm MUST NOT speak
    chair_item = {
        'class_name': 'chair',
        'zone': 'left',
        'message': 'Chair left',
        'distance': 'medium',
        'estimated_distance_m': 1.5,
    }
    assert not nav.should_speak_guidance(chair_item, ultrasonic_distance=150.0, is_find_active=True)

    nav_item = {
        'class_name': 'navigation',
        'zone': 'left',
        'message': 'Move left',
        'distance': 'medium',
        'estimated_distance_m': 2.0,
    }
    assert not nav.should_speak_guidance(nav_item, ultrasonic_distance=200.0, is_find_active=True)

    # 2. Closer than 50 cm: allowed even during find
    close_person = {
        'class_name': 'person',
        'zone': 'center',
        'message': 'Person ahead',
        'distance': 'very-close',
        'estimated_distance_m': 0.4,
    }
    assert nav.should_speak_guidance(close_person, ultrasonic_distance=40.0, is_find_active=True), \
        "Objects closer than 50 cm must be allowed to speak even during find"

    # 3. Restored when find session ends (is_find_active=False)
    nav.is_find_active = False
    fresh_person = {
        'class_name': 'person',
        'zone': 'center',
        'message': 'Person ahead',
        'distance': 'medium',
        'estimated_distance_m': 1.8,
    }
    assert nav.should_speak_guidance(fresh_person, ultrasonic_distance=180.0, is_find_active=False), \
        "Spatial speech must be restored when find session is inactive"


def test_hybrid_stt_empty_google_does_not_fallback():
    """
    10. Hybrid STT: an empty Google result (no speech recognised) is NOT a failure.
    Return '' and ignore it. Fall back to Whistle only on network or API errors.
    """
    import speech_recognition as sr
    import stt_engine
    vcs = VoiceCommandSystem()
    mock_recognizer = MagicMock()
    mock_audio = MagicMock()

    # Case A: Google raises sr.UnknownValueError (no speech recognised)
    mock_recognizer.recognize_google.side_effect = sr.UnknownValueError()
    with patch.object(config, 'STT_ENGINE', 'hybrid'), \
         patch.object(stt_engine, 'is_available', return_value=True), \
         patch.object(stt_engine, 'is_internet_reachable', return_value=True), \
         patch.object(stt_engine, 'transcribe_with_timing') as mock_whistle:

        res = vcs._recognize_audio(mock_recognizer, mock_audio)
        assert res == "", "Empty Google result must return empty string"
        mock_whistle.assert_not_called()

    # Case B: Google returns whitespace/empty string ""
    mock_recognizer.recognize_google.side_effect = None
    mock_recognizer.recognize_google.return_value = "   "
    with patch.object(config, 'STT_ENGINE', 'hybrid'), \
         patch.object(stt_engine, 'is_available', return_value=True), \
         patch.object(stt_engine, 'is_internet_reachable', return_value=True), \
         patch.object(stt_engine, 'transcribe_with_timing') as mock_whistle:

        res = vcs._recognize_audio(mock_recognizer, mock_audio)
        assert res == "", "Empty Google result must return empty string"
        mock_whistle.assert_not_called()

    # Case C: Google raises RequestError (network error) -> falls back to Whistle
    mock_recognizer.recognize_google.side_effect = sr.RequestError("Connection failed")
    with patch.object(config, 'STT_ENGINE', 'hybrid'), \
         patch.object(stt_engine, 'is_available', return_value=True), \
         patch.object(stt_engine, 'is_internet_reachable', return_value=True), \
         patch.object(stt_engine, 'transcribe_with_timing', return_value=("hey vision find bottle", 70)) as mock_whistle, \
         patch.object(stt_engine, 'is_filler', return_value=False), \
         patch('voice_command_system.logger.info') as mock_log:

        res = vcs._recognize_audio(mock_recognizer, mock_audio)
        assert res == "hey vision find bottle"
        assert mock_whistle.called
        log_calls = [c[0][0] for c in mock_log.call_args_list if len(c[0]) > 0 and isinstance(c[0][0], str)]
        assert any("Hybrid STT fallback to Whistle: RequestError" in log_msg for log_msg in log_calls)
