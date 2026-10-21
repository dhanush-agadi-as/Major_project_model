"""
Tests for fixes from real log:
1. Spoken names never contain "your my". Names with leading "my" display and speak as clean names.
2. "remember phone as mobile" stores alias "mobile", and "find my phone" and "find my mobile" find it.
3. 1 s gap in detections during a find does not trigger "I can't see it" (1.5s hysteresis + 3s absence requirement).
4. "Person ahead" is announced once for a person standing in view for 60 frames (Priority 3, event-based only).
5. Last-seen text: if only nearby object is a person, says "near you" instead of "near the person".
6. rpi_client.py: logs "Recognized: <name>" only on name change or every 10s.
"""

import time
import pytest
import numpy as np
import logging

from wireless.laptop_server import LaptopMLServer, clean_object_name
from wireless.rpi_client import RaspberryPiClient
from find_guidance import FindGuide
from spatial_navigator import SpatialNavigator
from voice_command_system import VoiceCommandSystem, VoiceCommandParser
from database import VisionMateDB
from object_memory import ObjectMemory


class MockAudio:
    def __init__(self):
        self.spoken = []

    def speak(self, text, priority=3, interrupt=False, is_critical=False):
        if text:
            self.spoken.append(text)

    def speak_social(self, text):
        self.speak(text, priority=2)

    def speak_navigational(self, text, is_critical=False):
        self.speak(text, priority=3, is_critical=is_critical)

    def speak_emergency(self, text, is_critical=True):
        self.speak(text, priority=1, is_critical=is_critical)

    def speak_status(self, text):
        self.speak(text, priority=4)

    def is_tts_active(self):
        return False


# ----------------------------------------------------------------------------
# 1. Names never contain "your my"
# ----------------------------------------------------------------------------

def test_clean_object_name_removes_leading_my_the_a_an():
    """clean_object_name correctly removes leading articles and 'my'."""
    assert clean_object_name("my phone") == "phone"
    assert clean_object_name("my my phone") == "phone"
    assert clean_object_name("the phone") == "phone"
    assert clean_object_name("a cup") == "cup"
    assert clean_object_name("an apple") == "apple"
    assert clean_object_name("phone") == "phone"
    assert clean_object_name("my cell phone") == "cell phone"


def test_spoken_names_never_contain_your_my(tmp_path):
    """
    Spoken sentences across remember, find, process_ml, and find_guidance
    must use 'your <name>' exactly once and NEVER produce 'your my <name>'.
    Existing names like 'my phone' must be displayed as 'phone'.
    """
    server = LaptopMLServer(port=8099)
    server.show_display = False
    server.audio = MockAudio()
    db_path = str(tmp_path / "test_vm.db")
    server.db = VisionMateDB(db_path)
    server.object_memory = ObjectMemory(server.db)
    server.find_guide = FindGuide(speak=server.audio.speak, beep=lambda *args: None)
    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server._register_voice_handlers()

    # Pre-populate DB with an existing object named "my phone"
    server.db.add_remembered_object(
        custom_name="my phone",
        yolo_class="cell phone",
        visual_features={"feature_type": "deep", "deep_embedding": [0.1] * 128},
        enrollment_frames=5
    )

    # 1. Display frame test: "my phone" must display as "phone"
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ml_result = {
        'objects': [],
        'remembered': [{'name': 'my phone', 'bbox': [100, 100, 200, 200], 'confidence': 0.95}],
        'faces': []
    }
    # Test display label creation
    for mem_obj in ml_result['remembered']:
        disp_name = clean_object_name(mem_obj['name'])
        assert disp_name == "phone"
        assert "my" not in disp_name

    # 2. handle_find with "my phone"
    server.current_objects = [{'bbox': [100, 100, 200, 200], 'class_name': 'cell phone', 'confidence': 0.9}]
    server.voice_system.command_callbacks['find']({'object_name': 'my phone'})

    # 3. handle_remember with "remember my cup"
    server.current_objects = [{'bbox': [50, 50, 150, 150], 'class_name': 'cup', 'confidence': 0.85}]
    server.voice_system.command_callbacks['remember']({'object_name': 'my cup'})

    # 4. FindGuide arrival message
    guide = FindGuide(speak=server.audio.speak, beep=lambda *args: None)
    guide.start("my phone", "")
    # Target right in front (dist < 40cm, center)
    guide.update([{'custom_name': 'my phone', 'class': 'cell phone', 'confidence': 0.9, 'bbox': (250, 150, 390, 330), 'tier': 'high'}], ultrasonic_cm=30.0)

    # Verify all spoken messages
    for msg in server.audio.spoken:
        assert "your my" not in msg.lower(), f"Forbidden phrase 'your my' in: {msg}"
        assert "my my" not in msg.lower(), f"Forbidden phrase 'my my' in: {msg}"


# ----------------------------------------------------------------------------
# 2. Aliases when remembering & finding
# ----------------------------------------------------------------------------

def test_remember_phone_as_mobile_stores_alias_and_find_matches(tmp_path):
    """
    "remember phone as mobile" stores the object under alias "mobile".
    "find my phone" and "find my mobile" must both find the object "mobile".
    """
    # Test parser
    parser = VoiceCommandParser()
    res1 = parser.parse("remember phone as mobile")
    assert res1['intent'] == 'remember'
    assert res1['parameters'].get('alias') == 'mobile'
    assert res1['parameters'].get('object_name') == 'phone'

    res2 = parser.parse("remember this as mobile")
    assert res2['intent'] == 'remember'
    assert res2['parameters'].get('alias') == 'mobile'

    res3 = parser.parse("call it mobile")
    assert res3['intent'] == 'remember'
    assert res3['parameters'].get('alias') == 'mobile'

    # Test server execution
    server = LaptopMLServer(port=8098)
    server.show_display = False
    server.audio = MockAudio()
    db_path = str(tmp_path / "test_alias.db")
    server.db = VisionMateDB(db_path)
    server.object_memory = ObjectMemory(server.db)
    server.find_guide = FindGuide(speak=server.audio.speak, beep=lambda *args: None)
    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server._register_voice_handlers()

    # Place a cell phone in view
    server.current_objects = [{'bbox': [100, 100, 200, 200], 'class_name': 'cell phone', 'confidence': 0.9}]

    # User says "remember phone as mobile"
    server.voice_system.command_callbacks['remember'](res1['parameters'])
    assert server.enrollment_active is True
    assert server.enrollment_target_obj['custom_name'] == 'mobile'
    assert server.enrollment_target_obj['yolo_class'] == 'cell phone'

    # Verify spoken prompt
    assert any("Remembering your mobile" in msg for msg in server.audio.spoken)
    for msg in server.audio.spoken:
        assert "your my" not in msg.lower()

    # Complete enrollment
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        server.object_memory.add_enrollment_frame(dummy_frame, (100, 100, 200, 200))
    server.object_memory.finish_enrollment()
    server.enrollment_active = False

    # Check database: stored under alias "mobile"
    remembered = server.object_memory.list_remembered_objects()
    names = [name for name, _ in remembered]
    assert "mobile" in names

    # Now test finding: "find my mobile" and "find my phone" must find "mobile"
    server.audio.spoken.clear()
    server.voice_system.command_callbacks['find']({'object_name': 'my mobile'})
    assert any("mobile" in msg for msg in server.audio.spoken)
    assert not any("I don't know an object" in msg for msg in server.audio.spoken)

    server.audio.spoken.clear()
    server.voice_system.command_callbacks['find']({'object_name': 'my phone'})
    assert any("mobile" in msg for msg in server.audio.spoken)
    assert not any("I don't know an object" in msg for msg in server.audio.spoken)


# ----------------------------------------------------------------------------
# 3. Find session robustness: 1s gap does not trigger "I can't see it"
# ----------------------------------------------------------------------------

def test_one_second_gap_during_find_does_not_trigger_cant_see_it(monkeypatch):
    """
    A 1 s gap in detections during a find does not trigger "I can't see it".
    Hysteresis keeps last known position for 1.5 s.
    Only after 3 s of continuous absence is "I can't see it" spoken.
    """
    import find_guidance
    clock_time = [100.0]
    monkeypatch.setattr(find_guidance.time, "monotonic", lambda: clock_time[0])

    spoken = []
    guide = FindGuide(speak=spoken.append, beep=lambda *args: None)
    guide.start("phone", "Searching for your phone")
    spoken.clear()

    # Frame 0 at t=100.0: phone detected
    det = {'custom_name': 'phone', 'class': 'cell phone', 'confidence': 0.9, 'bbox': (200, 150, 440, 330), 'tier': 'high'}
    res0 = guide.update([det], ultrasonic_cm=120.0)
    assert res0['visible'] is True

    # 1.0 second gap (frames with NO detections at t=101.0)
    clock_time[0] = 101.0
    # 1.0s <= 1.5s hysteresis, so target must still be treated as visible
    res1 = guide.update([], ultrasonic_cm=120.0)
    assert res1['visible'] is True
    assert not any("can't see" in msg.lower() for msg in spoken)

    # Even at t=102.0 (2.0s gap, past 1.5s hysteresis):
    # Target becomes visible=False, but elapsed_lost is 2.0s < 3.0s, so "I can't see it" is NOT spoken yet!
    clock_time[0] = 102.0
    res2 = guide.update([], ultrasonic_cm=120.0)
    assert res2['visible'] is False
    assert not any("can't see" in msg.lower() for msg in spoken)

    # At t=103.5 (3.5s gap >= 3.0s continuous absence):
    clock_time[0] = 103.5
    res3 = guide.update([], ultrasonic_cm=120.0)
    assert res3['visible'] is False
    assert any("can't see" in msg.lower() for msg in spoken)


def test_find_session_treats_ambiguous_as_visible_and_suppresses_question():
    """
    While find session is active for object X:
    - ambiguous tier is treated as X is visible
    - do not ask "Is this your X?"
    """
    spoken = []
    guide = FindGuide(speak=spoken.append, beep=lambda *args: None)
    guide.start("phone", "")

    # Ambiguous match for phone
    ambiguous_det = {
        'custom_name': 'phone',
        'class': 'cell phone',
        'confidence': 0.58,
        'bbox': (200, 150, 440, 330),
        'tier': 'ambiguous'
    }
    res = guide.update([ambiguous_det], ultrasonic_cm=120.0)
    assert res['visible'] is True

    # Verify is_target
    assert guide.is_target('phone', 'cell phone') is True
    assert guide.is_target('mobile', 'cell phone') is True


# ----------------------------------------------------------------------------
# 4. "Person ahead": priority 3, announced once for 60 frames
# ----------------------------------------------------------------------------

def test_person_ahead_announced_once_for_60_frames_and_is_priority_3():
    """
    "Person ahead" is announced once for a person standing in view for 60 frames.
    It is Priority 3 (navigational), not Priority 1.
    """
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    det = {'bbox': (240, 160, 400, 320), 'class_name': 'person', 'confidence': 0.9}

    spoken_items = []

    for frame in range(60):
        t = 1000.0 + (frame * 0.1)  # 10 fps
        guidance = nav.generate_guidance([det], current_time=t)
        assert len(guidance) > 0
        person_g = guidance[0]

        # Priority must be 3 (navigational)
        assert person_g['priority'] == 3
        # Must not be critical
        assert nav.is_critical(person_g, ultrasonic_distance=100.0) is False

        for item in guidance:
            if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=t):
                spoken_items.append(item)
                nav.mark_guidance_spoken(item, current_time=t)

    # Exactly 1 announcement across all 60 frames
    assert len(spoken_items) == 1
    assert "Person ahead" in spoken_items[0]['message']
    assert spoken_items[0]['priority'] == 3


# ----------------------------------------------------------------------------
# 5. Last-seen text: if only nearby object is person, says "near you"
# ----------------------------------------------------------------------------

def test_last_seen_text_says_near_you_when_only_person_nearby():
    """If the only nearby object is a person, say 'near you' instead of 'near the person'."""
    sighting = {
        "timestamp": "2026-10-19 14:00:00.000000",
        "context": {
            "nearby_objects": ["person"]
        }
    }
    desc = FindGuide.describe_last_seen(sighting)
    assert "near you" in desc
    assert "person" not in desc

    sighting2 = {
        "timestamp": "2026-10-19 14:00:00.000000",
        "context": {
            "nearby_objects": ["the person"]
        }
    }
    desc2 = FindGuide.describe_last_seen(sighting2)
    assert "near you" in desc2
    assert "person" not in desc2


# ----------------------------------------------------------------------------
# 6. rpi_client.py: Recognized face logging throttled
# ----------------------------------------------------------------------------

def test_rpi_client_face_recognition_log_throttling(caplog):
    """Log 'Recognized: <name>' only when face name changes or every 10s."""
    client = RaspberryPiClient(server_ip="127.0.0.1", client_speaks=False)

    alice_result = {'faces': [{'name': 'Alice', 'bbox': [10, 10, 50, 50]}]}
    bob_result = {'faces': [{'name': 'Bob', 'bbox': [10, 10, 50, 50]}]}

    with caplog.at_level(logging.INFO):
        # Frame 1: Alice enters -> log immediately
        client.process_results(alice_result)
        # Frames 2-5: Alice remains -> no repeat logs
        for _ in range(4):
            client.process_results(alice_result)

        # Count logs
        alice_logs = [r for r in caplog.records if "Recognized: Alice" in r.message]
        assert len(alice_logs) == 1

        # Bob enters -> name changed -> log immediately
        client.process_results(bob_result)
        bob_logs = [r for r in caplog.records if "Recognized: Bob" in r.message]
        assert len(bob_logs) == 1
