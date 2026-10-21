"""
Test suite for 'remember' intent enhancements and widened wake words.
Validates:
1. Acceptance of "remember", "remember this", "remember this <object>", "remember my <object>",
   "save this", "learn this" without failing when object name is missing.
2. Enrollment behavior with:
   - Exactly one enrollable object in view (speaks "Remembering your <class>. Hold it steady.")
   - Several in view (speaks "Which one? Say the name" and disambiguates in 8s window)
   - None in view (speaks "I don't see an object to remember. Hold it in front of the camera.")
   - Person in view (only enrolls if explicitly "remember this person")
3. Widened wake words from config.WAKE_WORDS and mishearings:
   - "hey december", "hey decision", "a decision", "decision", "revision", "division", "hey visions"
   - Fuzzy matching (ratio >= 0.75 on first 2 words) and stripping matched wake phrase.
"""

import sys
import os
import time
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from voice_command_system import VoiceCommandParser, VoiceCommandSystem
from wireless.laptop_server import LaptopMLServer


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

    def messages_text(self):
        return [m[1] for m in self.messages]

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
        self.status_messages.append(text)

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


# ─────────────────────────────────────────────────────────────
# 1. Wake Words & Mishearings Tests
# ─────────────────────────────────────────────────────────────

def test_wake_words_in_config():
    """Verify that all required wake phrases and mishearings are in config.WAKE_WORDS."""
    required = [
        "hey vision", "hey visions", "vision", "revision", "division",
        "hey december", "hey decision", "a decision", "decision",
        "virginia", "eh vision", "hey virginia", "a visions"
    ]
    for w in required:
        assert w in config.WAKE_WORDS, f"'{w}' missing from config.WAKE_WORDS"


def test_strip_wake_word_mishearings():
    """Verify stripping of new mishearings and standard wake phrases."""
    cases = [
        ("hey december take a note", True, "take a note"),
        ("hey decision find my phone", True, "find my phone"),
        ("a decision navigate", True, "navigate"),
        ("decision remember", True, "remember"),
        ("revision list all items", True, "list all items"),
        ("division take a note", True, "take a note"),
        ("hey visions shut down", True, "shut down"),
        ("hey vision remember this phone", True, "remember this phone"),
        ("vision stop navigation", True, "stop navigation"),
        ("virginia remember this cup", True, "remember this cup"),
        ("eh vision find my phone", True, "find my phone"),
        ("hey virginia take a note", True, "take a note"),
        ("a visions shut down", True, "shut down"),
    ]
    for utterance, expected_wake, expected_rem in cases:
        has_wake, rem = VoiceCommandParser.strip_wake_word(utterance)
        assert has_wake is expected_wake, f"Failed wake match for '{utterance}'"
        assert rem == expected_rem, f"Expected remainder '{expected_rem}', got '{rem}' for '{utterance}'"


def test_fuzzy_wake_word_matching():
    """Verify fuzzy matching with ratio >= 0.75 on first two words."""
    # "hay december" is very close to "hey december"
    has_wake, rem = VoiceCommandParser.strip_wake_word("hay december take a note")
    assert has_wake is True
    assert rem == "take a note"

    # "hey decishun" is close to "hey decision"
    has_wake, rem = VoiceCommandParser.strip_wake_word("hey decishun find my keys")
    assert has_wake is True
    assert rem == "find my keys"


def test_wake_mishearing_take_note_test_case():
    """Test 4 requirement: 'hey december take a note' is treated as wake phrase plus take_note."""
    parser = VoiceCommandParser()
    res = parser.parse("hey december take a note")
    assert res["wake_word"] is True
    assert res["intent"] == "take_note"
    assert res["parameters"]["text"] == ""


# ─────────────────────────────────────────────────────────────
# 2. 'remember' Intent Parsing Tests
# ─────────────────────────────────────────────────────────────

def test_remember_intent_parsing_variations():
    """Verify all required forms of 'remember' parse to remember intent and never fail when object name is missing."""
    parser = VoiceCommandParser()
    cases = [
        ("remember", ""),
        ("remember this", ""),
        ("save this", ""),
        ("learn this", ""),
        ("enroll this", ""),
        ("remember this phone", "phone"),
        ("remember my phone", "phone"),
        ("remember this cell phone", "cell phone"),
        ("remember my keys", "keys"),
        ("save this bottle", "bottle"),
        ("learn this cup", "cup"),
        ("enroll this backpack", "backpack"),
        ("remember wallet", "wallet"),
    ]

    for phrase, expected_obj in cases:
        # Test without wake phrase
        res_direct = parser.parse(phrase)
        assert res_direct["intent"] == "remember", f"Failed direct for '{phrase}'"
        assert res_direct["parameters"]["object_name"] == expected_obj, f"Wrong obj for '{phrase}'"

        # Test with wake phrase "hey vision"
        res_wake = parser.parse(f"hey vision {phrase}")
        assert res_wake["intent"] == "remember", f"Failed with wake for 'hey vision {phrase}'"
        assert res_wake["parameters"]["object_name"] == expected_obj
        assert res_wake["wake_word"] is True

        # Test with wake phrase "hey december"
        res_december = parser.parse(f"hey december {phrase}")
        assert res_december["intent"] == "remember"
        assert res_december["parameters"]["object_name"] == expected_obj
        assert res_december["wake_word"] is True


def test_remember_never_fails_on_missing_object_name():
    """Ensure parser never fails or crashes when object name is missing."""
    parser = VoiceCommandParser()
    for phrase in ["hey vision remember", "hey vision remember this", "hey vision save this", "hey vision learn this"]:
        res = parser.parse(phrase)
        assert res["intent"] == "remember"
        assert "object_name" in res["parameters"]
        assert res["parameters"]["object_name"] == ""


# ─────────────────────────────────────────────────────────────
# 3. Server 'handle_remember' Execution Tests
# ─────────────────────────────────────────────────────────────

def _setup_mock_server():
    """Create a LaptopMLServer instance with mocked ML models and dummy audio."""
    server = LaptopMLServer()
    server.audio = DummyAudio()
    server.audio_enabled = True
    server.object_memory = MagicMock()
    server.object_memory.start_enrollment = MagicMock()
    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server.voice_system.open_dictation_window = MagicMock()
    server.voice_system.open_disambiguation_window = MagicMock()
    server._register_voice_handlers()
    return server


def test_remember_one_phone_in_view():
    """
    Test 4: 'hey vision remember' with one phone in view starts enrollment of the phone.
    Speaks: 'Remembering your <class>. Hold it steady.'
    """
    server = _setup_mock_server()
    server.current_objects = [
        {"class_name": "cell phone", "bbox": [100, 100, 200, 300], "confidence": 0.88}
    ]

    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision remember")
    assert parsed["intent"] == "remember"

    handler = server.voice_system.command_callbacks["remember"]
    handler(parsed["parameters"])

    assert server.enrollment_active is True
    assert server.enrollment_target_obj == {
        "custom_name": "cell phone",
        "yolo_class": "cell phone"
    }
    server.object_memory.start_enrollment.assert_called_once_with("cell phone", "cell phone")
    assert any("Remembering your cell phone. Hold it and turn it slowly." in msg or "Remembering your cell phone. Hold it steady." in msg for msg in server.audio.social_messages)


def test_remember_this_unmatched_word_single_trackable_object():
    """
    "remember this <word>": if <word> doesn't match a detected class (or alias)
    but exactly one trackable object is in view, enroll that object and say "Remembering your <class>".
    """
    # Case A: exactly one trackable object in view (cup), word is 'gadget'
    server = _setup_mock_server()
    server.current_objects = [
        {"class_name": "cup", "bbox": [50, 50, 100, 100], "confidence": 0.85}
    ]

    parser = VoiceCommandParser()
    parsed = parser.parse("remember this gadget")
    assert parsed["intent"] == "remember"
    assert parsed["parameters"]["object_name"] == "gadget"

    handler = server.voice_system.command_callbacks["remember"]
    handler(parsed["parameters"])

    assert server.enrollment_active is True
    assert server.enrollment_target_obj["custom_name"] == "cup"
    assert server.enrollment_target_obj["yolo_class"] == "cup"
    assert "gadget" in server.enrollment_target_obj["aliases"]
    server.object_memory.start_enrollment.assert_called_once_with("cup", "cup", aliases=["gadget"])
    assert any("Remembering your cup. Hold it and turn it slowly." in msg or "Remembering your cup. Hold it steady." in msg for msg in server.audio.social_messages)

    # Case B: two trackable objects in view (cup and laptop) -> does not enroll, speaks 'No gadget detected in view'
    server = _setup_mock_server()
    server.current_objects = [
        {"class_name": "cup", "bbox": [50, 50, 100, 100], "confidence": 0.85},
        {"class_name": "laptop", "bbox": [150, 150, 350, 350], "confidence": 0.90}
    ]
    handler = server.voice_system.command_callbacks["remember"]
    handler(parsed["parameters"])
    assert server.enrollment_active is False
    assert any("No gadget detected in view" in msg for msg in server.audio.social_messages)

    # Case C: zero trackable objects (only person in view) -> does not enroll, speaks 'No gadget detected in view'
    server = _setup_mock_server()
    server.current_objects = [
        {"class_name": "person", "bbox": [10, 10, 200, 400], "confidence": 0.90}
    ]
    handler = server.voice_system.command_callbacks["remember"]
    handler(parsed["parameters"])
    assert server.enrollment_active is False
    assert any("No gadget detected in view" in msg for msg in server.audio.social_messages)


def test_remember_no_object_in_view():
    """
    Test 4: 'hey vision remember' with no object in view.
    Speaks: 'I don't see an object to remember. Hold it in front of the camera.'
    """
    server = _setup_mock_server()
    server.current_objects = []

    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision remember")

    handler = server.voice_system.command_callbacks["remember"]
    handler(parsed["parameters"])

    assert server.enrollment_active is False
    server.object_memory.start_enrollment.assert_not_called()
    assert any("I don't see an object to remember. Hold it in front of the camera." in msg for msg in server.audio.social_messages)


def test_remember_only_person_in_view():
    """
    When no object name is given, person is NOT enrollable.
    With only person in view, it treats it as none:
    Speaks: 'I don't see an object to remember. Hold it in front of the camera.'
    """
    server = _setup_mock_server()
    server.current_objects = [
        {"class_name": "person", "bbox": [50, 50, 250, 400], "confidence": 0.92}
    ]

    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision remember")

    handler = server.voice_system.command_callbacks["remember"]
    handler(parsed["parameters"])

    assert server.enrollment_active is False
    server.object_memory.start_enrollment.assert_not_called()
    assert any("I only see a person. Say remember person to remember their face." in msg for msg in server.audio.social_messages)


def test_remember_person_and_phone_in_view():
    """
    When both a person and a phone are in view, person is ignored.
    Exactly one enrollable object (phone) is used directly.
    """
    server = _setup_mock_server()
    server.current_objects = [
        {"class_name": "person", "bbox": [10, 10, 200, 400], "confidence": 0.95},
        {"class_name": "cell phone", "bbox": [250, 250, 350, 400], "confidence": 0.89}
    ]

    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision remember this")

    handler = server.voice_system.command_callbacks["remember"]
    handler(parsed["parameters"])

    assert server.enrollment_active is True
    assert server.enrollment_target_obj["yolo_class"] == "cell phone"
    server.object_memory.start_enrollment.assert_called_once_with("cell phone", "cell phone")
    assert any("Remembering your cell phone. Hold it and turn it slowly." in msg or "Remembering your cell phone. Hold it steady." in msg for msg in server.audio.social_messages)


def test_remember_explicit_this_person():
    """
    'remember this person' starts the face flow instead of object flow.
    """
    server = _setup_mock_server()
    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision remember this person")
    assert parsed["intent"] in ("remember", "enroll_face")

    handler = server.voice_system.command_callbacks[parsed["intent"]]
    handler(parsed["parameters"])

    assert any("Who is this? Say their name" in msg for msg in server.audio.social_messages)
    server.voice_system.open_dictation_window.assert_called_once()
    assert server.voice_system.open_dictation_window.call_args[1].get("timeout") == 8.0


def test_remember_person_without_this_rejected():
    """
    'remember person' starts the face flow instead of object flow.
    """
    server = _setup_mock_server()
    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision remember person")
    assert parsed["intent"] in ("remember", "enroll_face")

    handler = server.voice_system.command_callbacks[parsed["intent"]]
    handler(parsed["parameters"])

    assert any("Who is this? Say their name" in msg for msg in server.audio.social_messages)
    server.voice_system.open_dictation_window.assert_called_once()
    assert server.voice_system.open_dictation_window.call_args[1].get("timeout") == 8.0


def test_remember_multiple_objects_disambiguation():
    """
    When several objects are in view:
    - Speaks 'Which one? Say the name'
    - Opens 8s disambiguation window
    - Matching response starts enrollment for the chosen class
    """
    server = _setup_mock_server()
    server.current_objects = [
        {"class_name": "cup", "bbox": [50, 50, 100, 100], "confidence": 0.85},
        {"class_name": "laptop", "bbox": [150, 150, 350, 350], "confidence": 0.90}
    ]

    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision remember")

    handler = server.voice_system.command_callbacks["remember"]
    handler(parsed["parameters"])

    # Should ask which one
    assert any("Which one? Say the name" in msg for msg in server.audio.social_messages)
    assert server.enrollment_active is False

    # Check that open_disambiguation_window or open_dictation_window was called with timeout=8.0
    call_args = None
    if server.voice_system.open_disambiguation_window.called:
        call_args = server.voice_system.open_disambiguation_window.call_args
    elif server.voice_system.open_dictation_window.called:
        call_args = server.voice_system.open_dictation_window.call_args

    assert call_args is not None
    callback = call_args[0][0]
    timeout = call_args[1].get("timeout", call_args[0][1] if len(call_args[0]) > 1 else None)
    assert timeout == 8.0

    # User speaks "cup" (no wake word needed)
    callback("cup")

    assert server.enrollment_active is True
    assert server.enrollment_target_obj == {
        "custom_name": "cup",
        "yolo_class": "cup"
    }
    server.object_memory.start_enrollment.assert_called_once_with("cup", "cup")
    assert any("Remembering your cup. Hold it and turn it slowly." in msg or "Remembering your cup. Hold it steady." in msg for msg in server.audio.social_messages)


def test_remember_already_enrolling():
    """If enrollment is already in progress, speaks 'Already enrolling an object'."""
    server = _setup_mock_server()
    server.enrollment_active = True

    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision remember")

    handler = server.voice_system.command_callbacks["remember"]
    handler(parsed["parameters"])

    assert any("Already enrolling an object" in msg for msg in server.audio.messages_text())


# ─────────────────────────────────────────────────────────────
# 4. Find Intent, Object Matching & Command Reply Tests
# ─────────────────────────────────────────────────────────────

def test_remember_handler_runs_without_optional_name_error():
    """Ensure handle_remember runs without NameError: name 'Optional' is not defined."""
    server = _setup_mock_server()
    server.current_objects = []
    handler = server.voice_system.command_callbacks["remember"]
    # Run with empty params and with specific object params
    handler({})
    handler({"object_name": "phone", "raw_command": "remember this phone"})
    assert any("I don't see an object to remember" in msg for msg in server.audio.messages_text())


def test_find_with_no_object_name_asks_and_opens_8s_window():
    """'hey vision find' with no object name speaks 'What should I find?' and opens 8s window."""
    server = _setup_mock_server()
    server.object_memory.list_remembered_objects.return_value = [
        ("phone", "cell phone"),
        ("mug", "cup"),
    ]
    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision find")
    assert parsed["intent"] == "find"
    assert parsed["parameters"]["object_name"] == ""

    handler = server.voice_system.command_callbacks["find"]
    handler(parsed["parameters"])

    assert any("What should I find?" in m for m in server.audio.messages_text())
    call_args = (server.voice_system.open_disambiguation_window.call_args or
                 server.voice_system.open_dictation_window.call_args)
    assert call_args is not None
    timeout = call_args[1].get("timeout", call_args[0][1] if len(call_args[0]) > 1 else None)
    assert timeout == 8.0


def test_find_with_single_remembered_object_auto_selects():
    """'hey vision find' with exactly 1 remembered object uses it directly."""
    server = _setup_mock_server()
    server.object_memory.list_remembered_objects.return_value = [
        ("phone", "cell phone")
    ]
    server.object_memory.get_object_location.return_value = {
        "location": "table",
        "timestamp": time.time() - 10
    }
    server.find_guide = MagicMock()
    server.find_guide.describe_last_seen.return_value = "on the table"

    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision find")
    assert parsed["intent"] == "find"
    assert parsed["parameters"]["object_name"] == ""

    handler = server.voice_system.command_callbacks["find"]
    handler(parsed["parameters"])

    # Disambiguation window was NOT opened because there was only 1 object
    assert not server.voice_system.open_disambiguation_window.called
    assert not server.voice_system.open_dictation_window.called
    # Guidance started for phone
    server.find_guide.start.assert_called_once_with("phone", "")
    assert any("Your phone was" in m for m in server.audio.messages_text())


def test_find_pan_with_only_phone_answers_with_spoken_message():
    """'find pan' with only phone remembered rejects pan and speaks error with remembered list."""
    server = _setup_mock_server()
    server.object_memory.list_remembered_objects.return_value = [
        ("phone", "cell phone")
    ]
    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision find pan")
    assert parsed["intent"] == "find"
    assert parsed["parameters"]["object_name"] == "pan"

    handler = server.voice_system.command_callbacks["find"]
    handler(parsed["parameters"])

    # Pan is not accepted as a new object, and an informative reply is spoken
    assert any("I don't know an object called pan. I remember: phone" in m for m in server.audio.messages_text())
    assert len(server.audio.messages) > 0


def test_find_synonyms_phone_mobile_cellphone():
    """Synonyms 'mobile', 'cell phone', 'my phone' match remembered 'phone'."""
    server = _setup_mock_server()
    server.object_memory.list_remembered_objects.return_value = [
        ("phone", "cell phone")
    ]
    server.object_memory.get_object_location.return_value = None
    server.item_tracker = MagicMock()
    server.item_tracker.query_item_location.return_value = None
    server.find_guide = MagicMock()

    parser = VoiceCommandParser()
    for phrase in ["hey vision find mobile", "hey vision find cell phone", "hey vision find my phone"]:
        server.audio.messages.clear()
        server.find_guide.start.reset_mock()
        parsed = parser.parse(phrase)
        handler = server.voice_system.command_callbacks["find"]
        handler(parsed["parameters"])
        server.find_guide.start.assert_called_once_with("phone", "")
        assert any("I remember your phone but haven't spotted it yet" in m for m in server.audio.messages_text())


def test_command_reply_spoken_during_quiet_mode():
    """Command replies are NEVER suppressed by quiet mode, while routine announcements are suppressed."""
    from audio_feedback import AudioFeedback
    af = AudioFeedback(use_gtts=False)
    # Quiet mode active (e.g. dictation or within 3s of wake word)
    af.set_quiet_mode_checker(lambda: True)

    # 1. Routine announcements: "Person ahead", "Hello Bharath", "found your mug"
    af.speak_navigational("Person ahead")
    af.speak_social("Hello Bharath")
    af.speak_social("found your mug")
    queued_texts = [item[1] for item in list(af.audio_queue.queue) if item[1] != "__STOP__"]
    assert "Person ahead" not in queued_texts
    assert "Hello Bharath" not in queued_texts
    assert "found your mug" not in queued_texts

    # 2. Command replies: "What should I find?", "Shutting down", etc.
    af.speak_command_reply("What should I find?")
    af.speak_social("Shutting down", is_command_reply=True)
    queued_texts = [item[1] for item in list(af.audio_queue.queue) if item[1] != "__STOP__"]
    assert "What should I find?" in queued_texts
    assert "Shutting down" in queued_texts


def test_bare_shutdown_does_nothing():
    """Bare 'shutdown', 'shut down', or 'quit' without wake word does nothing and is ignored."""
    server = LaptopMLServer()
    server.audio = DummyAudio()
    server.audio_enabled = True
    server.running = True
    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server.voice_system.enabled = True
    server.voice_system.running = True
    server._register_voice_handlers()

    # 1. Bare shutdown without wake word does nothing
    res = server.voice_system.process_text("shutdown")
    assert res is None
    assert server.running is True
    assert len(server.audio.messages) == 0

    res2 = server.voice_system.process_text("shut down")
    assert res2 is None
    assert server.running is True
    assert len(server.audio.messages) == 0

    res3 = server.voice_system.process_text("quit")
    assert res3 is None
    assert server.running is True
    assert len(server.audio.messages) == 0

    # 2. Wake phrase + shutdown triggers shutdown
    res_wake = server.voice_system.process_text("hey vision shutdown")
    assert res_wake is not None
    assert res_wake["intent"] == "quit"
    assert server.running is False
    assert any("Shutting down" in m for m in server.audio.messages_text())

