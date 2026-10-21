"""
Unit and integration tests for the 9 Voice, STT, and Enrollment updates:
1. Whistle keywords fixed vocabulary & db name filtering
2. Canonical names for "remember <name>" (cello/sell/selfour -> cell phone, no mishearing aliases)
3. Yes/no vocabulary (yeah, yep, etc.) and direct find resolution for sim >= 0.6
4. "remember person" / "remember me" / "remember this person" starts face flow
5. Audio capture pre/post buffer & thresholds
6. STT_ENGINE = "hybrid" fallback to Whistle on error or offline
7. Transcript logging DEBUG vs INFO
8. Shutdown mishearings and single-word fuzzy matching (immediate shutdown)
"""

import sys
import os
import time
import logging
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
import speech_recognition as sr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import stt_engine
from voice_command_system import VoiceCommandParser, VoiceCommandSystem, VoiceCapture, YES_WORDS, NO_WORDS, SHUTDOWN_VOCABULARY
from wireless.laptop_server import LaptopMLServer, clean_object_name, calculate_name_similarity


class DummyAudio:
    """Mock audio feedback for tracking spoken messages."""
    def __init__(self):
        self.messages = []
        self.social_messages = []
        self.status_messages = []
        self.is_speaking = False
        self.is_playing = False
        self.last_speech_end_time = 0.0

    def speak(self, text, priority=None, is_command_reply=False):
        self.messages.append(text)

    def speak_social(self, text, is_command_reply=False):
        self.social_messages.append(text)
        self.messages.append(text)

    def speak_status(self, text):
        self.status_messages.append(text)
        self.messages.append(text)

    def speak_command_reply(self, text, priority=None):
        self.messages.append(text)
        self.social_messages.append(text)


def _setup_mock_server():
    """Create a LaptopMLServer instance with mocked ML models and dummy audio."""
    server = LaptopMLServer()
    server.audio = DummyAudio()
    server.audio_enabled = True
    server.object_memory = MagicMock()
    server.object_memory.start_enrollment = MagicMock()
    server.object_memory.list_remembered_objects = MagicMock(return_value=[])
    server.object_memory.get_object_aliases = MagicMock(return_value=[])
    server.object_memory.get_object_location = MagicMock(return_value=None)
    server.item_tracker = MagicMock()
    server.item_tracker.query_item_location = MagicMock(return_value=None)
    server.item_tracker.update_tracking = MagicMock(return_value=[])
    server.object_detector = MagicMock()
    server.object_detector.detect = MagicMock(return_value=[])
    server.spatial_navigator = MagicMock()
    server.spatial_navigator.generate_guidance = MagicMock(return_value=[])
    server.face_recognizer = MagicMock()
    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server.voice_system.open_dictation_window = MagicMock()
    server.voice_system.open_disambiguation_window = MagicMock()
    server.voice_system.open_yes_no_window = MagicMock()
    server._register_voice_handlers()
    return server


# ─── 1. Canonical Names: "cello", "sell", "selfour" enroll as "cell phone" ──────

def test_canonical_name_cello_enrols_as_cell_phone():
    """'remember cello' with cell phone in view enroled as 'cell phone' with no mishearing alias."""
    server = _setup_mock_server()
    server.current_objects = [
        {"class_name": "cell phone", "bbox": [100, 100, 200, 300], "confidence": 0.88}
    ]

    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision remember cello")
    assert parsed["intent"] == "remember"
    assert parsed["parameters"]["object_name"] == "cello"

    handler = server.voice_system.command_callbacks["remember"]
    handler(parsed["parameters"])

    assert server.enrollment_active is True
    assert server.enrollment_target_obj["custom_name"] == "cell phone"
    assert server.enrollment_target_obj["yolo_class"] == "cell phone"
    # Mishearing "cello" must NEVER be stored as an alias
    assert "cello" not in server.enrollment_target_obj.get("aliases", [])
    server.object_memory.start_enrollment.assert_called_once_with("cell phone", "cell phone")
    assert any("Remembering your cell phone. Hold it and turn it slowly." in msg or "Remembering your cell phone. Hold it steady." in msg for msg in server.audio.messages)


def test_canonical_name_sell_enrols_as_cell_phone():
    """'remember sell' with cell phone in view enroled as 'cell phone' with no mishearing alias."""
    server = _setup_mock_server()
    server.current_objects = [
        {"class_name": "cell phone", "bbox": [100, 100, 200, 300], "confidence": 0.88}
    ]

    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision remember sell")
    assert parsed["intent"] == "remember"

    handler = server.voice_system.command_callbacks["remember"]
    handler(parsed["parameters"])

    assert server.enrollment_active is True
    assert server.enrollment_target_obj["custom_name"] == "cell phone"
    assert "sell" not in server.enrollment_target_obj.get("aliases", [])
    server.object_memory.start_enrollment.assert_called_once_with("cell phone", "cell phone")


def test_canonical_name_selfour_enrols_as_cell_phone():
    """'remember selfour' with cell phone in view enroled as 'cell phone' with no mishearing alias."""
    server = _setup_mock_server()
    server.current_objects = [
        {"class_name": "cell phone", "bbox": [100, 100, 200, 300], "confidence": 0.88}
    ]

    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision remember selfour")
    assert parsed["intent"] == "remember"

    handler = server.voice_system.command_callbacks["remember"]
    handler(parsed["parameters"])

    assert server.enrollment_active is True
    assert server.enrollment_target_obj["custom_name"] == "cell phone"
    assert "selfour" not in server.enrollment_target_obj.get("aliases", [])
    server.object_memory.start_enrollment.assert_called_once_with("cell phone", "cell phone")


def test_canonical_name_real_alias_retained():
    """'remember phone' or 'remember mobile' retains real alias."""
    server = _setup_mock_server()
    server.current_objects = [
        {"class_name": "cell phone", "bbox": [100, 100, 200, 300], "confidence": 0.88}
    ]

    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision remember mobile")
    assert parsed["intent"] == "remember"

    handler = server.voice_system.command_callbacks["remember"]
    handler(parsed["parameters"])

    assert server.enrollment_active is True
    assert server.enrollment_target_obj["custom_name"] == "cell phone"
    # Real alias "mobile" should be present in aliases
    assert "mobile" in server.enrollment_target_obj.get("aliases", [])


# ─── 2. Yes/No Vocabulary: "yeah", "yep", "sure", etc. ──────────────────────────

@pytest.mark.parametrize("yes_word", ["yes", "yeah", "yep", "ya", "yup", "sure", "ok", "okay", "correct", "right", "please"])
def test_yes_vocabulary_acceptance(yes_word):
    """All required words count as YES in yes/no confirmation window."""
    vcs = VoiceCommandSystem()
    on_yes_mock = MagicMock()
    on_no_mock = MagicMock()
    vcs.open_yes_no_window(on_yes=on_yes_mock, on_no=on_no_mock, timeout=10.0)

    # Process spoken word
    result = vcs.process_text(yes_word)
    assert result is not None
    assert result["intent"] == "yes"
    on_yes_mock.assert_called_once()
    on_no_mock.assert_not_called()
    assert vcs.yes_no_active is False


@pytest.mark.parametrize("no_word", ["no", "nope", "nah", "cancel", "wrong"])
def test_no_vocabulary_rejection(no_word):
    """All required words count as NO in yes/no confirmation window."""
    vcs = VoiceCommandSystem()
    on_yes_mock = MagicMock()
    on_no_mock = MagicMock()
    vcs.open_yes_no_window(on_yes=on_yes_mock, on_no=on_no_mock, timeout=10.0)

    result = vcs.process_text(no_word)
    assert result is not None
    assert result["intent"] == "no"
    on_no_mock.assert_called_once()
    on_yes_mock.assert_not_called()
    assert vcs.yes_no_active is False


# ─── 3. Face Flow for "remember person", "remember me", "remember this person" ───

def test_remember_person_starts_face_flow():
    """'remember person' starts face flow, asks 'Who is this? Say their name', opens 8s window."""
    server = _setup_mock_server()
    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision remember person")
    assert parsed["intent"] in ("remember", "enroll_face")

    handler = server.voice_system.command_callbacks[parsed["intent"]]
    handler(parsed["parameters"])

    assert any("Who is this? Say their name" in msg for msg in server.audio.messages)
    server.voice_system.open_dictation_window.assert_called_once()
    assert server.voice_system.open_dictation_window.call_args[1].get("timeout") == 8.0


def test_remember_me_starts_face_flow():
    """'remember me' starts face flow."""
    server = _setup_mock_server()
    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision remember me")
    assert parsed["intent"] in ("remember", "enroll_face")

    handler = server.voice_system.command_callbacks[parsed["intent"]]
    handler(parsed["parameters"])

    assert any("Who is this? Say their name" in msg for msg in server.audio.messages)
    server.voice_system.open_dictation_window.assert_called_once()
    assert server.voice_system.open_dictation_window.call_args[1].get("timeout") == 8.0


def test_remember_this_person_starts_face_flow():
    """'remember this person' starts face flow."""
    server = _setup_mock_server()
    parser = VoiceCommandParser()
    parsed = parser.parse("hey vision remember this person")
    assert parsed["intent"] in ("remember", "enroll_face")

    handler = server.voice_system.command_callbacks[parsed["intent"]]
    handler(parsed["parameters"])

    assert any("Who is this? Say their name" in msg for msg in server.audio.messages)
    server.voice_system.open_dictation_window.assert_called_once()
    assert server.voice_system.open_dictation_window.call_args[1].get("timeout") == 8.0


def test_face_flow_no_face_visible_speaks_camera_instruction():
    """After confirming name, if no face is visible when face flow runs, speaks 'I can't see a face. Face the camera.'"""
    server = _setup_mock_server()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    server.current_frame = dummy_frame

    # Face recognizer returns no faces
    server.face_recognizer.detect_faces = MagicMock(return_value=[])

    parser = VoiceCommandParser()
    parsed = parser.parse("remember person")
    handler = server.voice_system.command_callbacks[parsed["intent"]]
    handler(parsed["parameters"])

    # Extract the on_dictate callback passed to open_dictation_window
    on_dictate_cb = server.voice_system.open_dictation_window.call_args[1]["on_dictate"]

    # Provide name "Alice"
    on_dictate_cb("Alice")

    # Flow asks confirmation: "I heard Alice. Is that right?" and opens yes/no window
    assert any("I heard Alice. Is that right?" in msg for msg in server.audio.messages)
    server.voice_system.open_yes_no_window.assert_called_once()
    on_yes_cb = server.voice_system.open_yes_no_window.call_args[1]["on_yes"]

    # Confirm YES
    on_yes_cb()

    assert any("I can't see a face. Face the camera." in msg for msg in server.audio.messages)
    server.face_recognizer.enroll_face.assert_not_called()


def test_face_flow_face_visible_enrols_face():
    """After confirming name, if face is visible when face flow runs, enrols face and speaks 'Enrolled {name}'."""
    server = _setup_mock_server()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    server.current_frame = dummy_frame

    # Face recognizer detects a face and succeeds in enrollment
    server.face_recognizer.detect_faces = MagicMock(return_value=[{"bbox": [10, 10, 100, 100]}])
    server.face_recognizer.enroll_face = MagicMock(return_value=True)

    parser = VoiceCommandParser()
    parsed = parser.parse("remember this person")
    handler = server.voice_system.command_callbacks[parsed["intent"]]
    handler(parsed["parameters"])

    on_dictate_cb = server.voice_system.open_dictation_window.call_args[1]["on_dictate"]
    on_dictate_cb("Alice")

    # Confirms "I heard Alice. Is that right?"
    assert any("I heard Alice. Is that right?" in msg for msg in server.audio.messages)
    server.voice_system.open_yes_no_window.assert_called_once()
    on_yes_cb = server.voice_system.open_yes_no_window.call_args[1]["on_yes"]

    # Confirm YES
    on_yes_cb()

    server.face_recognizer.enroll_face.assert_called_once_with(dummy_frame, "Alice")
    assert any("Enrolled Alice" in msg for msg in server.audio.messages)


# ─── 4. STT Hybrid Engine Fallback ──────────────────────────────────────────────

def test_hybrid_stt_falls_back_to_whistle_when_google_fails():
    """STT_ENGINE = 'hybrid' tries Google first, falls back to Whistle when Google raises exception."""
    vcs = VoiceCommandSystem()
    mock_recognizer = MagicMock()
    mock_audio = MagicMock()

    # Google raises RequestError
    mock_recognizer.recognize_google.side_effect = sr.RequestError("Network dropped")

    with patch.object(config, 'STT_ENGINE', 'hybrid'), \
         patch.object(stt_engine, 'is_available', return_value=True), \
         patch.object(stt_engine, 'is_internet_reachable', return_value=True), \
         patch.object(stt_engine, 'transcribe_with_timing', return_value=("hey vision find my bottle", 85)), \
         patch.object(stt_engine, 'is_filler', return_value=False):

        result = vcs._recognize_audio(mock_recognizer, mock_audio)

        assert mock_recognizer.recognize_google.called
        assert result == "hey vision find my bottle"


def test_hybrid_stt_uses_whistle_directly_when_offline():
    """STT_ENGINE = 'hybrid' uses Whistle directly when is_internet_reachable is False."""
    vcs = VoiceCommandSystem()
    mock_recognizer = MagicMock()
    mock_audio = MagicMock()

    with patch.object(config, 'STT_ENGINE', 'hybrid'), \
         patch.object(stt_engine, 'is_available', return_value=True), \
         patch.object(stt_engine, 'is_internet_reachable', return_value=False), \
         patch.object(stt_engine, 'transcribe_with_timing', return_value=("hey vision shutdown", 90)), \
         patch.object(stt_engine, 'is_filler', return_value=False):

        result = vcs._recognize_audio(mock_recognizer, mock_audio)

        # Google should NOT have been called
        mock_recognizer.recognize_google.assert_not_called()
        assert result == "hey vision shutdown"


# ─── 5. Single Remembered Object Direct Find (ratio >= 0.6) ─────────────────────

def test_single_remembered_object_direct_find_ratio_gte_0_6():
    """When a single remembered object matches spoken name with ratio >= 0.6, use it directly without asking."""
    server = _setup_mock_server()
    # Exactly 1 remembered object: "water bottle"
    server.object_memory.list_remembered_objects.return_value = [("water bottle", "bottle")]
    server.find_guide = MagicMock()
    server.find_guide.start = MagicMock()

    # Query is "bottle" (calculate_name_similarity("bottle", "water bottle") >= 0.6)
    handler = server.voice_system.command_callbacks["find"]
    handler({"object_name": "bottle"})

    # Should NOT open yes/no window
    server.voice_system.open_yes_no_window.assert_not_called()
    assert server.find_guide.start.called
    assert any("bottle" in msg for msg in server.audio.messages)


def test_single_remembered_object_weak_match_asks_confirmation():
    """When similarity is between 0.35 and 0.6, asks 'Did you mean your {name}?'."""
    server = _setup_mock_server()
    server.object_memory.list_remembered_objects.return_value = [("television", "tv")]

    # Query with similarity in [0.35, 0.6)
    with patch("wireless.laptop_server.calculate_name_similarity", return_value=0.45):
        handler = server.voice_system.command_callbacks["find"]
        handler({"object_name": "teliv"})

        assert any("Did you mean your television?" in msg for msg in server.audio.messages)
        server.voice_system.open_yes_no_window.assert_called_once()


# ─── 6. Shutdown Mishearings and Single-Word Fuzzy Match ─────────────────────────

@pytest.mark.parametrize("phrase", [
    "shadow", "shut dawn", "shut doubt", "shut out", "shot down",
    "shutting down", "shut it down", "stop everything", "quit", "exit", "close"
])
def test_shutdown_mishearing_phrases_trigger_quit(phrase):
    """All 13 shutdown phrases/mishearings trigger 'quit' intent immediately."""
    parser = VoiceCommandParser()
    parsed = parser.parse(f"hey vision {phrase}")
    assert parsed["intent"] == "quit"


@pytest.mark.parametrize("word", ["shutt", "quitt", "shot", "exitt"])
def test_shutdown_single_word_fuzzy_match(word):
    """Any single word after wake word with ratio >= 0.55 against shutdown vocab triggers quit."""
    parser = VoiceCommandParser()
    parsed = parser.parse(f"hey vision {word}")
    assert parsed["intent"] == "quit"


def test_shutdown_executes_immediately_no_confirmation():
    """Shutdown executes immediately with no confirmation window required."""
    server = _setup_mock_server()
    server._shutdown = MagicMock()

    handler = server.voice_system.command_callbacks["quit"]
    handler({})

    server._shutdown.assert_called_once()
    server.voice_system.open_yes_no_window.assert_not_called()


# ─── 7. Audio Capture Pre/Post Buffer & Thresholds ──────────────────────────────

def test_audio_capture_settings():
    """VoiceCapture initializes with pause_threshold=0.8, non_speaking_duration=0.5, 0.4s pre, 0.3s post."""
    capture = VoiceCapture()
    assert capture.recognizer.pause_threshold == 0.8
    assert capture.recognizer.non_speaking_duration == 0.5
    assert getattr(capture.recognizer, 'non_speaking_before', None) == 0.4
    assert getattr(capture.recognizer, 'non_speaking_after', None) == 0.3


# ─── 8. Transcript Logging: DEBUG vs INFO ────────────────────────────────────────

def test_transcript_logging_debug_when_no_wake_and_no_window(caplog):
    """Transcripts containing no wake word and no open window are logged at DEBUG, not INFO."""
    vcs = VoiceCommandSystem()

    with caplog.at_level(logging.DEBUG):
        caplog.clear()
        vcs.process_text("just random background noise conversation")

        # Must NOT log "Processing voice input" at INFO
        info_records = [r for r in caplog.records if r.levelno == logging.INFO and "Processing voice input" in r.message]
        assert len(info_records) == 0

        # Must log at DEBUG
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG and "Processing voice input" in r.message]
        assert len(debug_records) > 0


def test_transcript_logging_info_when_wake_word_present(caplog):
    """Transcripts containing wake word are logged at INFO."""
    vcs = VoiceCommandSystem()

    with caplog.at_level(logging.INFO):
        caplog.clear()
        vcs.process_text("hey vision help")

        info_records = [r for r in caplog.records if r.levelno == logging.INFO and "Processing voice input" in r.message]
        assert len(info_records) > 0


# ─── 9. Whistle Keywords: Fixed Vocab & DB Filtering ────────────────────────────

def test_whistle_default_keywords_include_fixed_vocab():
    """Whistle DEFAULT_KEYWORDS includes fixed object vocab, shutdown, and commands."""
    for kw in ["cell phone", "phone", "mobile", "bottle", "chair", "keys", "clock", "tv", "couch", "person", "shutdown", "shut down"]:
        assert kw in stt_engine.DEFAULT_KEYWORDS


def test_get_db_names_filters_mishearings():
    """get_db_names only includes remembered objects that are real class names or aliases, not mishearings."""
    mock_db = MagicMock()
    mock_db.get_all_faces.return_value = [(1, "Alice", None)]
    # Includes real object and a mishearing object
    mock_db.get_all_remembered_objects.return_value = [
        (1, "cell phone", "cell phone"),
        (2, "cello", "cell phone"),       # Mishearing as custom name
        (3, "water bottle", "bottle")
    ]
    mock_db.cursor = None

    names = stt_engine.get_db_names(mock_db)
    assert "Alice" in names
    assert "cell phone" in names
    assert "water bottle" in names
    # "cello" must NOT be in STT keywords
    assert "cello" not in names


# ─── 10. Face Name Confirmation, Retries & Hints ────────────────────────────────

def test_face_enrollment_confirmation_retries_and_cancellation():
    """Face flow confirms 'I heard <name>. Is that right?', retries on 'no' up to 2 times, then cancels."""
    server = _setup_mock_server()
    handler = server.voice_system.command_callbacks["enroll_face"]
    handler({"name": ""})

    assert any("Who is this? Say their name" in msg for msg in server.audio.messages)
    dictate_cb1 = server.voice_system.open_dictation_window.call_args[1]["on_dictate"]

    # Utterance 1: "Bob"
    dictate_cb1("Bob")
    assert any("I heard Bob. Is that right?" in msg for msg in server.audio.messages)
    on_no_cb1 = server.voice_system.open_yes_no_window.call_args[1]["on_no"]

    # Reject 1 -> retries_left = 1
    on_no_cb1()
    dictate_cb2 = server.voice_system.open_dictation_window.call_args[1]["on_dictate"]

    # Utterance 2: "Charlie"
    dictate_cb2("Charlie")
    assert any("I heard Charlie. Is that right?" in msg for msg in server.audio.messages)
    on_no_cb2 = server.voice_system.open_yes_no_window.call_args[1]["on_no"]

    # Reject 2 -> retries_left = 0
    on_no_cb2()
    dictate_cb3 = server.voice_system.open_dictation_window.call_args[1]["on_dictate"]

    # Utterance 3: "David"
    dictate_cb3("David")
    assert any("I heard David. Is that right?" in msg for msg in server.audio.messages)
    on_no_cb3 = server.voice_system.open_yes_no_window.call_args[1]["on_no"]

    # Reject 3 -> max retries exceeded -> Cancelled
    on_no_cb3()
    assert any("Cancelled" in msg for msg in server.audio.messages)


def test_face_name_fuzzy_matches_hint():
    """If heard face name fuzzy-matches FACE_NAME_HINTS (ratio >= 0.6), use hint spelling."""
    server = _setup_mock_server()
    handler = server.voice_system.command_callbacks["enroll_face"]
    handler({"name": ""})

    dictate_cb = server.voice_system.open_dictation_window.call_args[1]["on_dictate"]
    # "barath" vs "Bharath" ratio ~ 0.86 >= 0.6
    dictate_cb("barath")
    assert any("I heard Bharath. Is that right?" in msg for msg in server.audio.messages)

    # Reset messages and test "dhanus" vs "Dhanush" ratio ~ 0.92 >= 0.6
    server.audio.messages.clear()
    handler({"name": ""})
    dictate_cb = server.voice_system.open_dictation_window.call_args[1]["on_dictate"]
    dictate_cb("dhanus")
    assert any("I heard Dhanush. Is that right?" in msg for msg in server.audio.messages)


def test_dictation_window_opens_after_tts():
    """Dictation window timer starts counting down only AFTER TTS finishes speaking."""
    audio = DummyAudio()
    vcs = VoiceCommandSystem(audio_feedback=audio)
    audio.is_speaking = True
    vcs.open_dictation_window(on_dictate=lambda t: None, timeout=8.0)

    # While speaking, timer has not started
    time.sleep(0.08)
    assert audio.is_speaking is True
    assert vcs.dictation_active is True

    # Stop speaking
    audio.is_speaking = False
    audio.last_speech_end_time = time.time() - 0.3
    time.sleep(0.12)

    # Official timer is now running
    assert vcs.dictation_active is True
    assert vcs.dictation_until <= time.time() + 8.1
    assert vcs.dictation_until >= time.time() + 7.5


# ─── 11. Face Management: Forget & Rename Faces, Find Face ──────────────────────

def test_forget_checks_objects_first_then_faces():
    """'forget <name>' checks object memory first, then enrolled face names."""
    server = _setup_mock_server()
    server.object_memory.list_remembered_objects.return_value = [("phone", "cell phone")]
    server.db = MagicMock()
    server.db.get_all_faces.return_value = [(1, "Bharath", None)]

    # 1. Forget remembered object "phone"
    server.voice_system.command_callbacks["forget"]({"object_name": "phone"})
    server.object_memory.delete_object.assert_called_with("phone")
    assert any("Forgot your phone" in msg for msg in server.audio.messages)

    # 2. Forget enrolled face "Bharath"
    server.voice_system.command_callbacks["forget"]({"object_name": "Bharath"})
    server.face_recognizer.delete_face.assert_called_with("Bharath")
    assert any("Forgot Bharath" in msg for msg in server.audio.messages)


def test_forget_face_and_rename_face_commands():
    """'forget face <name>' and 'rename face <old> to <new>' operate on face storage."""
    server = _setup_mock_server()
    server.db = MagicMock()
    server.db.get_all_faces.return_value = [(1, "Bharath", None)]
    server.face_recognizer.delete_face = MagicMock(return_value=True)
    server.face_recognizer.rename_face = MagicMock(return_value=True)

    # Forget face
    server.voice_system.command_callbacks["forget_face"]({"face_name": "Bharath"})
    server.face_recognizer.delete_face.assert_called_with("Bharath")
    assert any("Forgot Bharath" in msg for msg in server.audio.messages)

    # Rename face
    server.voice_system.command_callbacks["rename_face"]({"old_name": "Bharath", "new_name": "Bob"})
    server.face_recognizer.rename_face.assert_called_with("Bharath", "Bob")
    assert any("Renamed face Bharath to Bob" in msg for msg in server.audio.messages)


def test_find_face_in_view_and_not_in_view():
    """'find <face name>' says '<name> is in front of you' or 'I can't see <name>'."""
    server = _setup_mock_server()
    server.db = MagicMock()
    server.db.get_all_faces.return_value = [(1, "Bharath", None)]

    # Case 1: In view
    server.last_recognized_faces = ["Bharath"]
    server.voice_system.command_callbacks["find"]({"object_name": "Bharath"})
    assert any("Bharath is in front of you" in msg for msg in server.audio.messages)

    # Case 2: Not in view
    server.audio.messages.clear()
    server.last_recognized_faces = []
    server.face_recognizer.recognize_faces = MagicMock(return_value=[])
    server.voice_system.command_callbacks["find"]({"object_name": "Bharath"})
    assert any("I can't see Bharath" in msg for msg in server.audio.messages)


# ─── 12. Greetings: Once Per Appearance & 60s Cooldown ──────────────────────────

def test_greetings_once_per_appearance_and_60s_cooldown():
    """Greet each known person once per appearance; require 10s out of view and min 60s cooldown."""
    server = _setup_mock_server()
    server.audio_enabled = True
    server.is_quiet_mode_active = MagicMock(return_value=False)
    server.face_recognizer.recognize_faces = MagicMock(
        return_value=[{"name": "Alice", "bbox": [10, 10, 100, 100], "confidence": 0.9}]
    )
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1 at t=100.0 -> First appearance -> Greeted
    with patch("time.time", return_value=100.0):
        server.process_ml(dummy_frame, None, False)
    assert any("Hello Alice" in msg for msg in server.audio.social_messages)

    # Frame 2 at t=105.0 -> Still in view (same appearance) -> NOT repeated
    server.audio.social_messages.clear()
    with patch("time.time", return_value=105.0):
        server.process_ml(dummy_frame, None, False)
    assert not any("Hello Alice" in msg for msg in server.audio.social_messages)

    # Frame 3 at t=112.0 -> Out of view for only 7s (< 10s) -> Same appearance -> NOT repeated
    with patch("time.time", return_value=112.0):
        server.process_ml(dummy_frame, None, False)
    assert not any("Hello Alice" in msg for msg in server.audio.social_messages)

    # Frame 4 at t=130.0 -> Out of view for 18s (>= 10s -> new appearance), BUT only 30s elapsed (< 60s cooldown) -> NOT repeated
    with patch("time.time", return_value=130.0):
        server.process_ml(dummy_frame, None, False)
    assert not any("Hello Alice" in msg for msg in server.audio.social_messages)

    # Frame 5 at t=175.0 -> 63s out of view (> 10s) and 75s since last greeting (> 60s cooldown) -> GREETED
    with patch("time.time", return_value=175.0):
        server.process_ml(dummy_frame, None, False)
    assert any("Hello Alice" in msg for msg in server.audio.social_messages)


# ─── 13. Remember Only Person in View ───────────────────────────────────────────

def test_remember_only_person_in_view_speaks_remember_person():
    """'remember' with only a person in view says 'I only see a person. Say remember person to remember their face.'"""
    server = _setup_mock_server()
    server.current_objects = [{"class_name": "person", "bbox": [10, 10, 200, 300], "confidence": 0.9}]
    server.voice_system.command_callbacks["remember"]({"object_name": ""})
    assert any("I only see a person. Say remember person to remember their face." in msg for msg in server.audio.messages)


# ─── 14. Wake Phrases: 'have a vision', 'have vision', Trailing Noise ────────────

def test_wake_phrases_have_a_vision_and_trailing_noise():
    """Fuzzy matching recognizes 'have a vision', 'have vision', 'hey vision' with trailing noise."""
    parser = VoiceCommandParser()

    # Exact variations
    m1, r1 = parser.strip_wake_word("have a vision")
    assert m1 is True and r1 == ""

    m2, r2 = parser.strip_wake_word("have vision")
    assert m2 is True and r2 == ""

    # With trailing noise words
    m3, r3 = parser.strip_wake_word("hey vision uh")
    assert m3 is True and r3 == ""

    m4, r4 = parser.strip_wake_word("have a vision um find my phone")
    assert m4 is True and r4 == "find my phone"

    m5, r5 = parser.strip_wake_word("have vision ah navigate")
    assert m5 is True and r5 == "navigate"


# ─── 15. Clean Shutdown: Websocket Before DB & Quiet Frame Handler ──────────────

def test_shutdown_order_and_quiet_frame_handler_on_closed_db(caplog):
    """Websocket server is closed BEFORE database, and frame handler returns quietly on closed DB with no error log."""
    server = _setup_mock_server()
    server.server = MagicMock()
    server.db = MagicMock()
    call_order = []
    server.server.close.side_effect = lambda: call_order.append("server")
    server.db.close.side_effect = lambda: call_order.append("db")

    server.cleanup()
    assert "server" in call_order
    assert "db" in call_order
    assert call_order.index("server") < call_order.index("db")

    # Frame handler with closed DB returns quietly without error
    import asyncio
    import sqlite3

    async def _simulate_frame_handling():
        mock_ws = MagicMock()
        mock_ws.remote_address = ("127.0.0.1", 12345)
        # Message with valid json
        import json, base64
        msg = json.dumps({'frame': base64.b64encode(b'dummy').decode('ascii'), 'timestamp': time.time()})
        mock_ws.__aiter__.return_value = [msg]

        server.running = True
        server.db.is_closed = True
        server.process_ml = MagicMock(side_effect=sqlite3.ProgrammingError("Cannot operate on a closed database."))

        with caplog.at_level(logging.DEBUG):
            caplog.clear()
            await server.process_frame(mock_ws)

    asyncio.run(_simulate_frame_handling())

    # Verify no ERROR log was produced for closed database
    error_logs = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(error_logs) == 0

