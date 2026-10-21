"""
Unit tests for VisionMate Voice-Only Interface.
Tests:
1. Intent parsing with and without the wake phrase
2. Yes/no window logic (confirm/reject without wake word, 6s window)
3. Echo-suppression flag (is_speaking, last_speech_end_time, is_tts_active)
4. "find my mobile" and "stop finding" through the wake-word path
5. Voice notes (take_note, read_notes, delete_note, clear_notes, auto-read)
"""

import sys
import os
import time
import pytest

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from voice_command_system import VoiceCommandParser, VoiceCommandSystem
from audio_feedback import AudioFeedback
import speech_notes


class DummyAudioFeedback:
    """Mock audio feedback for deterministic testing."""
    def __init__(self):
        self.is_speaking = False
        self.is_playing = False
        self.last_speech_end_time = 0.0
        self.spoken = []
        self.flush_called = False

    def speak_status(self, text: str):
        self.spoken.append(("status", text))

    def speak_social(self, text: str):
        self.spoken.append(("social", text))

    def speak_navigational(self, text: str):
        self.spoken.append(("navigational", text))

    def speak_emergency(self, text: str):
        self.spoken.append(("emergency", text))

    def flush_non_emergency(self):
        self.flush_called = True
        self.spoken = [(p, t) for p, t in self.spoken if p == "emergency"]
        self.is_speaking = False
        self.is_playing = False
        self.last_speech_end_time = time.time()

    def is_tts_active(self) -> bool:
        if self.is_speaking or self.is_playing:
            return True
        return (time.time() - self.last_speech_end_time) < 0.25


def test_intent_parsing_without_wake_phrase():
    """Verify that command parser recognizes direct commands without wake word."""
    parser = VoiceCommandParser()

    test_cases = [
        ("find my keys", "find", {"object_name": "keys"}),
        ("find my mobile", "find", {"object_name": "mobile"}),
        ("stop finding", "stop_find", {}),
        ("stop search", "stop_find", {}),
        ("remember this mug", "remember", {"object_name": "mug"}),
        ("enroll face Alice", "enroll_face", {"name": "alice"}),
        ("take note buy bread", "take_note", {"text": "buy bread"}),
        ("read notes", "read_notes", {}),
        ("delete note 1", "delete_note", {"note_id": "1"}),
        ("clear notes", "clear_notes", {}),
        ("navigate", "navigate", {}),
        ("stop navigation", "stop_navigate", {}),
        ("navigate to Majestic", "navigate_to", {"destination": "majestic"}),
        ("stop gps", "stop_gps", {}),
        ("advance gps", "advance_gps", {}),
        ("toggle audio", "audio_toggle", {}),
        ("list all items", "list", {}),
        ("show database", "database_contents", {}),
        ("shut down", "quit", {}),
        ("quit", "quit", {}),
        ("help", "help", {}),
    ]

    for utterance, expected_intent, expected_params in test_cases:
        res = parser.parse(utterance)
        assert res["intent"] == expected_intent, f"Failed for '{utterance}': got {res['intent']}, expected {expected_intent}"
        assert res["wake_word"] is False
        for k, v in expected_params.items():
            assert res["parameters"].get(k) == v, f"Param mismatch for '{utterance}': {res['parameters']}"


def test_intent_parsing_with_wake_phrase():
    """Verify that wake phrases ('hey vision', 'vision', 'a vision', 'hey visions') are stripped and parsed."""
    parser = VoiceCommandParser()

    wake_cases = [
        ("hey vision find my mobile", "find", {"object_name": "mobile"}),
        ("vision find my phone", "find", {"object_name": "phone"}),
        ("a vision remember this laptop", "remember", {"object_name": "laptop"}),
        ("hey visions stop finding", "stop_find", {}),
        ("hey vision stop search", "stop_find", {}),
        ("vision take note check locks", "take_note", {"text": "check locks"}),
        ("hey vision read notes", "read_notes", {}),
        ("a vision shut down", "quit", {}),
        ("hey vision enroll face John", "enroll_face", {"name": "john"}),
        ("hey vision take a note", "take_note", {"text": ""}),
        ("a vision take a note", "take_note", {"text": ""}),
        ("division taken out", "take_note", {"text": ""}),
        ("ek vision shutdown", "quit", {}),
    ]

    for utterance, expected_intent, expected_params in wake_cases:
        res = parser.parse(utterance)
        assert res["intent"] == expected_intent, f"Failed for '{utterance}': got {res['intent']}, expected {expected_intent}"
        assert res["wake_word"] is True
        for k, v in expected_params.items():
            assert res["parameters"].get(k) == v, f"Param mismatch for '{utterance}': {res['parameters']}"


def test_wake_word_alone_behavior():
    """When user says ONLY the wake word, parser returns wake_word_only and system enters 6s listening window."""
    parser = VoiceCommandParser()
    beep_played = []

    def mock_beep():
        beep_played.append(True)

    audio = DummyAudioFeedback()
    vcs = VoiceCommandSystem(audio_feedback=audio, beep_func=mock_beep)
    vcs.enabled = True

    # Test wake word alone detection in parser
    for wake_word in ["hey vision", "vision", "a vision", "hey visions"]:
        res = parser.parse(wake_word)
        assert res["intent"] == "wake_word_only"
        assert res["wake_word"] is True

    # Test process_text when wake word is spoken alone
    res2 = vcs.process_text("hey vision")
    assert res2["intent"] == "wake_word_only"
    assert len(beep_played) == 1
    assert vcs.awaiting_command_until > time.time()  # 6s window opened

    # Next utterance within 6 seconds is parsed as command without needing wake word
    received_commands = []
    vcs.register_command_handler("find", lambda params: received_commands.append(params))
    res3 = vcs.process_text("find my mobile")
    assert res3["intent"] == "find"
    assert len(received_commands) == 1
    assert received_commands[0]["object_name"] == "mobile"
    assert vcs.awaiting_command_until == 0.0  # Window closed after command


def test_find_my_mobile_and_stop_finding_wake_path():
    """Explicitly verify that 'find my mobile' and 'stop finding' work through the wake-word path."""
    audio = DummyAudioFeedback()
    vcs = VoiceCommandSystem(audio_feedback=audio)
    vcs.enabled = True

    find_calls = []
    stop_calls = []

    vcs.register_command_handler("find", lambda params: find_calls.append(params))
    vcs.register_command_handler("stop_find", lambda params: stop_calls.append(params))

    # Single-sentence wake commands
    vcs.process_text("hey vision find my mobile")
    assert len(find_calls) == 1
    assert find_calls[0].get("object_name") == "mobile"

    vcs.process_text("hey vision stop finding")
    assert len(stop_calls) == 1

    vcs.process_text("vision find my mobile")
    assert len(find_calls) == 2

    vcs.process_text("hey visions stop finding")
    assert len(stop_calls) == 2


def test_yes_no_window_logic():
    """Verify that yes/no confirmation window works without wake word and respects timeout."""
    audio = DummyAudioFeedback()
    vcs = VoiceCommandSystem(audio_feedback=audio)
    vcs.enabled = True

    yes_triggered = []
    no_triggered = []

    def on_yes():
        yes_triggered.append(True)

    def on_no():
        no_triggered.append(True)

    # 1. Open window and say "yes" (or variants "yeah", "correct", "haan", "yes bank")
    vcs.open_yes_no_window(on_yes=on_yes, on_no=on_no, timeout=6.0)
    assert vcs.yes_no_active is True
    res = vcs.process_text("yes bank")
    assert res["intent"] == "yes"
    assert len(yes_triggered) == 1
    assert len(no_triggered) == 0
    assert vcs.yes_no_active is False

    # 2. Open window and test "haan"
    vcs.open_yes_no_window(on_yes=on_yes, on_no=on_no, timeout=6.0)
    assert vcs.yes_no_active is True
    res = vcs.process_text("haan")
    assert res["intent"] == "yes"
    assert len(yes_triggered) == 2

    # 3. Open window and say "no" (or variants "wrong", "nope", "cancel", "nah")
    vcs.open_yes_no_window(on_yes=on_yes, on_no=on_no, timeout=6.0)
    assert vcs.yes_no_active is True
    res = vcs.process_text("wrong")
    assert res["intent"] == "no"
    assert len(no_triggered) == 1
    assert vcs.yes_no_active is False

    # 4. Conflicted input ("yes no" / "sure nope") -> ignored, window stays open
    vcs.open_yes_no_window(on_yes=on_yes, on_no=on_no, timeout=6.0)
    res = vcs.process_text("yes no")
    assert res is None
    assert vcs.yes_no_active is True
    assert len(yes_triggered) == 2
    assert len(no_triggered) == 1

    # 5. Test timeout expiration -> speaks "Cancelled" and closes
    vcs.yes_no_until = time.time() - 1.0  # Force timeout
    res = vcs.process_text("hello")
    assert res is None
    assert vcs.yes_no_active is False
    assert any(msg == "Cancelled" for _, msg in audio.spoken)


def test_echo_suppression_flag():
    """Verify audio_feedback is_speaking flag and echo suppression window."""
    audio = AudioFeedback(use_gtts=False)

    # Initially not speaking
    assert audio.is_speaking is False
    assert audio.is_tts_active() is False

    # When speaking
    audio.is_speaking = True
    assert audio.is_tts_active() is True

    # When speaking finishes
    audio.is_speaking = False
    audio.last_speech_end_time = time.time()
    # Within 0.5s, is_tts_active is still True (echo suppression active)
    assert audio.is_tts_active() is True

    # After > 0.5s, is_tts_active returns False
    audio.last_speech_end_time = time.time() - 0.6
    assert audio.is_tts_active() is False


def test_speech_notes_and_auto_read(tmp_path):
    """Test voice notes creation, auto-read back, reading, deletion, and clearing."""
    notes_file = str(tmp_path / "notes.json")
    mgr = speech_notes.SpeechNotesManager(notes_file=notes_file)

    # 1. Take note with AUTO_READ_NOTES = True
    config.AUTO_READ_NOTES = True
    res = mgr.take_note("buy groceries")
    assert "Saved. You said: buy groceries" in res
    assert len(mgr.notes) == 1

    # 2. Take second note
    res2 = mgr.take_note("call mechanic")
    assert "Saved. You said: call mechanic" in res2
    assert len(mgr.notes) == 2

    # 3. Read notes
    read_text = mgr.read_notes()
    assert "You have 2 notes" in read_text
    assert "buy groceries" in read_text
    assert "call mechanic" in read_text

    # 4. Delete note by index
    del_res = mgr.delete_note(1)
    assert "Deleted note" in del_res
    assert len(mgr.notes) == 1

    # 5. Clear notes
    clr_res = mgr.clear_notes()
    assert "All notes cleared" in clr_res
    assert len(mgr.notes) == 0



def test_take_note_tolerant_inputs():
    """Verify that take_note maps tolerant inputs accurately."""
    parser = VoiceCommandParser()

    tolerant_cases = [
        "take a note",
        "take note",
        "make a note",
        "note this",
        "dictate",
        "take a note please",
        "taken out",
    ]
    for inp in tolerant_cases:
        res = parser.parse(inp)
        assert res["intent"] == "take_note", f"Failed for '{inp}': got {res['intent']}"
        assert res["parameters"].get("text") == ""

    # With note content
    content_cases = [
        ("take note buy bread", "buy bread"),
        ("take a note buy milk", "buy milk"),
        ("take a note please buy milk", "buy milk"),
        ("taken out buy groceries", "buy groceries"),
        ("make a note to call mom", "call mom"),
        ("note this check doors", "check doors"),
        ("dictate pick up dry cleaning", "pick up dry cleaning"),
    ]
    for inp, expected_text in content_cases:
        res = parser.parse(inp)
        assert res["intent"] == "take_note", f"Failed for '{inp}': got {res['intent']}"
        assert res["parameters"].get("text") == expected_text, f"Text mismatch for '{inp}': {res['parameters']}"

    # Read / Delete / Clear tolerant tests
    assert parser.parse("read notes")["intent"] == "read_notes"
    assert parser.parse("read my notes")["intent"] == "read_notes"
    assert parser.parse("list my notes")["intent"] == "read_notes"
    assert parser.parse("check notes")["intent"] == "read_notes"
    assert parser.parse("delete note 1")["intent"] == "delete_note"
    assert parser.parse("remove note 2")["intent"] == "delete_note"
    assert parser.parse("clear notes")["intent"] == "clear_notes"
    assert parser.parse("clear all notes")["intent"] == "clear_notes"
    assert parser.parse("delete all notes")["intent"] == "clear_notes"


def test_fuzzy_wake_word_and_mishearings():
    """Verify that all wake phrases and close fuzzy mishearings are accepted."""
    parser = VoiceCommandParser()

    # All 9 accepted wake phrases
    phrases = [
        "hey vision",
        "a vision",
        "ek vision",
        "hey visions",
        "he vision",
        "hay vision",
        "hi vision",
        "division",
        "vision",
    ]
    for p in phrases:
        has_wake, remainder = parser.strip_wake_word(f"{p} find my keys")
        assert has_wake is True, f"Failed wake phrase: '{p}'"
        assert remainder == "find my keys"

    # Specific tests from instructions
    res1 = parser.parse("hey vision take a note")
    assert res1["intent"] == "take_note"
    assert res1["wake_word"] is True

    res2 = parser.parse("a vision take a note")
    assert res2["intent"] == "take_note"
    assert res2["wake_word"] is True

    res3 = parser.parse("division taken out")
    assert res3["intent"] == "take_note"
    assert res3["wake_word"] is True

    res4 = parser.parse("ek vision shutdown")
    assert res4["intent"] == "quit"
    assert res4["wake_word"] is True

    # "yes bank" intent mapping
    res5 = parser.parse("yes bank")
    assert res5["intent"] == "yes"

    # Confirm find my mobile and stop finding still map correctly
    res_find = parser.parse("find my mobile")
    assert res_find["intent"] == "find"
    assert res_find["parameters"].get("object_name") == "mobile"

    res_stop = parser.parse("stop finding")
    assert res_stop["intent"] == "stop_find"


def test_quit_confirmation_window():
    """Verify shutdown only occurs after a YES inside confirmation window, never on shutdown alone."""
    audio = DummyAudioFeedback()
    vcs = VoiceCommandSystem(audio_feedback=audio)
    vcs.enabled = True

    shutdown_called = []

    def mock_shutdown():
        shutdown_called.append(True)

    def handle_quit(params):
        audio.speak_social("Say yes to confirm")
        vcs.open_yes_no_window(on_yes=mock_shutdown, on_no=lambda: audio.speak_social("Cancelled"), timeout=6.0)

    vcs.register_command_handler("quit", handle_quit)

    # 1. Plain "shutdown" without wake word (outside window) is ignored in voice-only mode
    res = vcs.process_text("shutdown")
    assert res is None
    assert len(shutdown_called) == 0

    # 2. "ek vision shutdown" triggers quit command, which asks for confirmation
    vcs.process_text("ek vision shutdown")
    assert len(shutdown_called) == 0
    assert vcs.yes_no_active is True
    assert any(msg == "Say yes to confirm" for _, msg in audio.spoken)

    # 3. Say "no" -> cancelled, never shuts down
    vcs.process_text("no")
    assert len(shutdown_called) == 0
    assert vcs.yes_no_active is False

    # 4. Trigger shutdown again, confirm with "yes bank"
    vcs.process_text("ek vision shutdown")
    assert vcs.yes_no_active is True
    vcs.process_text("yes bank")
    assert len(shutdown_called) == 1
    assert vcs.yes_no_active is False


def test_rpi_client_default_ip():
    """Verify RaspberryPiClient defaults to 127.0.0.1 and prints target IP."""
    from wireless.rpi_client import RaspberryPiClient
    client_default = RaspberryPiClient()
    assert client_default.server_ip == "127.0.0.1"
    assert client_default.server_url == "ws://127.0.0.1:8081"

    client_custom = RaspberryPiClient(server_ip="192.168.1.50", server_port=9000)
    assert client_custom.server_ip == "192.168.1.50"
    assert client_custom.server_url == "ws://192.168.1.50:9000"


def test_empty_take_note_dictation_flow(monkeypatch, tmp_path):
    """Empty take_note opens dictation window; next utterance is saved and auto-read."""
    notes_file = str(tmp_path / "notes.json")
    monkeypatch.setattr(speech_notes, "NOTES_FILE", notes_file)
    speech_notes.clear_notes()

    audio = DummyAudioFeedback()
    vcs = VoiceCommandSystem(audio_feedback=audio)
    vcs.enabled = True

    # 1. User says "hey vision take a note"
    res = vcs.process_text("hey vision take a note")
    assert res is not None
    assert res["intent"] == "take_note"
    assert vcs.dictation_active is True
    assert any("speak your note" in msg for _, msg in audio.spoken)

    # 2. Next utterance without wake word is the note text
    res2 = vcs.process_text("buy milk and eggs")
    assert vcs.dictation_active is False
    assert any("buy milk and eggs" in msg for _, msg in audio.spoken)

    # 3. Note is saved in speech_notes
    all_notes = speech_notes.get_manager().notes
    assert len(all_notes) == 1
    assert all_notes[0]["text"] == "buy milk and eggs"


def test_ambiguous_question_throttle_60s():
    """Ambiguous question 'Is this your X?' is not repeated within 60s."""
    import numpy as np
    from wireless.laptop_server import LaptopMLServer

    server = LaptopMLServer(port=8089)
    server.audio_enabled = True
    server.audio = DummyAudioFeedback()
    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server.voice_system.enabled = True

    class DummyDetector:
        def detect(self, frame):
            return [{'class_name': 'bottle', 'bbox': [100, 100, 200, 200], 'confidence': 0.9}]
    server.object_detector = DummyDetector()

    class DummyTracker:
        def update_tracking(self, detections):
            return []
    server.item_tracker = DummyTracker()

    class DummyMemory:
        pending_confirmation = None
        def recognize_object(self, frame, bbox, yolo_class):
            return ("thermos", 0.70, "ambiguous")
        def update_object_sighting(self, **kwargs):
            return True
    server.object_memory = DummyMemory()

    class DummyFaceRecognizer:
        def recognize_faces(self, frame):
            return []
    server.face_recognizer = DummyFaceRecognizer()
    server.find_guide = None
    server.spatial_navigator = None

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # First frame: triggers ambiguous question
    server.process_ml(dummy_frame, ultrasonic_distance=100.0, button_pressed=False)
    assert any("Is this your thermos?" in msg for _, msg in server.audio.spoken)
    spoken_count_1 = len(server.audio.spoken)

    # Close yes/no window so window_open isn't what's blocking
    server.voice_system.yes_no_active = False

    # Second frame immediately: should be muted by the 60s cooldown
    server.process_ml(dummy_frame, ultrasonic_distance=100.0, button_pressed=False)
    spoken_count_2 = len(server.audio.spoken)
    assert spoken_count_2 == spoken_count_1


def test_command_inside_yes_no_window_cancels_and_executes():
    """'a vision shutdown' inside an open yes/no window cancels window and runs quit."""
    audio = DummyAudioFeedback()
    vcs = VoiceCommandSystem(audio_feedback=audio)
    vcs.enabled = True

    quit_called = []
    vcs.register_command_handler("quit", lambda params: quit_called.append(True))

    yes_called = []
    vcs.open_yes_no_window(on_yes=lambda: yes_called.append(True), timeout=10.0)
    assert vcs.yes_no_active is True

    res = vcs.process_text("a vision shutdown")
    assert res is not None
    assert res["intent"] == "quit"
    assert vcs.yes_no_active is False
    assert len(quit_called) == 1
    assert len(yes_called) == 0


@pytest.mark.asyncio
async def test_connection_closed_on_send_handled():
    """ConnectionClosed on websocket send is handled at INFO level without exception."""
    from wireless.laptop_server import LaptopMLServer
    import websockets.exceptions
    import json

    server = LaptopMLServer(port=8089)
    server.show_display = False

    class MockWS:
        def __init__(self):
            self.remote_address = ("127.0.0.1", 12345)
            self.messages = [
                json.dumps({
                    'image': '',
                    'ultrasonic': 100.0,
                    'button': False,
                    'timestamp': time.time()
                })
            ]
        def __aiter__(self):
            return self
        async def __anext__(self):
            if self.messages:
                return self.messages.pop(0)
            raise StopAsyncIteration
        async def send(self, data):
            raise websockets.exceptions.ConnectionClosedOK(rcvd=None, sent=None)

    mock_ws = MockWS()
    server.process_ml = lambda frame, dist, btn: {'objects': [], 'faces': [], 'remembered': []}
    # This should run and complete cleanly without raising
    await server.process_frame(mock_ws)


def test_recognize_object_returns_none_when_nothing_enrolled():
    """recognize_object returns None immediately when no remembered object is enrolled for that YOLO class."""
    from object_memory import ObjectMemory
    import numpy as np

    class EmptyDB:
        def get_remembered_objects_by_class(self, yolo_class):
            return []

    memory = ObjectMemory(db=EmptyDB())
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (50, 50, 150, 150)

    result = memory.recognize_object(dummy_frame, bbox, "keys")
    assert result is None


def test_guidance_repeat_suppression_within_20s():
    """The same 'Person ahead' message is not repeated within 20 seconds."""
    from spatial_navigator import SpatialNavigator

    nav = SpatialNavigator(frame_width=640, frame_height=480)
    # Person centered, medium size (not very-close)
    det = {'bbox': (250, 180, 390, 300), 'class_name': 'person', 'confidence': 0.9}

    t0 = 100.0
    guidance = nav.generate_guidance([det], current_time=t0)
    assert len(guidance) > 0
    top = guidance[0]
    assert "Person ahead" in top['message']

    # First time: should speak
    assert nav.should_speak_guidance(top, ultrasonic_distance=100.0, current_time=t0) is True
    nav.mark_guidance_spoken(top, current_time=t0)

    # Object stays continuously in view across 10 seconds
    for t in range(101, 111):
        nav.generate_guidance([det], current_time=float(t))

    guidance1 = nav.generate_guidance([det], current_time=110.0)
    top1 = guidance1[0]
    assert nav.should_speak_guidance(top1, ultrasonic_distance=100.0, current_time=110.0) is False

    # Object continues to stay in view until 21s elapsed (event-based guidance: no time-based repeats)
    for t in range(111, 122):
        nav.generate_guidance([det], current_time=float(t))

    guidance2 = nav.generate_guidance([det], current_time=121.0)
    top2 = guidance2[0]
    assert nav.should_speak_guidance(top2, ultrasonic_distance=100.0, current_time=121.0) is False


def test_guidance_spoken_again_after_object_leaves_for_3s_and_returns():
    """Guidance is spoken again if object leaves view for > 3s and re-enters, even if within 20s."""
    from spatial_navigator import SpatialNavigator

    nav = SpatialNavigator(frame_width=640, frame_height=480)
    det = {'bbox': (250, 180, 390, 300), 'class_name': 'chair', 'confidence': 0.9}

    t0 = 100.0
    guidance = nav.generate_guidance([det], current_time=t0)
    top = guidance[0]
    assert nav.should_speak_guidance(top, ultrasonic_distance=100.0, current_time=t0) is True
    nav.mark_guidance_spoken(top, current_time=t0)

    # Object absent for > 3.0s (at t0 + 4.0s)
    t_return = t0 + 4.0
    guidance_return = nav.generate_guidance([det], current_time=t_return)
    top_return = guidance_return[0]

    # Even though only 4s elapsed (< 20s), it was absent > 3s so it speaks again
    assert nav.should_speak_guidance(top_return, ultrasonic_distance=100.0, current_time=t_return) is True


def test_critical_warning_spoken_during_quiet_mode():
    """During quiet mode, routine announcements are suppressed but critical (<50 cm) warnings are spoken."""
    import numpy as np
    from wireless.laptop_server import LaptopMLServer
    from spatial_navigator import SpatialNavigator

    server = LaptopMLServer(port=8091)
    server.show_display = False
    server.audio_enabled = True
    server.navigation_enabled = True
    server.audio = DummyAudioFeedback()
    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server.voice_system.enabled = True

    det = {'bbox': [250, 180, 390, 300], 'class_name': 'person', 'confidence': 0.9}

    class DummyDetector:
        def detect(self, frame):
            return [det]
    server.object_detector = DummyDetector()

    class DummyTracker:
        def update_tracking(self, detections):
            return []
    server.item_tracker = DummyTracker()

    class DummyMemory:
        def recognize_object(self, frame, bbox, yolo_class):
            return None
    server.object_memory = DummyMemory()

    class DummyFaceRecognizer:
        def recognize_faces(self, frame):
            return []
    server.face_recognizer = DummyFaceRecognizer()

    server.spatial_navigator = SpatialNavigator(frame_width=640, frame_height=480)
    server.find_guide = None

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Enter quiet mode via dictation window
    server.voice_system.open_dictation_window(on_dictate=lambda text: None, timeout=10.0)
    assert server.is_quiet_mode_active() is True

    # 1. Routine warning (ultrasonic = 100 cm, not critical): should be suppressed in quiet mode
    server.process_ml(dummy_frame, ultrasonic_distance=100.0, button_pressed=False)
    assert len(server.audio.spoken) == 0

    # 2. Critical warning (ultrasonic = 40 cm < 50 cm): must bypass quiet mode and speak immediately
    server.process_ml(dummy_frame, ultrasonic_distance=40.0, button_pressed=False)
    assert len(server.audio.spoken) > 0
    priorities = [p for p, _ in server.audio.spoken]
    assert "emergency" in priorities


def test_wake_word_flushes_queue():
    """On wake word detected, non-emergency items in audio queue are flushed."""
    from audio_feedback import AudioFeedback, PRIORITY_EMERGENCY, PRIORITY_NAVIGATIONAL, PRIORITY_STATUS

    # Direct AudioFeedback test
    audio = AudioFeedback(use_gtts=False)
    audio.audio_queue.put((PRIORITY_NAVIGATIONAL, "Person ahead"))
    audio.audio_queue.put((PRIORITY_STATUS, "Ready"))
    audio.audio_queue.put((PRIORITY_EMERGENCY, "STOP! Obstacle"))

    audio.flush_non_emergency()

    items = []
    while not audio.audio_queue.empty():
        items.append(audio.audio_queue.get_nowait())

    non_stop_items = [it for it in items if it[1] != "__STOP__"]
    # Emergency item was preserved (either still waiting in queue or already processed by background worker)
    assert len(non_stop_items) == 1 or "STOP! Obstacle" in getattr(audio, 'last_spoken_time_by_text', {})
    if len(non_stop_items) == 1:
        assert non_stop_items[0][0] == PRIORITY_EMERGENCY
        assert non_stop_items[0][1] == "STOP! Obstacle"

    # VoiceCommandSystem test: wake word triggers flush
    dummy_audio = DummyAudioFeedback()
    dummy_audio.spoken = [("navigational", "Person ahead"), ("status", "Ready")]
    vcs = VoiceCommandSystem(audio_feedback=dummy_audio)
    vcs.enabled = True

    vcs.process_text("hey vision find my mobile")
    assert dummy_audio.flush_called is True
    assert len(dummy_audio.spoken) == 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])


