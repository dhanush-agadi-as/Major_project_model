"""
Tests for direct shutdown without confirmation, window speech recognition, and logging.

Covers:
1. "hey vision shutdown" and "revision shutdown" trigger shutdown directly with REQUIRE_SHUTDOWN_CONFIRM False.
2. "hey vision shutdown now please" works.
3. Text that doesn't contain a shutdown word never shuts down.
4. When REQUIRE_SHUTDOWN_CONFIRM True, old yes/no flow is preserved.
5. An utterance heard during an open dictation window is saved as the note without needing a wake word.
6. An utterance heard during an open yes/no window is routed to yes/no handler without needing a wake word.
7. INFO logging for every audio chunk and recognition result while a window is open.
8. Background listener is non-blocking and mute flag clears when TTS finishes (0.25s window maintained).
"""

import os
import sys
import time
import logging
from unittest.mock import MagicMock
import pytest
import speech_recognition as sr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from voice_command_system import VoiceCommandParser, VoiceCommandSystem
from wireless.laptop_server import LaptopMLServer
import speech_notes


class DummyAudio:
    """Mock audio feedback for testing."""
    def __init__(self):
        self.is_speaking = False
        self.is_playing = False
        self.last_speech_end_time = 0.0
        self.spoken = []

    def speak_social(self, text: str):
        self.spoken.append(("social", text))

    def speak_status(self, text: str):
        self.spoken.append(("status", text))

    def speak_navigational(self, text: str):
        self.spoken.append(("navigational", text))

    def speak_emergency(self, text: str):
        self.spoken.append(("emergency", text))

    def is_tts_active(self) -> bool:
        if self.is_speaking or self.is_playing:
            return True
        return (time.time() - self.last_speech_end_time) < 0.25


# ============================================================================
# Requirement 1 & 3: Shutdown without confirmation & exact word matching
# ============================================================================

def test_hey_vision_shutdown_triggers_direct_shutdown(monkeypatch):
    """'hey vision shutdown' triggers direct shutdown when REQUIRE_SHUTDOWN_CONFIRM is False."""
    monkeypatch.setattr(config, "REQUIRE_SHUTDOWN_CONFIRM", False)
    server = LaptopMLServer(port=8092)
    server.show_display = False
    server.audio = DummyAudio()
    server.audio_enabled = True
    server.running = True

    # Mock voice system and trackers
    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server.voice_system.enabled = True
    server.voice_system.running = True
    server.gps_tracker = MagicMock()
    server.ultrasonic = MagicMock()
    server.server = MagicMock()

    server._register_voice_handlers()

    # User says "hey vision shutdown"
    res = server.voice_system.process_text("hey vision shutdown")

    assert res is not None
    assert res['intent'] == 'quit'
    # Check shutdown occurred
    assert server.running is False
    assert any(msg == "Shutting down" for _, msg in server.audio.spoken)
    assert server.voice_system.running is False
    assert server.gps_tracker.stop.called is True
    assert server.ultrasonic.stop.called is True
    assert server.server.close.called is True
    assert server.voice_system.yes_no_active is False


def test_revision_shutdown_triggers_direct_shutdown(monkeypatch):
    """'revision shutdown' triggers direct shutdown when REQUIRE_SHUTDOWN_CONFIRM is False."""
    monkeypatch.setattr(config, "REQUIRE_SHUTDOWN_CONFIRM", False)
    server = LaptopMLServer(port=8093)
    server.show_display = False
    server.audio = DummyAudio()
    server.audio_enabled = True
    server.running = True

    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server.voice_system.enabled = True
    server.voice_system.running = True
    server.gps_tracker = MagicMock()
    server.ultrasonic = MagicMock()
    server.server = MagicMock()

    server._register_voice_handlers()

    res = server.voice_system.process_text("revision shutdown")

    assert res is not None
    assert res['intent'] == 'quit'
    assert server.running is False
    assert any(msg == "Shutting down" for _, msg in server.audio.spoken)
    assert server.voice_system.yes_no_active is False


def test_hey_vision_shutdown_now_please_triggers_direct_shutdown(monkeypatch):
    """'hey vision shutdown now please' triggers direct shutdown."""
    monkeypatch.setattr(config, "REQUIRE_SHUTDOWN_CONFIRM", False)
    server = LaptopMLServer(port=8094)
    server.show_display = False
    server.audio = DummyAudio()
    server.audio_enabled = True
    server.running = True

    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server.voice_system.enabled = True
    server.voice_system.running = True
    server.gps_tracker = MagicMock()
    server.ultrasonic = MagicMock()
    server.server = MagicMock()

    server._register_voice_handlers()

    res = server.voice_system.process_text("hey vision shutdown now please")

    assert res is not None
    assert res['intent'] == 'quit'
    assert server.running is False
    assert any(msg == "Shutting down" for _, msg in server.audio.spoken)


def test_text_without_shutdown_words_never_shuts_down(monkeypatch):
    """Text that doesn't contain an exact shutdown word ('shutdown', 'shut down', 'quit') never shuts down."""
    monkeypatch.setattr(config, "REQUIRE_SHUTDOWN_CONFIRM", False)
    server = LaptopMLServer(port=8095)
    server.show_display = False
    server.audio = DummyAudio()
    server.audio_enabled = True
    server.running = True

    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server.voice_system.enabled = True
    server.voice_system.running = True
    server.gps_tracker = MagicMock()
    server.ultrasonic = MagicMock()
    server.server = MagicMock()

    server._register_voice_handlers()

    non_shutdown_phrases = [
        "hey vision shoot down",
        "hey vision turn off system",
        "hey vision power off",
        "hey vision navigation",
        "hey vision find my keys",
        "shutdown",  # No wake phrase outside window
        "shut down",  # No wake phrase outside window
        "quit",       # No wake phrase outside window
    ]

    for phrase in non_shutdown_phrases:
        res = server.voice_system.process_text(phrase)
        intent = res['intent'] if res else None
        assert intent != 'quit', f"Phrase '{phrase}' incorrectly triggered quit intent!"
        assert server.running is True, f"Phrase '{phrase}' caused server shutdown!"
        assert server.gps_tracker.stop.called is False
        assert server.ultrasonic.stop.called is False


def test_require_shutdown_confirm_true_keeps_yes_no_flow(monkeypatch):
    """When REQUIRE_SHUTDOWN_CONFIRM is True, asks 'Say yes to confirm' and opens yes/no window."""
    monkeypatch.setattr(config, "REQUIRE_SHUTDOWN_CONFIRM", True)
    server = LaptopMLServer(port=8096)
    server.show_display = False
    server.audio = DummyAudio()
    server.audio_enabled = True
    server.running = True

    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server.voice_system.enabled = True
    server.voice_system.running = True
    server.gps_tracker = MagicMock()
    server.ultrasonic = MagicMock()
    server.server = MagicMock()

    server._register_voice_handlers()

    # User says "hey vision shutdown"
    res = server.voice_system.process_text("hey vision shutdown")
    assert res is not None
    assert res['intent'] == 'quit'
    # Server should NOT have shut down yet
    assert server.running is True
    assert server.voice_system.yes_no_active is True
    assert any(msg == "Say yes to confirm" for _, msg in server.audio.spoken)

    # Saying "no" cancels shutdown
    server.voice_system.process_text("no")
    assert server.running is True
    assert server.voice_system.yes_no_active is False
    assert any(msg == "Shutdown cancelled" for _, msg in server.audio.spoken)


# ============================================================================
# Requirement 2 & 3: Dictation window saves note without wake word
# ============================================================================

def test_utterance_during_open_dictation_window_saved_as_note(tmp_path, monkeypatch):
    """An utterance heard during an open dictation window is saved as the note without needing a wake word."""
    test_notes_file = str(tmp_path / "notes.json")
    mgr = speech_notes.SpeechNotesManager(test_notes_file)
    monkeypatch.setattr(speech_notes, "_manager", mgr)

    audio = DummyAudio()
    vcs = VoiceCommandSystem(audio_feedback=audio)
    vcs.enabled = True

    saved_notes = []

    def on_dictate(text):
        saved_notes.append(text)
        speech_notes.take_note(text)

    # Open dictation window
    vcs.open_dictation_window(on_dictate=on_dictate, timeout=10.0)
    assert vcs.dictation_active is True

    # User speaks directly without wake word
    res = vcs.process_text("pick up apples and milk from grocery")

    assert res is not None
    assert res['intent'] == 'dictated_note'
    assert res['parameters']['text'] == "pick up apples and milk from grocery"
    assert vcs.dictation_active is False
    assert len(saved_notes) == 1
    assert saved_notes[0] == "pick up apples and milk from grocery"

    # Verify saved in notes module
    all_notes = speech_notes.list_notes()
    assert len(all_notes) == 1
    assert all_notes[0]['text'] == "pick up apples and milk from grocery"


def test_dictation_window_with_wake_word_strips_wake_word():
    """If user includes a wake word during dictation window, it is stripped cleanly."""
    audio = DummyAudio()
    vcs = VoiceCommandSystem(audio_feedback=audio)
    vcs.enabled = True

    received = []
    vcs.open_dictation_window(on_dictate=lambda t: received.append(t), timeout=10.0)

    vcs.process_text("hey vision buy fresh bread")
    assert vcs.dictation_active is False
    assert received == ["buy fresh bread"]


# ============================================================================
# Requirement 2: Yes/No window routing without wake word
# ============================================================================

def test_yes_no_window_routes_without_wake_word():
    """Utterances during a yes/no window are routed to handlers without needing a wake word."""
    audio = DummyAudio()
    vcs = VoiceCommandSystem(audio_feedback=audio)
    vcs.enabled = True

    yes_called = []
    no_called = []

    # 1. Test "yes"
    vcs.open_yes_no_window(
        on_yes=lambda: yes_called.append(True),
        on_no=lambda: no_called.append(True),
        timeout=10.0
    )
    vcs.process_text("yes")
    assert len(yes_called) == 1
    assert len(no_called) == 0
    assert vcs.yes_no_active is False

    # 2. Test "nope"
    vcs.open_yes_no_window(
        on_yes=lambda: yes_called.append(True),
        on_no=lambda: no_called.append(True),
        timeout=10.0
    )
    vcs.process_text("nope")
    assert len(yes_called) == 1
    assert len(no_called) == 1
    assert vcs.yes_no_active is False

    # 3. Test "hey vision yes"
    vcs.open_yes_no_window(
        on_yes=lambda: yes_called.append(True),
        on_no=lambda: no_called.append(True),
        timeout=10.0
    )
    vcs.process_text("hey vision yes")
    assert len(yes_called) == 2
    assert len(no_called) == 1
    assert vcs.yes_no_active is False


# ============================================================================
# Requirement 2: Window logging, mute clearing, non-blocking listener
# ============================================================================

def test_window_audio_chunk_and_recognition_logging(caplog):
    """INFO logs are emitted for every audio chunk and recognition result while a window is open."""
    audio = DummyAudio()
    vcs = VoiceCommandSystem(audio_feedback=audio)
    vcs.enabled = True
    vcs.running = True

    caplog.set_level(logging.INFO)

    vcs.open_dictation_window(on_dictate=lambda t: None, timeout=10.0)

    # Mock recognizer and audio data
    mock_recognizer = MagicMock()
    mock_recognizer.recognize_google.return_value = "meeting at noon"
    mock_audio = MagicMock()

    # Call _bg_audio_callback in sync mode for test determinism
    vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)

    records = [r.message for r in caplog.records]
    assert any("Audio chunk captured while window is open" in m for m in records)
    assert any("Recognition result while window is open: 'meeting at noon'" in m for m in records)


def test_window_no_speech_detected_logging(caplog):
    """INFO logs 'No speech detected while window is open' when UnknownValueError occurs."""
    audio = DummyAudio()
    vcs = VoiceCommandSystem(audio_feedback=audio)
    vcs.enabled = True
    vcs.running = True

    caplog.set_level(logging.INFO)

    vcs.open_yes_no_window(on_yes=lambda: None, on_no=lambda: None, timeout=10.0)

    mock_recognizer = MagicMock()
    mock_recognizer.recognize_google.side_effect = sr.UnknownValueError()
    mock_audio = MagicMock()

    vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)

    records = [r.message for r in caplog.records]
    assert any("Audio chunk captured while window is open" in m for m in records)
    assert any("No speech detected while window is open" in m for m in records)


def test_tts_mute_cleared_keeps_025s_post_speech_window():
    """Verify TTS mute is active during speech and for exactly 0.25s after, then clears."""
    audio = DummyAudio()
    vcs = VoiceCommandSystem(audio_feedback=audio)

    # 1. Not speaking -> inactive
    audio.is_speaking = False
    audio.is_playing = False
    audio.last_speech_end_time = time.time() - 1.0
    assert vcs._is_tts_active() is False

    # 2. Speaking -> active
    audio.is_speaking = True
    assert vcs._is_tts_active() is True

    # 3. Finished speaking -> active within 0.25s
    audio.is_speaking = False
    audio.last_speech_end_time = time.time()
    assert vcs._is_tts_active() is True

    # 4. 0.3s later -> cleared
    audio.last_speech_end_time = time.time() - 0.30
    assert vcs._is_tts_active() is False
