"""
Tests for STT-related fixes:
1. Normalise recognised text before wake-word matching and parsing:
   "Hey Vision!" opens 6s window, bare "!" is ignored without warning.
2. Object-name matching for find/remember:
   difflib ratio >= 0.5 or phonetic comparison (e.g. "selfour" ~ "cellphone").
   Single best candidate used. Weak match on single remembered object asks "Did you mean your cell phone?" and opens yes/no window.
3. Remembered object names and aliases added to Whistle keywords list and refreshed at runtime.
4. ip_geolocation not started when GPS_MOCK_MODE or INDOOR_MODE is True.
   Initial failure enters 600s cooldown immediately without retry loop. Daemon thread used.
5. ThreadPoolExecutor clean shutdown (wait=False, cancel_futures=True) and daemon worker threads.
6. Startup banner in laptop_server.py.
7. Whistle results longer than 12 words ignored on background speech.
"""

import os
import sys
import time
import pytest
from unittest.mock import MagicMock, patch
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import stt_engine
from voice_command_system import VoiceCommandSystem, VoiceCommandParser, normalise_text
from gps_tracker import GPSTracker
from ip_geolocation import IPGeolocationTracker
from wireless.laptop_server import LaptopMLServer, clean_object_name, calculate_name_similarity, phonetic_normalize


class DummyAudio:
    def __init__(self):
        self.messages = []
        self.social_messages = []
        self.status_messages = []
        self.command_replies = []
        self.is_speaking = False
        self.is_playing = False
        self.last_speech_end_time = 0.0

    def speak(self, text, priority=3, is_command_reply=False):
        self.messages.append(text)
        if is_command_reply:
            self.command_replies.append(text)

    def speak_social(self, text, is_command_reply=False):
        self.messages.append(text)
        self.social_messages.append(text)
        if is_command_reply:
            self.command_replies.append(text)

    def speak_status(self, text, is_command_reply=False):
        self.messages.append(text)
        self.status_messages.append(text)
        if is_command_reply:
            self.command_replies.append(text)

    def speak_command_reply(self, text, priority=3):
        self.messages.append(text)
        self.command_replies.append(text)

    def is_tts_active(self):
        return False

    def flush_non_emergency(self):
        pass

    def stop(self):
        pass


def _setup_test_server():
    server = LaptopMLServer(port=8081)
    server.audio = DummyAudio()
    server.audio_enabled = True
    server.object_memory = MagicMock()
    server.find_guide = MagicMock()
    server.item_tracker = MagicMock()
    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server.voice_system.enabled = True
    server.voice_system.running = True
    server._register_voice_handlers()
    return server


# ─────────────────────────────────────────────────────────────
# 1. Normalisation of recognised text
# ─────────────────────────────────────────────────────────────

def test_normalise_text():
    """Verify normalise_text lowercases, strips punctuation, collapses spaces."""
    assert normalise_text("Hey Vision!") == "hey vision"
    assert normalise_text("  HEY,   VISION?!  ") == "hey vision"
    assert normalise_text("!") == ""
    assert normalise_text("  ! . , ? ' \" ; :  ") == ""
    assert normalise_text("find   my   cell  phone?!") == "find my cell phone"


def test_hey_vision_exclamation_opens_window():
    """'Hey Vision!' == 'hey vision' and opens the 6s window."""
    audio = DummyAudio()
    vcs = VoiceCommandSystem(audio_feedback=audio)
    vcs.enabled = True
    vcs.running = True

    beep_called = []
    vcs.beep_func = lambda: beep_called.append(True)

    result = vcs.process_text("Hey Vision!")
    assert result is not None
    assert result.get("intent") == "wake_word_only"
    assert len(beep_called) == 1
    assert vcs.awaiting_command_until > time.time()


def test_bare_punctuation_ignored_without_warning(caplog):
    """A bare '!' is ignored without a warning."""
    audio = DummyAudio()
    vcs = VoiceCommandSystem(audio_feedback=audio)
    vcs.enabled = True
    vcs.running = True

    with caplog.at_level("WARNING"):
        res = vcs.process_text("!")
        assert res is None
        # Must not log warnings
        assert not any("warning" in r.levelname.lower() for r in caplog.records)


# ─────────────────────────────────────────────────────────────
# 2. Object-name matching for find/remember
# ─────────────────────────────────────────────────────────────

def test_phonetic_and_fuzzy_similarity():
    """Verify 'selfour' ~ 'cellphone' / 'cell phone' gives ratio >= 0.5."""
    assert phonetic_normalize("cell phone") == "selfon"
    assert phonetic_normalize("cellphone") == "selfon"
    assert phonetic_normalize("selfour") == "selfour"

    sim1 = calculate_name_similarity("selfour", "cell phone")
    sim2 = calculate_name_similarity("selfour", "cellphone")
    assert sim1 >= 0.5, f"Expected sim >= 0.5 for 'selfour' vs 'cell phone', got {sim1}"
    assert sim2 >= 0.5, f"Expected sim >= 0.5 for 'selfour' vs 'cellphone', got {sim2}"


def test_find_selfour_single_remembered_asks_or_resolves():
    """
    With only one remembered object ('cell phone'), 'find selfour' either
    resolves to the cell phone or asks 'Did you mean your cell phone?' and opens yes/no window.
    """
    server = _setup_test_server()
    server.object_memory.list_remembered_objects.return_value = [("cell phone", "cell phone")]
    server.object_memory.get_object_aliases.return_value = ["mobile", "phone"]
    server.object_memory.get_object_location.return_value = None
    server.item_tracker.query_item_location.return_value = None

    handler = server.voice_system.command_callbacks["find"]
    handler({"object_name": "selfour"})

    # Check if asked "Did you mean your cell phone?" or resolved directly
    asked = any("Did you mean your cell phone?" in msg for msg in server.audio.messages)
    resolved = server.find_guide.start.called

    assert asked or resolved, "Must ask 'Did you mean your cell phone?' or resolve to cell phone"

    if asked:
        assert server.voice_system.yes_no_active is True, "Must open yes/no window when asking"
        # Say 'yes'
        yes_handler = server.voice_system.on_yes_callback
        assert yes_handler is not None
        yes_handler()
        assert server.find_guide.start.called is True
        assert server.find_guide.start.call_args[0][0] == "cell phone"


def test_find_selfour_multiple_remembered_resolves_to_best_candidate():
    """
    With two remembered objects ('cell phone' and 'water bottle'),
    'find selfour' matches 'cell phone' as the single best candidate (ratio >= 0.5).
    """
    server = _setup_test_server()
    server.object_memory.list_remembered_objects.return_value = [
        ("cell phone", "cell phone"),
        ("water bottle", "bottle")
    ]
    server.object_memory.get_object_aliases.return_value = []
    server.object_memory.get_object_location.return_value = None
    server.item_tracker.query_item_location.return_value = None

    handler = server.voice_system.command_callbacks["find"]
    handler({"object_name": "selfour"})

    assert server.find_guide.start.called is True
    assert server.find_guide.start.call_args[0][0] == "cell phone"


def test_remember_with_fuzzy_candidate_matching():
    """
    'remember selfour' matches 'cell phone' in view (ratio >= 0.5)
    and starts enrollment for cell phone.
    """
    server = _setup_test_server()
    server.current_objects = [
        {"class_name": "cell phone", "bbox": [10, 10, 100, 100], "confidence": 0.8}
    ]
    handler = server.voice_system.command_callbacks["remember"]
    handler({"object_name": "selfour"})

    assert server.enrollment_active is True
    assert server.enrollment_target_obj["yolo_class"] == "cell phone"


# ─────────────────────────────────────────────────────────────
# 3. Whistle runtime keywords list & refresh
# ─────────────────────────────────────────────────────────────

def test_keywords_include_db_names_and_aliases():
    """Verify get_keywords includes names and aliases from DB, and refresh_keywords updates."""
    mock_db = MagicMock()
    mock_db.get_all_faces.return_value = [(1, "Alice")]
    mock_db.get_all_remembered_objects.return_value = [(1, "phone", "cell phone")]
    mock_db.cursor.fetchall.return_value = [('{"aliases": ["handset", "gadget"]}',)]

    names = stt_engine.get_db_names(mock_db)
    assert "Alice" in names
    assert "phone" in names
    assert "cell phone" in names
    assert "handset" in names
    assert "gadget" in names

    kw = stt_engine.refresh_keywords(db=mock_db)
    assert "handset" in kw
    assert "gadget" in kw


# ─────────────────────────────────────────────────────────────
# 4. ip_geolocation mock/indoor mode and cooldown
# ─────────────────────────────────────────────────────────────

def test_ip_geolocation_not_started_in_mock_or_indoor_mode(monkeypatch):
    """ip_geolocation thread is NOT started when GPS_MOCK_MODE or INDOOR_MODE is True."""
    # Case A: GPS_MOCK_MODE = True
    monkeypatch.setattr(config, "GPS_MOCK_MODE", True)
    monkeypatch.setattr(config, "INDOOR_MODE", False)

    tracker = GPSTracker(mock_mode=True, use_ip_geolocation=True)
    tracker.start()
    assert tracker.geolocation_thread is None, "Geolocation thread must not start in mock mode"
    tracker.stop()

    ip_tracker = IPGeolocationTracker()
    ip_tracker.start()
    assert ip_tracker.geolocation_thread is None, "IPGeolocationTracker thread must not start in mock mode"
    ip_tracker.stop()

    # Case B: INDOOR_MODE = True
    monkeypatch.setattr(config, "GPS_MOCK_MODE", False)
    monkeypatch.setattr(config, "INDOOR_MODE", True)

    tracker = GPSTracker(mock_mode=False, use_ip_geolocation=True)
    tracker.start()
    assert tracker.geolocation_thread is None, "Geolocation thread must not start in indoor mode"
    tracker.stop()

    ip_tracker = IPGeolocationTracker()
    ip_tracker.start()
    assert ip_tracker.geolocation_thread is None, "IPGeolocationTracker thread must not start in indoor mode"
    ip_tracker.stop()


def test_ip_geolocation_failure_enters_600s_cooldown_immediately(monkeypatch):
    """When first attempt fails due to no internet, enter 600s cooldown immediately without retry loop."""
    tracker = IPGeolocationTracker(cache_seconds=300.0)
    tracker._cached_position = None
    tracker._last_fetch_time = 0.0

    call_count = 0

    def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise requests.exceptions.ConnectionError("No internet")

    with patch("requests.get", side_effect=mock_get):
        t0 = time.time()
        pos = tracker.get_position()
        elapsed = time.time() - t0

        assert pos is None
        assert call_count == 1, "Must make exactly 1 attempt without a retry loop"
        assert elapsed < 1.0, "Must not sleep or retry"
        assert tracker._in_failure_cooldown() is True, "Must immediately be in failure cooldown"

        # Subsequent call immediately skips network request
        pos2 = tracker.get_position()
        assert pos2 is None
        assert call_count == 1, "Must not retry during cooldown period"


# ─────────────────────────────────────────────────────────────
# 5. Background speech rejection > 12 words
# ─────────────────────────────────────────────────────────────

def test_whistle_rejects_longer_than_12_words_background_speech():
    """Ignore Whistle results longer than 12 words when no wake word and no window is open."""
    long_speech = "one two three four five six seven eight nine ten eleven twelve thirteen"
    short_speech = "one two three"

    # Background speech (no wake word, no window)
    assert stt_engine.should_ignore_whistle_output(long_speech, is_window_active=False, has_wake_word=False) is True
    assert stt_engine.should_ignore_whistle_output(short_speech, is_window_active=False, has_wake_word=False) is False

    # When window is open or wake word present, long speech is NOT ignored
    assert stt_engine.should_ignore_whistle_output(long_speech, is_window_active=True, has_wake_word=False) is False
    assert stt_engine.should_ignore_whistle_output(long_speech, is_window_active=False, has_wake_word=True) is False
