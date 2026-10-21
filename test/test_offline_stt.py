"""
Test suite for offline speech recognition (Cactus Whistle) and STT engine selection.
Tests:
1. The STT engine is selected from config ("whistle", "google", "auto").
2. The fallback to Google happens when Whistle returns empty (in auto mode, online).
3. Whistle filler text ("thank you", "you", ".") is ignored on non-speech, but allowed
   when a wake word or open window is active.
4. The app works with no internet in whistle mode (100% offline).
5. If cactus-needle fails to import or is unavailable, a warning is logged and Google STT is used.
"""

import sys
import os
import time
import pytest
from unittest.mock import MagicMock, patch
import speech_recognition as sr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import stt_engine
from voice_command_system import VoiceCommandSystem, VoiceCapture


class DummyAudioFeedback:
    def __init__(self):
        self.messages = []
        self.is_speaking = False
        self.is_playing = False
        self.last_speech_end_time = 0.0

    def speak(self, text, priority=3, is_command_reply=False):
        self.messages.append(text)

    def speak_social(self, text, is_command_reply=False):
        self.messages.append(text)

    def speak_status(self, text, is_command_reply=False):
        self.messages.append(text)

    def speak_command_reply(self, text, priority=3):
        self.messages.append(text)

    def is_tts_active(self):
        return False

    def flush_non_emergency(self):
        pass

    def stop(self):
        pass


def _create_vcs():
    audio = DummyAudioFeedback()
    vcs = VoiceCommandSystem(audio_feedback=audio)
    vcs.enabled = True
    vcs.running = True
    return vcs


# ─────────────────────────────────────────────────────────────
# 1. Engine Selection from config.STT_ENGINE
# ─────────────────────────────────────────────────────────────

def test_engine_selected_from_config(monkeypatch):
    """Verify STT engine selection ('whistle', 'google', 'auto') from config."""
    vcs = _create_vcs()
    mock_recognizer = MagicMock()
    mock_audio = MagicMock()

    # Case A: config.STT_ENGINE = "whistle"
    monkeypatch.setattr(config, "STT_ENGINE", "whistle")
    with patch("stt_engine.is_available", return_value=True), \
         patch("stt_engine.transcribe_with_timing", return_value=("hey vision list", 50)) as mock_whistle:
        vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)
        assert mock_whistle.called is True, "Whistle must be called when STT_ENGINE is 'whistle'"
        assert mock_recognizer.recognize_google.called is False, "Google must NOT be called in 'whistle' mode"

    # Case B: config.STT_ENGINE = "google"
    mock_recognizer.reset_mock()
    monkeypatch.setattr(config, "STT_ENGINE", "google")
    mock_recognizer.recognize_google.return_value = "hey vision list"
    with patch("stt_engine.is_available", return_value=True), \
         patch("stt_engine.transcribe_with_timing") as mock_whistle:
        vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)
        assert mock_whistle.called is False, "Whistle must NOT be called in 'google' mode"
        assert mock_recognizer.recognize_google.called is True, "Google must be called in 'google' mode"

    # Case C: config.STT_ENGINE = "auto" with non-empty Whistle output
    mock_recognizer.reset_mock()
    monkeypatch.setattr(config, "STT_ENGINE", "auto")
    with patch("stt_engine.is_available", return_value=True), \
         patch("stt_engine.transcribe_with_timing", return_value=("hey vision help", 40)) as mock_whistle:
        vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)
        assert mock_whistle.called is True, "Whistle must be called first in 'auto' mode"
        assert mock_recognizer.recognize_google.called is False, "Google must NOT be called when Whistle succeeds"


# ─────────────────────────────────────────────────────────────
# 2. Fallback when Whistle returns empty
# ─────────────────────────────────────────────────────────────

def test_fallback_when_whistle_returns_empty(monkeypatch):
    """
    In 'auto' mode, when Whistle returns empty and internet is reachable,
    fallback to Google STT occurs.
    """
    vcs = _create_vcs()
    mock_recognizer = MagicMock()
    mock_audio = MagicMock()
    mock_recognizer.recognize_google.return_value = "hey vision navigate"

    monkeypatch.setattr(config, "STT_ENGINE", "auto")

    processed_texts = []
    vcs.process_text = lambda t: processed_texts.append(t)

    with patch("stt_engine.is_available", return_value=True), \
         patch("stt_engine.transcribe_with_timing", return_value=("", 30)) as mock_whistle, \
         patch("stt_engine.is_internet_reachable", return_value=True):
        
        vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)

        assert mock_whistle.called is True, "Whistle must be called first"
        assert mock_recognizer.recognize_google.called is True, "Google must be called as fallback when Whistle is empty"
        assert processed_texts == ["hey vision navigate"], "Fallback text must be processed"


# ─────────────────────────────────────────────────────────────
# 3. Filler text rejection on non-speech
# ─────────────────────────────────────────────────────────────

def test_filler_text_ignored(monkeypatch):
    """
    Whistle outputs that are filler on non-speech (e.g. 'thank you', 'you', '.')
    are ignored unless a wake word or open window is active.
    """
    vcs = _create_vcs()
    mock_recognizer = MagicMock()
    mock_audio = MagicMock()

    monkeypatch.setattr(config, "STT_ENGINE", "whistle")
    processed_texts = []
    vcs.process_text = lambda t: processed_texts.append(t)

    # 1. Non-speech filler with no wake word and no window active -> ignored!
    fillers = ["thank you", "thank you.", "you", "you.", ".", "..."]
    for filler in fillers:
        processed_texts.clear()
        with patch("stt_engine.is_available", return_value=True), \
             patch("stt_engine.transcribe_with_timing", return_value=(filler, 20)):
            vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)
            assert len(processed_texts) == 0, f"Filler '{filler}' must be ignored when no window/wake active"

    # 2. Filler word WITH wake word (e.g. 'hey vision thank you') -> NOT ignored!
    processed_texts.clear()
    with patch("stt_engine.is_available", return_value=True), \
         patch("stt_engine.transcribe_with_timing", return_value=("hey vision thank you", 25)):
        vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)
        assert len(processed_texts) == 1, "Filler with wake word must NOT be ignored"
        assert "thank you" in processed_texts[0].lower()

    vcs.open_dictation_window(on_dictate=lambda txt: None, timeout=5.0)
    assert vcs.dictation_active is True
    processed_texts.clear()
    with patch("stt_engine.is_available", return_value=True), \
         patch("stt_engine.transcribe_with_timing", return_value=("thank you", 20)):
        vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)
        assert len(processed_texts) == 1, "Filler while dictation window is open must NOT be ignored"
    vcs.close_dictation_window()


# ─────────────────────────────────────────────────────────────
# 4. App works with no internet in whistle mode
# ─────────────────────────────────────────────────────────────

def test_app_works_with_no_internet_in_whistle_mode(monkeypatch):
    """
    In whistle mode, with internet completely unreachable, the app transcribes
    and processes commands 100% offline without failing.
    """
    vcs = _create_vcs()
    mock_recognizer = MagicMock()
    # If Google was mistakenly called offline, it would raise RequestError
    mock_recognizer.recognize_google.side_effect = sr.RequestError("No internet connection")
    mock_audio = MagicMock()

    monkeypatch.setattr(config, "STT_ENGINE", "whistle")

    command_executed = []
    vcs.register_command_handler("remember", lambda params: command_executed.append(params))

    # Simulate offline internet
    with patch("stt_engine.is_available", return_value=True), \
         patch("stt_engine.is_internet_reachable", return_value=False), \
         patch("stt_engine.transcribe_with_timing", return_value=("hey vision remember this phone", 50)):
        
        vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)

        assert mock_recognizer.recognize_google.called is False, "Google must NEVER be called in whistle mode"
        assert len(command_executed) == 1, "Remember command must be executed offline"
        assert command_executed[0].get("object_name") == "phone"


# ─────────────────────────────────────────────────────────────
# 5. Fallback to Google STT if cactus-needle is unavailable
# ─────────────────────────────────────────────────────────────

def test_needle_import_failure_falls_back_to_google(monkeypatch):
    """
    If cactus-needle fails to import or is unavailable, Whistle is bypassed
    and Google STT is used instead.
    """
    vcs = _create_vcs()
    mock_recognizer = MagicMock()
    mock_audio = MagicMock()
    mock_recognizer.recognize_google.return_value = "hey vision help"

    monkeypatch.setattr(config, "STT_ENGINE", "whistle")

    processed_texts = []
    vcs.process_text = lambda t: processed_texts.append(t)

    with patch("stt_engine.is_available", return_value=False):
        vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)
        assert mock_recognizer.recognize_google.called is True, (
            "Must fall back to Google STT when cactus-needle is unavailable"
        )
        assert processed_texts == ["hey vision help"]


# ─────────────────────────────────────────────────────────────
# 6. Keywords passed in every needle.transcribe call
# ─────────────────────────────────────────────────────────────

def test_keywords_passed_in_every_needle_transcribe():
    """Verify that keywords are passed in needle.transcribe during warm_up and transcribe."""
    import numpy as np
    with patch("needle.transcribe") as mock_needle_transcribe:
        mock_needle_transcribe.return_value = {"text": "hello"}
        # Force warm up
        stt_engine._MODEL_WARMED_UP = False
        stt_engine.warm_up()
        assert mock_needle_transcribe.called is True
        warmup_call_kwargs = mock_needle_transcribe.call_args[1]
        assert "keywords" in warmup_call_kwargs, "warm_up must pass keywords to needle.transcribe"
        assert isinstance(warmup_call_kwargs["keywords"], list)
        assert "vision" in warmup_call_kwargs["keywords"]

        # Call transcribe
        mock_needle_transcribe.reset_mock()
        stt_engine.transcribe_with_timing(np.zeros(16000, dtype=np.float32))
        assert mock_needle_transcribe.called is True
        transcribe_call_kwargs = mock_needle_transcribe.call_args[1]
        assert "keywords" in transcribe_call_kwargs, "transcribe must pass keywords to needle.transcribe"
        assert isinstance(transcribe_call_kwargs["keywords"], list)


# ─────────────────────────────────────────────────────────────
# 7. HF_HUB_OFFLINE set to 1 after cached
# ─────────────────────────────────────────────────────────────

def test_hf_hub_offline_set_after_cached():
    """Verify that os.environ['HF_HUB_OFFLINE'] is set to '1' when model is cached."""
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE must be set to '1' after model is cached"


# ─────────────────────────────────────────────────────────────
# 8. Auto mode offline is Whistle only (no Google call)
# ─────────────────────────────────────────────────────────────

def test_auto_mode_offline_only_whistle(monkeypatch):
    """
    In default 'auto' mode, when Whistle returns empty and internet is unreachable,
    Google STT is NOT called (offline it is Whistle only).
    """
    vcs = _create_vcs()
    mock_recognizer = MagicMock()
    mock_audio = MagicMock()

    monkeypatch.setattr(config, "STT_ENGINE", "auto")

    with patch("stt_engine.is_available", return_value=True), \
         patch("stt_engine.transcribe_with_timing", return_value=("", 20)), \
         patch("stt_engine.is_internet_reachable", return_value=False):

        vcs._bg_audio_callback(mock_recognizer, mock_audio, _sync=True)
        assert mock_recognizer.recognize_google.called is False, "Offline auto mode must not call Google STT"

