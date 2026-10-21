"""
Tests for object voice commands and shutdown improvements:
1. "forget <object>" / "delete <object>" / "remove <object>" deletes the remembered object and says "Forgot your <object>".
2. "update <object>" / "re-remember <object>" / "remember <object> again" deletes and re-enrolls it from current view.
3. If the name is unmatched, fuzzy-match it (same rule as find).
4. "remember <object>" when it already exists says:
   "You already have a <object> remembered. Say 'update <object>' to replace it, or 'forget <object>' to remove it."
5. Enrollment detection retry: when "remember" is spoken and no object is detected in current frame,
   keep trying for up to 2 s (use latest 5 frames) before saying:
   "I don't see a <object>. Hold it in front of the camera." (or "I don't see an object to remember. Hold it in front of the camera.")
6. Shutdown: OpenVINO telemetry singleton shutdown and 1s force-exit watchdog fallback.
"""

import os
import sys
import time
import threading
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_command_system import VoiceCommandSystem, VoiceCommandParser
from wireless.laptop_server import LaptopMLServer, shutdown_openvino_telemetry


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


def _setup_server():
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


def test_parse_forget_and_update_commands():
    """Verify parser extracts intent and parameters for forget, delete, remove, update, and re-remember."""
    parser = VoiceCommandParser()

    for cmd in ["forget cell phone", "delete cell phone", "remove cell phone", "forget my cell phone", "please delete the cell phone"]:
        parsed = parser.parse(cmd)
        assert parsed["intent"] == "forget", f"Failed for {cmd}"
        assert parsed["parameters"]["object_name"] == "cell phone", f"Failed for {cmd}"

    for cmd in ["update cell phone", "re-remember cell phone", "re remember cell phone", "remember cell phone again", "update my cell phone"]:
        parsed = parser.parse(cmd)
        assert parsed["intent"] == "update", f"Failed for {cmd}"
        assert parsed["parameters"]["object_name"] == "cell phone", f"Failed for {cmd}"

    # Verify notes commands are not intercepted
    assert parser.parse("delete note 1")["intent"] == "delete_note"
    assert parser.parse("remove note 2")["intent"] == "delete_note"
    assert parser.parse("clear notes")["intent"] == "clear_notes"


def test_forget_cell_phone_removes_it():
    """'forget cell phone' removes it and says 'Forgot your cell phone'."""
    server = _setup_server()
    server.object_memory.list_remembered_objects.return_value = [("cell phone", "cell phone")]
    server.object_memory.delete_object.return_value = True

    server.voice_system.process_text("hey vision forget cell phone")

    server.object_memory.delete_object.assert_called_with("cell phone")
    assert any("Forgot your cell phone" in m for m in server.audio.messages)


def test_forget_fuzzy_matching():
    """'delete cellphone' fuzzy-matches 'cell phone' and deletes it."""
    server = _setup_server()
    server.object_memory.list_remembered_objects.return_value = [("cell phone", "cell phone")]
    server.object_memory.delete_object.return_value = True

    server.voice_system.process_text("hey vision delete cellphone")

    server.object_memory.delete_object.assert_called_with("cell phone")
    assert any("Forgot your cell phone" in m for m in server.audio.messages)


def test_remember_cell_phone_when_enrolled_gives_friendly_message():
    """'remember cell phone' when enrolled gives the friendly message and doesn't say 'Error...'."""
    server = _setup_server()
    server.object_memory.list_remembered_objects.return_value = [("cell phone", "cell phone")]
    server.current_objects = [{"class_name": "cell phone", "bbox": [10, 10, 50, 50], "confidence": 0.9}]

    server.voice_system.process_text("hey vision remember cell phone")

    expected_msg = (
        "You already have a cell phone remembered. "
        "Say 'update cell phone' to replace it, or 'forget cell phone' to remove it."
    )
    assert any(expected_msg in m for m in server.audio.messages)
    assert not any("Error" in m for m in server.audio.messages)
    assert not server.enrollment_active


def test_update_cell_phone_re_enrolls_it():
    """'update cell phone' deletes the existing object and re-enrolls it from the current view."""
    server = _setup_server()
    remembered_list = [("cell phone", "cell phone")]
    server.object_memory.list_remembered_objects.side_effect = lambda: list(remembered_list)

    def _mock_delete(name):
        nonlocal remembered_list
        remembered_list = [r for r in remembered_list if r[0] != name]
        return True

    server.object_memory.delete_object.side_effect = _mock_delete
    server.current_objects = [{"class_name": "cell phone", "bbox": [10, 10, 50, 50], "confidence": 0.9}]

    server.voice_system.process_text("hey vision update cell phone")

    assert server.object_memory.delete_object.called
    assert server.enrollment_active
    assert any("Remembering your cell phone. Hold it and turn it slowly." in m or "Remembering your cell phone. Hold it steady." in m for m in server.audio.messages)


def test_remember_retries_for_2s_when_no_object_detected_at_first():
    """'remember' retries for 2 s when no object is detected at first, succeeding when object appears."""
    server = _setup_server()
    server.object_memory.list_remembered_objects.return_value = []
    server.current_objects = []
    server.recent_frames.clear()

    # Simulate object appearing in frame after 0.25 seconds
    def _appear_object():
        time.sleep(0.25)
        server.current_objects = [{"class_name": "cell phone", "bbox": [10, 10, 50, 50], "confidence": 0.9}]

    th = threading.Thread(target=_appear_object, daemon=True)
    th.start()

    t0 = time.time()
    server.voice_system.process_text("hey vision remember cell phone")
    elapsed = time.time() - t0

    assert 0.2 <= elapsed < 1.8
    assert server.enrollment_active
    assert any("Remembering your cell phone. Hold it and turn it slowly." in m or "Remembering your cell phone. Hold it steady." in m for m in server.audio.messages)


def test_remember_retries_and_times_out_after_2s():
    """'remember cell phone' retries for 2 s and says 'I don't see a cell phone. Hold it in front of the camera.'"""
    server = _setup_server()
    server.object_memory.list_remembered_objects.return_value = []
    server.current_objects = []
    server.recent_frames.clear()

    t0 = time.time()
    server.voice_system.process_text("hey vision remember cell phone")
    elapsed = time.time() - t0

    assert elapsed >= 1.9
    assert any("I don't see a cell phone. Hold it in front of the camera." in m for m in server.audio.messages)
    assert not server.enrollment_active


def test_remember_bare_retries_and_times_out_after_2s():
    """Bare 'remember' retries for 2 s and says 'I don't see an object to remember. Hold it in front of the camera.'"""
    server = _setup_server()
    server.object_memory.list_remembered_objects.return_value = []
    server.current_objects = []
    server.recent_frames.clear()

    t0 = time.time()
    server.voice_system.process_text("hey vision remember")
    elapsed = time.time() - t0

    assert elapsed >= 1.9
    assert any("I don't see an object to remember. Hold it in front of the camera." in m for m in server.audio.messages)
    assert not server.enrollment_active


def test_remember_uses_latest_5_frames():
    """'remember' checks recent_frames and enrolls an object detected in recent frames."""
    server = _setup_server()
    server.object_memory.list_remembered_objects.return_value = []
    server.current_objects = []
    server.recent_frames.clear()
    server.recent_frames.append({
        'frame': None,
        'objects': [{"class_name": "cell phone", "bbox": [10, 10, 50, 50], "confidence": 0.9}],
        'timestamp': time.time()
    })

    t0 = time.time()
    server.voice_system.process_text("hey vision remember cell phone")
    elapsed = time.time() - t0

    assert elapsed < 1.0
    assert server.enrollment_active
    assert any("Remembering your cell phone. Hold it and turn it slowly." in m or "Remembering your cell phone. Hold it steady." in m for m in server.audio.messages)


def test_shutdown_openvino_telemetry():
    """Verify shutdown_openvino_telemetry shuts down Telemetry singleton executor."""
    mock_executor = MagicMock()
    mock_sender = MagicMock(executor=mock_executor)
    mock_instance = MagicMock(sender=mock_sender)

    with patch('openvino_telemetry.main.Telemetry', create=True) as mock_telemetry:
        mock_telemetry._SingletonMetaClass__single_instance = mock_instance
        res = shutdown_openvino_telemetry()
        assert res is True
        mock_executor.shutdown.assert_called_with(wait=False, cancel_futures=True)


def test_help_text_includes_forget_and_update():
    """Verify get_help_text includes forget and update commands."""
    parser = VoiceCommandParser()
    help_text = parser.get_help_text().lower()
    assert "update [object]" in help_text
    assert "forget [object]" in help_text
