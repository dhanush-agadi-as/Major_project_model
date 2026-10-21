"""
Tests for:
1. Destructive voice commands:
   - "forget/delete <name>" executes directly when match ratio >= 0.8.
   - Between 0.5 and 0.8: asks "Do you want me to forget your <name>?" and opens yes/no window.
   - Below 0.5: says "I don't know an object called <name>".
2. Active find session stopping:
   - When an object is forgotten or updated, stop active find session ("Stopped looking for <name>")
     and refresh the in-RAM embedding cache.
   - Sighting updates of non-existent objects log "Object '<name>' not found in database" once at most.
3. Rolling 30-frame FPS & stage averages:
   - Rolling FPS reflects the instantaneous rate over the last 30 frames.
   - embed_ms, compare_ms, db_ms are rolling averages over the last 30 frames (rolling to 0 when unused).
4. Startup thread diagnostics:
   - print_thread_diagnostics prints thread counts (torch, OpenVINO, STT).
5. Scoped torch threads:
   - Model inference scopes torch thread count without permanently altering the global setting.
"""

import os
import sys
import time
import collections
from unittest.mock import MagicMock
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from voice_command_system import VoiceCommandSystem, VoiceCommandParser
from wireless.laptop_server import LaptopMLServer, clean_object_name
from object_memory import ObjectMemory
from find_guidance import FindGuide
from database import VisionMateDB
from deep_feature_extractor import ScopedTorchThreads


class MockAudio:
    def __init__(self):
        self.messages = []
        self.spoken = []
        self.command_replies = []
        self.social_messages = []
        self.is_speaking = False

    def speak(self, text, priority=3, is_command_reply=False):
        if text:
            self.messages.append(text)
            self.spoken.append(text)
        if is_command_reply:
            self.command_replies.append(text)

    def speak_social(self, text, is_command_reply=False):
        if text:
            self.messages.append(text)
            self.spoken.append(text)
            self.social_messages.append(text)
        if is_command_reply:
            self.command_replies.append(text)

    def speak_command_reply(self, text, priority=config.PRIORITY_SOCIAL):
        if text:
            self.messages.append(text)
            self.spoken.append(text)
            self.command_replies.append(text)

    def is_tts_active(self):
        return False

    def get_queue_size(self):
        return 0


def _setup_server():
    server = LaptopMLServer(port=8081)
    server.audio = MockAudio()
    server.audio_enabled = True
    server.object_memory = MagicMock()
    server.find_guide = MagicMock()
    server.find_guide.is_active = False
    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server.voice_system.open_yes_no_window = MagicMock()
    server._register_voice_handlers()
    return server


# ─────────────────────────────────────────────────────────────
# 1. Destructive Voice Commands Tests
# ─────────────────────────────────────────────────────────────

def test_forget_direct_execution_high_match_ratio():
    """Ratio >= 0.8 executes directly without opening yes/no window."""
    server = _setup_server()
    server.object_memory.list_remembered_objects.return_value = [("wallet", "wallet")]
    server.object_memory.delete_object.return_value = True

    # "forget wallet" matches "wallet" exactly (ratio = 1.0 >= 0.8)
    server.voice_system.process_text("hey vision forget wallet")

    server.voice_system.open_yes_no_window.assert_not_called()
    server.object_memory.delete_object.assert_called_once_with("wallet")
    assert any("Forgot your wallet" in m for m in server.audio.messages)


def test_delete_direct_execution_synonym_match():
    """'delete phone' matches enrolled 'cell phone' (synonym ratio = 1.0 >= 0.8)."""
    server = _setup_server()
    server.object_memory.list_remembered_objects.return_value = [("cell phone", "cell phone")]
    server.object_memory.delete_object.return_value = True

    server.voice_system.process_text("hey vision delete phone")

    server.voice_system.open_yes_no_window.assert_not_called()
    server.object_memory.delete_object.assert_called_once_with("cell phone")
    assert any("Forgot your cell phone" in m for m in server.audio.messages)


def test_forget_ambiguous_asks_confirmation_and_opens_window():
    """Between 0.5 and 0.8: asks 'Do you want me to forget your <name>?' and opens yes/no window."""
    server = _setup_server()
    # "wallet" vs "wal" (similarity is 0.67, which is 0.5 <= ratio < 0.8)
    server.object_memory.list_remembered_objects.return_value = [("wallet", "wallet")]

    # Spoken: "forget wal"
    server.voice_system.process_text("hey vision forget wal")

    # Should NOT delete directly
    server.object_memory.delete_object.assert_not_called()

    # Must ask confirmation: "Do you want me to forget your wallet?"
    assert any("Do you want me to forget your wallet?" in m for m in server.audio.messages)
    server.voice_system.open_yes_no_window.assert_called_once()

    # Simulate user answering "yes" via on_yes callback passed to open_yes_no_window
    kwargs = server.voice_system.open_yes_no_window.call_args[1]
    on_yes_cb = kwargs["on_yes"]
    on_yes_cb()

    server.object_memory.delete_object.assert_called_once_with("wallet")
    assert any("Forgot your wallet" in m for m in server.audio.messages)


def test_forget_low_confidence_rejects():
    """Below 0.5: says 'I don't know an object called <name>'."""
    server = _setup_server()
    server.object_memory.list_remembered_objects.return_value = [("keys", "keys")]

    server.voice_system.process_text("hey vision forget submarine")

    server.voice_system.open_yes_no_window.assert_not_called()
    server.object_memory.delete_object.assert_not_called()
    assert any("I don't know an object called submarine" in m for m in server.audio.messages)


# ─────────────────────────────────────────────────────────────
# 2. Active Find Target Cancellation Tests
# ─────────────────────────────────────────────────────────────

def test_forget_active_find_target_stops_session_and_speaks():
    """When an object is forgotten, any active find session for it is stopped with 'Stopped looking for <name>'."""
    server = _setup_server()
    server.object_memory.list_remembered_objects.return_value = [("keys", "keys")]
    server.object_memory.delete_object.return_value = True

    # Real FindGuide instance
    find_guide = FindGuide(speak=server.audio.speak, beep=lambda *args: None)
    server.find_guide = find_guide
    find_guide.start("keys", "keys")
    assert find_guide.is_active is True

    # Execute forget
    server.voice_system.process_text("hey vision forget keys")

    assert find_guide.is_active is False
    assert any("Stopped looking for keys" in m for m in server.audio.messages)
    assert any("Forgot your keys" in m for m in server.audio.messages)


def test_update_active_find_target_stops_session_and_speaks():
    """When an object is updated, any active find session for it is stopped with 'Stopped looking for <name>'."""
    server = _setup_server()
    server.object_memory.list_remembered_objects.return_value = [("wallet", "wallet")]
    server.object_memory.delete_object.return_value = True

    find_guide = FindGuide(speak=server.audio.speak, beep=lambda *args: None)
    server.find_guide = find_guide
    find_guide.start("wallet", "wallet")
    assert find_guide.is_active is True

    # Execute update
    server.voice_system.process_text("hey vision update wallet")

    assert find_guide.is_active is False
    assert any("Stopped looking for wallet" in m for m in server.audio.messages)


# ─────────────────────────────────────────────────────────────
# 3. DB Not-Found Logged Once at Most Tests
# ─────────────────────────────────────────────────────────────

def test_object_not_found_logged_once_at_most(tmp_path, caplog):
    """update_object_sighting logs 'Object not found in database' once at most, never per frame."""
    db_path = str(tmp_path / "not_found_test.db")
    db = VisionMateDB(db_path=db_path)
    mem = ObjectMemory(db=db)

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Call update_object_sighting 10 times for a non-existent object
    with caplog.at_level("ERROR"):
        for i in range(10):
            res = mem.update_object_sighting(
                custom_name="phantom_gadget",
                image=dummy_frame,
                confidence=0.9
            )
            assert res is False

    # Check logs: the phrase must appear at most once
    not_found_logs = [r for r in caplog.records if "not found in database" in r.message]
    assert len(not_found_logs) == 1
    assert "Object 'phantom_gadget' not found in database" in not_found_logs[0].message

    # Calling refresh_cache() clears the logged set so future novel lookups can log again
    mem.refresh_cache()
    assert len(mem._logged_not_found) == 0

    db.close()


# ─────────────────────────────────────────────────────────────
# 4. Rolling 30-Frame FPS and Stage Averages Tests
# ─────────────────────────────────────────────────────────────

def test_rolling_fps_reflects_last_30_frames():
    """get_rolling_fps computes instantaneous FPS over the last 30 frames."""
    server = LaptopMLServer(port=8082)

    # With fewer than 2 frames, returns 0.0
    assert server.get_rolling_fps() == 0.0

    # Simulate 30 frames arriving at exactly 30 FPS (interval = 1/30 s = 0.03333 s)
    t = 1000.0
    interval = 1.0 / 30.0
    for _ in range(31):
        server._rolling_frame_times.append(t)
        t += interval

    fps = server.get_rolling_fps()
    assert abs(fps - 30.0) < 0.1

    # Simulate sudden speed-up to 60 FPS over the next 30 frames
    t_fast = server._rolling_frame_times[-1]
    fast_interval = 1.0 / 60.0
    for _ in range(31):
        t_fast += fast_interval
        server._rolling_frame_times.append(t_fast)

    fast_fps = server.get_rolling_fps()
    assert abs(fast_fps - 60.0) < 0.1


def test_rolling_stage_averages_roll_to_zero_when_unused():
    """embed_ms, compare_ms, db_ms roll down to 0.0 when unused across frames."""
    server = LaptopMLServer(port=8083)

    # Initial values should be 0.0
    assert server.embed_ms == 0.0
    assert server.compare_ms == 0.0
    assert server.db_ms == 0.0

    # Add 5 active frames with heavy stage times
    for _ in range(5):
        server._rolling_embed_ms.append(25.0)
        server._rolling_compare_ms.append(10.0)
        server._rolling_db_ms.append(5.0)

    assert server.embed_ms == 25.0
    assert server.compare_ms == 10.0
    assert server.db_ms == 5.0

    # Simulate 30 idle frames where embedding, comparison, and DB write are unused (0.0 ms)
    for _ in range(30):
        server._rolling_embed_ms.append(0.0)
        server._rolling_compare_ms.append(0.0)
        server._rolling_db_ms.append(0.0)

    # All metrics must roll down to exactly 0.0
    assert server.embed_ms == 0.0
    assert server.compare_ms == 0.0
    assert server.db_ms == 0.0


# ─────────────────────────────────────────────────────────────
# 5. Startup Thread Diagnostics Line Test
# ─────────────────────────────────────────────────────────────

def test_startup_thread_diagnostics_format():
    """print_thread_diagnostics outputs [STARTUP] Thread counts with torch, OpenVINO, and STT."""
    server = LaptopMLServer(port=8084)
    line = server.print_thread_diagnostics()

    assert line.startswith("[STARTUP] Thread counts:")
    assert "torch=" in line
    assert "OpenVINO=" in line
    assert "STT=" in line


# ─────────────────────────────────────────────────────────────
# 6. ScopedTorchThreads Context Manager Test
# ─────────────────────────────────────────────────────────────

def test_scoped_torch_threads():
    """ScopedTorchThreads sets thread count within context and restores original upon exit."""
    import torch
    orig = torch.get_num_threads()

    with ScopedTorchThreads(2):
        assert torch.get_num_threads() == 2

    # Restored to original count
    assert torch.get_num_threads() == orig
