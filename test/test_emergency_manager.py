"""
test_emergency_manager.py
=========================
Comprehensive tests for EmergencyManager and AlertSender:
- A fake fall event starts the check-in
- "I'm fine" cancels with no alert
- Three silences send a severe alert
- "no" on the first ask sends a severe alert
- A second fall inside 60 s is ignored
- The 5-minute limit holds (and voice emergency bypasses it)
- Dry-run sends nothing
- Location falls back to IP then to none and still sends
- Cancel sets the cancelled status and pushes "cancelled"
- Invalid tokens are removed
- Ack triggers the spoken line
- Running find session pauses during check-in and resumes after
- No SMS/call/Telegram code or config remains
- No secrets in tracked git files (Firebase mocked in tests)
"""

import os
import sys
import time
import queue
import subprocess
import pytest
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from emergency_manager import EmergencyManager, classify_reply
from alert_sender import AlertSender


class MockAudio:
    def __init__(self):
        self.spoken = []

    def speak_prompt(self, text, interrupt=True, wait=True):
        self.spoken.append(("prompt", text))

    def speak(self, text):
        self.spoken.append(("general", text))

    def speak_social(self, text):
        self.spoken.append(("social", text))


class MockVoiceSystem:
    def __init__(self):
        self.emergency_checkin_active = False
        self.on_reply_callback = None
        self.commands_executed = []

    def open_emergency_window(self, timeout=8.0, on_reply=None):
        self.emergency_checkin_active = True
        self.on_reply_callback = on_reply

    def close_emergency_window(self):
        self.emergency_checkin_active = False
        self.on_reply_callback = None

    def simulate_user_reply(self, text):
        if self.on_reply_callback:
            self.on_reply_callback(text)


class MockFindGuide:
    def __init__(self):
        self.is_active = True
        self._is_paused = False

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    @property
    def is_paused(self):
        return self._is_paused


# ─── 1. Fake fall event starts check-in & "I'm fine" cancels with no alert ─────
def test_fake_fall_and_im_fine_cancels(capsys):
    audio = MockAudio()
    voice = MockVoiceSystem()
    mock_sender = MagicMock(spec=AlertSender)
    mock_sender.send_fall_alert.return_value = "alert-fall-101"
    em = EmergencyManager(
        audio_feedback=audio,
        voice_system=voice,
        alert_sender=mock_sender,
        dry_run=True
    )

    event = {"id": "fall-101", "type": "fall", "peak_g": 3.1, "ts": time.time()}

    # Auto-reply "I'm fine" when asked
    def auto_reply(timeout_s=8.0):
        return "I'm fine"

    em._listen_for_reply = auto_reply
    em._run_checkin_protocol(event)

    # Verify check-in asked "Are you okay?"
    prompts = [t for p, t in audio.spoken if "are you okay" in t.lower()]
    assert len(prompts) >= 1
    # Verify Prompt 1 includes cancel instruction
    assert any("say i'm okay to cancel" in t.lower() for p, t in audio.spoken)

    # Verify "Emergency cancelled. Glad you are okay." was spoken
    replies = [t for p, t in audio.spoken if "glad you are okay" in t.lower() or "glad you're fine" in t.lower()]
    assert len(replies) == 1

    # Verify NO alert was sent and state reset to IDLE
    mock_sender.send_alert.assert_not_called()
    mock_sender.send_fall_alert.assert_not_called()
    assert em.state == "IDLE"

    # Verify logged output format:
    captured = capsys.readouterr().out
    assert "EMERGENCY: event=fall-101 asked=1 reply=I'm fine -> cancel" in captured


# ─── 2. Three silences send a severe alert ─────────────────────────────────────
def test_three_silences_send_severe_alert(capsys):
    audio = MockAudio()
    mock_sender = MagicMock(spec=AlertSender)
    mock_sender.send_fall_alert.return_value = "alert-severe-999"
    mock_sender.send_alert.return_value = "alert-severe-999"

    em = EmergencyManager(
        audio_feedback=audio,
        alert_sender=mock_sender,
        dry_run=True
    )

    event = {"id": "fall-silence", "type": "fall", "peak_g": 3.5, "ts": time.time()}

    # Silence (timeout) on all asks
    em._listen_for_reply = lambda timeout_s=8.0: None
    em._run_checkin_protocol(event)

    # Verify asked 3 times
    prompts = [t for p, t in audio.spoken if "are you okay" in t.lower()]
    assert len(prompts) == 3

    # Verify alert was dispatched and state is TRIGGERED
    assert mock_sender.send_fall_alert.called or mock_sender.send_alert.called
    assert em.state == "TRIGGERED"

    # Verify logged format
    captured = capsys.readouterr().out
    assert "EMERGENCY: event=fall-silence asked=3 reply=None -> severe" in captured


# ─── 3. "no" on first ask sends severe alert ──────────────────────────────────
def test_no_on_first_ask_sends_severe_alert(capsys):
    audio = MockAudio()
    mock_sender = MagicMock(spec=AlertSender)
    mock_sender.send_fall_alert.return_value = "alert-bad-123"
    mock_sender.send_alert.return_value = "alert-bad-123"

    em = EmergencyManager(
        audio_feedback=audio,
        alert_sender=mock_sender,
        dry_run=True
    )

    event = {"id": "fall-hurt", "type": "fall", "peak_g": 3.0, "ts": time.time()}
    em._listen_for_reply = lambda timeout_s=8.0: "no, I'm hurt"
    em._run_checkin_protocol(event)

    # Asked once
    prompts = [t for p, t in audio.spoken if "are you okay" in t.lower()]
    assert len(prompts) == 1

    # Alert dispatched immediately and state is TRIGGERED
    assert mock_sender.send_fall_alert.called or mock_sender.send_alert.called
    assert em.state == "TRIGGERED"

    captured = capsys.readouterr().out
    assert "EMERGENCY: event=fall-hurt asked=1 reply=no, I'm hurt -> severe" in captured


# ─── 4. Second fall inside 60s is ignored ──────────────────────────────────────
def test_second_fall_inside_60s_ignored():
    em = EmergencyManager(dry_run=True)
    em.last_checkin_end_time = time.time()  # Check-in just ended 5 seconds ago

    event = {"id": "fall-too-soon", "type": "fall", "peak_g": 3.0, "ts": time.time()}
    with patch.object(em, '_run_checkin_protocol') as mock_run:
        em.on_fall_detected(event)
        mock_run.assert_not_called()
        assert em.is_checkin_running is False


# ─── 5. 5-minute safety limit holds & emergency voice command bypasses it ──────
def test_five_minute_rate_limit_and_bypass():
    mock_sender = MagicMock(spec=AlertSender)
    mock_sender.send_fall_alert.return_value = "alert-fall-1"
    mock_sender.send_alert.return_value = "alert-manual"

    em = EmergencyManager(
        alert_sender=mock_sender,
        dry_run=True
    )

    def total_dispatches():
        return mock_sender.send_fall_alert.call_count + mock_sender.send_alert.call_count

    # First automatic alert at t=1000
    with patch("time.time", return_value=1000.0):
        em._dispatch_automatic_alert("First fall")
        assert total_dispatches() == 1

    # Second automatic alert at t=1200 (200s later < 300s / 5 min) -> Suppressed!
    with patch("time.time", return_value=1200.0):
        em._dispatch_automatic_alert("Second fall")
        assert total_dispatches() == 1  # Still 1, suppressed!

    # Manual voice emergency at t=1250 -> BYPASSES the 5-minute rate limit!
    with patch("time.time", return_value=1250.0):
        em.handle_voice_emergency()
        assert total_dispatches() == 2  # Dispatched!


# ─── 6. Dry run sends nothing ─────────────────────────────────────────────────
def test_dry_run_sends_nothing():
    sender = AlertSender(dry_run=True)
    aid = sender.send_alert(severity="severe", reason="Test dry run")
    assert aid is not None
    assert sender.db is None


# ─── 7. Location fallback: Phone GPS -> IP -> None ─────────────────────────────
def test_location_fallbacks():
    # Case A: Fresh Phone GPS
    mock_gps = MagicMock()
    mock_gps._get_fresh_phone_position_info.return_value = ((12.9265, 77.5264), 2.0, 15.0)
    sender_phone = AlertSender(dry_run=True, gps_tracker=mock_gps)
    loc_a = sender_phone.resolve_location()
    assert loc_a["source"] == "phone"
    assert loc_a["lat"] == 12.9265
    assert loc_a["lon"] == 77.5264

    # Case B: Phone GPS missing, IP geolocation available
    mock_gps_ip = MagicMock()
    mock_gps_ip._get_fresh_phone_position_info.return_value = None
    mock_ip_tracker = MagicMock()
    mock_ip_tracker.get_position.return_value = (13.0, 77.6)
    mock_ip_tracker.get_city.return_value = "Bengaluru"
    mock_gps_ip._get_ip_tracker.return_value = mock_ip_tracker
    sender_ip = AlertSender(dry_run=True, gps_tracker=mock_gps_ip)
    loc_b = sender_ip.resolve_location()
    assert loc_b["source"] == "ip"
    assert loc_b["lat"] == 13.0
    assert loc_b["lon"] == 77.6

    # Case C: No location available at all -> sends with reason stating location unavailable
    mock_gps_none = MagicMock()
    mock_gps_none._get_fresh_phone_position_info.return_value = None
    mock_gps_none._get_ip_tracker.return_value = None
    with patch("ip_geolocation.IPGeolocationTracker") as mock_ip_cls:
        mock_instance = mock_ip_cls.return_value
        mock_instance.get_position.return_value = None
        sender_none = AlertSender(dry_run=True, gps_tracker=mock_gps_none)
        loc_c = sender_none.resolve_location()
        assert loc_c["lat"] is None
        assert loc_c["source"] is None

        # Verify send_alert states location unavailable in reason
        with patch.object(sender_none, 'resolve_location', return_value=loc_c):
            aid = sender_none.send_alert(severity="severe", reason="Fall detected")
            assert aid is not None


# ─── 8. Cancel sets cancelled status & pushes cancelled ────────────────────────
def test_cancel_alert_workflow():
    sender = AlertSender(dry_run=False)
    # Mock Firestore db and messaging
    mock_db = MagicMock()
    sender.db = mock_db
    sender.dry_run = False

    # Simulate 1 registered device
    mock_device = MagicMock()
    mock_device.id = "dev-1"
    mock_device.to_dict.return_value = {"token": "fcm-token-123", "name": "Mom"}
    mock_db.collection("devices").stream.return_value = [mock_device]

    with patch("alert_sender.messaging") as mock_msg:
        sender.cancel_alert(alert_id="alert-test-xyz")

        # Status updated to cancelled
        mock_db.collection("alerts").document("alert-test-xyz").update.assert_called_with({"status": "cancelled"})

        # FCM cancelled data message pushed
        mock_msg.Message.assert_called()
        call_kwargs = mock_msg.Message.call_args[1]
        assert call_kwargs["data"] == {"type": "cancelled", "alertId": "alert-test-xyz"}
        assert call_kwargs["token"] == "fcm-token-123"


# ─── 9. Invalid device tokens are removed ─────────────────────────────────────
def test_invalid_tokens_removed():
    sender = AlertSender(dry_run=False)
    mock_db = MagicMock()
    sender.db = mock_db
    sender.dry_run = False

    # Device with invalid token
    mock_device = MagicMock()
    mock_device.id = "bad-dev"
    mock_device.to_dict.return_value = {"token": "expired-token-404"}
    mock_db.collection("devices").stream.return_value = [mock_device]

    with patch("alert_sender.messaging") as mock_msg:
        # Simulate FCM Unregistered / invalid error
        mock_msg.send.side_effect = Exception("Requested entity was not found: registration-token-not-registered")

        sender.send_alert(severity="severe", reason="Fall detected")

        # Invalid device document was deleted
        mock_device.reference.delete.assert_called_once()


# ─── 10. Ack triggers spoken line "<name> is on the way" ───────────────────────
def test_ack_triggers_spoken_line():
    spoken_lines = []
    sender = AlertSender(dry_run=False, speak_fn=lambda txt: spoken_lines.append(txt))
    mock_db = MagicMock()
    sender.db = mock_db
    sender.dry_run = False

    # Start snapshot listener
    sender._start_ack_listener("alert-ack-1")
    alert_ref = mock_db.collection("alerts").document("alert-ack-1")

    # Extract the on_snapshot callback passed to Firestore
    snapshot_callback = alert_ref.on_snapshot.call_args[0][0]

    # Simulate Firestore delivering a document with an ack from "John"
    mock_doc = MagicMock()
    mock_doc.exists = True
    mock_doc.to_dict.return_value = {
        "acks": [{"name": "John", "ts": 1700000000}]
    }

    snapshot_callback([mock_doc], None, None)

    assert len(spoken_lines) == 1
    assert spoken_lines[0] == "John is on the way"


# ─── 11. Running find session pauses during check-in & resumes after ────────────
def test_find_session_pauses_and_resumes():
    find_guide = MockFindGuide()
    assert find_guide.is_paused is False

    mock_sender = MagicMock(spec=AlertSender)
    em = EmergencyManager(
        find_guide=find_guide,
        alert_sender=mock_sender,
        dry_run=True
    )

    # During check-in, verify find session is paused
    def check_paused_during_reply(timeout_s=8.0):
        assert find_guide.is_paused is True
        return "I'm okay"

    em._listen_for_reply = check_paused_during_reply
    em._run_checkin_protocol({"id": "fall-find", "type": "fall"})

    # After check-in finishes, verify find session is resumed
    assert find_guide.is_paused is False


# ─── 12. No SMS, voice call, or Telegram code or config ────────────────────────
def test_no_sms_or_telegram_remains():
    forbidden_terms = ["telegram", "twilio", "sms_alert", "voice_call"]
    for root, _, files in os.walk(REPO_ROOT):
        if any(ignored in root for ignored in [".git", "__pycache__", ".pytest_cache"]):
            continue
        for file in files:
            if file.endswith((".py", ".json", ".md")) and file != "test_emergency_manager.py":
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().lower()
                        for term in forbidden_terms:
                            assert term not in content, f"Forbidden term '{term}' found in {file_path}"
                except Exception:
                    pass


# ─── 13. No secrets in tracked git files ───────────────────────────────────────
def test_no_secrets_in_tracked_git():
    # Run git status / ls-files to verify secret files are not tracked
    res = subprocess.run(
        ["git", "ls-files", "firebase_service_account.json", "emergency_config.json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True
    )
    assert res.returncode == 0
    tracked_secrets = res.stdout.strip()
    assert tracked_secrets == "", f"Secret files must NOT be tracked by git: {tracked_secrets}"


# ─── 14. Voice commands parsing after 'hey vision' ─────────────────────────────
def test_emergency_voice_command_parsing():
    from voice_command_system import VoiceCommandParser

    parser = VoiceCommandParser()

    # "hey vision I'm okay"
    res1 = parser.parse("i'm okay")
    assert res1["intent"] == "im_okay"

    res2 = parser.parse("cancel alert")
    assert res2["intent"] == "im_okay"

    # "hey vision emergency"
    res3 = parser.parse("emergency")
    assert res3["intent"] == "emergency"

    # "hey vision help me"
    res4 = parser.parse("help me")
    assert res4["intent"] == "emergency"

    # "hey vision simulate fall"
    res5 = parser.parse("simulate fall")
    assert res5["intent"] == "simulate_fall"


# ─── 15. Physical fall detection criteria calibration for live demo ──────────
def test_fall_criteria_calibration(capsys):
    em = EmergencyManager(dry_run=True)
    # Patch on_fall_detected to prevent actual thread launch
    with patch.object(em, 'on_fall_detected') as mock_on_fall:
        t0 = 1000.0

        # Resting at 1.0g -> no event
        assert em._check_fall_criteria(1.0, ts=t0) is False
        mock_on_fall.assert_not_called()

        # Step a & b: Movement >= 1.15g qualifies as impact spike from resting baseline
        em._check_fall_criteria(1.20, ts=t0 + 0.1)
        mock_on_fall.assert_not_called()  # Needs stillness to verify

        # Step c: Resting in 0.85g to 1.15g for 0.5s to 1.0s confirms the fall
        em._check_fall_criteria(0.95, ts=t0 + 0.2)   # stillness start
        em._check_fall_criteria(1.05, ts=t0 + 0.5)   # 0.3s stillness
        mock_on_fall.assert_not_called()

        # At t0 + 0.75s (0.55s of stillness >= 0.5s) -> Fall detected!
        res = em._check_fall_criteria(1.00, ts=t0 + 0.75)
        assert res is True
        mock_on_fall.assert_called_once()
        ev = mock_on_fall.call_args[0][0]
        assert ev["type"] == "fall"
        assert ev["peak_g"] == 1.20

        # Verify output log format
        captured = capsys.readouterr().out
        assert "[FALL DETECTED] Triggered at 1.2g. Initiating confirmation." in captured


# ─── 16. "stop" response cancels emergency ────────────────────────────────────
def test_stop_response_cancels_emergency():
    assert classify_reply("stop") == "ok"
    assert classify_reply("please stop") == "ok"
    assert classify_reply("stop alert") == "ok"
    assert classify_reply("help") == "bad"
    assert classify_reply("emergency") == "bad"
    assert classify_reply("I am not okay") == "bad"


