"""
Tests for 'where am I' real phone GPS integration, source priority,
voice parsing, and non-blocking background polling.

Requirements verified:
1. Phone position is used when fresh (<30s), logging source, age, accuracy,
   and speaking street-level place name. If accuracy > 100m, speaks "approximately near".
2. Stale position (>30s) falls back to IP geolocation with "approximate location" spoken note.
3. Mock position is NOT used when GPS_MOCK_MODE is False (speaks prompt to open location page).
   Mock position is ONLY used when GPS_MOCK_MODE is True.
4. Voice parsing: "where am", "where", "where I", "where am I", "whereabouts",
   "my location", "what is this place" parse as where_am_i, but ONLY when whole command
   (so "where is my phone" triggers find, not where_am_i).
5. Polling thread runs in background and never blocks the main loop.
"""

import time
import threading
import logging
from unittest.mock import MagicMock, patch
import pytest

import config
from gps_tracker import GPSTracker
from voice_command_system import VoiceCommandParser


# ---------------------------------------------------------------------------
# 1. Phone position used when fresh
# ---------------------------------------------------------------------------

def test_phone_position_used_when_fresh(caplog):
    """
    Phone GPS fix is fresher than 30 s:
    - get_position() returns phone fix
    - last_source is 'phone'
    - logs 'GPS position source used: phone (age=<s>, accuracy=<m>)'
    - speaks street-level reverse geocode name
    - accuracy <= 100m uses 'You are near <place>'
    """
    tracker = GPSTracker(mock_mode=False, start_phone_polling=False)
    now = time.time()
    tracker.set_phone_position(12.9716, 77.5946, accuracy=15.0, timestamp=now - 2.0)

    assert tracker.get_position() == (12.9716, 77.5946)
    assert tracker.last_source == "phone"

    with patch('ip_geolocation.reverse_geocode', return_value="Brigade Road, Ashok Nagar, Bengaluru"):
        with caplog.at_level(logging.INFO):
            caplog.clear()
            speech = tracker.get_where_am_i_speech()

    assert speech == "You are near Brigade Road, Ashok Nagar, Bengaluru"
    assert any("GPS position source used: phone" in r.message for r in caplog.records)
    # Check that age and accuracy are present in log
    phone_log = [r.message for r in caplog.records if "GPS position source used: phone" in r.message][0]
    assert "age=" in phone_log
    assert "accuracy=15m" in phone_log or "accuracy=15" in phone_log
    tracker.stop()


def test_phone_position_fresh_with_low_accuracy(caplog):
    """
    Phone GPS fix is fresh (<30s) but accuracy is worse than 100m (>100m):
    - speaks 'You are approximately near ...'
    """
    tracker = GPSTracker(mock_mode=False, start_phone_polling=False)
    now = time.time()
    tracker.set_phone_position(12.9716, 77.5946, accuracy=150.0, timestamp=now - 5.0)

    with patch('ip_geolocation.reverse_geocode', return_value="MG Road, Bengaluru"):
        with caplog.at_level(logging.INFO):
            caplog.clear()
            speech = tracker.get_where_am_i_speech()

    assert speech == "You are approximately near MG Road, Bengaluru"
    assert any("GPS position source used: phone" in r.message for r in caplog.records)
    phone_log = [r.message for r in caplog.records if "GPS position source used: phone" in r.message][0]
    assert "accuracy=150m" in phone_log or "accuracy=150" in phone_log
    tracker.stop()


# ---------------------------------------------------------------------------
# 2. Stale position falls back to IP
# ---------------------------------------------------------------------------

def test_stale_phone_position_falls_back_to_ip(caplog):
    """
    Phone position older than 30s is ignored; falls back to IP geolocation:
    - get_position() returns IP fix
    - last_source is 'ip'
    - logs 'GPS position source used: ip'
    - speaks 'You are approximately near <place>. Approximate location.'
    """
    tracker = GPSTracker(mock_mode=False, start_phone_polling=False)
    # Stale phone fix (40s ago)
    tracker.set_phone_position(13.0827, 80.2707, accuracy=10.0, timestamp=time.time() - 40.0)

    # Mock IP tracker
    mock_ip_tracker = MagicMock()
    mock_ip_tracker.get_position.return_value = (12.9265, 77.5264)
    mock_ip_tracker.get_city.return_value = "Bengaluru, Karnataka"
    tracker.ip_tracker = mock_ip_tracker

    pos = tracker.get_position()
    assert pos == (12.9265, 77.5264)
    assert tracker.last_source == "ip"

    with patch('ip_geolocation.reverse_geocode', return_value="RR Nagar, Bengaluru"):
        with caplog.at_level(logging.INFO):
            caplog.clear()
            speech = tracker.get_where_am_i_speech()

    assert any("GPS position source used: ip" in r.message for r in caplog.records)
    assert "approximate location" in speech.lower()
    assert "RR Nagar, Bengaluru" in speech or "Bengaluru" in speech
    tracker.stop()


# ---------------------------------------------------------------------------
# 3. Mock is NOT used when GPS_MOCK_MODE is False
# ---------------------------------------------------------------------------

def test_mock_not_used_when_gps_mock_mode_false(monkeypatch, caplog):
    """
    When GPS_MOCK_MODE is False and neither phone nor IP is available:
    - Never silently return mock position
    - get_position() returns None
    - speaks: "I can't get your location. Open the location page on your phone."
    """
    monkeypatch.setattr(config, 'GPS_MOCK_MODE', False)

    tracker = GPSTracker(mock_mode=False, start_phone_polling=False)
    # Ensure no phone position
    tracker._phone_position = None

    # Mock IP tracker returning None (no internet or IP failed)
    mock_ip_tracker = MagicMock()
    mock_ip_tracker.get_position.return_value = None
    mock_ip_tracker.get_city.return_value = None
    tracker.ip_tracker = mock_ip_tracker

    pos = tracker.get_position()
    assert pos is None, "Must not return mock position when GPS_MOCK_MODE is False"

    with caplog.at_level(logging.INFO):
        caplog.clear()
        speech = tracker.get_where_am_i_speech()

    assert speech == "I can't get your location. Open the location page on your phone."
    assert not any("GPS position source used: mock" in r.message for r in caplog.records)
    tracker.stop()


def test_mock_used_only_when_gps_mock_mode_true(monkeypatch, caplog):
    """
    Mock is ONLY used when GPS_MOCK_MODE is True.
    """
    monkeypatch.setattr(config, 'GPS_MOCK_MODE', True)

    tracker = GPSTracker(mock_mode=True, start_phone_polling=False)
    tracker._phone_position = None

    mock_ip_tracker = MagicMock()
    mock_ip_tracker.get_position.return_value = None
    tracker.ip_tracker = mock_ip_tracker

    pos = tracker.get_position()
    assert pos is not None
    assert tracker.last_source == "mock"

    with patch.object(tracker, 'get_current_place_name', return_value="RR Nagar, Bangalore"):
        with caplog.at_level(logging.INFO):
            caplog.clear()
            speech = tracker.get_where_am_i_speech()

    assert any("GPS position source used: mock" in r.message for r in caplog.records)
    assert "You are near RR Nagar, Bangalore" in speech
    tracker.stop()


# ---------------------------------------------------------------------------
# 4. Voice parsing: where am, where, etc. as where_am_i (whole command only)
# ---------------------------------------------------------------------------

def test_voice_parsing_where_am_i_exact_whole_commands():
    """
    After wake word (or standalone), treat:
    "where am", "where", "where I", "where am I", "whereabouts",
    "my location", "what is this place" as where_am_i.
    """
    parser = VoiceCommandParser()

    phrases = [
        "where am",
        "where",
        "where I",
        "where i",
        "where am I",
        "where am i",
        "whereabouts",
        "my location",
        "what is this place",
        "where is this",
        "current location",
    ]

    for phrase in phrases:
        # Without wake word
        res = parser.parse(phrase)
        intent = res['intent'] if isinstance(res, dict) else res[0]
        assert intent == 'where_am_i', f"Expected 'where_am_i' for '{phrase}', got '{intent}'"

        # With wake word
        wake_phrase = f"hey vision {phrase}"
        res_wake = parser.parse(wake_phrase)
        intent_wake = res_wake['intent'] if isinstance(res_wake, dict) else res_wake[0]
        assert intent_wake == 'where_am_i', f"Expected 'where_am_i' for '{wake_phrase}', got '{intent_wake}'"


def test_where_only_matches_whole_command_and_does_not_hijack_other_commands():
    """
    "where" does NOT trigger anything else, and commands like "where is my phone"
    must NOT trigger where_am_i.
    """
    parser = VoiceCommandParser()

    # Commands where 'where' is part of find/navigate:
    res_phone = parser.parse("hey vision where is my phone")
    assert res_phone['intent'] == 'find', f"Expected 'find', got {res_phone['intent']}"
    assert res_phone['parameters'].get('object_name') == 'phone'

    res_keys = parser.parse("where is my keys")
    assert res_keys['intent'] == 'find', f"Expected 'find', got {res_keys['intent']}"

    res_it = parser.parse("hey vision where is it")
    assert res_it['intent'] == 'find', f"Expected 'find', got {res_it['intent']}"

    # Whole command "where" must trigger where_am_i
    res_where = parser.parse("hey vision where")
    assert res_where['intent'] == 'where_am_i', f"Expected 'where_am_i', got {res_where['intent']}"


# ---------------------------------------------------------------------------
# 5. Polling thread never blocks
# ---------------------------------------------------------------------------

def test_polling_thread_never_blocks_main_thread():
    """
    The background polling thread runs asynchronously and never blocks the main loop,
    even when requests time out or take a long time.
    """
    # Simulate a slow/hanging response in requests.get for phone GPS endpoint
    def slow_requests_get(url, *args, **kwargs):
        if "api/location" in str(url) or "127.0.0.1" in str(url):
            time.sleep(1.0)
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "source": "phone",
                "latitude": 12.9265,
                "longitude": 77.5264,
                "accuracy": 10.0,
                "timestamp": time.time()
            }
            return resp
        # Return immediate failed response for other calls
        resp = MagicMock()
        resp.status_code = 500
        return resp

    with patch('requests.get', side_effect=slow_requests_get):
        tracker = GPSTracker(mock_mode=False, start_phone_polling=True)

        # Calling get_position() from main thread must be instant and never block
        t0 = time.time()
        for _ in range(10):
            _ = tracker.get_position()
        elapsed = time.time() - t0

        assert elapsed < 0.1, f"Main thread was blocked! Elapsed: {elapsed:.3f}s"

        # Background thread is running and alive
        assert tracker._phone_poll_thread is not None
        assert tracker._phone_poll_thread.is_alive()
        assert tracker._phone_poll_thread.daemon is True

        # Stop tracker cleanly
        t_stop0 = time.time()
        tracker.stop()
        stop_elapsed = time.time() - t_stop0
        assert stop_elapsed < 1.0, f"Stopping tracker took too long ({stop_elapsed:.3f}s)"


def test_polling_thread_updates_position_asynchronously():
    """
    Background polling thread receives updates from mobile_backend endpoint and
    updates tracker._phone_position asynchronously.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "source": "phone",
        "latitude": 13.0827,
        "longitude": 80.2707,
        "accuracy": 12.0,
        "timestamp": time.time()
    }

    with patch('requests.get', return_value=mock_resp):
        tracker = GPSTracker(mock_mode=False, start_phone_polling=False)
        # Execute one poll
        tracker._poll_phone_gps_once()

        assert tracker._get_fresh_phone_position() == (13.0827, 80.2707)
        assert tracker.get_position() == (13.0827, 80.2707)
        tracker.stop()
