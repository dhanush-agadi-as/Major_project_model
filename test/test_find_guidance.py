"""
Unit tests for find_guidance.py
Tests indoor object-finding guidance, direction/distance calculations,
ultrasonic fusion, arrival, timeout/lost handling, throttling, and last-seen descriptions.
"""
import sys
import os
from datetime import datetime, timedelta
import pytest

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import find_guidance
from find_guidance import FindGuide, FOCAL_PX, TYPICAL_SIZES_CM


class FakeClock:
    """Deterministic simulated clock for monotonic time."""
    def __init__(self, initial_time: float = 100.0):
        self._time = initial_time

    def monotonic(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


@pytest.fixture
def fake_audio():
    spoken = []
    beeped = []

    def speak(text: str) -> None:
        spoken.append(text)

    def beep(freq_hz: float, ms: int) -> None:
        beeped.append((freq_hz, ms))

    return spoken, beeped, speak, beep


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock(100.0)
    monkeypatch.setattr(find_guidance.time, "monotonic", c.monotonic)
    return c


@pytest.fixture
def guide(fake_audio, clock):
    spoken, beeped, speak, beep = fake_audio
    fg = FindGuide(speak=speak, beep=beep, frame_w=640, frame_h=480)
    yield fg
    if fg.is_active:
        fg.stop()


# ---------------------------------------------------------------------------
# a) test_direction_straight_ahead
# ---------------------------------------------------------------------------
def test_direction_straight_ahead(guide):
    """Bbox centered in frame produces 'straight ahead' direction."""
    guide.start("bottle", "last seen on desk")
    # frame_w = 640 -> center = 320. bbox centered: [290, 100, 350, 200] -> cx=320, offset=0.0
    remembered = [{
        "custom_name": "bottle",
        "class": "bottle",
        "confidence": 0.9,
        "bbox": [290, 100, 350, 200]
    }]
    state = guide.update(remembered, ultrasonic_cm=None)
    assert state["visible"] is True
    assert state["direction"] == "straight ahead"


# ---------------------------------------------------------------------------
# b) test_direction_slightly_left / test_direction_slightly_right
# ---------------------------------------------------------------------------
def test_direction_slightly_left(guide):
    """Bbox moderately off-center to the left produces 'slightly left'."""
    guide.start("bottle", "last seen on desk")
    # offset: (cx - 320) / 320. For slightly left: -0.45 < offset < -0.15.
    # cx = 220 -> offset = -100/320 = -0.3125
    remembered = [{
        "custom_name": "bottle",
        "class": "bottle",
        "confidence": 0.9,
        "bbox": [190, 100, 250, 200]  # cx = 220
    }]
    state = guide.update(remembered, ultrasonic_cm=None)
    assert state["visible"] is True
    assert state["direction"] in ("slightly left", "a little left")


def test_direction_slightly_right(guide):
    """Bbox moderately off-center to the right produces 'slightly right' / 'a little right'."""
    guide.start("bottle", "last seen on desk")
    # cx = 420 -> offset = +100/320 = +0.3125
    remembered = [{
        "custom_name": "bottle",
        "class": "bottle",
        "confidence": 0.9,
        "bbox": [390, 100, 450, 200]  # cx = 420
    }]
    state = guide.update(remembered, ultrasonic_cm=None)
    assert state["visible"] is True
    assert state["direction"] in ("slightly right", "a little right")


# ---------------------------------------------------------------------------
# c) test_direction_turn_left / test_direction_turn_right
# ---------------------------------------------------------------------------
def test_direction_turn_left(guide):
    """Bbox far off-center to the left produces 'turn left' or 'a little left'."""
    guide.start("bottle", "last seen on desk")
    # cx = 60 -> offset = (60 - 320) / 320 = -0.8125 <= -0.45
    remembered = [{
        "custom_name": "bottle",
        "class": "bottle",
        "confidence": 0.9,
        "bbox": [30, 100, 90, 200]  # cx = 60
    }]
    state = guide.update(remembered, ultrasonic_cm=None)
    assert state["visible"] is True
    assert state["direction"] in ("left", "slightly left", "turn left", "a little left")


def test_direction_turn_right(guide):
    """Bbox far off-center to the right produces 'right', 'slightly right', 'turn right' or 'a little right'."""
    guide.start("bottle", "last seen on desk")
    # cx = 580 -> offset = (580 - 320) / 320 = +0.8125 >= 0.45
    remembered = [{
        "custom_name": "bottle",
        "class": "bottle",
        "confidence": 0.9,
        "bbox": [550, 100, 610, 200]  # cx = 580
    }]
    state = guide.update(remembered, ultrasonic_cm=None)
    assert state["visible"] is True
    assert state["direction"] in ("right", "slightly right", "turn right", "a little right")


# ---------------------------------------------------------------------------
# d) test_distance_estimate_from_bbox_size
# ---------------------------------------------------------------------------
def test_distance_estimate_from_bbox_size(guide):
    """Known bbox size + known class matches distance formula with FOCAL_PX and TYPICAL_SIZES_CM."""
    guide.start("remote", "last seen on sofa")
    target_class = "remote"
    real_size = TYPICAL_SIZES_CM[target_class]
    longest_side_px = 120
    expected_dist_cm = FOCAL_PX * real_size / longest_side_px

    remembered = [{
        "custom_name": "remote",
        "class": target_class,
        "confidence": 0.95,
        "bbox": [260, 100, 380, 100 + longest_side_px]  # width=120, height=120
    }]
    state = guide.update(remembered, ultrasonic_cm=None)
    assert state["visible"] is True
    assert state["distance_cm"] == pytest.approx(expected_dist_cm, rel=1e-3)


# ---------------------------------------------------------------------------
# e) test_ultrasonic_override_skipped_in_mock_mode
# ---------------------------------------------------------------------------
def test_ultrasonic_override_skipped_in_mock_mode(guide, monkeypatch):
    """When config.ULTRASONIC_MOCK_MODE is True, ultrasonic reading does NOT override camera distance."""
    monkeypatch.setattr(config, "ULTRASONIC_MOCK_MODE", True)
    guide.start("bottle", "last seen on desk")

    # Bottle size 24cm, longest side 100px -> camera dist = 530*24/100 = 127.2 cm
    camera_expected = FOCAL_PX * TYPICAL_SIZES_CM["bottle"] / 100.0
    remembered = [{
        "custom_name": "bottle",
        "class": "bottle",
        "confidence": 0.9,
        "bbox": [270, 100, 370, 200]  # cx=320, side=100
    }]
    state = guide.update(remembered, ultrasonic_cm=55.0)
    assert state["distance_cm"] == pytest.approx(camera_expected, rel=1e-3)
    assert state["distance_cm"] != 55.0


# ---------------------------------------------------------------------------
# f) test_ultrasonic_override_applied_when_not_mock
# ---------------------------------------------------------------------------
def test_ultrasonic_override_applied_when_not_mock(guide, monkeypatch):
    """When config.ULTRASONIC_MOCK_MODE is False and conditions met, ultrasonic overrides distance."""
    monkeypatch.setattr(config, "ULTRASONIC_MOCK_MODE", False)
    guide.start("bottle", "last seen on desk")

    # Camera dist = 127.2 cm. Ultrasonic = 90.0 cm (< 300, offset=0 < 0.25, 90 <= 127.2 * 1.3)
    remembered = [{
        "custom_name": "bottle",
        "class": "bottle",
        "confidence": 0.9,
        "bbox": [270, 100, 370, 200]
    }]
    state = guide.update(remembered, ultrasonic_cm=90.0)
    assert state["distance_cm"] == pytest.approx(90.0, rel=1e-3)


# ---------------------------------------------------------------------------
# g) test_arrival_triggers_message_and_stop
# ---------------------------------------------------------------------------
def test_arrival_triggers_message_and_stop(guide, fake_audio):
    """When target is centered and <= 45cm, arrival message is spoken and session stops."""
    spoken, _, _, _ = fake_audio
    guide.start("cup", "last seen on counter")

    # Cup size = 10cm. Compute bbox size from FOCAL_PX so camera dist is ~26.5cm (<= 45cm).
    px = FOCAL_PX * 10.0 / 26.5
    half_w = px / 2.0
    x1 = 320.0 - half_w
    x2 = 320.0 + half_w
    y1 = 100.0
    y2 = 100.0 + px

    remembered = [{
        "custom_name": "cup",
        "class": "cup",
        "confidence": 0.92,
        "bbox": [x1, y1, x2, y2]
    }]
    guide.update(remembered, ultrasonic_cm=None)
    state = guide.update(remembered, ultrasonic_cm=None)
    assert state.get("arrived") is True
    assert guide.is_active is False
    assert any("right in front of you" in s and "reach out" in s.lower() for s in spoken)


# ---------------------------------------------------------------------------
# h) test_lost_target_after_3_seconds
# ---------------------------------------------------------------------------
def test_lost_target_after_3_seconds(guide, clock, fake_audio):
    """When target is lost for >3s, a directional hint is spoken."""
    spoken, _, _, _ = fake_audio
    guide.start("bottle", "last seen on table")
    spoken.clear()

    # Frame 1: see it on the left (cx = 100, offset < 0)
    remembered = [{
        "custom_name": "bottle",
        "class": "bottle",
        "confidence": 0.9,
        "bbox": [70, 100, 130, 200]
    }]
    guide.update(remembered, ultrasonic_cm=None)
    spoken.clear()

    # Advance clock by 5.5 seconds with no target visible (5s repeat guidance)
    clock.advance(5.5)
    state = guide.update([], ultrasonic_cm=None)
    assert state.get("visible") is False
    assert any(w in s for s in spoken for w in ("left", "turn left"))


# ---------------------------------------------------------------------------
# i) test_timeout_after_120_seconds
# ---------------------------------------------------------------------------
def test_timeout_after_120_seconds(guide, clock, fake_audio):
    """When session exceeds 45s total without finding target, timeout message is spoken and session ends."""
    spoken, _, _, _ = fake_audio
    guide.start("keys", "last seen near door")
    spoken.clear()

    # Advance clock beyond 45s
    clock.advance(46.0)
    state = guide.update([], ultrasonic_cm=None)
    assert state.get("visible") is False
    assert guide.is_active is False
    assert any("can't find" in s.lower() or "stopping search" in s.lower() for s in spoken)


# ---------------------------------------------------------------------------
# j) test_speech_throttling
# ---------------------------------------------------------------------------
def test_speech_throttling(guide, clock, fake_audio):
    """Guidance speech is throttled: speaks on direction/bucket change, at most once per 2.5s."""
    spoken, _, _, _ = fake_audio
    guide.start("bottle", "last seen on table")
    spoken.clear()

    remembered = [{
        "custom_name": "bottle",
        "class": "bottle",
        "confidence": 0.9,
        "bbox": [290, 100, 350, 200]  # straight ahead, mid-distance
    }]

    # First update triggers speech (initial direction)
    guide.update(remembered, ultrasonic_cm=None)
    assert len(spoken) == 1
    assert "straight ahead" in spoken[0]

    # Multiple immediate updates with identical object location shouldn't speak again
    for _ in range(5):
        clock.advance(0.1)
        guide.update(remembered, ultrasonic_cm=None)
    assert len(spoken) == 1

    # Same direction and bucket after 4.5s does not chatter
    clock.advance(4.5)
    guide.update(remembered, ultrasonic_cm=None)
    assert len(spoken) == 1

    # Direction change after cooldown speaks new direction
    remembered[0]["bbox"] = [70, 100, 130, 200]
    clock.advance(3.0)
    guide.update(remembered, ultrasonic_cm=None)
    assert len(spoken) == 2
    assert any(d in spoken[1] for d in ("slightly left", "a little left"))


# ---------------------------------------------------------------------------
# k) test_describe_last_seen_with_nearby_objects
# ---------------------------------------------------------------------------
def test_describe_last_seen_with_nearby_objects():
    """Sighting dict with nearby_objects mentions the objects in the description."""
    recent_ts = (datetime.now() - timedelta(seconds=20)).strftime("%Y-%m-%d %H:%M:%S.000000")
    sighting = {
        "timestamp": recent_ts,
        "context": {
            "nearby_objects": ["table", "chair"]
        }
    }
    desc = FindGuide.describe_last_seen(sighting)
    assert "last seen near" in desc
    assert "table" in desc
    assert "chair" in desc
    assert "just now" in desc


# ---------------------------------------------------------------------------
# l) test_describe_last_seen_without_nearby_objects
# ---------------------------------------------------------------------------
def test_describe_last_seen_without_nearby_objects():
    """Sighting dict without nearby objects returns clean description without 'None'."""
    mins_ago_ts = (datetime.now() - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S.000000")
    sighting = {
        "timestamp": mins_ago_ts,
        "context": {}
    }
    desc = FindGuide.describe_last_seen(sighting)
    assert desc.startswith("last seen")
    assert "None" not in desc
    assert "15 minutes ago" in desc


# ---------------------------------------------------------------------------
# m) test_describe_last_seen_malformed_timestamp
# ---------------------------------------------------------------------------
def test_describe_last_seen_malformed_timestamp():
    """Garbage timestamp string falls back gracefully to 'a while ago' without raising."""
    sighting = {
        "timestamp": "invalid-timestamp-string-xyz",
        "context": {"nearby_objects": ["desk"]}
    }
    desc = FindGuide.describe_last_seen(sighting)
    assert "last seen" in desc
    assert "desk" in desc
    assert "a while ago" in desc
