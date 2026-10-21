"""
Unit and integration tests for Pi hardware and camera stream architecture:
1. open_camera helper for 0 (webcam) and for URL (stream)
2. CameraStreamReader returns only the latest frame
3. Reconnection behavior after failed reads or stream stalls
4. Connection timeout error output and diagnostic checklist
5. PIDistanceClient staleness check (None when > 500 ms)
6. PIDistanceClient out-of-range rejection (2 - 400 cm)
7. PIDistanceClient 3-reading median smoothing
8. Clean thread and resource shutdown
9. Verification that no literal VideoCapture(0) exists outside open_camera()
"""

import os
import sys
import time
import re
import subprocess
from unittest.mock import MagicMock
import pytest
import numpy as np

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import camera_source
from camera_source import open_camera, parse_camera_source, CameraStreamReader
from pi_distance_client import PIDistanceClient


# ─── 1. Tests for open_camera (0 vs URL) ──────────────────────────────────────

def test_parse_camera_source():
    """Verify parsing of digit strings to int and URLs to str."""
    assert parse_camera_source(0) == 0
    assert parse_camera_source("0") == 0
    assert parse_camera_source("1") == 1
    assert parse_camera_source("http://10.117.98.217:5000/video") == "http://10.117.98.217:5000/video"
    assert parse_camera_source("rtsp://example.com/stream") == "rtsp://example.com/stream"


def test_open_camera_for_int_webcam(monkeypatch):
    """Verify open_camera(0) configures buffer size 1 and sets width/height on webcams."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    captured_args = []
    def fake_video_capture(*args, **kwargs):
        captured_args.append(args)
        return mock_cap

    monkeypatch.setattr(camera_source.cv2, "VideoCapture", fake_video_capture)

    cap = open_camera(0, width=640, height=480, fps=30)
    assert cap is mock_cap

    # Check that VideoCapture was called with 0
    assert len(captured_args) == 1
    assert captured_args[0][0] == 0
    if sys.platform == "win32":
        assert captured_args[0][1] == camera_source.cv2.CAP_DSHOW

    # Check buffer size set to 1
    mock_cap.set.assert_any_call(camera_source.cv2.CAP_PROP_BUFFERSIZE, 1)
    mock_cap.set.assert_any_call(camera_source.cv2.CAP_PROP_FRAME_WIDTH, 640)
    mock_cap.set.assert_any_call(camera_source.cv2.CAP_PROP_FRAME_HEIGHT, 480)
    mock_cap.set.assert_any_call(camera_source.cv2.CAP_PROP_FPS, 30)


def test_open_camera_for_url_stream(monkeypatch):
    """Verify open_camera(url) uses CAP_FFMPEG, buffer size 1, and never sets width/height."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    captured_args = []
    def fake_video_capture(*args, **kwargs):
        captured_args.append(args)
        return mock_cap

    monkeypatch.setattr(camera_source.cv2, "VideoCapture", fake_video_capture)

    url = "http://10.117.98.217:5000/video"
    cap = open_camera(url, width=640, height=480, fps=30)
    assert cap is mock_cap

    # Check VideoCapture called with URL and CAP_FFMPEG
    assert len(captured_args) == 1
    assert captured_args[0][0] == url
    assert captured_args[0][1] == camera_source.cv2.CAP_FFMPEG

    # Buffer size 1 set
    mock_cap.set.assert_any_call(camera_source.cv2.CAP_PROP_BUFFERSIZE, 1)

    # Must NOT set width or height on network stream
    set_prop_ids = [call_args[0][0] for call_args in mock_cap.set.call_args_list]
    assert camera_source.cv2.CAP_PROP_FRAME_WIDTH not in set_prop_ids
    assert camera_source.cv2.CAP_PROP_FRAME_HEIGHT not in set_prop_ids


# ─── 2. Tests for CameraStreamReader (Latest frame, Reconnect, Timeout) ──────

def test_frame_reader_returns_latest_frame(monkeypatch):
    """Verify CameraStreamReader drops intermediate frames and yields only the latest frame."""
    frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 10
    frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 50
    frame3 = np.ones((480, 640, 3), dtype=np.uint8) * 100

    frames_queue = [frame1, frame2, frame3]

    class FakeCap:
        def __init__(self):
            self.opened = True
        def isOpened(self):
            return self.opened
        def read(self):
            if frames_queue:
                return True, frames_queue.pop(0)
            return True, frame3
        def set(self, *args):
            return True
        def release(self):
            self.opened = False

    monkeypatch.setattr(camera_source, "open_camera", lambda *args, **kwargs: FakeCap())

    reader = CameraStreamReader(source="http://10.117.98.217:5000/video", connect_timeout=1.0)
    reader.start()

    try:
        # Give reader thread time to consume all frames in queue
        time.sleep(0.1)
        latest_frame, frame_id = reader.read()
        assert latest_frame is not None
        # Latest frame must be frame3 (mean pixel value 100)
        assert np.array_equal(latest_frame, frame3)
        assert frame_id >= 3
    finally:
        reader.stop()


def test_frame_reader_reconnects_after_failed_reads(monkeypatch):
    """Verify reader detects stream loss (failed read or stall) and re-opens stream."""
    reconnected = [0]
    frame_ok = np.ones((480, 640, 3), dtype=np.uint8)

    class ReconnectCap:
        def __init__(self, reads_before_fail=1):
            self.reads_left = reads_before_fail
            self.opened = True
        def isOpened(self):
            return self.opened
        def read(self):
            if self.reads_left > 0:
                self.reads_left -= 1
                return True, frame_ok
            return False, None
        def set(self, *args):
            return True
        def release(self):
            self.opened = False

    caps = [ReconnectCap(reads_before_fail=1), ReconnectCap(reads_before_fail=999)]

    def mock_open_cam(*args, **kwargs):
        reconnected[0] += 1
        if caps:
            return caps.pop(0)
        return ReconnectCap(reads_before_fail=999)

    monkeypatch.setattr(camera_source, "open_camera", mock_open_cam)

    reader = CameraStreamReader(source="http://10.117.98.217:5000/video", connect_timeout=1.0)
    reader.start()

    try:
        # Give reader time to hit read failure and reconnect (retries every 2s)
        time.sleep(2.5)
        # Should have attempted reconnect and opened new cap
        assert reconnected[0] >= 2
    finally:
        reader.stop()


def test_frame_reader_connect_timeout_exits(monkeypatch):
    """Verify reader prints diagnostic checklist and exits when connection fails within 10s."""
    mock_bad_cap = MagicMock()
    mock_bad_cap.isOpened.return_value = False
    monkeypatch.setattr(camera_source, "open_camera", lambda *args, **kwargs: mock_bad_cap)

    reader = CameraStreamReader(source="http://10.117.98.217:5000/video", connect_timeout=0.2, exit_on_fail=False)
    with pytest.raises(RuntimeError) as exc_info:
        reader.start()

    assert "could not open camera stream" in str(exc_info.value)


# ─── 3. Tests for PIDistanceClient (Staleness, Range, Median) ─────────────────

def test_distance_client_returns_none_when_stale():
    """Verify get_distance() returns None when data is older than 500 ms."""
    client = PIDistanceClient(host="10.117.98.217", stale_timeout_ms=500)

    # Simulate fresh reading
    client._process_reading({"distance_cm": 45.0, "age_ms": 50, "ok": True})
    assert client.get_distance() == 45.0

    # Simulate server reporting age > 500 ms
    client._process_reading({"distance_cm": 45.0, "age_ms": 550, "ok": True})
    # Server age > 500ms -> ignored / stale
    # Next, simulate local age > 500 ms
    client._last_received_time = time.monotonic() - 0.6
    client._last_server_age_ms = 100
    assert client.get_distance() is None


def test_distance_client_ignores_out_of_range():
    """Verify distance client ignores readings < 2.0 cm or > 400.0 cm."""
    client = PIDistanceClient(host="10.117.98.217")

    # Valid reading at 50 cm
    client._process_reading({"distance_cm": 50.0, "age_ms": 20, "ok": True})
    assert client.get_distance() == 50.0

    # Under minimum (1.5 cm < 2 cm)
    client._process_reading({"distance_cm": 1.5, "age_ms": 20, "ok": True})
    # Remains 50.0 (out of range ignored)
    assert client.get_distance() == 50.0

    # Over maximum (450 cm > 400 cm)
    client._process_reading({"distance_cm": 450.0, "age_ms": 20, "ok": True})
    # Remains 50.0 (out of range ignored)
    assert client.get_distance() == 50.0


def test_distance_client_median_smoothing():
    """Verify median smoothing of the last 3 valid readings."""
    client = PIDistanceClient(host="10.117.98.217")

    # Send 3 readings: 50.0, 70.0, 60.0
    client._process_reading({"distance_cm": 50.0, "age_ms": 10, "ok": True})
    assert client.get_distance() == 50.0

    client._process_reading({"distance_cm": 70.0, "age_ms": 10, "ok": True})
    assert client.get_distance() == 60.0  # median(50, 70) = 60.0

    client._process_reading({"distance_cm": 60.0, "age_ms": 10, "ok": True})
    assert client.get_distance() == 60.0  # median(50, 70, 60) = 60.0

    # Next reading pushes out the first (window of 3: [70, 60, 20])
    client._process_reading({"distance_cm": 20.0, "age_ms": 10, "ok": True})
    assert client.get_distance() == 60.0  # median(70, 60, 20) = 60.0

    # Next reading: [60, 20, 25] -> median is 25.0
    client._process_reading({"distance_cm": 25.0, "age_ms": 10, "ok": True})
    assert client.get_distance() == 25.0


def test_distance_client_stops_cleanly():
    """Verify stop() shuts down thread and cleans up session."""
    client = PIDistanceClient(host="10.117.98.217")
    client.start()
    th = client._thread
    assert th is not None
    assert th.is_alive()

    client.stop()
    assert client._stop_event.is_set()
    assert client._thread is None
    assert not th.is_alive()


# ─── 4. Verification that no camera index 0 VideoCapture exists in project ───

def test_no_videocapture_0_left_in_project():
    """
    Ensure no direct camera-index-0 calls remain anywhere in project code
    outside camera_source.py.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_ident = "VideoCapture" + r"\s*\(\s*0\s*\)"
    pattern = re.compile(target_ident)

    offending_files = []

    for root, dirs, files in os.walk(repo_root):
        # Skip virtualenvs, git, cache
        if any(ignored in root for ignored in [".git", "__pycache__", "venv", ".venv", ".pytest_cache"]):
            continue

        for f in files:
            if not f.endswith(".py"):
                continue

            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, repo_root).replace("\\", "/")

            # camera_source.py is the single sanctioned location
            if rel_path in ("camera_source.py", "test/test_pi_hardware_and_camera.py"):
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
                    if pattern.search(content):
                        offending_files.append(rel_path)
            except Exception:
                pass

    assert offending_files == [], f"Found VideoCapture with 0 in files: {offending_files}"


def test_laptop_server_constructor_and_defaults():
    """Verify LaptopMLServer.__init__ accepts host and port and uses config defaults."""
    from wireless.laptop_server import LaptopMLServer
    import config

    # Default constructor uses config defaults
    s1 = LaptopMLServer()
    assert s1.host == getattr(config, "SERVER_HOST", "0.0.0.0")
    assert s1.port == getattr(config, "SERVER_PORT", 8081)

    # Keyword host and port
    s2 = LaptopMLServer(host="127.0.0.1", port=9099)
    assert s2.host == "127.0.0.1"
    assert s2.port == 9099

    # Keyword port only (backward compatibility with existing tests)
    s3 = LaptopMLServer(port=8082)
    assert s3.host == getattr(config, "SERVER_HOST", "0.0.0.0")
    assert s3.port == 8082


def test_dry_run_laptop_server_and_rpi_client():
    """Run both wireless/rpi_client.py and wireless/laptop_server.py with --dry-run and fail on exception."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. RPi client dry-run
    rpi_res = subprocess.run(
        [sys.executable, os.path.join(repo_root, "wireless", "rpi_client.py"), "--dry-run"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30
    )
    assert rpi_res.returncode == 0, f"rpi_client --dry-run failed with code {rpi_res.returncode}:\n{rpi_res.stderr}\n{rpi_res.stdout}"
    assert "DRY RUN OK" in rpi_res.stdout, f"'DRY RUN OK' not found in stdout:\n{rpi_res.stdout}"

    # 2. Laptop server dry-run
    server_res = subprocess.run(
        [sys.executable, os.path.join(repo_root, "wireless", "laptop_server.py"), "--dry-run"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=90
    )
    assert server_res.returncode == 0, f"laptop_server --dry-run failed with code {server_res.returncode}:\n{server_res.stderr}\n{server_res.stdout}"
    assert "DRY RUN OK" in server_res.stdout, f"'DRY RUN OK' not found in stdout:\n{server_res.stdout}"

