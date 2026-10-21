"""
Camera Source Module
====================
Shared camera helper and background stream reader for VisionMate:
- Local webcam: integer index -> cv2.VideoCapture(source, cv2.CAP_DSHOW / default backend)
- Network stream: string URL -> cv2.VideoCapture(url, cv2.CAP_FFMPEG)
- Sets cv2.CAP_PROP_BUFFERSIZE = 1 (keeps only the latest frame in buffer)
- Does NOT set width/height on network streams
- CameraStreamReader: background daemon thread reading frames with 3s timeout
  auto-reconnection every 2s, 10s initial connection timeout with diagnostic steps,
  and clean shutdown.
"""

import sys
import os
import time
import threading
import logging
from typing import Optional, Tuple, Union
import numpy as np
import cv2

try:
    import config
except ImportError:
    config = None

logger = logging.getLogger(__name__)


def parse_camera_source(source: Optional[Union[int, str]] = None) -> Union[int, str]:
    """
    Resolve camera source parameter:
    - If None, uses config.CAMERA_SOURCE or environment variable VISIONMATE_CAMERA (fallback 0).
    - If string of digits (e.g. "0"), converts to int.
    - Otherwise returns the string (URL).
    """
    if source is None:
        env_cam = os.environ.get("VISIONMATE_CAMERA")
        if env_cam is not None:
            source = env_cam.strip()
        elif config is not None:
            source = getattr(config, "CAMERA_SOURCE", 0)
        else:
            source = 0

    if isinstance(source, str):
        s = source.strip()
        if s.isdigit():
            return int(s)
        return s

    return source


def open_camera(
    source: Optional[Union[int, str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
) -> cv2.VideoCapture:
    """
    Open camera source with consistent backend and buffer configuration:
    - int -> local webcam: cv2.VideoCapture(source, cv2.CAP_DSHOW) on Windows or default backend
    - str -> network stream: cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    - Sets CAP_PROP_BUFFERSIZE = 1.
    - Never sets width/height on network streams.

    Args:
        source: Optional camera index or stream URL.
        width: Optional requested width (local webcams only).
        height: Optional requested height (local webcams only).
        fps: Optional frame rate hint (local webcams only).

    Returns:
        cv2.VideoCapture instance.
    """
    resolved_source = parse_camera_source(source)

    if isinstance(resolved_source, int):
        # Local webcam: use CAP_DSHOW on Windows for fast startup
        if sys.platform == "win32":
            cap = cv2.VideoCapture(resolved_source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(resolved_source)

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            cap.set(cv2.CAP_PROP_FPS, fps)

    else:
        # Network stream (URL): use CAP_FFMPEG
        cap = cv2.VideoCapture(resolved_source, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # Do NOT set width/height on network streams

    return cap


class CameraStreamReader:
    """
    Background daemon thread reader for low-latency camera stream:
    - Keeps only the LATEST frame in memory.
    - If read() fails or nothing new arrives for 3 s, releases and reopens every 2 s.
    - Logs "Camera stream lost, reconnecting..." / "Camera stream connected (<w>x<h>)".
    - If stream cannot open within 10 s on initial connect, prints diagnostic checklist and exits.
    - Stops cleanly on shutdown.
    """

    def __init__(
        self,
        source: Optional[Union[int, str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        connect_timeout: float = 10.0,
        exit_on_fail: bool = True,
    ):
        self.source = parse_camera_source(source)
        self.width = width
        self.height = height
        self.fps = fps
        self.connect_timeout = connect_timeout
        self.exit_on_fail = exit_on_fail

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_time: float = 0.0
        self._frame_id: int = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.frame_width: int = 0
        self.frame_height: int = 0
        self.connected: bool = False
        self._frames_captured: int = 0

    def start(self) -> "CameraStreamReader":
        """Connect to stream (with 10 s timeout) and start daemon reader thread."""
        ok = self._initial_connect(timeout=self.connect_timeout)
        if not ok:
            if self.exit_on_fail:
                pi_host = getattr(config, "PI_HOST", "10.117.98.217") if config else "10.117.98.217"
                print("\n" + "=" * 70, file=sys.stderr)
                print(f"ERROR: Cannot connect to camera stream at: {self.source}", file=sys.stderr)
                print("Could not open stream within 10 seconds. Please check:", file=sys.stderr)
                print("  1. The Raspberry Pi is powered on and connected to the same network.", file=sys.stderr)
                print(f"  2. 'python3 pi_hub.py' is running on the Pi ({pi_host}:5000).", file=sys.stderr)
                print(f"  3. The stream URL opens in your browser: {self.source}", file=sys.stderr)
                print("=" * 70 + "\n", file=sys.stderr)
                sys.exit(1)
            else:
                raise RuntimeError(f"CameraStreamReader: could not open camera stream {self.source} within {self.connect_timeout}s")

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop, name="CameraStreamReaderThread", daemon=True
        )
        self._thread.start()
        return self

    def _initial_connect(self, timeout: float) -> bool:
        """Attempt to open camera and fetch first frame within timeout."""
        start_t = time.monotonic()
        while (time.monotonic() - start_t) < timeout:
            cap = open_camera(self.source, width=self.width, height=self.height, fps=self.fps)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    self._cap = cap
                    h, w = frame.shape[:2]
                    self.frame_width = w
                    self.frame_height = h
                    with self._lock:
                        self._latest_frame = frame
                        self._latest_frame_time = time.monotonic()
                        self._frame_id += 1
                        self._frames_captured += 1
                        self.connected = True
                    logger.info("Camera stream connected (%dx%d)", w, h)
                    return True
                cap.release()
            time.sleep(0.5)
        return False

    def _reader_loop(self):
        """Daemon thread: grabs frames continuously, maintaining only the latest frame."""
        while not self._stop_event.is_set():
            now = time.monotonic()

            if self._cap is None or not self._cap.isOpened():
                self._handle_reconnect()
                continue

            if hasattr(self._cap, 'grab'):
                grabbed = self._cap.grab()
                if not grabbed:
                    read_failed = True
                else:
                    ret, frame = self._cap.retrieve()
                    read_failed = not (ret and frame is not None and frame.size > 0)
            else:
                ret, frame = self._cap.read()
                read_failed = not (ret and frame is not None and frame.size > 0)
            if not read_failed:
                with self._lock:
                    self._latest_frame = frame
                    self._latest_frame_time = now
                    self._frame_id += 1
                    self._frames_captured += 1
                    self.connected = True

            # If read() fails or nothing new arrives for 3 s -> reconnect
            if read_failed or (now - self._latest_frame_time) >= 3.0:
                logger.info("Camera stream lost, reconnecting...")
                self.connected = False
                if self._cap is not None:
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None
                self._handle_reconnect()

    def _handle_reconnect(self):
        """Release and retry opening stream every 2 seconds until connected or stopped."""
        while not self._stop_event.is_set():
            if self._stop_event.wait(2.0):
                break
            cap = open_camera(self.source, width=self.width, height=self.height, fps=self.fps)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    self._cap = cap
                    h, w = frame.shape[:2]
                    self.frame_width = w
                    self.frame_height = h
                    now = time.monotonic()
                    with self._lock:
                        self._latest_frame = frame
                        self._latest_frame_time = now
                        self._frame_id += 1
                        self._frames_captured += 1
                        self.connected = True
                    logger.info("Camera stream connected (%dx%d)", w, h)
                    break
                try:
                    cap.release()
                except Exception:
                    pass

    def read(self) -> Tuple[Optional[np.ndarray], int]:
        """Return the latest frame and monotonically increasing frame_id."""
        with self._lock:
            return self._latest_frame, self._frame_id

    def get_frame(self) -> Optional[np.ndarray]:
        """Return only the latest frame."""
        with self._lock:
            return self._latest_frame

    def is_opened(self) -> bool:
        """Check if reader is currently connected and receiving frames."""
        return self.connected and (time.monotonic() - self._latest_frame_time) < 3.0

    def stop(self):
        """Stop reader daemon thread and release camera resources cleanly."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self.connected = False
        logger.info("CameraStreamReader stopped. %d frames captured.", self._frames_captured)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
