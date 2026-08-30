"""
Threaded camera reader for VisionMate.

Cross-platform webcam reader that works on:
- Windows (CAP_DSHOW)
- Linux / Docker / Raspberry Pi (CAP_V4L2)
- macOS (CAP_ANY fallback)

Runs cv2.VideoCapture.read() in a background thread so AI inference
never blocks waiting for webcam frames.
"""

import cv2
import threading
import time
import logging
import platform

logger = logging.getLogger(__name__)


class CameraReader:
    """
    Background-threaded VideoCapture wrapper.

    Continuously captures frames in a daemon thread.
    Always returns the latest frame (no queue, no stale frames).
    """

    def __init__(
        self,
        cam_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backend: int | None = None,
        warmup_frames: int = 3,
    ):
        self._cam_index = cam_index
        self._width = width
        self._height = height
        self._fps = fps
        self._warmup_frames = warmup_frames

        # -------------------------------
        # Select backend automatically
        # -------------------------------
        if backend is None:
            system = platform.system()

            if system == "Windows":
                self._backend = cv2.CAP_DSHOW
            elif system == "Linux":
                self._backend = cv2.CAP_V4L2
            else:
                self._backend = cv2.CAP_ANY
        else:
            self._backend = backend

        self._cap = None
        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()

        self._running = False
        self._thread = None

        # Statistics
        self._last_read_ms = 0.0
        self._frames_captured = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """
        Open webcam using backend fallback.

        Returns
        -------
        CameraReader
        """

        # Try preferred backend first, then fallback.
        backends_to_try = [self._backend, cv2.CAP_ANY]

        for backend in backends_to_try:
            logger.info(f"Trying camera backend: {backend}")

            self._cap = cv2.VideoCapture(self._cam_index, backend)

            if self._cap.isOpened():
                logger.info(f"Camera opened successfully using backend {backend}")
                break

            self._cap.release()
            self._cap = None

        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(
                f"CameraReader: cannot open camera index {self._cam_index}"
            )

        # Camera properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        # Warmup frames
        for _ in range(self._warmup_frames):
            self._cap.read()

        # Start reader thread
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="CameraReaderThread",
            daemon=True,
        )
        self._thread.start()

        logger.info(
            "CameraReader started (index=%d, %dx%d @ %dfps)",
            self._cam_index,
            self._width,
            self._height,
            self._fps,
        )

        return self

    def stop(self):
        """Stop reader thread and release webcam."""

        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        logger.info(
            "CameraReader stopped. %d frames captured.",
            self._frames_captured,
        )

    def read(self):
        """
        Return latest captured frame.

        Returns
        -------
        frame, frame_id
        """

        with self._lock:
            return self._frame, self._frame_id

    @property
    def last_read_ms(self):
        return self._last_read_ms

    @property
    def frames_captured(self):
        return self._frames_captured

    def is_opened(self):
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Internal reader thread
    # ------------------------------------------------------------------

    def _reader_loop(self):
        """Continuously read frames."""

        while self._running:

            t0 = time.perf_counter()
            ret, frame = self._cap.read()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if not ret:
                logger.warning("CameraReader: cap.read() returned False")
                time.sleep(0.01)
                continue

            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._frames_captured += 1
                self._last_read_ms = elapsed_ms

    # ------------------------------------------------------------------
    # Context Manager Support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()