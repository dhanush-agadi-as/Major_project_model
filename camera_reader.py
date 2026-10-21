"""
Threaded camera reader for VisionMate.

Runs cv2.VideoCapture.read() in a background thread so the AI
inference pipeline never blocks waiting for the next webcam frame.
Frames are always the *latest* captured one — stale frames are
discarded, not queued.

Usage
-----
    from camera_reader import CameraReader

    cam = CameraReader(cam_index=0)
    cam.start()

    while True:
        frame = cam.read()
        if frame is None:
            continue            # first frame not yet available
        # … run inference on frame …

    cam.stop()
"""

import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)


class CameraReader:
    """
    Background-threaded VideoCapture wrapper.

    The reader thread continuously grabs frames from the webcam.
    The inference thread calls :meth:`read` to get the most-recent
    frame without blocking on I/O.

    Parameters
    ----------
    cam_index : int
        OpenCV camera index (0 = default webcam).
    width, height : int
        Requested capture resolution (may be ignored by the driver).
    fps : int
        Requested frame-rate hint sent to the driver.
    backend : int
        OpenCV backend flag (e.g. ``cv2.CAP_DSHOW`` on Windows).
    warmup_frames : int
        Number of frames to read silently after open before making
        them available to callers (discards first-few dark frames).
    """

    def __init__(
        self,
        cam_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backend: int = cv2.CAP_DSHOW,
        warmup_frames: int = 3,
    ):
        self._cam_index = cam_index
        self._width = width
        self._height = height
        self._fps = fps
        self._backend = backend
        self._warmup_frames = warmup_frames

        self._cap: cv2.VideoCapture | None = None
        self._frame = None          # latest captured frame (BGR ndarray)
        self._frame_id = 0          # increments each new frame
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # Timing stats (updated by reader thread)
        self._last_read_ms: float = 0.0
        self._frames_captured: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "CameraReader":
        """Open the camera and start the reader thread."""
        from camera_source import open_camera
        self._cap = open_camera(self._cam_index, width=self._width, height=self._height, fps=self._fps)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"CameraReader: cannot open camera source {self._cam_index}"
            )
        try:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
        logger.info("Camera negotiated: %dx%d @ %.1f fps",
                    int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    self._cap.get(cv2.CAP_PROP_FPS))

        # Warm up: discard a few frames that may be over-exposed / dark
        for _ in range(self._warmup_frames):
            self._cap.read()

        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop, name="CameraReaderThread", daemon=True
        )
        self._thread.start()
        logger.info("CameraReader started (index=%d, %dx%d @ %dfps)",
                    self._cam_index, self._width, self._height, self._fps)
        return self

    def stop(self):
        """Stop the reader thread and release the webcam."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        logger.info("CameraReader stopped. %d frames captured.", self._frames_captured)

    def read(self):
        """
        Return the most-recent frame, or ``None`` if none yet.

        This call **never blocks** — it returns whatever was last grabbed
        by the background thread.

        Returns
        -------
        frame : np.ndarray | None
            BGR image array, or ``None`` before the first frame arrives.
        frame_id : int
            Monotonically increasing counter; compare with previous
            call to detect whether a new frame is available.
        """
        with self._lock:
            return self._frame, self._frame_id

    @property
    def last_read_ms(self) -> float:
        """Average ms the reader thread spent waiting in cap.read()."""
        return self._last_read_ms

    @property
    def frames_captured(self) -> int:
        return self._frames_captured

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Internal reader loop
    # ------------------------------------------------------------------

    def _reader_loop(self):
        """Background thread: continuously grab frames and store latest."""
        while self._running:
            t0 = time.perf_counter()
            ret, frame = self._cap.read()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

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
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
