"""
Pi Distance Client
==================
Polls the Raspberry Pi ESP32 ultrasonic distance endpoint at 10 Hz:
- Endpoint: http://<PI_HOST>:5000/distance
- Daemon thread with 0.3s HTTP request timeout
- Discards readings outside 2 - 400 cm
- Smooths using median of the last 3 valid readings
- Returns None when stale (> 500 ms) or unavailable
- Clean shutdown
"""

import time
import threading
import collections
import statistics
import logging
from typing import Optional, Dict, Any
import requests

try:
    import config
except ImportError:
    config = None

logger = logging.getLogger(__name__)


class PIDistanceClient:
    """
    Client for polling ultrasonic distance readings from Raspberry Pi hub.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 5000,
        poll_rate_hz: float = 10.0,
        stale_timeout_ms: int = 500,
        request_timeout_s: float = 0.3,
    ):
        if host is None:
            if config is not None:
                host = getattr(config, "PI_HOST", "10.117.98.217")
            else:
                host = "10.117.98.217"

        if host.startswith("http://") or host.startswith("https://"):
            if "/distance" in host:
                self.url = host
            else:
                self.url = f"{host.rstrip('/')}/distance"
        else:
            self.url = f"http://{host}:{port}/distance"

        self.poll_interval = 1.0 / max(poll_rate_hz, 1.0)
        self.stale_timeout_s = stale_timeout_ms / 1000.0
        self.request_timeout_s = request_timeout_s

        self._recent_valid = collections.deque(maxlen=3)
        self._smoothed_distance_cm: Optional[float] = None
        self._raw_distance_cm: Optional[float] = None
        self._last_received_time: float = 0.0
        self._last_server_age_ms: Optional[int] = None
        self._server_ok: bool = False

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[requests.Session] = None

    def start(self) -> "PIDistanceClient":
        """Start background daemon polling thread."""
        self._stop_event.clear()
        self._session = requests.Session()
        self._thread = threading.Thread(
            target=self._poll_loop, name="PIDistanceClientThread", daemon=True
        )
        self._thread.start()
        logger.info("PIDistanceClient started polling %s at %.0f Hz", self.url, 1.0 / self.poll_interval)
        return self

    def _poll_loop(self):
        """Worker loop polling /distance endpoint at 10 Hz."""
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            self._poll_once()
            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, self.poll_interval - elapsed)
            if self._stop_event.wait(sleep_t):
                break

    def _poll_once(self):
        """Perform a single HTTP GET request and update state."""
        try:
            resp = self._session.get(self.url, timeout=self.request_timeout_s)
            if resp.status_code == 200:
                data = resp.json()
                self._process_reading(data)
            else:
                with self._lock:
                    self._server_ok = False
        except Exception as e:
            logger.debug("PIDistanceClient poll error: %s", e)
            with self._lock:
                self._server_ok = False

    def _process_reading(self, data: Dict[str, Any]):
        """
        Process payload from Pi:
        {"distance_cm": <float or null>, "age_ms": <int>, "ok": true}
        """
        raw_dist = data.get("ultrasonic_distance_cm")
        if raw_dist is None:
            raw_dist = data.get("distance_cm", data.get("distance"))
        if raw_dist is None:
            for sub in ("ultrasonic", "sensors"):
                if isinstance(data.get(sub), dict):
                    raw_dist = (
                        data[sub].get("ultrasonic_distance_cm")
                        or data[sub].get("distance_cm")
                        or data[sub].get("distance")
                    )
                    if raw_dist is not None:
                        break

        age_ms = data.get("age_ms", 999999)
        server_ok = bool(data.get("ok", False))
        now = time.monotonic()

        with self._lock:
            self._server_ok = server_ok
            self._last_server_age_ms = age_ms

            if raw_dist is None or age_ms > 1500 or not server_ok:
                # Sensor report is invalid, absent or stale on Pi
                return

            try:
                val = float(raw_dist)
            except (ValueError, TypeError):
                return

            # Range check: ignore readings outside 2 - 400 cm
            if val < 2.0 or val > 400.0:
                return

            # Valid reading: update smoothing window
            self._raw_distance_cm = val
            self._recent_valid.append(val)
            self._smoothed_distance_cm = float(statistics.median(self._recent_valid))
            self._last_received_time = now

    def get_distance(self) -> Optional[float]:
        """
        Return the latest smoothed ultrasonic distance in cm.
        Returns None if unavailable or if the reading is older than 500 ms.
        """
        now = time.monotonic()
        with self._lock:
            if self._smoothed_distance_cm is None:
                return None

            # Local staleness check (> 500 ms since last valid reading received)
            if (now - self._last_received_time) > self.stale_timeout_s:
                return None

            # Server staleness check (> 500 ms reported by Pi)
            if self._last_server_age_ms is not None and self._last_server_age_ms > 500:
                return None

            return self._smoothed_distance_cm

    def get_reading_details(self) -> Dict[str, Any]:
        """Return dict with full status for debugging/telemetry."""
        with self._lock:
            dist = self.get_distance()
            return {
                "distance_cm": dist,
                "raw_cm": self._raw_distance_cm,
                "server_age_ms": self._last_server_age_ms,
                "server_ok": self._server_ok,
                "buffer_len": len(self._recent_valid),
            }

    def stop(self):
        """Stop polling thread and close HTTP session cleanly."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
        logger.info("PIDistanceClient stopped")

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
