"""
Obstacle Alerter Module
=======================
Handles ultrasonic obstacle warnings, object naming, proximity beeps,
and camera lag fallback for VisionMate.

Requirements:
- Polls SENSORS_URL (~10 Hz, 0.5s timeout, reused session)
- URL is always config.SENSORS_URL (http://raspberrypi.local:5000/sensors); never changed on retry
- Range check: 2 - 400 cm; discard invalid / negative / out of bounds
- Detect frozen values (>= 3s identical reading): no warning, logs once
- Failure handling: no valid reading for 5s -> warning logged once, silent
- 3-read median filter; ignores single spikes
- 2 reads in a row <= 100 cm: warn; <= 40 cm: stop
- Object naming: checks latest ML detections (<500 ms old) for center-third object (|offset| < 0.33, conf >= 0.5)
- Rate limiting: at most once per 3s; repeat if >15 cm delta or 5s pass; reset when >120 cm
- Proximity beeps: 20-100 cm (interval 1.0s at 100cm down to 0.2s at 40cm); continuous < 20cm; muted during find sessions
- Camera fallback: frame age > 1s -> "Camera lagging, obstacle alert still on"; "Camera back" on recovery
- Logging: "ULTRA: raw=<> smooth=<> level=<clear|warn|stop> label=<name|none>"
"""

import time
import threading
import collections
import statistics
import logging
from typing import Optional, Callable, Dict, Any, List, Tuple
import requests

try:
    import config
except ImportError:
    config = None

logger = logging.getLogger(__name__)


def default_beep_tone(freq_hz: float = 1000, ms: int = 150):
    """Emit a system beep tone via winsound on Windows or silent fallback."""
    try:
        import winsound
        winsound.Beep(int(freq_hz), int(ms))
    except Exception:
        pass


def calculate_beep_interval(dist_cm: Optional[float]) -> Optional[float]:
    """
    Calculate proximity beep interval in seconds:
    - 100 cm -> 1.0 s
    - 70 cm  -> 0.6 s
    - 40 cm  -> 0.2 s
    - Below 40 cm to 20 cm -> continuous fast beeps (0.12 s)
    - Below 20 cm -> 0.0 s (solid tone)
    - Silent (None) when null, over 100 cm, or under 2 cm
    """
    if dist_cm is None:
        return None
    if dist_cm < 2.0 or dist_cm > 100.0:
        return None
    if dist_cm < 20.0:
        return 0.0
    if dist_cm <= 40.0:
        if dist_cm == 40.0:
            return 0.2
        return 0.12
    # Linear interpolation between (40cm, 0.2s) and (100cm, 1.0s)
    return round(0.2 + (dist_cm - 40.0) * (0.8 / 60.0), 4)


class ObstacleAlerter:
    """
    Ultrasonic obstacle warning and proximity beeping system.
    """

    def __init__(
        self,
        sensors_url: Optional[str] = None,
        poll_rate_hz: float = 10.0,
        request_timeout: float = 0.5,
        speak_fn: Optional[Callable[[str, int], None]] = None,
        beep_fn: Optional[Callable[[int, int], None]] = None,
        get_latest_detections_fn: Optional[Callable[[], Tuple[List[Dict[str, Any]], float]]] = None,
        get_frame_age_fn: Optional[Callable[[], float]] = None,
        is_find_active_fn: Optional[Callable[[], bool]] = None,
        beeps_enabled: Optional[bool] = None,
    ):
        pi_host = getattr(config, "PI_HOST", "raspberrypi.local") if config else "raspberrypi.local"
        default_url = getattr(config, "SENSORS_URL", f"http://{pi_host}:5000/sensors") if config else f"http://{pi_host}:5000/sensors"
        self.primary_url = sensors_url or default_url
        self.fallback_url = None
        self.current_url = self.primary_url

        self.poll_interval = 1.0 / max(poll_rate_hz, 1.0)
        self.request_timeout = request_timeout

        self.speak_fn = speak_fn
        self.beep_fn = beep_fn or default_beep_tone
        self.get_latest_detections_fn = get_latest_detections_fn
        self.get_frame_age_fn = get_frame_age_fn
        self.is_find_active_fn = is_find_active_fn

        if beeps_enabled is not None:
            self.beeps_enabled = beeps_enabled
        elif config is not None:
            self.beeps_enabled = getattr(config, "OBSTACLE_BEEPS", True)
        else:
            self.beeps_enabled = True

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._beep_thread: Optional[threading.Thread] = None
        self._session: Optional[requests.Session] = None

        # Data history & state
        self._recent_reads = collections.deque(maxlen=3)
        self._raw_distance: Optional[float] = None
        self._smooth_distance: Optional[float] = None
        self._current_level = "clear"

        # Startup raw JSON logging
        self._logged_startup_json = False

        # Frozen sensor detection
        self._last_raw_val: Optional[float] = None
        self._frozen_start_time: Optional[float] = None
        self._is_frozen = False
        self._is_frozen_logged = False

        # 5-second failure handling & 10-second unresponsive sensor alert
        self._last_valid_time = time.monotonic()
        self._no_data_logged = False
        self._distance_unresponsive_spoken = False

        # Warning rate limiting & hysteresis
        self._last_warn_time = 0.0
        self._last_warn_dist: Optional[float] = None
        self._last_warn_level = "clear"

        # Camera lag state
        self._camera_lag_reported = False

        # Logging throttling
        self._last_ultra_log_time = 0.0
        self._last_logged_level = "clear"

    def set_beeps_enabled(self, enabled: bool):
        """Toggle proximity beeps on or off."""
        with self._lock:
            self.beeps_enabled = enabled
        logger.info("Obstacle proximity beeps set to: %s", "ON" if enabled else "OFF")

    def start(self) -> "ObstacleAlerter":
        """Start background polling and proximity beep threads."""
        self._stop_event.clear()
        self._session = requests.Session()

        self._poll_thread = threading.Thread(
            target=self._poll_loop, name="ObstaclePollThread", daemon=True
        )
        self._poll_thread.start()

        self._beep_thread = threading.Thread(
            target=self._beep_loop, name="ObstacleBeepThread", daemon=True
        )
        self._beep_thread.start()

        sensor_status = "OK" if self._smooth_distance is not None else "NO DATA"
        logger.info("Obstacle beeps: %s, sensor: %s", "ON" if self.beeps_enabled else "OFF", sensor_status)
        logger.info("ObstacleAlerter started polling %s", self.current_url)
        return self

    def stop(self):
        """Clean shutdown of threads and session."""
        self._stop_event.set()
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=1.0)
            self._poll_thread = None
        if self._beep_thread and self._beep_thread.is_alive():
            self._beep_thread.join(timeout=1.0)
            self._beep_thread = None
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
        logger.info("ObstacleAlerter stopped")

    def _speak(self, text: str, priority: int = 1):
        """Invoke speak function if available."""
        if self.speak_fn:
            try:
                self.speak_fn(text, priority)
            except TypeError:
                try:
                    self.speak_fn(text)
                except Exception as e:
                    logger.debug("Error speaking obstacle warning: %s", e)
            except Exception as e:
                logger.debug("Error speaking obstacle warning: %s", e)

    def _poll_loop(self):
        """Continuous polling loop running at ~10 Hz without blocking."""
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            self._poll_once()
            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, self.poll_interval - elapsed)
            if self._stop_event.wait(sleep_t):
                break

    def _poll_once(self):
        """Fetch sensor readings via HTTP GET and process."""
        if not self._session:
            return

        resp_text = None
        data = None
        try:
            resp = self._session.get(self.current_url, timeout=self.request_timeout)
            if resp.status_code == 200:
                resp_text = resp.text
                data = resp.json()
            else:
                self._try_switch_to_fallback()
        except Exception:
            self._try_switch_to_fallback()

        self.process_sensor_payload(data, resp_text)

    def _try_switch_to_fallback(self):
        """Never change URL on retry/recovery."""
        pass

    def process_sensor_payload(self, data: Optional[Dict[str, Any]], raw_text: Optional[str] = None):
        """
        Process sensor JSON payload, update smoothing, evaluate alerts.
        Can be called directly for unit testing.
        """
        now = time.monotonic()

        # Log raw JSON once at startup
        if data is not None and not self._logged_startup_json:
            logger.info("Ultrasonic raw JSON: %s", raw_text or str(data))
            self._logged_startup_json = True

        raw_val = None
        if isinstance(data, dict):
            # Parse field: ultrasonic_distance_cm, fallback distance_cm / distance
            # Also check nested under "ultrasonic" or "sensors"
            candidates = [
                data.get("ultrasonic_distance_cm"),
                data.get("distance_cm"),
                data.get("distance"),
            ]
            for sub in ("ultrasonic", "sensors"):
                sub_d = data.get(sub)
                if isinstance(sub_d, dict):
                    candidates.extend([
                        sub_d.get("ultrasonic_distance_cm"),
                        sub_d.get("distance_cm"),
                        sub_d.get("distance"),
                    ])

            for c in candidates:
                if c is not None:
                    try:
                        val = float(c)
                        raw_val = val
                        break
                    except (ValueError, TypeError):
                        pass

            # Range validation: null, negative, < 2 cm or > 400 cm means no reading
            if raw_val is not None:
                if raw_val < 2.0 or raw_val > 400.0:
                    raw_val = None

            # Server status check
            if data.get("ok") is False or data.get("ultrasonic_distance_cm") is None and "ultrasonic_distance_cm" in data:
                if raw_val is None or data.get("ok") is False:
                    raw_val = None

        with self._lock:
            self._raw_distance = raw_val

            # Frozen sensor detection
            if raw_val is not None:
                if self._last_raw_val is not None and abs(raw_val - self._last_raw_val) < 1e-4:
                    if self._frozen_start_time is None:
                        self._frozen_start_time = now
                    elif (now - self._frozen_start_time) >= 3.0:
                        if not self._is_frozen_logged:
                            logger.warning("Ultrasonic: value frozen")
                            self._is_frozen_logged = True
                        self._is_frozen = True
                else:
                    self._last_raw_val = raw_val
                    self._frozen_start_time = now
                    self._is_frozen = False
                    self._is_frozen_logged = False
            else:
                self._last_raw_val = None
                self._frozen_start_time = None
                self._is_frozen = False

            # Update valid reading window if not frozen
            if raw_val is not None and not self._is_frozen:
                self._last_valid_time = now
                self._no_data_logged = False
                self._recent_reads.append(raw_val)

            # Failure handling: 5 seconds without valid data
            if (now - self._last_valid_time) >= 5.0:
                if not self._no_data_logged:
                    logger.warning("Ultrasonic: no data from /sensors")
                    self._no_data_logged = True

                # If distance stays null for 10 s, say once "Distance sensor not responding" and log it, then stay quiet.
                if (now - self._last_valid_time) >= 10.0:
                    if not self._distance_unresponsive_spoken:
                        logger.warning("Distance sensor not responding")
                        self._speak("Distance sensor not responding", priority=1)
                        self._distance_unresponsive_spoken = True

                self._recent_reads.clear()
                self._smooth_distance = None
                self._current_level = "clear"
                self._check_log_ultra("clear", None, None)
                return
            else:
                if raw_val is not None and not self._is_frozen:
                    self._distance_unresponsive_spoken = False

            if self._is_frozen:
                self._smooth_distance = None
                self._current_level = "clear"
                self._check_log_ultra("clear", None, raw_val)
                return

            if not self._recent_reads:
                self._smooth_distance = None
                self._current_level = "clear"
                self._check_log_ultra("clear", None, raw_val)
                return

            # Median of the last 3 reads, ignoring single spikes
            smooth_dist = float(statistics.median(self._recent_reads))
            self._smooth_distance = smooth_dist

            # Level evaluation based on consecutive reads
            # <= 100 cm for 2 reads in a row: warn level. <= 40 cm: stop level.
            recent_list = list(self._recent_reads)
            level = "clear"
            if len(recent_list) >= 2:
                last_two = recent_list[-2:]
                if all(r <= 40.0 for r in last_two) or (smooth_dist <= 40.0 and last_two[-1] <= 40.0):
                    level = "stop"
                elif all(r <= 100.0 for r in last_two) or (smooth_dist <= 100.0 and last_two[-1] <= 100.0):
                    level = "warn"

            self._current_level = level

            # Reset hysteresis: when distance is above 120 cm, reset warn state so next object warns immediately
            if smooth_dist > 120.0 or (raw_val is not None and raw_val > 120.0):
                self._last_warn_time = 0.0
                self._last_warn_dist = None
                self._last_warn_level = "clear"

            # Check ML detection for object naming in center third
            obj_name = self._find_center_object()

            # Dispatch spoken warnings
            self._evaluate_spoken_warning(level, smooth_dist, obj_name, now)

            # Check camera lag fallback
            self._evaluate_camera_fallback(now)

            # Log ULTRA status
            self._check_log_ultra(level, obj_name, raw_val)

    def _find_center_object(self) -> Optional[str]:
        """
        Check if latest ML detections (< 500 ms old) include an object
        covering the center third of the frame (|offset| < 0.33) with confidence >= 0.5.
        """
        if not self.get_latest_detections_fn:
            return None

        try:
            detections, det_time = self.get_latest_detections_fn()
        except Exception:
            return None

        now = time.monotonic()
        if (now - det_time) > 0.5 or not detections:
            return None

        frame_w = getattr(config, "CAMERA_WIDTH", 640) if config else 640

        best_label = None
        best_conf = 0.0

        for obj in detections:
            conf = obj.get("confidence", 0.0)
            if conf < 0.5:
                continue

            bbox = obj.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            # Normalized center offset from frame center: (|offset| < 0.33)
            cx = (bbox[0] + bbox[2]) / 2.0
            offset = (cx - (frame_w / 2.0)) / float(frame_w)

            # Center third condition
            if abs(offset) < 0.33:
                if conf > best_conf:
                    best_conf = conf
                    label = obj.get("class") or obj.get("label") or obj.get("name") or obj.get("class_name")
                    if label:
                        best_label = str(label).strip().capitalize()

        return best_label

    def _evaluate_spoken_warning(self, level: str, smooth_dist: float, obj_name: Optional[str], now: float):
        """
        Trigger spoken TTS warnings with throttling and object naming:
        - Warn at most once per 3 s.
        - Repeat same level only if distance changes by > 15 cm or 5 s pass.
        - Silent when clear.
        """
        if level == "clear":
            return

        dt = now - self._last_warn_time
        dist_change = abs(smooth_dist - self._last_warn_dist) if self._last_warn_dist is not None else 999.0
        level_changed = (level != self._last_warn_level)

        should_speak = False
        if level_changed:
            if dt >= 3.0 or self._last_warn_level == "clear":
                should_speak = True
        else:
            if dt >= 3.0 and (dist_change > 15.0 or dt >= 5.0):
                should_speak = True

        if should_speak:
            p_emergency = getattr(config, "PRIORITY_EMERGENCY", 1) if config else 1
            p_navigational = getattr(config, "PRIORITY_NAVIGATIONAL", 4) if config else 4

            if level == "stop":
                phrase = f"Stop, {obj_name} very close" if obj_name else "Stop, object very close"
                self._speak(phrase, p_emergency)
            elif level == "warn":
                phrase = f"{obj_name} ahead, close" if obj_name else "Object ahead"
                self._speak(phrase, p_navigational)

            self._last_warn_time = now
            self._last_warn_dist = smooth_dist
            self._last_warn_level = level

    def _evaluate_camera_fallback(self, now: float):
        """
        Camera fallback: if newest frame is older than 1 s or stream lost,
        say once "Camera lagging, obstacle alert still on".
        Say "Camera back" once when recovered.
        """
        if not self.get_frame_age_fn:
            return

        try:
            frame_age_s = self.get_frame_age_fn()
        except Exception:
            frame_age_s = 999.0

        p_status = getattr(config, "PRIORITY_STATUS", 5) if config else 5

        if frame_age_s > 1.0:
            if not self._camera_lag_reported:
                self._speak("Camera lagging, obstacle alert still on", p_status)
                self._camera_lag_reported = True
        else:
            if self._camera_lag_reported:
                self._speak("Camera back", p_status)
                self._camera_lag_reported = False

    def _check_log_ultra(self, level: str, obj_name: Optional[str], raw_val: Optional[float]):
        """
        Log: 'ULTRA: raw=<> smooth=<> level=<clear|warn|stop> label=<name|none>'
        at most once per second and on every level change.
        """
        now = time.monotonic()
        level_changed = (level != self._last_logged_level)
        time_elapsed = (now - self._last_ultra_log_time) >= 1.0

        if level_changed or time_elapsed:
            raw_str = f"{raw_val:.1f}" if raw_val is not None else "None"
            smooth_str = f"{self._smooth_distance:.1f}" if self._smooth_distance is not None else "None"
            label_str = obj_name.lower() if obj_name else "none"

            logger.info("ULTRA: raw=%s smooth=%s level=%s label=%s", raw_str, smooth_str, level, label_str)
            self._last_ultra_log_time = now
            self._last_logged_level = level

    def _beep_loop(self):
        """
        Proximity beeps worker:
        - 20-100 cm: interval scales 1.0s (at 100cm) to 0.2s (at 40cm), 0.12s for 20-40cm
        - < 20 cm: solid tone
        - Muted during find sessions (find guidance beeps take priority)
        - Silent when clear, null, or out of range (<2cm or >100cm)
        """
        while not self._stop_event.is_set():
            with self._lock:
                beeps_on = self.beeps_enabled
                dist = self._smooth_distance
                is_frozen = self._is_frozen

            is_finding = False
            if self.is_find_active_fn:
                try:
                    is_finding = self.is_find_active_fn()
                except Exception:
                    is_finding = False

            if not beeps_on or is_finding or dist is None or is_frozen:
                time.sleep(0.05)
                continue

            interval = calculate_beep_interval(dist)
            if interval is None:
                time.sleep(0.05)
                continue

            if interval == 0.0:
                # Below 20 cm: solid tone
                try:
                    self.beep_fn(1000, 150)
                except Exception:
                    pass
                time.sleep(0.01)
            else:
                try:
                    self.beep_fn(1000, 150)
                except Exception:
                    pass
                if self._stop_event.wait(interval):
                    break

    def play_test_pattern(self, duration_s: float = 6.0):
        """
        Play the full beep pattern from 100 cm down to 20 cm in ~6 s without the sensor.
        Used for voice command 'test beep'.
        """
        steps = [
            (100.0, 1.0),
            (70.0, 0.6),
            (50.0, 0.33),
            (40.0, 0.2),
            (30.0, 0.12),
            (25.0, 0.12),
            (20.0, 0.12),
            (15.0, 0.0),
        ]
        start_t = time.monotonic()
        for dist, _ in steps:
            if self._stop_event.is_set():
                break
            interval = calculate_beep_interval(dist)
            if interval == 0.0:
                tone_end = time.monotonic() + 1.2
                while time.monotonic() < tone_end and not self._stop_event.is_set():
                    try:
                        self.beep_fn(1000, 150)
                    except Exception:
                        pass
                    time.sleep(0.01)
            else:
                try:
                    self.beep_fn(1000, 150)
                except Exception:
                    pass
                time.sleep(interval if interval is not None else 0.1)
            if (time.monotonic() - start_t) >= duration_s:
                break

    def get_distance(self) -> Optional[float]:
        """Return current smoothed ultrasonic distance in cm."""
        with self._lock:
            return self._smooth_distance

    def get_raw_distance(self) -> Optional[float]:
        """Return raw ultrasonic distance in cm."""
        with self._lock:
            return self._raw_distance

    def get_level(self) -> str:
        """Return current warning level ('clear', 'warn', 'stop')."""
        with self._lock:
            return self._current_level
