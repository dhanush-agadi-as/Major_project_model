"""
Find Guidance
=============
Indoor real-time object-finding guidance for a blind user using only the
camera feed already processed by laptop_server.py.

How it works
------------
1. The caller invokes ``FindGuide.start(name, last_seen_text)`` when the user
   issues a "find <object>" voice command.
2. Every frame, ``FindGuide.update(remembered, ultrasonic_cm)`` is called from
   ``process_ml``.  It searches ``remembered`` for the target, computes a
   left/right direction and coarse distance, throttles speech to avoid
   chattering, and plays a proximity beep whose rate and pitch scale with
   estimated distance.
3. When the object is within arm's reach (<=45 cm, centred in frame) or the
   caller invokes ``FindGuide.stop()``, the session ends cleanly.

Integration snippet for laptop_server.py::

    from find_guidance import FindGuide, default_beep

    # In LaptopMLServer.__init__:
    self.find_guide = FindGuide(
        speak=self.audio.speak_navigational,
        beep=default_beep,
        frame_w=config.FRAME_WIDTH,
        frame_h=config.FRAME_HEIGHT,
    )

    # In handle_find() (voice command handler):
    last_seen_desc = self.find_guide.describe_last_seen(location)
    self.find_guide.start(object_name, last_seen_desc)

    # In process_ml(), after building remembered list:
    self.find_guide.update(remembered, ultrasonic_cm)
"""

import logging
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# TODO: recalibrate for actual camera in use, see calibrate_focal.py
FOCAL_PX: float = 1100.0

# Approximate real-world longest dimension (cm) for common YOLO classes.
# Used for the pinhole camera distance estimate:  d = f * size / pixels
TYPICAL_SIZES_CM: Dict[str, float] = {
    "cell phone": 15,
    "remote":     18,
    "mouse":      11,
    "bottle":     24,
    "cup":        10,
    "book":       22,
    "laptop":     33,
    "backpack":   45,
    "handbag":    30,
    "scissors":   18,
    "clock":      25,
}
_DEFAULT_SIZE_CM: float = 20.0

# Beep frequency range (Hz) and interval range (seconds)
_BEEP_FREQ_FAR:   float = 400.0
_BEEP_FREQ_CLOSE: float = 1200.0
_BEEP_IVRL_FAR:   float = 1.2   # seconds between beeps when >= 300 cm away
_BEEP_IVRL_CLOSE: float = 0.12  # seconds between beeps when <= 50 cm away
_BEEP_DURATION_MS: int  = 80

# Distance (cm) thresholds
_DIST_FAR  = 300.0
_DIST_NEAR = 50.0
_DIST_BUCKET_FAR   = 150.0  # > 150 cm -> "about 2 meters"
_DIST_BUCKET_MID   = 70.0   # > 70 cm  -> "about 1 meter"
# <= 70 cm -> "within arm's reach"
_DIST_ARRIVAL = 45.0        # trigger arrival announcement

# Speech throttle
_SPEECH_COOLDOWN_S = 4.0

# Lost-target timeouts
_LOST_SPEAK_AFTER_S  = 3.0
_LOST_REPEAT_AFTER_S = 30.0
_TIMEOUT_TOTAL_S     = 120.0


# ---------------------------------------------------------------------------
# Module-level default beep helper
# ---------------------------------------------------------------------------

def default_beep(freq_hz: float, ms: int) -> None:
    """
    Emit a synchronous system beep using ``winsound`` (Windows built-in).

    This is the recommended callable to pass as the ``beep`` argument to
    ``FindGuide`` on Windows.  On non-Windows platforms the call is silently
    ignored.  The function never raises.
    """
    try:
        import winsound  # noqa: PLC0415 – platform-conditional import
        winsound.Beep(int(freq_hz), int(ms))
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Helper: linear interpolation
# ---------------------------------------------------------------------------

def _lerp(val: float, lo_in: float, hi_in: float,
          lo_out: float, hi_out: float) -> float:
    """Linearly map *val* from [lo_in, hi_in] to [lo_out, hi_out], clamped."""
    t = max(0.0, min(1.0, (val - lo_in) / (hi_in - lo_in + 1e-9)))
    return lo_out + t * (hi_out - lo_out)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FindGuide:
    """
    Camera-only indoor object-finding guidance for a blind user.

    The caller injects ``speak`` and ``beep`` callables so this module stays
    decoupled from the audio subsystem and can be unit-tested with stubs.

    Thread safety
    -------------
    ``start()`` and ``stop()`` may be called from a voice-handling thread
    while ``update()`` is called from the frame-processing thread.  All shared
    state is protected by ``self._lock``.
    """

    def __init__(
        self,
        speak: Callable[[str], None],
        beep:  Callable[[float, int], None],
        frame_w: int = 640,
        frame_h: int = 480,
    ) -> None:
        """
        Initialise the guidance system.

        Args:
            speak:   Callable that converts text to speech, e.g.
                     ``audio.speak_navigational``.
            beep:    Callable ``(freq_hz, duration_ms)`` that emits a tone,
                     e.g. ``default_beep``.
            frame_w: Width of the camera frame in pixels.
            frame_h: Height of the camera frame in pixels.
        """
        self.speak   = speak
        self.beep    = beep
        self.frame_w = frame_w
        self.frame_h = frame_h

        self.is_active: bool = False
        self._lock = threading.Lock()

        # Target identity
        self._target_name: str = ""
        self._last_seen_text: str = ""

        # Timing / speech state (all reset in start())
        self._start_time: float         = 0.0
        self._last_seen_time: float     = 0.0
        self._last_speech_time: float   = 0.0
        self._last_spoken_direction: Optional[str] = None
        self._last_spoken_distance_bucket: Optional[str] = None

        # Lost-target episode flags (reset in start())
        self._lost_3s_spoken: bool  = False
        self._lost_30s_spoken: bool = False

        # Last known horizontal offset sign for directional hints when lost
        self._last_offset_sign: int = 0  # -1, 0, +1

        # Shared distance for the beep thread (written under lock in update())
        self._current_distance_cm: Optional[float] = None

        # Beep thread handles
        self._stop_event:   threading.Event  = threading.Event()
        self._beep_thread:  Optional[threading.Thread] = None

        logger.debug("FindGuide initialised (frame %dx%d)", frame_w, frame_h)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, name: str, last_seen_text: str) -> None:
        """
        Begin a find session for *name*.

        Args:
            name:           Object name to search for (matched
                            case-insensitively against ``custom_name`` and
                            ``class`` fields in *remembered*).
            last_seen_text: Human-friendly description of where the object was
                            last seen; spoken immediately via ``self.speak``.
        """
        with self._lock:
            self._target_name    = name.strip().lower()
            self._last_seen_text = last_seen_text
            self.is_active       = True

            now = time.monotonic()
            self._start_time     = now
            self._last_seen_time = now
            self._last_speech_time = 0.0
            self._last_spoken_direction      = None
            self._last_spoken_distance_bucket = None

            self._lost_3s_spoken  = False
            self._lost_30s_spoken = False
            self._last_offset_sign = 0
            self._current_distance_cm = None

            # Restart beep thread
            self._stop_event.clear()

        # Speak outside the lock so TTS never holds the lock
        self.speak(last_seen_text)

        self._beep_thread = threading.Thread(
            target=self._beep_worker, daemon=True, name="FindGuide-beep"
        )
        self._beep_thread.start()
        logger.info("FindGuide: started search for '%s'", name)

    def stop(self, reason: str = "") -> None:
        """
        End the find session.

        Args:
            reason: If non-empty, spoken aloud before stopping.
        """
        with self._lock:
            self.is_active = False
            self._stop_event.set()

        if self._beep_thread is not None:
            self._beep_thread.join(timeout=2.0)
            self._beep_thread = None

        if reason:
            self.speak(reason)

        logger.info("FindGuide: stopped. reason=%r", reason)

    def update(
        self,
        remembered: List[Dict[str, Any]],
        ultrasonic_cm: Optional[float],
    ) -> Dict[str, Any]:
        """
        Process one camera frame.

        Called once per frame from ``laptop_server.process_ml``.  Thread-safe
        against concurrent ``start()``/``stop()`` calls.

        Args:
            remembered:    List of currently-visible object dicts.  Each dict
                           must contain at minimum: ``custom_name`` (str),
                           ``class`` (str), ``confidence`` (float),
                           ``bbox`` (tuple ``(x1, y1, x2, y2)``).
            ultrasonic_cm: Reading from the ultrasonic sensor in centimetres,
                           or ``None`` if unavailable.

        Returns:
            A state dict:
            ``{'visible': False}`` when the target is not in the current frame,
            ``{'visible': True, 'arrived': True}`` on arrival, or
            ``{'visible': True, 'direction': str, 'distance_cm': float,
               'distance_bucket': str}`` otherwise.
            Returns ``{}`` immediately if the session is not active.
        """
        with self._lock:
            if not self.is_active:
                return {}
            target = self._target_name

        # ── 1. Search for target in remembered ────────────────────────────
        best: Optional[Dict[str, Any]] = None
        best_conf: float = -1.0
        for obj in remembered:
            cn    = (obj.get("custom_name") or "").strip().lower()
            cls   = (obj.get("class")       or "").strip().lower()
            conf  = float(obj.get("confidence", 0.0))
            if (cn == target or cls == target) and conf > best_conf:
                best      = obj
                best_conf = conf

        if best is None:
            return self._handle_lost()

        # ── 2. Target found — extract geometry ────────────────────────────
        x1, y1, x2, y2 = best["bbox"]
        cx = (x1 + x2) / 2.0
        # cy kept for future use
        offset = (cx - self.frame_w / 2.0) / (self.frame_w / 2.0 + 1e-9)

        now = time.monotonic()
        with self._lock:
            self._last_seen_time  = now
            self._last_offset_sign = int(round(offset / max(abs(offset), 1e-9)))
            # Reset lost-episode flags so they fire again if target disappears
            self._lost_3s_spoken  = False
            self._lost_30s_spoken = False

        # ── 3. Direction ──────────────────────────────────────────────────
        abs_off = abs(offset)
        if abs_off < 0.15:
            direction = "straight ahead"
        elif abs_off < 0.45:
            direction = "slightly left" if offset < 0 else "slightly right"
        else:
            direction = "turn left" if offset < 0 else "turn right"

        # ── 4. Distance estimate (pinhole model) ──────────────────────────
        cls_name       = (best.get("class") or "").strip().lower()
        real_size_cm   = TYPICAL_SIZES_CM.get(cls_name, _DEFAULT_SIZE_CM)
        bbox_px        = max(x2 - x1, y2 - y1)
        distance_cm    = FOCAL_PX * real_size_cm / max(bbox_px, 1)

        # ── 5. Ultrasonic cross-check ─────────────────────────────────────
        if (
            ultrasonic_cm is not None
            and ultrasonic_cm < 300
            and abs_off < 0.25
            and not config.ULTRASONIC_MOCK_MODE
            and ultrasonic_cm <= distance_cm * 1.3
        ):
            distance_cm = float(ultrasonic_cm)

        # ── 6. Distance bucket ────────────────────────────────────────────
        if distance_cm > _DIST_BUCKET_FAR:
            bucket = "about 2 meters"
        elif distance_cm > _DIST_BUCKET_MID:
            bucket = "about 1 meter"
        else:
            bucket = "within arm's reach"

        # ── 7. Arrival check (before speech so arrival message wins) ──────
        if distance_cm <= _DIST_ARRIVAL and abs_off < 0.4:
            with self._lock:
                target_name = self._target_name
            self.speak(
                f"Your {target_name} is right in front of you. Reach out."
            )
            self.stop()
            return {"visible": True, "arrived": True}

        # ── 8. Update shared beep state ───────────────────────────────────
        with self._lock:
            self._current_distance_cm = distance_cm

        # ── 9. Speech throttling ──────────────────────────────────────────
        with self._lock:
            last_dir    = self._last_spoken_direction
            last_bucket = self._last_spoken_distance_bucket
            last_t      = self._last_speech_time

        should_speak = (
            direction != last_dir
            or bucket != last_bucket
            or (now - last_t) > _SPEECH_COOLDOWN_S
        )
        if should_speak:
            msg = f"{direction}, {bucket}"
            self.speak(msg)
            with self._lock:
                self._last_spoken_direction       = direction
                self._last_spoken_distance_bucket = bucket
                self._last_speech_time            = now

        return {
            "visible":       True,
            "direction":     direction,
            "distance_cm":   distance_cm,
            "distance_bucket": bucket,
        }

    # ------------------------------------------------------------------
    # Static / instance utility
    # ------------------------------------------------------------------

    @staticmethod
    def describe_last_seen(sighting_dict: Dict[str, Any]) -> str:
        """
        Build a human-friendly "last seen …" string from an object-memory
        sighting record.

        Args:
            sighting_dict: Dict with keys ``'context'`` (may contain
                           ``'nearby_objects': list[str]``) and
                           ``'timestamp'`` (sqlite datetime string,
                           e.g. ``"2026-10-02 15:04:24.414437"``).

        Returns:
            A string such as:
            ``"last seen near the table and chair, 2 hours ago"``
            ``"last seen 5 minutes ago"``
            ``"last seen a while ago"``
        """
        # ── Relative time ─────────────────────────────────────────────────
        raw_ts: str = sighting_dict.get("timestamp", "")
        try:
            # sqlite stores timestamps as "YYYY-MM-DD HH:MM:SS.ffffff"
            dt = datetime.strptime(raw_ts[:26], "%Y-%m-%d %H:%M:%S.%f")
            age_s = (datetime.now() - dt).total_seconds()
            if age_s < 60:
                rel_time = "just now"
            elif age_s < 3600:
                mins = int(age_s // 60)
                rel_time = f"{mins} minute{'s' if mins != 1 else ''} ago"
            elif age_s < 86400:
                hours = int(age_s // 3600)
                rel_time = f"{hours} hour{'s' if hours != 1 else ''} ago"
            else:
                rel_time = raw_ts or "a while ago"
        except Exception:  # noqa: BLE001
            rel_time = "a while ago"

        # ── Nearby objects ────────────────────────────────────────────────
        context = sighting_dict.get("context") or {}
        nearby: List[str] = context.get("nearby_objects", []) or []
        nearby = [o.strip() for o in nearby if o.strip()]

        if nearby:
            if len(nearby) == 2:
                nearby_phrase = f"the {nearby[0]} and {nearby[1]}"
            else:
                nearby_phrase = ", ".join(f"the {o}" for o in nearby)
            return f"last seen near {nearby_phrase}, {rel_time}"
        return f"last seen {rel_time}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_lost(self) -> Dict[str, Any]:
        """
        Called from ``update()`` when the target is not visible this frame.

        Manages the three-tier lost-target escalation:
          * 3 s  → directional hint ("turn slowly …")
          * 30 s → repeat the original ``last_seen_text``
          * 120 s → give up and stop

        Returns:
            ``{'visible': False}``
        """
        now = time.monotonic()

        with self._lock:
            last_seen   = self._last_seen_time
            start       = self._start_time
            spoken_3s   = self._lost_3s_spoken
            spoken_30s  = self._lost_30s_spoken
            offset_sign = self._last_offset_sign
            lst_text    = self._last_seen_text

        elapsed_lost  = now - last_seen
        elapsed_total = now - start

        # 120 s total timeout — give up
        if elapsed_total > _TIMEOUT_TOTAL_S:
            self.stop("Stopping search")
            return {"visible": False}

        # 30 s without seeing it — repeat last-known location hint
        if elapsed_lost > _LOST_REPEAT_AFTER_S and not spoken_30s:
            with self._lock:
                self._lost_30s_spoken = True
            self.speak(lst_text)

        # 3 s without seeing it — directional suggestion
        elif elapsed_lost > _LOST_SPEAK_AFTER_S and not spoken_3s:
            if offset_sign < 0:
                hint = "try turning left"
            elif offset_sign > 0:
                hint = "try turning right"
            else:
                hint = "turn slowly"
            with self._lock:
                self._lost_3s_spoken = True
            self.speak(f"I can't see it, {hint}")

        return {"visible": False}

    def _beep_worker(self) -> None:
        """
        Daemon thread: emits proximity beeps until ``_stop_event`` is set.

        Frequency and interval scale with ``_current_distance_cm``:
        * Far (>= 300 cm): slow (1.2 s interval), low pitch (400 Hz)
        * Close (<= 50 cm): fast (0.12 s interval), high pitch (1200 Hz)
        Linear interpolation between those extremes.
        """
        logger.debug("FindGuide beep thread started")
        while not self._stop_event.is_set():
            with self._lock:
                dist = self._current_distance_cm

            if dist is None:
                # No distance yet — slow quiet default
                freq_hz    = _BEEP_FREQ_FAR
                interval_s = _BEEP_IVRL_FAR
            else:
                # Clamp distance into the interpolation range
                d = max(_DIST_NEAR, min(_DIST_FAR, dist))
                freq_hz    = _lerp(d, _DIST_NEAR, _DIST_FAR,
                                   _BEEP_FREQ_CLOSE, _BEEP_FREQ_FAR)
                interval_s = _lerp(d, _DIST_NEAR, _DIST_FAR,
                                   _BEEP_IVRL_CLOSE, _BEEP_IVRL_FAR)

            try:
                self.beep(freq_hz, _BEEP_DURATION_MS)
            except Exception:  # noqa: BLE001
                pass

            # Wait for interval_s but wake immediately if stop is requested
            self._stop_event.wait(timeout=interval_s)

        logger.debug("FindGuide beep thread stopped")
