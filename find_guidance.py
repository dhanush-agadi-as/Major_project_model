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

import re
import logging
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any

import config

logger = logging.getLogger(__name__)


def _clean_obj_name(name: str) -> str:
    """Strip leading 'my', 'the', 'a', 'an' and return lower-cased stripped object name."""
    if not name:
        return ""
    return re.sub(r'^(?:(?:my|the|a|an)\s+)+', '', name.strip(), flags=re.IGNORECASE).strip().lower()

def _compute_iou(box1: Any, box2: Any) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    try:
        x1_a, y1_a, x2_a, y2_a = [float(v) for v in box1]
        x1_b, y1_b, x2_b, y2_b = [float(v) for v in box2]

        x1_i = max(x1_a, x1_b)
        y1_i = max(y1_a, y1_b)
        x2_i = min(x2_a, x2_b)
        y2_i = min(y2_a, y2_b)

        inter_w = max(0.0, x2_i - x1_i)
        inter_h = max(0.0, y2_i - y1_i)
        inter_area = inter_w * inter_h

        area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
        area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)
        union_area = area_a + area_b - inter_area
        if union_area <= 0.0:
            return 0.0
        return float(inter_area / union_area)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Pi camera (OV5647, ~53.5 degrees horizontal view) differs from the webcam.
# FOCAL_PX = 636 for 640 px wide frames.
# estimate from OV5647 spec; re-calibrate with tools/calibrate_focal.py
FOCAL_PX: float = getattr(config, 'FOCAL_PX', 636.0) if config else 636.0

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

DEFAULT_FIND_BEEP_PROFILE: Dict[str, Dict[str, Any]] = {
    "far":     {"min_dist": 200.0, "duration_ms": 60,  "interval_s": 1.2,  "freq_hz": 800},
    "mid":     {"min_dist": 100.0, "duration_ms": 90,  "interval_s": 0.7,  "freq_hz": 1000},
    "near":    {"min_dist": 60.0,  "duration_ms": 140, "interval_s": 0.4,  "freq_hz": 1200},
    "reach":   {"min_dist": 0.0,   "duration_ms": 220, "interval_s": 0.25, "freq_hz": 1400},
    "arrival": {"duration_ms": 600, "freq_hz": 1400},
}

# Distance (cm) thresholds
_DIST_BUCKET_FAR  = 200.0
_DIST_BUCKET_MID  = 100.0
_DIST_BUCKET_NEAR = 60.0
_DIST_ARRIVAL     = 55.0   # trigger arrival announcement

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
        is_speaking_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Initialise the guidance system.

        Args:
            speak:   Callable that converts text to speech, e.g.
                     ``audio.speak_prompt``.
            beep:    Callable ``(freq_hz, duration_ms)`` that emits a tone,
                     e.g. ``default_beep``.
            frame_w: Width of the camera frame in pixels.
            frame_h: Height of the camera frame in pixels.
            is_speaking_fn: Optional callable returning True if TTS speech is currently active.
        """
        self.speak   = speak
        self.beep    = beep
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._is_speaking_fn = is_speaking_fn
        self._beeps_paused = False
        self._is_paused = False

        self.is_active: bool = False
        self._lock = threading.Lock()


        # Target identity
        self._target_name: str = ""
        self._last_seen_text: str = ""

        # Timing / speech state (all reset in start())
        self._start_time: float         = 0.0
        self._last_seen_time: float     = 0.0
        self._last_detected_time: float = 0.0
        self._last_known_detection: Optional[Dict[str, Any]] = None
        self._last_speech_time: float   = 0.0
        self._last_direction_speech_time: float = 0.0
        self._last_lost_speech_time: float = 0.0
        self._last_spoken_direction: Optional[str] = None
        self._last_spoken_distance_bucket: Optional[str] = None

        # Lost-target episode flags (reset in start())
        self._lost_3s_spoken: bool  = False
        self._lost_30s_spoken: bool = False

        # Last known horizontal offset sign for directional hints when lost
        self._last_offset_sign: int = 0  # -1, 0, +1

        # Shared distance and visibility for the beep thread (written under lock in update())
        self._current_distance_cm: Optional[float] = None
        self._is_target_visible: bool = False

        # Beep thread handles
        self._stop_event:    threading.Event = threading.Event()
        self._visible_event: threading.Event = threading.Event()
        self._beep_thread:   Optional[threading.Thread] = None

        # Diagnostic log throttling (1 line per second)
        self._last_find_log_time: float = 0.0

        # Arrival hold & check
        self._arrival_hold_frames: int = 0
        self._last_arrival_check_log_time: float = 0.0

        # Reached then removed tracking
        self._reach_total_duration: float = 0.0
        self._last_frame_time: Optional[float] = None
        self._reached_then_removed_ended: bool = False

        # Lost search limits & phrase cycling
        self._lost_phrase_idx: int = 0

        logger.debug("FindGuide initialised (frame %dx%d)", frame_w, frame_h)

    def pause_beeps(self) -> None:
        """Pause proximity beeping while TTS speech is speaking."""
        self._beeps_paused = True

    def resume_beeps(self) -> None:
        """Resume proximity beeping after TTS speech finishes."""
        self._beeps_paused = False

    def pause(self) -> None:
        """Pause find guidance session (e.g. during emergency check-in)."""
        with self._lock:
            self._is_paused = True
            self.pause_beeps()
        logger.info("FindGuide: session paused")

    def resume(self) -> None:
        """Resume find guidance session after pause."""
        with self._lock:
            self._is_paused = False
            self.resume_beeps()
            now = time.monotonic()
            self._last_speech_time = now
            self._last_detected_time = now
        logger.info("FindGuide: session resumed")

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return getattr(self, '_is_paused', False)

    def set_frame_size(self, width: int, height: int) -> None:
        """Update frame dimensions dynamically from real camera stream."""
        self.frame_w = width
        self.frame_h = height


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_target(self, name: str, yolo_class: str = "") -> bool:
        """Check if an object name or YOLO class matches the current find target."""
        if not self.is_active or not self._target_name:
            return False
        clean_target = _clean_obj_name(self._target_name)
        clean_name = _clean_obj_name(name)
        clean_cls = _clean_obj_name(yolo_class)

        target_cls = getattr(self, '_target_class', '')
        if target_cls:
            clean_tgt_cls = _clean_obj_name(target_cls)
            if clean_cls and (clean_cls == clean_tgt_cls or clean_tgt_cls in clean_cls or clean_cls in clean_tgt_cls):
                return True

        if clean_name and (clean_name == clean_target or clean_target in clean_name or clean_name in clean_target):
            return True
        if clean_cls and (clean_cls == clean_target or clean_target in clean_cls or clean_cls in clean_target):
            return True

        phone_synonyms = {'phone', 'mobile', 'cell phone', 'cellphone', 'cell', 'smartphone'}
        if clean_target in phone_synonyms or (target_cls and _clean_obj_name(target_cls) in phone_synonyms):
            if clean_name in phone_synonyms or clean_cls in phone_synonyms:
                return True
        return False

    def get_beep_params(self, dist: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Return beep profile dict for given distance (or current distance) based on FIND_BEEP_PROFILE."""
        if dist is None:
            with self._lock:
                dist = self._current_distance_cm
                if not getattr(self, '_is_target_visible', False):
                    return None
        if dist is None:
            return None

        profile = getattr(config, 'FIND_BEEP_PROFILE', DEFAULT_FIND_BEEP_PROFILE)
        if dist > 200.0:
            return profile.get('far', DEFAULT_FIND_BEEP_PROFILE['far'])
        elif dist > 100.0:
            return profile.get('mid', DEFAULT_FIND_BEEP_PROFILE['mid'])
        elif dist >= 60.0:
            return profile.get('near', DEFAULT_FIND_BEEP_PROFILE['near'])
        else:
            return profile.get('reach', DEFAULT_FIND_BEEP_PROFILE['reach'])

    def get_beep_interval(self, dist: Optional[float] = None) -> float:
        """Calculate current beep interval in seconds based on distance."""
        params = self.get_beep_params(dist)
        if params is None:
            return 1.2
        return params.get("interval_s", 1.2)

    def get_beep_duration(self, dist: Optional[float] = None) -> int:
        """Calculate current beep duration in milliseconds based on distance."""
        params = self.get_beep_params(dist)
        if params is None:
            return 0
        return params.get("duration_ms", 60)

    @property
    def target_class(self) -> str:
        with self._lock:
            return getattr(self, '_target_class', '') or ''

    def start(self, name: str, last_seen_text: str = "", target_class: str = "") -> None:
        """
        Begin a find session for *name*.

        Args:
            name:           Object name to search for (matched
                            case-insensitively against ``custom_name`` and
                            ``class`` fields in *remembered*).
            last_seen_text: Human-friendly description of where the object was
                            last seen; spoken immediately via ``self.speak`` if provided.
            target_class:   Optional YOLO class of the object (e.g. 'cell phone').
        """
        with self._lock:
            self._target_name    = _clean_obj_name(name)
            self._last_seen_text = last_seen_text
            self.is_active       = True
            if target_class:
                self._target_class = _clean_obj_name(target_class)
            else:
                if self._target_name in {'phone', 'mobile', 'cell phone', 'cellphone', 'cell', 'smartphone'}:
                    self._target_class = 'cell phone'
                else:
                    self._target_class = self._target_name

            now = time.monotonic()
            self._start_time     = now
            self._last_seen_time = now
            self._last_detected_time = now
            self._last_known_detection = None
            self._last_speech_time = 0.0
            self._last_direction_speech_time = 0.0
            self._last_lost_speech_time = 0.0
            self._last_spoken_direction      = None
            self._last_spoken_distance_bucket = None

            self._lost_3s_spoken  = False
            self._lost_30s_spoken = False
            self._last_offset_sign = 0
            self._current_distance_cm = None
            self._is_target_visible = False
            self._last_find_log_time = 0.0

            # Arrival hold & check
            self._arrival_hold_frames = 0
            self._last_arrival_check_log_time = 0.0

            # Reached then removed tracking
            self._reach_total_duration = 0.0
            self._last_frame_time = None
            self._reached_then_removed_ended = False

            # Lost search limits & phrase cycling
            self._lost_phrase_idx = 0

            # Restart beep thread
            self._stop_event.clear()
            self._visible_event.clear()

        # Speak outside the lock so TTS never holds the lock
        if last_seen_text:
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
            self._is_target_visible = False
            self._stop_event.set()
            self._visible_event.set()
            self._current_distance_cm = None

        if self._beep_thread is not None:
            self._beep_thread.join(timeout=1.0)
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
            if not self.is_active or self._reached_then_removed_ended or self._is_paused:
                return {}

        # ── 1. Search for target in remembered ────────────────────────────
        # While find session is active, treat both high and ambiguous tiers as visible
        candidates = []
        for obj in remembered:
            cn    = (obj.get("custom_name") or "").strip().lower()
            cls   = (obj.get("class")       or "").strip().lower()
            tier  = (obj.get("tier")        or "").strip().lower()
            if tier and tier not in ('high', 'ambiguous'):
                continue
            if self.is_target(cn, cls):
                candidates.append(obj)

        best: Optional[Dict[str, Any]] = None
        if candidates:
            with self._lock:
                last_known = self._last_known_detection

            # Use IoU tracker to keep the same bbox identity across frames
            if last_known and "bbox" in last_known and len(candidates) > 1:
                best_iou = -1.0
                best_by_iou = None
                for c in candidates:
                    iou = _compute_iou(c["bbox"], last_known["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_by_iou = c
                    if best_iou > 0.15:
                        best = best_by_iou

            if best is None:
                # Pick candidate with highest confidence / similarity
                best = max(candidates, key=lambda o: float(o.get("confidence", 0.0)))

        now = time.monotonic()
        if self._last_frame_time is not None:
            dt = max(0.0, now - self._last_frame_time)
        else:
            dt = 0.0
        self._last_frame_time = now

        if best is not None:
            with self._lock:
                self._last_detected_time = now
                self._last_seen_time = now
                self._last_known_detection = best
                self._is_target_visible = True
                self._visible_event.set()
                self._lost_3s_spoken = False
                self._lost_30s_spoken = False
            active_obj = best
        else:
            # Hold the track for 1.5 s before visible=False, so a tilt doesn't drop it
            with self._lock:
                last_det = self._last_detected_time
                last_known = self._last_known_detection
            if last_known is not None and (now - last_det) <= 1.5:
                active_obj = last_known
                with self._lock:
                    self._is_target_visible = True
                    self._visible_event.set()
            else:
                self._arrival_hold_frames = 0
                with self._lock:
                    self._is_target_visible = False
                    self._visible_event.clear()
                    self._current_distance_cm = None
                if (now - self._last_find_log_time) >= 1.0:
                    self._last_find_log_time = now
                    logger.info(
                        "FIND: visible=False dir=None dist_cm=None bucket=None bbox=None beep=off"
                    )
                return self._handle_lost()

        # ── 2. Target found or kept via hysteresis — extract geometry ─────
        x1, y1, x2, y2 = active_obj["bbox"]
        cx = (x1 + x2) / 2.0
        # cy kept for future use
        offset = (cx - self.frame_w / 2.0) / (self.frame_w / 2.0 + 1e-9)
        abs_off = abs(offset)
        bbox_h = float(y2 - y1)
        h_ratio = bbox_h / float(self.frame_h)

        with self._lock:
            self._last_offset_sign = int(round(offset / max(abs(offset), 1e-9)))

        # ── 3. Direction (shortened phrases) ───────────────────────────────
        if abs_off < 0.15:
            direction = "straight ahead"
        elif offset < 0:
            direction = "slightly left"
        else:
            direction = "slightly right"

        # ── 4. Distance estimate (pinhole model) ──────────────────────────
        cls_name       = (active_obj.get("class") or "").strip().lower()
        real_size_cm   = TYPICAL_SIZES_CM.get(cls_name, _DEFAULT_SIZE_CM)
        bbox_px        = max(x2 - x1, y2 - y1)
        effective_focal = FOCAL_PX * (float(self.frame_w) / 640.0)
        distance_cm    = effective_focal * real_size_cm / max(bbox_px, 1)

        # ── 5. Ultrasonic cross-check ─────────────────────────────────────
        if (
            ultrasonic_cm is not None
            and ultrasonic_cm < 300
            and abs_off < 0.25
            and not config.ULTRASONIC_MOCK_MODE
            and ultrasonic_cm <= distance_cm * 1.3
        ):
            distance_cm = float(ultrasonic_cm)

        # Track total duration visible within reach (dist <= 60 cm or h_ratio >= 0.6)
        if distance_cm <= 60.0 or h_ratio >= 0.6:
            self._reach_total_duration += dt

        # ── 6. Distance bucket ────────────────────────────────────────────
        if distance_cm < 60.0:
            bucket = "within reach"
        elif distance_cm < 150.0:
            bucket = "close"
        else:
            bucket = "far"

        # ── 7. Arrival check (before speech so arrival message wins) ──────
        # Rule 1: dist <= 55 and |offset| < 0.55 for 2 frames
        # Rule 2: h_ratio >= 0.65 and |offset| < 0.7
        rule1_cand = (distance_cm <= 55.0 and abs_off < 0.55)
        if rule1_cand:
            self._arrival_hold_frames += 1
        else:
            self._arrival_hold_frames = 0

        rule1_fired = (self._arrival_hold_frames >= 2)
        rule2_fired = (h_ratio >= 0.65 and abs_off < 0.7)

        if rule1_fired and rule2_fired:
            matched_rule = "both"
        elif rule1_fired:
            matched_rule = "dist_hold"
        elif rule2_fired:
            matched_rule = "h_ratio"
        elif rule1_cand:
            matched_rule = "holding"
        else:
            matched_rule = "none"

        arrived = rule1_fired or rule2_fired

        if (now - self._last_arrival_check_log_time) >= 1.0 or arrived:
            self._last_arrival_check_log_time = now
            logger.info(
                f"ARRIVAL_CHECK dist={distance_cm:.1f} offset={offset:.2f} h_ratio={h_ratio:.2f} hold={self._arrival_hold_frames} rule={matched_rule}"
            )

        if arrived:
            with self._lock:
                target_name = _clean_obj_name(self._target_name)
            # One long 600 ms tone, then stop beeping and end find
            try:
                arrival_cfg = getattr(config, 'FIND_BEEP_PROFILE', DEFAULT_FIND_BEEP_PROFILE).get('arrival', {})
                arr_dur = arrival_cfg.get('duration_ms', 600)
                arr_freq = arrival_cfg.get('freq_hz', 1400)
                self.beep(arr_freq, arr_dur)
            except Exception:
                pass
            self.stop()
            logger.info("FIND: ARRIVED")
            self.speak(f"{target_name} is right in front of you, reach out")
            return {"visible": True, "arrived": True}

        # ── 8. Update shared beep state ───────────────────────────────────
        with self._lock:
            self._is_target_visible = True
            self._visible_event.set()
            self._current_distance_cm = distance_cm

        # ── 9. Speech throttling ──────────────────────────────────────────
        with self._lock:
            last_dir    = self._last_spoken_direction
            last_bucket = self._last_spoken_distance_bucket
            last_t      = self._last_speech_time
            last_lost_t = self._last_lost_speech_time

        dir_changed = (direction != last_dir)
        bucket_changed = (bucket != last_bucket and bucket in ("close", "within reach"))
        should_speak = (dir_changed or bucket_changed)

        # Speak direction only when changed or bucket changed, at most once per 2.5 s
        if should_speak and (now - last_t) >= 2.5 and (now - last_lost_t) >= 2.5:
            if dir_changed and bucket_changed:
                msg = f"{direction}, {bucket}"
            elif dir_changed:
                msg = direction
            else:
                msg = bucket
            self.speak(msg)
            with self._lock:
                self._last_spoken_direction       = direction
                self._last_spoken_distance_bucket = bucket
                self._last_speech_time            = now
                self._last_direction_speech_time  = now

        if (now - self._last_find_log_time) >= 1.0:
            self._last_find_log_time = now
            x1, y1, x2, y2 = active_obj["bbox"]
            bbox_str = f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}"
            interval_s = self.get_beep_interval(distance_cm)
            duration_ms = self.get_beep_duration(distance_cm)
            logger.info(
                "FIND: visible=True dir=%s dist_cm=%.1f bucket=%s bbox=%s beep_interval=%.2fs beep_duration=%dms",
                direction, distance_cm, bucket, bbox_str, interval_s, duration_ms
            )

        return {
            "visible":         True,
            "direction":       direction,
            "distance_cm":     distance_cm,
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
            ``"last seen near you, 5 minutes ago"``
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
            # If the only nearby object is a person, say "near you" instead of "near the person"
            nearby_lower = [o.lower() for o in nearby]
            if all(o in ['person', 'the person'] for o in nearby_lower):
                return f"last seen near you, {rel_time}"

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
        Called from ``update()`` when the target is not visible this frame (or past 1.5s hysteresis).

        Not visible:
        - Check reached-then-removed: if visible within reach for >= 1.5s total and absent >= 1.5s,
          end session: speak "Found your <name>", stop beeps and all guidance. Never go back to "left" afterwards.
        - After 45 s speak "I can't find your <name>. Say hey vision find <name> to try again." and end session.
        - Otherwise repeat guidance at most every 5 s and alternate phrases: "left", "right", "keep turning slowly".
        """
        now = time.monotonic()
        if self._reached_then_removed_ended or not self.is_active:
            return {"visible": False}

        with self._lock:
            last_detected = self._last_detected_time
            start         = self._start_time
            last_lost_t   = self._last_lost_speech_time
            target_name   = _clean_obj_name(self._target_name)

        elapsed_lost  = now - last_detected if last_detected > 0 else (now - start)
        elapsed_total = now - start

        # 1. Reached then removed check: visible within reach for >= 1.5s in total, and absent >= 1.5s
        if self._reach_total_duration >= 1.5 and elapsed_lost >= 1.5:
            self._reached_then_removed_ended = True
            self.stop()
            self.speak(f"Found your {target_name}")
            return {"visible": False, "ended": True, "reached_and_removed": True}

        # 2. 45 s search timeout — give up
        if elapsed_lost >= 45.0 or elapsed_total >= 45.0:
            self.stop()
            self.speak(f"I can't find your {target_name}. Say hey vision find {target_name} to try again.")
            return {"visible": False, "timeout": True}

        # 3. Repeat guidance at most every 5 s and alternate phrases
        if (now - last_lost_t) >= 5.0:
            phrases = ["left", "right", "keep turning slowly"]
            hint = phrases[self._lost_phrase_idx % len(phrases)]
            self._lost_phrase_idx += 1

            with self._lock:
                self._last_lost_speech_time = now
                self._last_speech_time = now
            self.speak(hint)

        return {"visible": False}

    def _beep_worker(self) -> None:
        """
        Daemon thread: emits proximity beeps while target is visible until ``_stop_event`` is set.

        Do NOT beep at all while the object is not visible (after hysteresis). Silence until seen again.
        While visible, beep interval and duration depend on the distance bucket defined in
        config.FIND_BEEP_PROFILE (or DEFAULT_FIND_BEEP_PROFILE).
        Beeps pause while speech is active and resume after.
        """
        logger.debug("FindGuide beep thread started")
        while not self._stop_event.is_set():
            if self._beeps_paused or (callable(self._is_speaking_fn) and self._is_speaking_fn()):
                self._stop_event.wait(timeout=0.05)
                continue

            with self._lock:
                visible = getattr(self, '_is_target_visible', False)
                dist = self._current_distance_cm

            if not visible or dist is None:
                # Silence until seen again (event-driven, idle thread does NOT poll)
                self._visible_event.wait()
                if self._stop_event.is_set():
                    break
                continue

            params = self.get_beep_params(dist)
            if params is None:
                self._stop_event.wait(timeout=0.05)
                continue

            freq_hz = params.get("freq_hz", 800)
            duration_ms = params.get("duration_ms", 60)
            interval_s = params.get("interval_s", 1.2)

            if self._stop_event.is_set():
                break

            try:
                with self._lock:
                    still_visible = getattr(self, '_is_target_visible', False)
                if not self._stop_event.is_set() and still_visible:
                    self.beep(freq_hz, duration_ms)
            except Exception:  # noqa: BLE001
                pass

            # Wait for interval_s but wake immediately if stop is requested
            self._stop_event.wait(timeout=interval_s)

        logger.debug("FindGuide beep thread stopped")
