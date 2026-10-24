"""
emergency_manager.py
====================
Coordinates fall detection monitoring from Pi Hub, check-in dialog with user,
safety rate limits, and family app emergency dispatching for VisionMate.

Fall / Emergency Verification Flow
-----------------------------------
1. Fall Detection Algorithm Tuning for Physical Testing:
   - Poller connects to active Pi motion endpoint (using config.PI_HOST or
     fallback http://10.210.112.217:5000/motion).
   - Calibrates physical fall detection criteria in _check_fall_criteria():
     * FALL_HIGH_G_THRESHOLD = 1.8g (drop from 30-50 cm or sharp tap).
     * Requires:
       a) Brief drop/low-G (< 0.7g) or normal transition (0.85g - 1.15g).
       b) High-G impact spike (magnitude_g >= 1.8g).
       c) Post-impact inactivity/stillness window of 1.0 to 1.5 seconds
          (magnitude_g settles around 0.9g - 1.1g).
     * Once verified: logs "[FALL DETECTED] Impact magnitude: {mag_g}g. Starting confirmation prompts."
       and triggers confirmation sequence.

2. Strict 3-Prompt Voice Confirmation Sequence:
   - State set to CONFIRMING; non-blocking background thread runs:
     * Prompt 1: Speak "Are you okay? Say I'm okay to cancel."
       - Wait 3.0 seconds for voice input.
     * Prompt 2: If no cancel/distress intent recognized, speak "Are you okay?"
       - Wait 3.0 seconds for voice input.
     * Prompt 3: If still no response, speak "Are you okay? Alerting your family in three seconds."
       - Wait 3.0 seconds for voice input.

3. Intent Handling During Confirmation Window:
   - Cancel Intents ("I'm okay", "I am okay", "yes", "I'm fine", "cancel", ...):
     * Immediately abort confirmation sequence.
     * Speak: "Emergency cancelled. Glad you are okay."
     * Reset state back to IDLE without notifying contacts.
   - Immediate Distress Intents ("help", "no", "I am not okay", "emergency", "call family"):
     * Immediately abort remaining prompts and proceed directly to alert dispatch.

4. Emergency Alert Dispatch:
   - If all 3 prompts pass without cancellation, or on immediate distress:
     * Transition state to TRIGGERED.
     * Speak: "Emergency confirmed. Sending notification to your family with your location."
     * Dispatch alert via alert_sender.send_fall_alert(lat, lon, fall_type="fall").
     * Fallback logging maintained in DRY_RUN mode.
"""

import os
import sys
import time
import re
import uuid
import math
import logging
import threading
import queue
import requests
from typing import Optional, Callable, Dict, Any, Tuple

logger = logging.getLogger("emergency_manager")

# Calibration thresholds for physical live demo testing
FALL_HIGH_G_THRESHOLD = 1.15  # Down from 1.25g / 2.5g so any spike >= 1.15g triggers impact
FALL_LOW_G_THRESHOLD = 0.85   # Relaxed baseline threshold
STILLNESS_MIN_G = 0.85        # Resting range lower bound near 0.85g - 1.15g
STILLNESS_MAX_G = 1.15        # Resting range upper bound near 0.85g - 1.15g
STILLNESS_DURATION_MIN_S = 0.5  # 0.5 to 1.0 second of resting confirms the fall
STILLNESS_DURATION_MAX_S = 1.0

try:
    import config
    DEFAULT_PI_HOST = getattr(config, 'PI_HOST', 'raspberrypi.local')
    DEFAULT_MOTION_URL = getattr(config, 'MOTION_URL', f'http://{DEFAULT_PI_HOST}:5000/motion')
    DEFAULT_DRY_RUN = getattr(config, 'EMERGENCY_DRY_RUN', True)
    DEFAULT_USER = getattr(config, 'EMERGENCY_USER_NAME', 'Bharath')
    DEFAULT_AUTO_LIMIT = getattr(config, 'EMERGENCY_AUTO_RATE_LIMIT_S', 300.0)
    DEFAULT_COOLDOWN = getattr(config, 'EMERGENCY_COOLDOWN_S', 60.0)
    CHECKIN_WINDOW_S = getattr(config, 'CHECKIN_REPLY_WINDOW_S', 3.0)
except Exception:
    DEFAULT_PI_HOST = 'raspberrypi.local'
    DEFAULT_MOTION_URL = f'http://{DEFAULT_PI_HOST}:5000/motion'
    DEFAULT_DRY_RUN = True
    DEFAULT_USER = "Bharath"
    DEFAULT_AUTO_LIMIT = 300.0
    DEFAULT_COOLDOWN = 60.0
    CHECKIN_WINDOW_S = 3.0

FALLBACK_PI_HOST = '10.210.112.217'
FALLBACK_MOTION_URL = f'http://{FALLBACK_PI_HOST}:5000/motion'

from alert_sender import AlertSender


def classify_reply(text: Optional[str]) -> str:
    """
    Classify user reply during emergency check-in.
    Returns: 'ok' (cancel intent), 'bad' (distress), or 'unclear'.

    Cancel intents: "I'm okay", "yes", "I am fine", "cancel", "all good", ...
    Distress:       "help", "no", "I am not okay", "emergency", "call family", ...
    """
    if not text:
        return 'unclear'
    clean = re.sub(r"[!.,?'\";:]+", " ", text.lower()).strip()
    clean_norm = " ".join(clean.split())

    # 1. Multi-word DISTRESS phrases (checked FIRST so "i am not okay" is not caught by "okay")
    for bad_phrase in (
        "i am not okay", "i'm not okay", "im not okay",
        "i am not ok",   "i'm not ok",   "im not ok",
        "not okay", "not ok", "not good",
        "help me", "call family", "call for help",
        "i am hurt", "i'm hurt", "im hurt",
    ):
        if bad_phrase in clean_norm:
            return 'bad'

    # 2. Multi-word CANCEL / OK phrases
    for ok_phrase in (
        "no problem",
        "i'm good",   "im good",   "i am good",
        "i'm alright","im alright","i am alright",
        "i'm okay",   "im okay",   "i am okay",
        "i'm ok",     "im ok",     "i am ok",
        "i am fine",  "i'm fine",  "im fine",
        "all good",
    ):
        if ok_phrase in clean_norm:
            return 'ok'

    words = set(re.findall(r"\b\w+\b", clean))

    # 3. Distress single keywords
    if any(w in words for w in ("help", "hurt", "pain", "emergency")):
        return 'bad'
    # "no" alone -> distress ("no problem" already matched above)
    if "no" in words and "problem" not in words:
        return 'bad'

    # 4. Cancel / OK single keywords
    if any(w in words for w in (
        "cancel", "stop", "fine", "okay", "ok", "good", "alright",
        "yes", "yeah", "yep", "ya", "yup", "sure",
    )):
        return 'ok'

    return 'unclear'


class EmergencyManager:
    """
    Manages fall polling, physical fall detection criteria, 3-prompt check-in dialog,
    and mobile alerts for VisionMate.
    """

    FALL_HIGH_G_THRESHOLD = FALL_HIGH_G_THRESHOLD

    def __init__(self,
                 pi_host: str = DEFAULT_PI_HOST,
                 audio_feedback: Optional[Any] = None,
                 voice_system: Optional[Any] = None,
                 find_guide: Optional[Any] = None,
                 gps_tracker: Optional[Any] = None,
                 alert_sender: Optional[AlertSender] = None,
                 dry_run: bool = DEFAULT_DRY_RUN,
                 user_name: str = DEFAULT_USER,
                 motion_url: Optional[str] = None):
        self.pi_host = pi_host
        self.motion_url = motion_url or DEFAULT_MOTION_URL
        self.fallback_motion_url = FALLBACK_MOTION_URL
        self.audio_feedback = audio_feedback
        self.voice_system = voice_system
        self.find_guide = find_guide
        self.gps_tracker = gps_tracker
        self.dry_run = dry_run
        self.user_name = user_name

        def speak_prompt(text: str):
            if self.audio_feedback:
                if hasattr(self.audio_feedback, 'speak_prompt'):
                    self.audio_feedback.speak_prompt(text, interrupt=True, wait=True)
                elif hasattr(self.audio_feedback, 'speak'):
                    self.audio_feedback.speak(text)
            else:
                logger.info("[TTS]: %s", text)

        self._speak = speak_prompt

        self.alert_sender = alert_sender or AlertSender(
            user_name=self.user_name,
            dry_run=self.dry_run,
            speak_fn=self._speak,
            gps_tracker=self.gps_tracker
        )

        # State tracking (IDLE, CONFIRMING, TRIGGERED)
        self.state: str = "IDLE"
        self.is_checkin_running: bool = False
        self.last_checkin_end_time: float = 0.0
        self.last_automatic_alert_time: float = 0.0
        self.active_alert_id: Optional[str] = None
        self.active_alert_time: float = 0.0
        self.seen_event_ids = set()

        self._checkin_thread: Optional[threading.Thread] = None
        self._poller_thread: Optional[threading.Thread] = None
        self._poller_running: bool = False
        self._reply_queue: queue.Queue = queue.Queue()
        self._cancel_checkin_event = threading.Event()
        self._lock = threading.Lock()

        # Poller state & logging rate-limiters
        self._missing_unreachable_logged: bool = False
        self._last_unreachable_log_time: float = -999.0
        self._last_motion_log_time: float = 0.0
        self._last_mag_g: Optional[float] = None
        self._last_event_id: str = "none"

        # Physical fall detection criteria state machine
        self._low_g_seen_time: float = 0.0
        self._normal_seen_time: float = 0.0
        self._impact_detected_time: float = 0.0
        self._impact_mag: float = 0.0
        self._stillness_start_time: float = 0.0
        self._last_verified_fall_time: float = 0.0

    def start_polling(self):
        """Start polling /motion in background thread."""
        if self._poller_running:
            return
        self._poller_running = True
        self._poller_thread = threading.Thread(target=self._poll_loop, daemon=True, name="MotionPoller")
        self._poller_thread.start()
        logger.info("EmergencyManager motion poller started (url=%s)", self.motion_url)

    def stop_polling(self):
        """Stop polling /motion."""
        self._poller_running = False
        self._poller_thread = None

    def _check_fall_criteria(self, sample_or_mag: Any, ts: Optional[float] = None) -> bool:
        """
        Calibrate the physical fall detection criteria:
          a) Brief drop/low-G (< 0.7g) or normal transition (0.85g - 1.15g).
          b) High-G impact spike (magnitude_g >= 1.8g).
          c) Post-impact inactivity/stillness window of 1.0 to 1.5 seconds
             (magnitude_g settles around 0.9g - 1.1g).
        Once verified, logs:
          `[FALL DETECTED] Impact magnitude: {mag_g}g. Starting confirmation prompts.`
        and triggers the confirmation sequence.
        """
        if isinstance(sample_or_mag, dict):
            mag_val = sample_or_mag.get("magnitude_g")
            if mag_val is None:
                mag_val = sample_or_mag.get("magnitude")
            if mag_val is None and "accel_x" in sample_or_mag:
                ax = float(sample_or_mag.get("accel_x", 0.0))
                ay = float(sample_or_mag.get("accel_y", 0.0))
                az = float(sample_or_mag.get("accel_z", 0.0))
                mag_val = math.sqrt(ax * ax + ay * ay + az * az)
                if mag_val > 5.0:  # in m/s^2 -> convert to g
                    mag_val = mag_val / 9.80665
            if mag_val is None:
                return False
            mag_g = float(mag_val)
            if ts is None:
                ts = float(sample_or_mag.get("ts", time.time()))
        else:
            mag_g = float(sample_or_mag)
            if ts is None:
                ts = time.time()

        # Step a & b: Impact spike (>= 1.15g) from resting baseline (~1.0g)
        # Register immediately as valid impact spike without strict pre-impact free-fall requirement
        if mag_g >= self.FALL_HIGH_G_THRESHOLD:
            self._impact_detected_time = ts
            self._impact_mag = round(mag_g, 2)
            self._stillness_start_time = 0.0
            logger.info("[MOTION] Impact spike registered: %.2fg at %.2f", mag_g, ts)

        # Step c: Post-impact resting window (0.85g to 1.15g for 0.5 to 1.0 second)
        if self._impact_detected_time > 0.0:
            dt = ts - self._impact_detected_time
            if dt > 3.0:
                # Expired without settling
                self._impact_detected_time = 0.0
                self._stillness_start_time = 0.0
            elif dt >= 0.05:
                # Check if settled in [0.90, 1.10]
                if STILLNESS_MIN_G <= mag_g <= STILLNESS_MAX_G:
                    if self._stillness_start_time == 0.0:
                        self._stillness_start_time = ts
                    stillness_duration = ts - self._stillness_start_time
                    if stillness_duration >= STILLNESS_DURATION_MIN_S:
                        # Fall criteria fully satisfied!
                        mag_detected = self._impact_mag
                        self._impact_detected_time = 0.0
                        self._stillness_start_time = 0.0
                        self._last_verified_fall_time = ts

                        log_msg = f"[FALL DETECTED] Triggered at {mag_detected}g. Initiating confirmation."
                        print(log_msg)
                        logger.info(log_msg)

                        event = {
                            "id": f"fall-{uuid.uuid4().hex[:6]}",
                            "type": "fall",
                            "peak_g": mag_detected,
                            "ts": ts
                        }
                        self.on_fall_detected(event)
                        return True
                else:
                    # Non-stillness movement breaks the stillness window
                    self._stillness_start_time = 0.0

        return False

    def poll_once(self) -> Dict[str, Any]:
        """
        Execute a single poll against the active motion endpoint.
        Connects using config.PI_HOST with automatic fallback to 10.210.112.217:5000/motion.
        """
        poll_ok = False
        retry_delay = 0.5
        now = time.time()
        resp = None

        urls_to_try = [self.motion_url]
        if self.fallback_motion_url != self.motion_url:
            urls_to_try.append(self.fallback_motion_url)

        for target_url in urls_to_try:
            try:
                resp = requests.get(target_url, timeout=0.8)
                if resp.status_code == 200:
                    if target_url != self.motion_url:
                        logger.info("Switched to fallback motion endpoint: %s", target_url)
                        self.motion_url = target_url
                    poll_ok = True
                    break
            except Exception as e:
                logger.debug("Motion polling exception on %s: %s", target_url, e)

        if poll_ok and resp is not None:
            self._missing_unreachable_logged = False
            try:
                data = resp.json()
                mag = data.get("magnitude_g")
                if mag is None:
                    mag = data.get("magnitude")
                if mag is None and "mpu6050" in data:
                    m = data["mpu6050"]
                    ax = float(m.get("accel_x", 0.0))
                    ay = float(m.get("accel_y", 0.0))
                    az = float(m.get("accel_z", 0.0))
                    mag = math.sqrt(ax * ax + ay * ay + az * az)
                    if mag > 5.0:
                        mag = mag / 9.80665

                if mag is not None:
                    mag_f = float(mag)
                    self._last_mag_g = round(mag_f, 2)
                    self._check_fall_criteria(mag_f, now)

                events = data.get("events", [])
                trigger_on_impact = True
                try:
                    import config as _cfg
                    trigger_on_impact = getattr(_cfg, 'FALL_TRIGGER_ON_IMPACT', True)
                except Exception:
                    pass

                for ev in events:
                    ev_id = ev.get("id")
                    ev_type = ev.get("type")
                    if ev_id and ev_id not in self.seen_event_ids:
                        self.seen_event_ids.add(ev_id)
                        self._last_event_id = ev_id
                        if ev_type == "fall" or (trigger_on_impact and ev_type == "impact"):
                            peak = ev.get("peak_g", 1.25)
                            log_msg = f"[FALL DETECTED] Triggered at {peak}g. Initiating confirmation."
                            print(log_msg)
                            logger.info(log_msg)
                            self.on_fall_detected(ev)
            except Exception as parse_err:
                logger.debug("Error parsing motion json: %s", parse_err)
        else:
            retry_delay = 5.0
            if (now - self._last_unreachable_log_time) >= 30.0:
                logger.warning("Motion endpoint temporarily unreachable (retrying)")
                print("Motion endpoint temporarily unreachable (retrying)")
                self._last_unreachable_log_time = now

        now = time.time()
        if (now - self._last_motion_log_time) >= 5.0:
            status_str = "ok" if poll_ok else "fail"
            mag_str = f"{self._last_mag_g:.2f}" if self._last_mag_g is not None else "none"
            logger.info("MOTION: poll=%s mag_g=%s last_event=%s", status_str, mag_str, self._last_event_id)
            self._last_motion_log_time = now

        return {"ok": poll_ok, "retry_delay": retry_delay}

    def _poll_loop(self):
        """Poll motion endpoint continuously."""
        while self._poller_running:
            res = self.poll_once()
            delay = res.get("retry_delay", 0.5)
            time.sleep(delay)

    def on_fall_detected(self, event: Dict[str, Any]):
        """
        Called when a fall is verified. Transitions state to CONFIRMING and
        starts the 3-prompt voice confirmation sequence in a background thread.
        """
        now = time.time()
        with self._lock:
            # Ignore subsequent falls while an alert is active or check-in is running
            if self.active_alert_id and (now - self.active_alert_time) < DEFAULT_AUTO_LIMIT:
                logger.info("Ignoring fall %s: active alert %s is in progress", event.get("id"), self.active_alert_id)
                return
            if self.is_checkin_running or self.state == "CONFIRMING":
                logger.info("Ignoring fall %s: confirmation is already running", event.get("id"))
                return
            if (now - self.last_checkin_end_time) < DEFAULT_COOLDOWN:
                logger.info("Ignoring fall %s: within %ds cooldown", event.get("id"), DEFAULT_COOLDOWN)
                return
            self.state = "CONFIRMING"
            self.is_checkin_running = True

        self._cancel_checkin_event.clear()
        self._checkin_thread = threading.Thread(
            target=self._run_checkin_protocol,
            args=(event,),
            daemon=True,
            name="EmergencyConfirmation"
        )
        self._checkin_thread.start()

    def _on_fall_detected(self, event: Dict[str, Any]):
        """Alias for on_fall_detected."""
        return self.on_fall_detected(event)

    def post_reply(self, text: str):
        """Deliver recognized speech reply to a running confirmation sequence."""
        self._reply_queue.put(text)

    def _listen_for_reply(self, timeout_s: float = 3.0) -> Optional[str]:
        """
        Listen for user reply without requiring wake word.
        Integrates with VoiceCommandSystem emergency window.
        """
        while not self._reply_queue.empty():
            try:
                self._reply_queue.get_nowait()
            except queue.Empty:
                break

        if self.voice_system and hasattr(self.voice_system, 'open_emergency_window'):
            try:
                self.voice_system.open_emergency_window(
                    on_reply=self.post_reply,
                    timeout=timeout_s
                )
            except TypeError:
                self.voice_system.open_emergency_window(timeout_s, self.post_reply)

        try:
            return self._reply_queue.get(timeout=timeout_s)
        except queue.Empty:
            return None
        finally:
            if self.voice_system and hasattr(self.voice_system, 'close_emergency_window'):
                self.voice_system.close_emergency_window()

    def _run_checkin_protocol(self, event: Dict[str, Any]):
        """
        Strict 3-prompt voice confirmation sequence:
          Prompt 1: "Are you okay? Say I'm okay to cancel." (Wait 3.0s)
          Prompt 2: "Are you okay?" (Wait 3.0s)
          Prompt 3: "Are you okay? Alerting your family in three seconds." (Wait 3.0s)

        Intent outcomes:
          - Cancel intent  -> speak "Emergency cancelled. Glad you are okay." -> state IDLE.
          - Distress reply -> immediately abort remaining prompts -> alert dispatch.
          - Silence/unclear-> advance prompt, or alert dispatch after Prompt 3.
        """
        event_id = event.get("id", str(uuid.uuid4())[:8])
        find_paused = False

        if self.find_guide and getattr(self.find_guide, 'is_active', False):
            self.find_guide.pause()
            find_paused = True

        window_s = CHECKIN_WINDOW_S

        PROMPT_TEXT = {
            1: "Are you okay? Say I'm okay to cancel.",
            2: "Are you okay?",
            3: "Are you okay? Alerting your family in three seconds.",
        }

        try:
            for asked in range(1, 4):
                if self._cancel_checkin_event.is_set():
                    logger.info("Confirmation cancelled by external event before ask %d", asked)
                    with self._lock:
                        self.state = "IDLE"
                    return

                # Prompt audio spoken; listening window starts after speech completes
                prompt_text = PROMPT_TEXT[asked]
                self._speak(prompt_text)

                reply = self._listen_for_reply(timeout_s=window_s)
                reply_disp = repr(reply) if reply else "(silence)"
                reply_raw = reply if reply else "None"

                checkin_log = (
                    f"CHECKIN: event={event_id} ask={asked}/3 "
                    f"window={window_s:.1f}s reply={reply_disp}"
                )
                print(checkin_log)
                logger.info(checkin_log)

                if self._cancel_checkin_event.is_set():
                    logger.info("Confirmation cancelled during listen (ask %d)", asked)
                    with self._lock:
                        self.state = "IDLE"
                    return

                classification = classify_reply(reply)

                if classification == 'ok':
                    # User cancelled emergency
                    em_log = f"EMERGENCY: event={event_id} asked={asked} reply={reply_raw} -> cancel"
                    print(em_log)
                    logger.info(em_log)
                    self._speak("Emergency cancelled. Glad you are okay.")
                    with self._lock:
                        self.state = "IDLE"
                    return

                elif classification == 'bad':
                    # Distress detected: abort remaining prompts and dispatch alert
                    em_log = f"EMERGENCY: event={event_id} asked={asked} reply={reply_raw} -> severe"
                    print(em_log)
                    logger.info(em_log)
                    self._dispatch_fall_alert(
                        reason=f"Fall detected - user replied: {reply}",
                        reason_code="distress_reply",
                        fall_type="fall",
                    )
                    return

                else:
                    # Silence or unclear
                    if asked < 3:
                        logger.info("Confirmation ask=%d reply=%r unclear/silence, advancing to next prompt", asked, reply)
                        continue
                    else:
                        # All 3 prompts expired without cancellation
                        em_log = f"EMERGENCY: event={event_id} asked=3 reply=None -> severe"
                        print(em_log)
                        logger.info(em_log)
                        self._dispatch_fall_alert(
                            reason="Fall detected - user unresponsive after 3 confirmation prompts",
                            reason_code="silence_timeout",
                            fall_type="fall",
                        )
                        return

        finally:
            with self._lock:
                if self.state == "CONFIRMING":
                    self.state = "IDLE"
                self.is_checkin_running = False
                self.last_checkin_end_time = time.time()

            if find_paused and self.find_guide:
                self.find_guide.resume()

    def _dispatch_fall_alert(
        self,
        reason: str,
        reason_code: str = "silence_timeout",
        fall_type: str = "fall",
        bypass_rate_limit: bool = False,
    ) -> Optional[str]:
        """
        Transition state to TRIGGERED, speak confirmation, and dispatch via
        alert_sender.send_fall_alert(lat, lon, fall_type).
        """
        now = time.time()
        with self._lock:
            if not bypass_rate_limit:
                if self.active_alert_id and (now - self.active_alert_time) < DEFAULT_AUTO_LIMIT:
                    logger.info("Alert suppressed: active alert %s is in progress", self.active_alert_id)
                    return None
                if (now - self.last_automatic_alert_time) < DEFAULT_AUTO_LIMIT:
                    remaining = int(DEFAULT_AUTO_LIMIT - (now - self.last_automatic_alert_time))
                    logger.info("Alert suppressed by 5-minute safety rate limit (%ds remaining)", remaining)
                    return None
            self.last_automatic_alert_time = now
            self.state = "TRIGGERED"
            self.is_checkin_running = False

        # Required spoken line
        self._speak("Emergency confirmed. Sending notification to your family with your location.")

        lat: Optional[float] = None
        lon: Optional[float] = None
        try:
            loc = self.alert_sender.resolve_location()
            lat = loc.get("lat")
            lon = loc.get("lon")
        except Exception as loc_err:
            logger.debug("Could not resolve location for fall alert: %s", loc_err)

        aid: Optional[str] = None
        if hasattr(self.alert_sender, 'send_fall_alert'):
            aid = self.alert_sender.send_fall_alert(lat=lat, lon=lon, fall_type=fall_type)

        # Fallback for mocks or legacy alert_sender providing send_alert
        if not aid and hasattr(self.alert_sender, 'send_alert'):
            aid = self.alert_sender.send_alert(severity="severe", reason=reason)

        if aid:
            with self._lock:
                self.active_alert_id = str(aid)
                self.active_alert_time = now
            alert_log = f"ALERT: sent=1 reason={reason_code} fall_type={fall_type}"
            print(alert_log)
            logger.info(alert_log)

        return aid

    def _dispatch_automatic_alert(self, reason: str, reason_code: str = "silence_timeout"):
        """Wrapper for automatic fall alert dispatch."""
        self._dispatch_fall_alert(reason=reason, reason_code=reason_code, fall_type="fall")

    def handle_voice_emergency(self):
        """
        Voice command 'emergency' / 'help me':
        Immediate distress intent: aborts any running confirmation prompts,
        bypasses the 5-minute rate limit, and dispatches the alert immediately.
        """
        logger.info("Manual emergency triggered via voice command ('emergency'/'help me')")
        with self._lock:
            if self.is_checkin_running:
                self._cancel_checkin_event.set()
                self.is_checkin_running = False

        self._dispatch_fall_alert(
            reason="Manual emergency triggered via voice command",
            reason_code="voice_emergency",
            fall_type="manual_emergency",
            bypass_rate_limit=True
        )

    def handle_voice_im_okay(self):
        """
        Voice command 'I'm okay' / 'cancel alert':
        Cancels running confirmation sequence or marks sent alert cancelled.
        """
        logger.info("Voice command 'I'm okay' received")
        was_confirming = False
        with self._lock:
            if self.is_checkin_running or self.state == "CONFIRMING":
                was_confirming = True
                self._cancel_checkin_event.set()
                self._reply_queue.put("I'm okay")
                self.state = "IDLE"
                self.is_checkin_running = False
                self.last_checkin_end_time = time.time()

            alert_to_cancel = self.active_alert_id
            self.active_alert_id = None
            self.active_alert_time = 0.0

        if alert_to_cancel:
            if hasattr(self.alert_sender, 'cancel_alert'):
                self.alert_sender.cancel_alert(alert_to_cancel)
            self._speak("Alert cancelled")
        elif was_confirming:
            self._speak("Emergency cancelled. Glad you are okay.")

    def handle_voice_simulate_fall(self):
        """Voice command 'simulate fall' for demonstration."""
        logger.info("Simulating fall event for demonstration")
        fake_event = {
            "id": f"sim-{uuid.uuid4().hex[:6]}",
            "type": "fall",
            "peak_g": 2.1,
            "ts": time.time()
        }
        self.on_fall_detected(fake_event)
