"""
Audio Feedback System
Priority-based Text-to-Speech for VisionMate
"""
import threading
import queue
import collections
from typing import Optional, Callable, Dict, List
import os
import sys
import tempfile
import time
import logging
import re

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False

if sys.platform == 'win32':
    try:
        import pythoncom
        import win32com.client
        WIN32COM_AVAILABLE = True
    except Exception:
        WIN32COM_AVAILABLE = False
else:
    WIN32COM_AVAILABLE = False

logger = logging.getLogger(__name__)

from config import (
    PRIORITY_EMERGENCY, PRIORITY_PROMPT, PRIORITY_SOCIAL,
    PRIORITY_NAVIGATIONAL, PRIORITY_STATUS
)


class AudioFeedback:
    def __init__(self, use_gtts: bool = False):
        """
        Initialize audio feedback system.
        TTS is handled by a single dedicated worker thread that owns a Windows SAPI
        voice (or falls back to pyttsx3/gTTS) to avoid COM threading issues.
        """
        self.use_gtts = use_gtts
        self.audio_queue = queue.PriorityQueue()
        self.is_playing = False
        self.is_speaking = False
        self.last_speech_end_time = 0.0
        self.current_priority = None
        self.current_speech_start_time = 0.0
        self.last_spoken_time_by_text = {}  # text -> timestamp when actually spoken
        self._quiet_mode_checker = None
        
        # Priority interruption events and synchronisation
        self._interrupt_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._events_lock = threading.Lock()
        self._done_events = collections.defaultdict(list)  # text -> list of threading.Event
        self._beep_controllers: List[tuple] = []  # [(pause_fn, resume_fn), ...]

        # Dedicated worker thread
        self.voice = None
        self.sapi_available = False
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True, name="TTS-Worker")
        self.playback_thread.start()

        print("[OK] Audio Feedback initialized")
        print(f"  Backend: {'gTTS' if use_gtts else 'Windows SAPI / pyttsx3 fallback'}")
        print(f"  Priority levels: EMERGENCY({PRIORITY_EMERGENCY}) > PROMPT({PRIORITY_PROMPT}) > SOCIAL({PRIORITY_SOCIAL}) > NAVIGATIONAL({PRIORITY_NAVIGATIONAL}) > STATUS({PRIORITY_STATUS})")

    def register_beep_controller(self, pause_fn: Callable[[], None], resume_fn: Callable[[], None]):
        """Register pause/resume callbacks so beeps are paused while speech is active."""
        with self._events_lock:
            self._beep_controllers.append((pause_fn, resume_fn))

    def unregister_beep_controller(self, pause_fn: Callable[[], None], resume_fn: Callable[[], None]):
        """Unregister previously registered beep controller callbacks."""
        with self._events_lock:
            self._beep_controllers = [pair for pair in self._beep_controllers if pair != (pause_fn, resume_fn)]

    def _notify_beeps_pause(self):
        with self._events_lock:
            controllers = list(self._beep_controllers)
        for pause_fn, _ in controllers:
            try:
                pause_fn()
            except Exception as e:
                logger.debug(f"Error pausing beeps: {e}")

    def _notify_beeps_resume(self):
        with self._events_lock:
            controllers = list(self._beep_controllers)
        for _, resume_fn in controllers:
            try:
                resume_fn()
            except Exception as e:
                logger.debug(f"Error resuming beeps: {e}")

    def set_quiet_mode_checker(self, checker_fn):
        """Set callback returning True when quiet mode is active."""
        self._quiet_mode_checker = checker_fn

    def _is_quiet_mode_active(self) -> bool:
        if callable(self._quiet_mode_checker):
            try:
                return bool(self._quiet_mode_checker())
            except Exception:
                return False
        return False

    def _is_critical_warning(self, text: str) -> bool:
        """Check if message represents a real critical warning (<50 cm)."""
        t = text.lower().strip()
        if 'stop' in t:
            return True
        m = re.search(r'obstacle\s+(\d+(?:\.\d+)?)\s*cm', t)
        if m:
            try:
                dist = float(m.group(1))
                if dist < 50.0:
                    return True
            except ValueError:
                pass
        return False

    def _is_text_in_queue(self, text: str) -> bool:
        """Check if identical text is currently waiting in the audio queue."""
        with self.audio_queue.mutex:
            for item in self.audio_queue.queue:
                if isinstance(item, tuple) and len(item) >= 2:
                    if item[1] == text:
                        return True
        return False

    def _is_routine_announcement(self, text: str) -> bool:
        """
        Check if message is a routine background announcement muted during quiet mode:
        e.g. 'Person ahead', 'Chair left', 'Hello Alice', 'Found your keys', 'Is this your keys?'
        """
        t = text.lower().strip()
        if t.startswith("hello "):
            return True
        if t.startswith("found your "):
            return True
        if t.startswith("is this your "):
            return True
        if self._is_critical_warning(text):
            return False
        # Spatial obstacle guidance: e.g. "Person ahead", "Chair left", etc.
        nav_directions = ('ahead', 'left', 'right', 'behind', 'center', 'front')
        if any(t.endswith(d) for d in nav_directions):
            return True
        return False

    def speak(self, text: str, priority: int = PRIORITY_NAVIGATIONAL, interrupt: bool = False,
              is_critical: bool = False, is_command_reply: bool = False, wait: bool = False):
        """
        Add text to speech queue with safety duplicate dropping, quiet mode gating,
        and priority interruption.
        
        Args:
            text: Text to speak
            priority: Priority level (1=highest, 5=lowest)
            interrupt: If True, interrupt current speech for high priority messages
            is_critical: If True, represents a real critical warning (<50 cm)
            is_command_reply: If True, this is a reply to a user voice command (never suppressed by quiet mode)
            wait: If True, block caller until this utterance finishes speaking
        """
        if not text:
            return

        clean_text = text.strip()

        # Check if real critical warning under 50 cm
        crit = is_critical or self._is_critical_warning(clean_text)

        # Quiet mode suppression
        is_routine = self._is_routine_announcement(clean_text)
        if not is_command_reply and is_routine and self._is_quiet_mode_active() and not crit:
            logger.debug(f"[AUDIO] Suppressed routine announcement during quiet mode: {clean_text}")
            return

        # Drop duplicate waiting in queue
        if self._is_text_in_queue(clean_text):
            logger.debug(f"[AUDIO] Dropped duplicate: {clean_text}")
            return

        # Cooldown check for routine duplicates
        if not is_command_reply:
            now = time.time()
            cooldown = 3.0 if crit else 15.0
            last_spoken = self.last_spoken_time_by_text.get(clean_text, 0.0)
            if (now - last_spoken) < cooldown:
                logger.debug(f"[AUDIO] Dropped duplicate: {clean_text}")
                return

        # High priority automatic interruption
        if priority == PRIORITY_EMERGENCY and crit:
            interrupt = True
        elif priority == PRIORITY_PROMPT:
            interrupt = True

        done_event = None
        if wait:
            done_event = threading.Event()
            with self._events_lock:
                self._done_events[clean_text].append(done_event)

        if interrupt:
            # Purge any queued items of lower priority than this utterance
            self._purge_lower_priority_from_queue(priority)
            # If current speech is lower priority than this utterance, interrupt it immediately
            if self.is_speaking and self.current_priority is not None and self.current_priority > priority:
                self._interrupt_current()

        # Enqueue (priority, text) tuple
        self.audio_queue.put((priority, clean_text))

        if wait and done_event is not None:
            done_event.wait(timeout=10.0)

    def _purge_lower_priority_from_queue(self, cutoff_priority: int):
        """Remove all queued messages with lower priority (higher integer value) than cutoff_priority."""
        import heapq
        dropped_texts = []
        with self.audio_queue.mutex:
            items = list(self.audio_queue.queue)
            preserved = []
            for item in items:
                if isinstance(item, tuple) and len(item) >= 2:
                    p, t = item[0], item[1]
                    if p <= cutoff_priority:
                        preserved.append(item)
                    else:
                        dropped_texts.append(t)
                else:
                    preserved.append(item)
            self.audio_queue.queue.clear()
            self.audio_queue.queue.extend(preserved)
            heapq.heapify(self.audio_queue.queue)

        # Release any callers waiting on purged texts
        with self._events_lock:
            for t in dropped_texts:
                for ev in self._done_events.pop(t, []):
                    ev.set()

    def _interrupt_current(self):
        """Signal interrupt event to immediately halt currently playing utterance."""
        self._interrupt_event.set()

    def speak_emergency(self, text: str, is_critical: bool = True, wait: bool = False):
        """Speak emergency message (highest priority, interrupts everything)"""
        self.speak(text, PRIORITY_EMERGENCY, interrupt=True, is_critical=is_critical, wait=wait)

    def speak_prompt(self, text: str, interrupt: bool = True, wait: bool = True):
        """
        Speak prompt/guidance message (PROMPT priority 2).
        Interrupts lower priorities, cannot be interrupted by low-priority speech.
        """
        self.speak(text, priority=PRIORITY_PROMPT, interrupt=interrupt, is_critical=False,
                   is_command_reply=True, wait=wait)

    def speak_social(self, text: str, is_command_reply: bool = False, wait: bool = False):
        """Speak social message (recognized person)"""
        self.speak(text, PRIORITY_SOCIAL, interrupt=False, is_critical=False,
                   is_command_reply=is_command_reply, wait=wait)

    def speak_command_reply(self, text: str, priority: int = PRIORITY_PROMPT, wait: bool = False):
        """Speak a reply to a voice command (PROMPT priority, never suppressed by quiet mode)"""
        self.speak(text, priority=priority, interrupt=(priority <= PRIORITY_PROMPT),
                   is_critical=False, is_command_reply=True, wait=wait)

    def speak_navigational(self, text: str, is_critical: bool = False, is_command_reply: bool = False, wait: bool = False):
        """Speak navigational message (obstacles)"""
        self.speak(text, PRIORITY_NAVIGATIONAL, interrupt=False, is_critical=is_critical,
                   is_command_reply=is_command_reply, wait=wait)

    def speak_status(self, text: str, is_command_reply: bool = False, wait: bool = False):
        """Speak status message (lowest priority)"""
        self.speak(text, PRIORITY_STATUS, interrupt=False, is_critical=False,
                   is_command_reply=is_command_reply, wait=wait)

    def _playback_worker(self):
        """Dedicated background thread for managing TTS engine and playing audio."""
        if sys.platform == 'win32':
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception as e:
                logger.warning(f"CoInitialize error: {e}")

        try:
            # Initialize Windows SAPI if available and not using gTTS
            if sys.platform == 'win32' and not self.use_gtts and WIN32COM_AVAILABLE:
                try:
                    import win32com.client
                    self.voice = win32com.client.Dispatch("SAPI.SpVoice")
                    try:
                        self.voice.Rate = 2
                    except Exception:
                        pass
                    self.sapi_available = True
                    logger.info("[OK] Windows SAPI voice initialized (Rate=+2)")
                except Exception as e:
                    try:
                        import comtypes.client
                        self.voice = comtypes.client.CreateObject("SAPI.SpVoice")
                        try:
                            self.voice.Rate = 2
                        except Exception:
                            pass
                        self.sapi_available = True
                        logger.info("[OK] Windows SAPI voice initialized via comtypes (Rate=+2)")
                    except Exception as e2:
                        logger.warning(f"Windows SAPI voice unavailable ({e}, {e2}), falling back to pyttsx3")
                        self.voice = None
                        self.sapi_available = False

            while not self._shutdown_event.is_set():
                try:
                    item = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if not isinstance(item, tuple) or len(item) < 2:
                    continue

                priority = item[0]
                text = item[1]

                if text == "__STOP__":
                    self._stop_current_internal()
                    self._signal_done(text)
                    continue

                self._interrupt_event.clear()
                self.current_priority = priority
                self.current_speech_start_time = time.time()
                self.is_playing = True
                self.is_speaking = True
                self.last_spoken_time_by_text[text] = time.time()
                self._notify_beeps_pause()

                logger.info(f"TTS start: {text}")
                print(f"TTS start: {text}")
                t0 = time.perf_counter()

                try:
                    self._play_text(text)
                except Exception as e:
                    logger.error(f"TTS error: {e}")
                    print(f"TTS error: {e}")
                finally:
                    t1 = time.perf_counter()
                    duration_ms = int((t1 - t0) * 1000)
                    logger.info(f"TTS finished: {text} ({duration_ms} ms)")
                    print(f"TTS finished: {text} ({duration_ms} ms)")

                    if len(text.strip()) > 10 and duration_ms < 150:
                        logger.warning("WARNING: TTS silent")
                        print("WARNING: TTS silent")

                    self.is_playing = False
                    self.is_speaking = False
                    self.last_speech_end_time = time.time()
                    self.current_priority = None
                    self.current_speech_start_time = 0.0
                    self._notify_beeps_resume()
                    self._signal_done(text)

        finally:
            if self.voice is not None:
                try:
                    del self.voice
                    self.voice = None
                except Exception:
                    pass
            if sys.platform == 'win32':
                try:
                    import pythoncom
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def _signal_done(self, text: str):
        with self._events_lock:
            events = self._done_events.pop(text, [])
        for ev in events:
            ev.set()

    def _play_text(self, text: str):
        """Play text using selected TTS engine."""
        if self.use_gtts and GTTS_AVAILABLE:
            self._play_gtts(text)
        elif self.sapi_available and self.voice is not None:
            self._play_sapi(text)
        elif PYTTSX3_AVAILABLE:
            self._play_pyttsx3_fresh(text)
        else:
            print(f"[AUDIO]: {text}")

    def _play_sapi(self, text: str):
        """Play text asynchronously using SAPI SpVoice with purge and interrupt monitoring."""
        # Flags: SVSFlagsAsync = 1, SVSFPurgeBeforeSpeak = 2
        try:
            self.voice.Speak(text, 1)
            while not self._interrupt_event.is_set():
                if self.voice.WaitUntilDone(50):
                    break
            if self._interrupt_event.is_set():
                # Purge speech immediately
                self.voice.Speak("", 2)
        except Exception as e:
            logger.error(f"SAPI speak error: {e}")
            raise

    def _play_pyttsx3_fresh(self, text: str):
        """Fallback: play text using a fresh pyttsx3 engine per utterance."""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
            del engine
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
            print(f"[AUDIO]: {text}")

    def _play_gtts(self, text: str):
        """Play text using gTTS."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_file = fp.name
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_file)
            if os.name == 'nt':
                os.system(f'start /min "" "{temp_file}"')
            else:
                os.system(f'mpg123 -q "{temp_file}"')
            try:
                os.remove(temp_file)
            except Exception:
                pass
        except Exception as e:
            print(f"[FAIL] gTTS error: {e}")
            print(f"[AUDIO]: {text}")

    def _stop_current_internal(self):
        """Stop current speech output immediately."""
        self._interrupt_event.set()
        if self.sapi_available and self.voice is not None:
            try:
                self.voice.Speak("", 2)  # SVSFPurgeBeforeSpeak = 2
            except Exception:
                pass
        self.is_playing = False
        self.is_speaking = False
        self.last_speech_end_time = time.time()
        self.current_priority = None
        self.current_speech_start_time = 0.0
        self._notify_beeps_resume()

    def _clear_queue(self):
        """Clear all pending messages from queue and release waiters."""
        dropped = []
        with self.audio_queue.mutex:
            while self.audio_queue.queue:
                try:
                    item = self.audio_queue.queue.pop(0)
                    if isinstance(item, tuple) and len(item) >= 2:
                        dropped.append(item[1])
                except IndexError:
                    break
        with self._events_lock:
            for text in dropped:
                for ev in self._done_events.pop(text, []):
                    ev.set()

    def stop(self):
        """Stop playback and clear pending messages."""
        self._clear_queue()
        self._stop_current_internal()
        self.audio_queue.put((0, "__STOP__"))

    def flush_non_emergency(self):
        """Immediately stop non-emergency speech and flush non-emergency items from audio queue."""
        emergency_items = []
        dropped_texts = []
        with self.audio_queue.mutex:
            while self.audio_queue.queue:
                try:
                    item = self.audio_queue.queue.pop(0)
                    if isinstance(item, tuple) and len(item) >= 2:
                        p, t = item[0], item[1]
                        if p == PRIORITY_EMERGENCY and t != "__STOP__":
                            emergency_items.append(item)
                        else:
                            dropped_texts.append(t)
                except IndexError:
                    break
            self.audio_queue.queue.extend(emergency_items)

        with self._events_lock:
            for t in dropped_texts:
                for ev in self._done_events.pop(t, []):
                    ev.set()

        if self.current_priority != PRIORITY_EMERGENCY:
            self._stop_current_internal()

    def get_queue_size(self) -> int:
        """Get number of pending messages."""
        return self.audio_queue.qsize()

    def is_busy(self) -> bool:
        """Check if currently speaking."""
        return bool(self.is_playing or self.is_speaking)

    def is_tts_active(self) -> bool:
        """Return True if TTS is currently speaking or finished within the 0.25s echo suppression window."""
        now = time.time()
        if self.is_speaking or self.is_playing:
            start_t = getattr(self, 'current_speech_start_time', 0.0)
            if start_t > 0 and (now - start_t) > 10.0:
                self.is_speaking = False
                self.is_playing = False
                self.last_speech_end_time = now
                self.current_speech_start_time = 0.0
                return False
            return True
        last_end = getattr(self, 'last_speech_end_time', 0.0)
        return last_end > 0 and (now - last_end) < 0.25


# Singleton instance
_audio_instance = None

def get_audio_feedback(use_gtts: bool = False) -> AudioFeedback:
    """Get singleton AudioFeedback instance."""
    global _audio_instance
    if _audio_instance is None:
        _audio_instance = AudioFeedback(use_gtts=use_gtts)
    return _audio_instance
