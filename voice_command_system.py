"""
Voice Command System for VisionMate
Continuous background wake-word listening, voice-only control, and intent parsing
"""

import speech_recognition as sr
import threading
import queue
import time
import logging
import sys
import math
import collections
import types
from typing import Optional, Callable, Dict, Any, Tuple
import re
import difflib
import socket

import config

logger = logging.getLogger(__name__)

try:
    import stt_engine
except Exception as e:
    logger.warning(f"Could not import stt_engine ({e}). Whistle STT disabled; using Google STT only.")
    stt_engine = None

# Accepted wake phrases moved to config.WAKE_WORDS, with fallback
WAKE_PHRASES = getattr(config, 'WAKE_WORDS', [
    "hey vision",
    "vision",
    "a vision",
    "have a vision",
    "have vision",
    "ek vision",
    "eh vision",
    "hey visions",
    "a visions",
    "he vision",
    "hay vision",
    "hi vision",
    "revision",
    "revisions",
    "division",
    "divisions",
    "hey december",
    "hey decision",
    "a decision",
    "decision",
    "virginia",
    "hey virginia",
    "bharath",
    "hey vis",
    "hey vishal",
    "call vis",
    "hey wisdom",
    "hey vission",
])


def strip_plural(word: str) -> str:
    """
    Strip plural endings for common objects, preserving inherently plural terms:
    - 'cell phones' -> 'cell phone'
    - 'phones' -> 'phone'
    - 'bottles' -> 'bottle'
    - 'chairs' -> 'chair'
    - 'cups' -> 'cup'
    - 'keys' remains 'keys'
    - 'glasses' remains 'glasses'
    """
    if not word:
        return ""
    w = word.strip()
    low = w.lower()
    if low in ("keys", "glasses", "sunglasses", "scissors"):
        return w
    if low == "cell phones":
        return "cell phone"
    if low.endswith("cell phones"):
        return w[:-1]
    if low.endswith("phones"):
        return w[:-1]
    if low.endswith("bottles"):
        return w[:-1]
    if low.endswith("chairs"):
        return w[:-1]
    if low.endswith("cups"):
        return w[:-1]
    if low.endswith("books"):
        return w[:-1]
    if low.endswith("boxes"):
        return w[:-2]
    if low.endswith("ies") and len(low) > 4:
        return w[:-3] + "y"
    if low.endswith("es") and not (low.endswith("ees") or low.endswith("ses") or low.endswith("sses")):
        return w[:-1]
    if low.endswith("s") and not low.endswith("ss") and len(low) > 3:
        return w[:-1]
    return w


def normalise_text(text: str) -> str:
    """
    Normalise recognised speech text BEFORE wake-word matching and parsing:
    - lowercase
    - strip punctuation (! . , ? ' " ; :)
    - collapse multiple spaces to a single space
    """
    if not text:
        return ""
    s = text.lower()
    s = re.sub(r"[!.,?'\";:]+", " ", s)
    return " ".join(s.split()).strip()


def get_wake_regex() -> re.Pattern:
    """Build compiled regex matching configured wake words at the start of text."""
    wake_words = getattr(config, 'WAKE_WORDS', WAKE_PHRASES)
    sorted_phrases = sorted(wake_words, key=lambda p: (len(p.split()), len(p)), reverse=True)
    regex_parts = [r'\s+'.join(re.escape(w) for w in p.split()) for p in sorted_phrases]
    return re.compile(r'^(?:' + '|'.join(regex_parts) + r')\b[,.]?\s*', re.IGNORECASE)


# Regular expression to match wake phrase variations at the start of an utterance
WAKE_REGEX = get_wake_regex()

# Filler words to strip before intent parsing:
# "do you", "can you", "please", "a", "an", "the", "this", "that", "my"
FILLER_WORDS_REGEX = re.compile(
    r'\b(?:do\s+you|can\s+you|please|this|that|the|an|my|a)\b',
    re.IGNORECASE
)

def strip_filler_words(text: str) -> str:
    """
    Strip filler words before intent parsing:
    'do you', 'can you', 'please', 'a', 'an', 'the', 'this', 'that', 'my'.
    """
    if not text:
        return ""
    res = FILLER_WORDS_REGEX.sub(' ', text)
    return " ".join(res.split()).strip()

# Tolerant confirmation word sets
FACE_NAME_HINTS = ["Bharath", "Dhanush"]

def apply_name_hints(name: str) -> str:
    """If a heard name fuzzy-matches a hint (ratio >= 0.6), use the hint's spelling."""
    if not name:
        return name
    name_clean = name.strip()
    name_lower = name_clean.lower()
    hints = getattr(config, 'FACE_NAME_HINTS', FACE_NAME_HINTS)
    best_ratio = 0.0
    best_hint = name_clean
    for hint in hints:
        r = difflib.SequenceMatcher(None, name_lower, hint.lower().strip()).ratio()
        if r >= 0.6 and r > best_ratio:
            best_ratio = r
            best_hint = hint
    return best_hint

YES_WORDS = {
    'yes', 'yeah', 'yep', 'ya', 'yup', 'sure', 'ok', 'okay', 'correct',
    'right', 'please', 'confirm', 'haan'
}
NO_WORDS = {'no', 'nope', 'nah', 'cancel', 'wrong'}

# Shutdown / quit phrases and mishearings
SHUTDOWN_VOCABULARY = [
    "shadow",
    "shut down",
    "shutdown",
    "shut dawn",
    "shut doubt",
    "shut out",
    "shot down",
    "shot up",
    "shutting down",
    "shutting",
    "shut it down",
    "shut it",
    "shuttle",
    "shut",
    "quit",
    "exit",
    "close",
    "stop everything",
]

KNOWN_VERB_ROOTS = {
    'find', 'finding', 'fine', 'fined', 'locate', 'where', 'driven',
    'remember', 'save', 'learn', 'enroll', 'call',
    'forget', 'delete', 'remove', 'erase', 'wipe', 'clear',
    'update', 'reremember', 're-remember',
    'navigate', 'navigating', 'navigation', 'guide', 'go', 'directions',
    'note', 'notes', 'dictate', 'read', 'write', 'create',
    'quit', 'exit', 'close', 'shutdown', 'shut', 'shutting', 'shuttle', 'shot', 'stop', 'cancel',
    'list', 'toggle', 'mute', 'unmute', 'show', 'check', 'advance', 'simulate',
    'obstacle', 'beeps', 'beep',
    'help', 'yes', 'no'
}


def _install_custom_listen(recognizer: sr.Recognizer):
    """
    Configure recognizer to keep 0.4s of audio before detected phrase and 0.3s after,
    with pause_threshold=0.8 and non_speaking_duration=0.5, preventing clipped leading words.
    """
    try:
        import audioop
    except ImportError:
        import pyaudioop as audioop

    def _custom_listen(self, source, timeout=None, phrase_time_limit=None, snowboy_configuration=None, stream=False):
        assert isinstance(source, sr.AudioSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before listening"
        assert self.pause_threshold >= self.non_speaking_duration >= 0

        seconds_per_buffer = float(source.CHUNK) / source.SAMPLE_RATE
        pause_buffer_count = int(math.ceil(self.pause_threshold / seconds_per_buffer))
        phrase_buffer_count = int(math.ceil(self.phrase_threshold / seconds_per_buffer))

        before_dur = getattr(self, 'non_speaking_before', 0.4)
        after_dur = getattr(self, 'non_speaking_after', 0.3)
        before_buffer_count = int(math.ceil(before_dur / seconds_per_buffer))
        after_buffer_count = int(math.ceil(after_dur / seconds_per_buffer))

        elapsed_time = 0
        buffer = b""
        while True:
            frames = collections.deque()

            if snowboy_configuration is None:
                while True:
                    elapsed_time += seconds_per_buffer
                    if timeout and elapsed_time > timeout:
                        raise sr.WaitTimeoutError("listening timed out while waiting for phrase to start")

                    buffer = source.stream.read(source.CHUNK)
                    if len(buffer) == 0:
                        break
                    frames.append(buffer)
                    if len(frames) > before_buffer_count:
                        frames.popleft()

                    energy = audioop.rms(buffer, source.SAMPLE_WIDTH)
                    if energy > self.energy_threshold:
                        break

                    if self.dynamic_energy_threshold:
                        damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer
                        target_energy = energy * self.dynamic_energy_ratio
                        self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)
            else:
                snowboy_location, snowboy_hot_word_files = snowboy_configuration
                buffer, delta_time = self.snowboy_wait_for_hot_word(snowboy_location, snowboy_hot_word_files, source, timeout)
                elapsed_time += delta_time
                if len(buffer) == 0:
                    break
                frames.append(buffer)

            pause_count, phrase_count = 0, 0
            phrase_start_time = elapsed_time

            if stream:
                yield sr.AudioData(b"".join(frames), source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                frames.clear()

            while True:
                elapsed_time += seconds_per_buffer
                if phrase_time_limit and elapsed_time - phrase_start_time > phrase_time_limit:
                    break

                buffer = source.stream.read(source.CHUNK)
                if len(buffer) == 0:
                    break
                frames.append(buffer)
                phrase_count += 1

                energy = audioop.rms(buffer, source.SAMPLE_WIDTH)
                if energy > self.energy_threshold:
                    pause_count = 0
                else:
                    pause_count += 1
                if pause_count > pause_buffer_count:
                    break

                if self.dynamic_energy_threshold:
                    damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer
                    target_energy = energy * self.dynamic_energy_ratio
                    self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)

                if stream:
                    yield sr.AudioData(buffer, source.SAMPLE_RATE, source.SAMPLE_WIDTH)

            phrase_count -= pause_count
            if phrase_count >= phrase_buffer_count or len(buffer) == 0:
                break

        if stream:
            yield sr.AudioData(buffer, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
        else:
            for i in range(pause_count - after_buffer_count):
                if frames:
                    frames.pop()
            frame_data = b"".join(frames)
            yield sr.AudioData(frame_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)

    recognizer._listen = types.MethodType(_custom_listen, recognizer)



def default_ready_beep():
    """Play a short ready beep using winsound (Windows) or terminal bell."""
    try:
        import winsound
        winsound.Beep(1000, 150)
    except Exception:
        sys.stdout.write('\a')
        sys.stdout.flush()


class ButtonHandler:
    """
    Legacy button handler kept for backward compatibility and manual trigger.
    In voice-only mode, continuous wake-word listening replaces physical keys.
    """
    def __init__(self, callback: Callable):
        self.callback = callback
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        logger.info("ButtonHandler ready (manual trigger mode)")

    def stop(self):
        self.running = False

    def trigger(self):
        if self.running:
            self.callback()


class VoiceCapture:
    """
    Captures audio and converts to text using speech recognition.
    """
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.is_listening = False

        # Adjust recognition settings for better accuracy
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Seconds of silence to consider end of phrase
        self.recognizer.non_speaking_duration = 0.5  # Seconds of non-speaking audio parameter
        self.recognizer.non_speaking_before = 0.4  # Keep 0.4s audio before detected phrase
        self.recognizer.non_speaking_after = 0.3   # Keep 0.3s audio after detected phrase
        _install_custom_listen(self.recognizer)

    def initialize_microphone(self) -> bool:
        """Initialize the microphone device."""
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                logger.info("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Microphone initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize microphone: {e}")
            return False

    def listen_and_recognize(self, timeout: int = 5) -> Optional[str]:
        """
        Listen for one-shot voice input and convert to text.
        """
        if not self.microphone:
            logger.warning("Microphone not initialized")
            return None

        try:
            self.is_listening = True
            logger.info("Listening for voice input...")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)

            logger.info("Processing speech...")
            stt_mode = getattr(config, 'STT_ENGINE', 'auto').lower().strip()
            text = None
            whistle_returned_empty = False

            if stt_mode == "hybrid":
                is_online = stt_engine.is_internet_reachable() if stt_engine else True
                fallback_to_whistle = False
                fallback_reason = None
                google_empty = False

                if not is_online:
                    fallback_to_whistle = True
                    fallback_reason = "Internet unreachable"
                elif hasattr(self.recognizer, 'recognize_google'):
                    t0 = time.perf_counter()
                    try:
                        g_text = self.recognizer.recognize_google(audio)
                        g_ms = int((time.perf_counter() - t0) * 1000)
                        logger.info(f"STT[engine=google ms={g_ms}]: '{g_text}'")
                        if g_text and g_text.strip():
                            text = g_text.strip()
                        else:
                            google_empty = True
                    except sr.UnknownValueError:
                        g_ms = int((time.perf_counter() - t0) * 1000)
                        logger.info(f"STT[engine=google ms={g_ms}]: ''")
                        google_empty = True
                    except sr.RequestError as e:
                        g_ms = int((time.perf_counter() - t0) * 1000)
                        logger.info(f"STT[engine=google ms={g_ms}]: ''")
                        fallback_to_whistle = True
                        fallback_reason = f"RequestError ({e})"
                    except (socket.timeout, TimeoutError) as e:
                        g_ms = int((time.perf_counter() - t0) * 1000)
                        logger.info(f"STT[engine=google ms={g_ms}]: ''")
                        fallback_to_whistle = True
                        fallback_reason = f"Timeout ({e})"
                    except Exception as e:
                        g_ms = int((time.perf_counter() - t0) * 1000)
                        logger.info(f"STT[engine=google ms={g_ms}]: ''")
                        err_str = str(e).lower()
                        if "unknownvalue" in err_str:
                            google_empty = True
                        elif any(w in err_str for w in ["network", "timeout", "connection", "request", "socket", "http"]):
                            fallback_to_whistle = True
                            fallback_reason = f"Network/API error ({e})"
                        else:
                            fallback_to_whistle = True
                            fallback_reason = f"API error ({e})"
                else:
                    fallback_to_whistle = True
                    fallback_reason = "Google recognizer unavailable"

                if google_empty:
                    # An empty Google result (no speech recognised) is NOT a failure. Return '' and ignore it.
                    return ""

                if fallback_to_whistle and stt_engine and stt_engine.is_available():
                    logger.info(f"Hybrid STT fallback to Whistle: {fallback_reason}")
                    try:
                        w_text, w_ms = stt_engine.transcribe_with_timing(audio)
                        logger.info(f"STT[engine=whistle ms={w_ms}]: '{w_text}'")
                        if w_text and not stt_engine.is_filler(w_text):
                            text = w_text
                    except Exception as e:
                        logger.warning(f"Whistle STT error: {e}")

            elif stt_mode in ("whistle", "auto") and stt_engine and stt_engine.is_available():
                try:
                    w_text, w_ms = stt_engine.transcribe_with_timing(audio)
                    logger.info(f"STT[engine=whistle ms={w_ms}]: '{w_text}'")
                    if not w_text:
                        whistle_returned_empty = True
                    elif not stt_engine.is_filler(w_text):
                        text = w_text
                except Exception as e:
                    logger.warning(f"Whistle STT error: {e}")
                    whistle_returned_empty = True

            if not text and stt_mode != "hybrid":
                should_try_google = False
                if stt_mode == "google":
                    should_try_google = True
                elif not stt_engine or not stt_engine.is_available():
                    should_try_google = True
                elif stt_mode == "auto":
                    if whistle_returned_empty and stt_engine.is_internet_reachable():
                        should_try_google = True

                if should_try_google and hasattr(self.recognizer, 'recognize_google'):
                    t0 = time.perf_counter()
                    try:
                        g_text = self.recognizer.recognize_google(audio)
                        g_ms = int((time.perf_counter() - t0) * 1000)
                        logger.info(f"STT[engine=google ms={g_ms}]: '{g_text}'")
                        text = g_text
                    except sr.UnknownValueError:
                        logger.warning("Could not understand audio")
                        return None
                    except sr.RequestError as e:
                        logger.error(f"Speech recognition service error: {e}")
                        return None

            if text:
                has_w = False
                wake_words = getattr(config, 'WAKE_WORDS', WAKE_PHRASES)
                norm = normalise_text(text)
                for w in wake_words:
                    if norm.startswith(w):
                        has_w = True
                        break
                if has_w:
                    logger.info(f"Recognized: '{text}'")
                else:
                    logger.debug(f"Recognized: '{text}'")
                return text.lower()
            return None
        except sr.WaitTimeoutError:
            logger.warning("Listening timeout - no speech detected")
            return None
        except Exception as e:
            logger.error(f"Error during voice capture: {e}")
            return None
        finally:
            self.is_listening = False


class VoiceCommandParser:
    """
    Parses voice commands and extracts intent and parameters.
    Supports both wake-phrase prefixed commands and plain commands.
    """
    def __init__(self, known_targets_provider: Optional[Callable[[], list]] = None):
        self.known_targets_provider = known_targets_provider

    # Priority-ordered command patterns
    PATTERNS = {
        'quit': [
            r'\b(?:shutdown|shut\s+down|shadow|shut\s+dawn|shut\s+doubt|shut\s+out|shot\s+down|shot\s+up|shutting\s+down|shutting|shut\s+it\s+down|shut\s+it|shuttle|shut|quit|exit|close|stop\s+everything)\b',
        ],
        'clear_notes': [
            r'^(?:please\s+)?(?:clear|delete|remove|erase|wipe)\s+(?:all\s+)?(?:the\s+|my\s+)?notes?\b(?:\s+please)?$',
        ],
        'delete_note': [
            r'^(?:please\s+)?(?:delete|remove|erase)\s+(?:the\s+)?note\s+(?:number\s+|#\s*)?(\w+)(?:\s+please)?$',
            r'^(?:please\s+)?(?:delete|remove|erase)\s+(?:the\s+)?note(?:\s+please)?$',
        ],
        'read_notes': [
            r'^(?:please\s+)?(?:read|list|check|show|get)\s+(?:out\s+)?(?:all\s+)?(?:the\s+|my\s+)?notes?\b(?:\s+please)?$',
            r'^(?:please\s+)?read\s+(?:out\s+)?(?:the\s+|my\s+)?notes?\b(?:\s+please)?$',
            r'^(?:please\s+)?what\s+are\s+my\s+notes\b(?:\s+please)?$',
        ],
        'take_note': [
            r'^(?:please\s+)?(?:take\s+(?:a\s+)?note|make\s+(?:a\s+)?note|taken\s+out|note\s+this|dictate)(?:\s+please)?(?:\s+(?:that\s+|to\s+)?(.+))?$',
            r'^(?:please\s+)?(?:add|save|record|write|create)\s+(?:a\s+)?note(?:\s+please)?(?:\s+(?:that\s+|to\s+)?(.+))?$',
            r'^note\s+(?:that\s+|to\s+)?(.+)$',
        ],
        'stop_find': [
            r'stop\s+find(?:ing)?\b',
            r'stop\s+search(?:ing)?\b',
            r'cancel\s+search\b',
            r'i\s+found\s+it\b',
        ],
        'stop_gps': [
            r'stop\s+gps\b',
            r'stop\s+gps\s+navigation\b',
            r'end\s+navigation\b',
        ],
        'stop_navigate': [
            r'stop\s+navigation\b',
            r'stop\s+navigating\b',
            r'turn\s+off\s+navigation\b',
        ],
        'forget_face': [
            r'^(?:please\s+)?(?:forget|delete|remove)\s+face\s+(?:my\s+|the\s+|an?\s+)?([\w\s]+?)(?:\s+please)?$',
            r'\b(?:forget|delete|remove)\s+face\s+([\w\s]+)',
        ],
        'rename_face': [
            r'^(?:please\s+)?rename\s+face\s+([\w\s]+?)\s+to\s+([\w\s]+?)(?:\s+please)?$',
            r'\brename\s+face\s+([\w\s]+?)\s+to\s+([\w\s]+)',
        ],
        'forget': [
            r'^(?:please\s+)?(?:forget|delete|remove)\s+(?!notes?\b)(?:my\s+|the\s+|an?\s+)?([\w\s]+?)(?:\s+please)?$',
            r'\b(?:forget|delete|remove)\s+(?!notes?\b)(?:my\s+|the\s+|an?\s+)?([\w\s]+)',
        ],
        'update': [
            r'^(?:please\s+)?update\s+(?:my\s+|the\s+|an?\s+)?([\w\s]+?)(?:\s+please)?$',
            r'^(?:please\s+)?(?:re-?remember|re\s+remember)\s+(?:my\s+|the\s+|an?\s+)?([\w\s]+?)(?:\s+please)?$',
            r'^(?:please\s+)?remember\s+(?:my\s+|the\s+|an?\s+)?([\w\s]+?)\s+again(?:\s+please)?$',
            r'\bupdate\s+(?:my\s+|the\s+|an?\s+)?([\w\s]+)',
            r'\b(?:re-?remember|re\s+remember)\s+(?:my\s+|the\s+|an?\s+)?([\w\s]+)',
            r'\bremember\s+(?:my\s+|the\s+|an?\s+)?([\w\s]+?)\s+again\b',
        ],
        'enroll_face': [
            r'enroll\s+face\s+as\s+([\w\s]+)',
            r'enroll\s+person\s+as\s+([\w\s]+)',
            r'enroll\s+face\s+([\w\s]+)',
            r'enroll\s+person\s+([\w\s]+)',
            r'enroll\s+this\s+person\b',
            r'enroll\s+face\b',
            r'enroll\s+person\b',
        ],
        'remember': [
            r'(?<!re-)(?<!re\s)\bremember\s+(?:(?:the|a|this|that|my)\s+)?(.+?)\s+as\s+([\w\s]+)',
            r'(?<!re-)(?<!re\s)\bremember\s+as\s+([\w\s]+)',
            r'save\s+(?:(?:the|a|this|that|my)\s+)?(.+?)\s+as\s+([\w\s]+)',
            r'save\s+as\s+([\w\s]+)',
            r'call\s+(?:(?:the|a|this|that|my)\s+)?(.+?)\s+as\s+([\w\s]+)',
            r'call\s+it\s+([\w\s]+)',
            r'call\s+this\s+([\w\s]+)',
            r'save\s+this\s+as\s+([\w\s]+)',
            r'learn\s+this\s+as\s+([\w\s]+)',
            r'(?<!re-)(?<!re\s)\bremember\s+(?:(?:the|a|this|that|my)\s+)?(person|me|face|this\s+person)\b',
            r'(?<!re-)(?<!re\s)\bremember\s+this\s+([\w\s]+)',
            r'(?<!re-)(?<!re\s)\bremember\s+my\s+([\w\s]+)',
            r'save\s+this\s+([\w\s]+)',
            r'learn\s+this\s+([\w\s]+)',
            r'enroll\s+this\s+([\w\s]+)',
            r'(?<!re-)(?<!re\s)\bremember\s+this\b',
            r'save\s+this\b',
            r'learn\s+this\b',
            r'enroll\s+this\b',
            r'(?<!re-)(?<!re\s)\bremember\s+([\w\s]+)',
            r'(?<!re-)(?<!re\s)\bremember\b',
        ],
        'where_am_i': [
            r'where\s+am\s+i\b',
            r'my\s+location\b',
            r'where\s+is\s+this\b',
            r'what\s+is\s+my\s+location\b',
            r'current\s+location\b',
        ],
        'find': [
            r'^(?:please\s+)?(?:find\s+the|find\s+a|fine\s+my|driven\s+by|fine|fined|finding|find)\s+(?:my\s+|the\s+|a\s+|an\s+)?([\w\s]+?)(?:\s+please)?$',
            r'\b(?:find\s+the|find\s+a|fine\s+my|driven\s+by|fine|fined|finding|find)\s+(?:my\s+|the\s+|a\s+|an\s+)?([\w\s]+)',
            r'where\s+is\s+(?:my\s+)?([\w\s]+)',
            r'locate\s+(?:my\s+)?([\w\s]+)',
            r'where\s+is\s+it\b',
            r'\b(?:find\s+the|find\s+a|fine\s+my|driven\s+by|fine|fined|finding|find)\s+my\b',
            r'\b(?:find\s+the|find\s+a|fine\s+my|driven\s+by|fine|fined|finding|find)\b',
            r'locate\b',
        ],
        'navigate_to': [
            r'navigate\s+to\s+([\w\s]+)',
            r'take\s+me\s+to\s+([\w\s]+)',
            r'go\s+to\s+([\w\s]+)',
            r'directions\s+to\s+([\w\s]+)',
        ],
        'navigate': [
            r'navigate\b',
            r'start\s+navigation\b',
            r'guide\s+me\b',
            r'help\s+me\s+walk\b',
        ],
        'list': [
            r'list\s+(?:all\s+)?(?:objects|items|things)\b',
            r'list\s+all\b',
            r'what\s+do\s+you\s+remember\b',
            r'show\s+me\s+everything\b',
        ],
        'audio_toggle': [
            r'toggle\s+audio\b',
            r'mute\s+audio\b',
            r'unmute\s+audio\b',
            r'turn\s+audio\s+(?:on|off)\b',
            r'audio\s+(?:on|off)\b',
        ],
        'obstacle_beeps': [
            r'^(?:please\s+)?(?:turn\s+)?obstacle\s+beeps?\s+(on|off)\b(?:\s+please)?$',
            r'\bobstacle\s+beeps?\s+(on|off)\b',
            r'^(?:please\s+)?toggle\s+obstacle\s+beeps?\b(?:\s+please)?$',
        ],
        'advance_gps': [
            r'advance\s+mock\s+gps\b',
            r'advance\s+gps\b',
            r'next\s+gps\b',
            r'simulate\s+movement\b',
        ],
        'database_contents': [
            r'show\s+database\b',
            r'show\s+db\b',
            r'database\s+contents\b',
            r'check\s+database\b',
            r'read\s+database\b',
        ],
        'toggle_voice': [
            r'toggle\s+voice\b',
            r'voice\s+mode\b',
        ],
        'yes': [
            r'\b(?:yes|yeah|yep|ya|yup|sure|ok|okay|correct|right|please|confirm|haan)\b',
        ],
        'no': [
            r'\b(?:no|nope|nah|cancel|wrong)\b',
        ],
        'help': [
            r'help\b',
            r'what\s+can\s+you\s+do\b',
            r'commands\b',
        ],
    }

    @staticmethod
    def strip_wake_word(text: str) -> Tuple[bool, str]:
        """
        Check if text begins with a wake phrase, and return (matched, stripped_remainder).
        Accepts: all phrases in config.WAKE_WORDS including mishearings:
        "have a vision", "have vision", "hey december", "hey decision", "a decision", "decision", "revision", "division", "hey visions".
        Uses regex and fuzzy matching (difflib ratio >= 0.75 on first 3, 2, or 1 words, with trailing noise tolerance).
        Strips the matched phrase and returns remainder.
        """
        if not text:
            return False, ""
        clean = normalise_text(text)
        if not clean:
            return False, ""
        wake_words = getattr(config, 'WAKE_WORDS', WAKE_PHRASES)

        TRAILING_NOISE = {'uh', 'um', 'ah', 'er', 'mm', 'hmm', 'ha', 'eh', 'oh', 'noise', 'a', 'the'}

        def _clean_remainder(rem: str) -> str:
            n_rem = normalise_text(rem)
            if not n_rem:
                return ""
            r_words = n_rem.split()
            # If all remainder words are trailing noise, treat as wake word alone
            if all(w in TRAILING_NOISE for w in r_words):
                return ""
            # Strip leading noise words
            while r_words and r_words[0] in TRAILING_NOISE:
                r_words.pop(0)
            return " ".join(r_words)

        wake_regex = get_wake_regex()
        match = wake_regex.match(clean)
        if match:
            remainder = clean[match.end():].strip().lstrip(',.- ')
            return True, _clean_remainder(remainder)

        words = clean.split()
        if not words:
            return False, clean

        best_ratio = 0.0
        matched_word_count = 0

        # Check 3 words against wake phrases
        if len(words) >= 3:
            first_three = f"{words[0]} {words[1]} {words[2]}".lower()
            for p in wake_words:
                p_words = p.lower().split()
                if len(p_words) == 3:
                    r = difflib.SequenceMatcher(None, first_three, p.lower()).ratio()
                    if r > best_ratio:
                        best_ratio = r
                        matched_word_count = 3
                elif len(p_words) == 2 and words[2].lower() in TRAILING_NOISE:
                    first_two = f"{words[0]} {words[1]}".lower()
                    r = difflib.SequenceMatcher(None, first_two, p.lower()).ratio()
                    if r > best_ratio:
                        best_ratio = r
                        matched_word_count = 3

        # Check 2 words against wake phrases
        if len(words) >= 2:
            first_two = f"{words[0]} {words[1]}".lower()
            for p in wake_words:
                if len(p.split()) == 2:
                    r = difflib.SequenceMatcher(None, first_two, p.lower()).ratio()
                    if r > best_ratio:
                        best_ratio = r
                        matched_word_count = 2

        # Check 1 word against 1-word wake phrases
        first_one = words[0].lower()
        for p in wake_words:
            if len(p.split()) == 1:
                r = difflib.SequenceMatcher(None, first_one, p.lower()).ratio()
                if r > best_ratio:
                    best_ratio = r
                    matched_word_count = 1

        if best_ratio >= 0.75:
            if matched_word_count == 3:
                rem = " ".join(words[3:])
                return True, _clean_remainder(rem)
            elif matched_word_count == 2:
                m = re.match(r'^\s*\S+\s+\S+[,.]?\s*', clean)
                rem = clean[m.end():].strip().lstrip(',.- ') if m else " ".join(words[2:])
                return True, _clean_remainder(rem)
            else:
                m = re.match(r'^\s*\S+[,.]?\s*', clean)
                rem = clean[m.end():].strip().lstrip(',.- ') if m else " ".join(words[1:])
                return True, _clean_remainder(rem)

        return False, clean

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse voice command text and extract intent and parameters.
        Runs matching on the text AFTER any wake phrase is stripped.
        """
        if not text:
            return {'intent': 'unknown', 'parameters': {}}

        clean_text = normalise_text(text)
        if not clean_text:
            return {'intent': 'unknown', 'parameters': {}}

        # Check if wake word is present
        has_wake, remainder = self.strip_wake_word(clean_text)

        if has_wake and not remainder:
            # Utterance was ONLY the wake word
            logger.debug(f"Matching text: '{clean_text}' -> Matched intent: 'wake_word_only' (wake_word=True)")
            return {
                'intent': 'wake_word_only',
                'parameters': {},
                'wake_word': True,
                'raw_text': clean_text
            }

        # Text to parse for intent runs on text AFTER the wake phrase is stripped
        command_target = remainder if has_wake else clean_text
        logger.debug(f"Matching text: '{command_target}' (wake_word={has_wake})")

        # Check shutdown vocabulary after wake word
        if has_wake:
            cmd_phrase = command_target.strip().lower()
            cmd_words = cmd_phrase.split()
            for s_word in SHUTDOWN_VOCABULARY:
                s_lower = s_word.lower()
                thresh = 0.75 if (s_lower in ('shuttle', 'shut', 'close') or len(s_lower) <= 5) else 0.65
                if cmd_phrase == s_lower or difflib.SequenceMatcher(None, cmd_phrase, s_lower).ratio() >= thresh:
                    logger.info(f"Shutdown phrase match: '{cmd_phrase}' matched '{s_word}'")
                    return {
                        'intent': 'quit',
                        'parameters': {},
                        'wake_word': True,
                        'raw_text': clean_text
                    }
            if len(cmd_words) == 1:
                w = cmd_words[0].lower()
                for s_word in SHUTDOWN_VOCABULARY:
                    s_lower = s_word.lower()
                    thresh = 0.75 if (s_lower in ('shuttle', 'shut', 'close') or len(s_lower) <= 5) else 0.65
                    if difflib.SequenceMatcher(None, w, s_lower).ratio() >= thresh:
                        logger.info(f"Shutdown single-word fuzzy match: '{w}' matched '{s_word}' (ratio >= {thresh})")
                        return {
                            'intent': 'quit',
                            'parameters': {},
                            'wake_word': True,
                            'raw_text': clean_text
                        }

        # Strip filler words before intent parsing:

        # "do you", "can you", "please", "a", "an", "the", "this", "that", "my"
        parsed_target = strip_filler_words(command_target)
        if not parsed_target:
            parsed_target = command_target

        for intent, patterns in self.PATTERNS.items():
            # For yes/no, ensure no conflict where both yes and no words are present
            if intent in ['yes', 'no']:
                words = set(re.findall(r'\b\w+\b', command_target.lower()))
                if (words & YES_WORDS) and (words & NO_WORDS):
                    continue
                if intent == 'yes' and not (words & YES_WORDS):
                    continue
                if intent == 'no' and not (words & NO_WORDS):
                    continue

            for pattern in patterns:
                match = re.search(pattern, parsed_target, re.IGNORECASE)
                if not match and parsed_target != command_target:
                    match = re.search(pattern, command_target, re.IGNORECASE)
                if match:
                    params = {}
                    if intent == 'remember':
                        params['raw_command'] = command_target
                        if len(match.groups()) >= 2 and match.group(2):
                            raw_thing = match.group(1).strip()
                            raw_name = match.group(2).strip()
                            raw_thing = re.sub(r'\s+please$', '', raw_thing, flags=re.IGNORECASE).strip()
                            raw_thing = re.sub(r'[.?!]+$', '', raw_thing).strip()
                            raw_name = re.sub(r'\s+please$', '', raw_name, flags=re.IGNORECASE).strip()
                            raw_name = re.sub(r'[.?!]+$', '', raw_name).strip()
                            raw_thing = re.sub(r'^(?:(?:this|that|the|my|a|an)\s+)+', '', raw_thing, flags=re.IGNORECASE).strip()
                            raw_thing = strip_plural(raw_thing)
                            if raw_thing.lower() in ('this', 'it', 'item', ''):
                                thing = 'this'
                            else:
                                thing = raw_thing
                            name = apply_name_hints(raw_name)
                            params['thing'] = thing
                            params['name'] = name
                            params['object_name'] = thing
                            params['alias'] = name
                        elif re.search(r'\b(?:as|call)\b', pattern):
                            raw_val = match.group(1).strip() if (match.groups() and match.group(1)) else ''
                            raw_val = re.sub(r'\s+please$', '', raw_val, flags=re.IGNORECASE).strip()
                            raw_val = re.sub(r'[.?!]+$', '', raw_val).strip()
                            name = apply_name_hints(raw_val)
                            params['thing'] = 'this'
                            params['name'] = name
                            params['object_name'] = 'this'
                            params['alias'] = name
                        else:
                            if match.groups() and match.group(1):
                                raw_val = match.group(1).strip()
                                raw_val = re.sub(r'\s+please$', '', raw_val, flags=re.IGNORECASE).strip()
                                raw_val = re.sub(r'[.?!]+$', '', raw_val).strip()
                                raw_val = re.sub(r'^(?:(?:this|that|the|my|a|an)\s+)+', '', raw_val, flags=re.IGNORECASE).strip()
                                raw_val = strip_plural(raw_val)
                                if raw_val.lower() in ('this', 'it', 'item'):
                                    raw_val = 'this'
                                params['thing'] = raw_val
                                params['name'] = ''
                                params['object_name'] = raw_val
                                params['alias'] = ''
                            else:
                                params['thing'] = ''
                                params['name'] = ''
                                params['object_name'] = ''
                                params['alias'] = ''
                    elif intent == 'forget_face':
                        if match.groups() and match.group(1):
                            raw_val = match.group(1).strip()
                            raw_val = re.sub(r'\s+please$', '', raw_val, flags=re.IGNORECASE).strip()
                            raw_val = re.sub(r'[.?!]+$', '', raw_val).strip()
                            params['face_name'] = raw_val
                            params['name'] = raw_val
                            params['object_name'] = raw_val
                        else:
                            params['face_name'] = ''
                            params['name'] = ''
                            params['object_name'] = ''
                    elif intent == 'rename_face':
                        if len(match.groups()) >= 2 and match.group(2):
                            old_val = match.group(1).strip()
                            new_val = match.group(2).strip()
                            old_val = re.sub(r'\s+please$', '', old_val, flags=re.IGNORECASE).strip()
                            old_val = re.sub(r'[.?!]+$', '', old_val).strip()
                            new_val = re.sub(r'\s+please$', '', new_val, flags=re.IGNORECASE).strip()
                            new_val = re.sub(r'[.?!]+$', '', new_val).strip()
                            params['old_name'] = old_val
                            params['new_name'] = new_val
                    elif intent in ('forget', 'update'):
                        if match.groups() and match.group(1):
                            raw_val = match.group(1).strip()
                            raw_val = re.sub(r'\s+please$', '', raw_val, flags=re.IGNORECASE).strip()
                            raw_val = re.sub(r'[.?!]+$', '', raw_val).strip()
                            params['object_name'] = raw_val
                        else:
                            params['object_name'] = ''
                    elif intent == 'find':
                        if match.groups() and match.group(1):
                            raw_val = match.group(1).strip()
                            raw_val = re.sub(r'\s+please$', '', raw_val, flags=re.IGNORECASE).strip()
                            raw_val = re.sub(r'[.?!]+$', '', raw_val).strip()
                            raw_val = re.sub(r'^(?:(?:my|the|a|an)\s+)+', '', raw_val, flags=re.IGNORECASE).strip()
                            raw_val = strip_plural(raw_val)
                            if raw_val.lower() == 'please':
                                raw_val = ''
                            params['object_name'] = raw_val
                        else:
                            params['object_name'] = ''
                    elif intent == 'where_am_i':
                        params = {}
                    elif intent == 'enroll_face':
                        if match.groups() and match.group(1):
                            raw_name = match.group(1).strip()
                            raw_name = re.sub(r'\s+please$', '', raw_name, flags=re.IGNORECASE).strip()
                            raw_name = re.sub(r'[.?!]+$', '', raw_name).strip()
                            if raw_name.lower().startswith('as '):
                                raw_name = raw_name[3:].strip()
                            if raw_name.lower() in ('person', 'me', 'this person', 'face'):
                                raw_name = ''
                            name = apply_name_hints(raw_name) if raw_name else ''
                            params['thing'] = 'person'
                            params['name'] = name
                            params['object_name'] = 'person'
                            params['alias'] = name
                        else:
                            params['thing'] = 'person'
                            params['name'] = ''
                            params['object_name'] = 'person'
                            params['alias'] = ''
                    elif intent == 'take_note':
                        if match.groups() and match.group(1):
                            raw_val = match.group(1).strip()
                            raw_val = re.sub(r'\s+please$', '', raw_val, flags=re.IGNORECASE).strip()
                            params['text'] = raw_val
                        else:
                            params['text'] = ''
                    elif intent == 'delete_note':
                        if match.groups() and match.group(1):
                            params['note_id'] = match.group(1).strip()
                        else:
                            params['note_id'] = ''
                    elif intent == 'navigate_to':
                        if match.groups() and match.group(1):
                            params['destination'] = match.group(1).strip()
                    elif intent == 'obstacle_beeps':
                        if match.groups() and match.group(1):
                            params['state'] = match.group(1).lower().strip()
                        else:
                            params['state'] = 'toggle'

                    logger.debug(f"Matching text: '{command_target}' -> Matched intent: '{intent}' (params: {params})")
                    logger.info(f"Parsed command - Intent: {intent}, Params: {params} (wake_word={has_wake})")
                    return {
                        'intent': intent,
                        'parameters': params,
                        'wake_word': has_wake,
                        'raw_text': clean_text
                    }

        # 4. Unparsed command fallback:
        # if the text after the wake word contains no known verb but fuzzy-matches (ratio >= 0.8) a remembered object,
        # alias or class name, treat it as "find <object>" and say "Looking for your <object>".
        # NEVER fires for words close to a shutdown/quit phrase.
        words = set(re.findall(r'\b\w+\b', command_target.lower()))
        has_known_verb = bool(words & KNOWN_VERB_ROOTS) or any(
            v in command_target.lower() for v in ('shut down', 'driven by', 'stop find', 'stop gps', 'stop nav', 'shot up', 'shot down', 'shutting', 'shut it', 'shuttle')
        )
        if not has_known_verb:
            clean_q = re.sub(r'^(?:(?:my|the|a|an|this|that)\s+)+', '', command_target.strip(), flags=re.IGNORECASE).strip().lower()
            clean_q = re.sub(r'[.?!]+$', '', clean_q).strip()

            # Guard: never fire unparsed fallback for words close to a shutdown/quit phrase
            is_close_to_shutdown = False
            for s_phrase in SHUTDOWN_VOCABULARY:
                s_low = s_phrase.lower()
                thresh = 0.75 if (s_low in ('shuttle', 'shut', 'close') or len(s_low) <= 5) else 0.70
                if clean_q == s_low or difflib.SequenceMatcher(None, clean_q, s_low).ratio() >= thresh:
                    is_close_to_shutdown = True
                    break
                for qw in clean_q.split():
                    for sw in s_low.split():
                        w_thresh = 0.75 if (sw in ('shuttle', 'shut', 'close') or len(sw) <= 5) else 0.70
                        if difflib.SequenceMatcher(None, qw, sw).ratio() >= w_thresh:
                            is_close_to_shutdown = True
                            break
                    if is_close_to_shutdown:
                        break
                if is_close_to_shutdown:
                    break

            if not is_close_to_shutdown and clean_q:
                # Unparsed fallback only matches remembered object names or known class names,
                # and NEVER when nothing is enrolled.
                p_targets = []
                if hasattr(self, 'known_targets_provider') and callable(self.known_targets_provider):
                    try:
                        p_targets = self.known_targets_provider() or []
                    except Exception as e:
                        logger.debug(f"Error fetching known targets from provider: {e}")

                candidates = set()
                if p_targets:
                    candidates.update(p_targets)
                    target_classes = getattr(config, 'TARGET_CLASSES', [])
                    for c in target_classes:
                        candidates.add(c)

                def _calc_sim(q: str, c: str) -> float:
                    if not q or not c:
                        return 0.0
                    if q == c:
                        return 1.0
                    r1 = difflib.SequenceMatcher(None, q, c).ratio()
                    q_ns = re.sub(r'\s+', '', q)
                    c_ns = re.sub(r'\s+', '', c)
                    r2 = difflib.SequenceMatcher(None, q_ns, c_ns).ratio()
                    return max(r1, r2)

                if candidates:
                    best_ratio = 0.0
                    best_candidate = None
                    for cand in candidates:
                        c_clean = cand.strip().lower()
                        r = _calc_sim(clean_q, c_clean)
                        if r > best_ratio:
                            best_ratio = r
                            best_candidate = cand

                    if best_ratio >= 0.8 and best_candidate:
                        logger.info(
                            f"Unparsed command fallback: '{command_target}' contains no verb, "
                            f"fuzzy-matched '{best_candidate}' (ratio {best_ratio:.2f} >= 0.8) -> treat as 'find {best_candidate}'"
                        )
                        return {
                            'intent': 'find',
                            'parameters': {
                                'object_name': best_candidate,
                                'fallback_phrase': True
                            },
                            'wake_word': has_wake,
                            'raw_text': clean_text
                        }


        logger.debug(f"Matching text: '{command_target}' -> Matched intent: 'unknown'")
        logger.warning(f"Could not parse command: '{command_target}'")
        return {
            'intent': 'unknown',
            'parameters': {'raw_text': command_target},
            'wake_word': has_wake
        }

    def get_help_text(self) -> str:
        """Get help text describing available voice-only commands."""
        return """
Voice-Only Commands (Say 'Hey Vision' first, or speak command after ready beep):
- "Remember this [object]" - Enroll object currently in camera view
- "Update [object]" / "Re-remember [object]" / "Remember [object] again" - Re-enroll an object from camera view
- "Forget [object]" / "Delete [object]" / "Remove [object]" - Remove a remembered object
- "Find my [object]" - Search and guide to an object ("stop finding" to cancel)
- "Enroll face [name]" - Enroll face currently in camera view
- "Take note [text]" - Save a voice note (automatically read back)
- "Read notes" - Read all saved voice notes
- "Delete note [number]" - Delete a specific voice note
- "Clear notes" - Delete all voice notes
- "Navigate" / "Stop navigation" - Toggle obstacle navigation
- "Navigate to [destination]" / "Stop GPS" - GPS walking directions
- "Advance GPS" - Advance mock GPS movement
- "Toggle audio" - Turn audio announcements on/off
- "List all items" - List all remembered objects
- "Show database" - Summarize database contents
- "Shut down" - Quit VisionMate (shuts down immediately)
- "Help" - Speak available commands

Confirmations: When asked a question (e.g., 'Is this your phone?'),
simply reply 'yes' or 'no' within 6 seconds without saying the wake word.
"""


class VoiceCommandSystem:
    """
    Main voice command system.
    Features:
    - Continuous background listening with listen_in_background
    - Wake word detection ("Hey Vision")
    - 6-second window when wake word spoken alone
    - 6-second confirmation window (Yes/No) without wake word
    - Echo suppression integration with AudioFeedback
    - Offline error handling with 30s throttled warning
    """
    def __init__(self, audio_feedback=None, beep_func=None):
        self.voice_capture = VoiceCapture()
        self.parser = VoiceCommandParser()
        self.button_handler = ButtonHandler(self._on_button_press)
        self.audio_feedback = audio_feedback
        self.beep_func = beep_func or default_ready_beep
        self.command_callbacks: Dict[str, Callable] = {}
        self.command_queue = queue.Queue()
        self.enabled = False
        self.running = False

        # Confirmation (Yes/No) window
        self.yes_no_active = False
        self.yes_no_until = 0.0
        self.on_yes_callback: Optional[Callable] = None
        self.on_no_callback: Optional[Callable] = None
        self.on_timeout_callback: Optional[Callable] = None
        self._yes_no_timer: Optional[threading.Timer] = None

        # Dictation window (10 seconds for voice notes)
        self.dictation_active = False
        self.dictation_until = 0.0
        self.on_dictate_callback: Optional[Callable[[str], None]] = None
        self._dictation_timer: Optional[threading.Timer] = None

        # Default handlers
        self.command_callbacks['take_note'] = self._default_handle_take_note
        self.command_callbacks['quit'] = self._default_handle_quit

        # Wake-word alone follow-up window (up to 6s)
        self.awaiting_command_until = 0.0

        # Offline tracking
        self.is_offline = False
        self.offline_announced = False
        self.last_offline_time = 0.0

        # Wake-word tracking and quiet mode
        self.last_wake_word_time = 0.0

        # Background listener stopper
        self._stop_listening = None

    def initialize(self) -> bool:
        """Initialize microphone and components."""
        logger.info("Initializing Voice Command System...")
        mic_ok = self.voice_capture.initialize_microphone()
        if not mic_ok:
            logger.warning("Microphone not available; running in simulation/fallback mode")
            return False
        logger.info("Voice Command System initialized successfully")
        return True

    def start(self):
        """Start continuous background listening."""
        self.enabled = True
        self.running = True
        if self.button_handler:
            self.button_handler.start()

        if self.voice_capture.microphone:
            try:
                self._stop_listening = self.voice_capture.recognizer.listen_in_background(
                    self.voice_capture.microphone,
                    self._bg_audio_callback,
                    phrase_time_limit=8
                )
                logger.info("Continuous background listening started with speech_recognition")
            except Exception as e:
                logger.error(f"Failed to start listen_in_background: {e}")
        else:
            logger.warning("VoiceCommandSystem started without active microphone")

    def stop(self):
        """Stop background listening."""
        self.enabled = False
        self.running = False
        self.close_yes_no_window()
        self.close_dictation_window()
        if self._stop_listening:
            try:
                self._stop_listening(wait_for_stop=False)
            except Exception:
                pass
            self._stop_listening = None
        if self.button_handler:
            self.button_handler.stop()
        logger.info("Voice Command System stopped")

    def register_command_handler(self, intent: str, callback: Callable):
        """Register a callback function for a specific command intent."""
        self.command_callbacks[intent] = callback
        logger.info(f"Registered voice handler for intent: {intent}")

    def open_yes_no_window(self, on_yes: Callable, on_no: Optional[Callable] = None, timeout: float = 10.0, on_timeout: Optional[Callable] = None):
        """
        Open a 10-second window where 'yes/correct/yeah/haan/...' and 'no/wrong/nope/cancel/...'
        trigger handlers directly without requiring the wake word.
        Timer starts AFTER the TTS finishes speaking.
        Does not open if a dictation window is currently active.
        """
        if self.dictation_active:
            logger.info("Cannot open yes/no window: dictation window is currently active")
            return

        if self._yes_no_timer:
            try:
                self._yes_no_timer.cancel()
            except Exception:
                pass
            self._yes_no_timer = None

        self.yes_no_active = True
        self.on_yes_callback = on_yes
        self.on_no_callback = on_no
        self.on_timeout_callback = on_timeout

        def _start_timer_after_tts():
            # Wait for TTS to finish speaking and clear 0.5s echo suppression
            # Give TTS up to 0.1s to register speaking if just queued
            time.sleep(0.05)
            while self._is_tts_active() and self.yes_no_active:
                time.sleep(0.05)
            if not self.yes_no_active:
                return
            self.yes_no_until = time.time() + timeout
            self._yes_no_timer = threading.Timer(timeout, self._on_yes_no_timeout)
            self._yes_no_timer.daemon = True
            self._yes_no_timer.start()
            logger.info(f"Opened {timeout}s yes/no confirmation window (after TTS completed)")

        # Temporary buffer until TTS completes and thread starts the official timer
        self.yes_no_until = time.time() + timeout + 5.0
        threading.Thread(target=_start_timer_after_tts, daemon=True).start()

    def close_yes_no_window(self):
        """Close the yes/no confirmation window."""
        if self._yes_no_timer:
            try:
                self._yes_no_timer.cancel()
            except Exception:
                pass
            self._yes_no_timer = None
        self.yes_no_active = False
        self.yes_no_until = 0.0
        self.on_yes_callback = None
        self.on_no_callback = None
        self.on_timeout_callback = None

    def _on_yes_no_timeout(self):
        """Handle yes/no window timeout by invoking on_timeout, announcing 'Cancelled', and closing window."""
        if not self.yes_no_active:
            return
        logger.info("Yes/No window timed out. Announcing 'Cancelled'.")
        cb = self.on_timeout_callback
        self.close_yes_no_window()
        if cb:
            try:
                cb()
            except Exception as e:
                logger.error(f"Error in on_timeout callback: {e}")
        if self.audio_feedback:
            try:
                if hasattr(self.audio_feedback, 'speak_social'):
                    self.audio_feedback.speak_social("Cancelled")
                elif hasattr(self.audio_feedback, 'speak_status'):
                    self.audio_feedback.speak_status("Cancelled")
            except Exception as e:
                logger.error(f"Error speaking Cancelled on timeout: {e}")

    def open_dictation_window(self, on_dictate: Callable[[str], None], timeout: float = 8.0):
        """
        Open an 8-second dictation window.
        The next utterance, WITHOUT a wake word, is captured as the note text.
        """
        if self._dictation_timer:
            try:
                self._dictation_timer.cancel()
            except Exception:
                pass
            self._dictation_timer = None

        self.dictation_active = True
        self.on_dictate_callback = on_dictate
        logger.info("Listening for dictation...")

        def _start_dictation_timer():
            time.sleep(0.05)
            while self._is_tts_active() and self.dictation_active:
                time.sleep(0.05)
            if not self.dictation_active:
                return
            self.dictation_until = time.time() + timeout
            self._dictation_timer = threading.Timer(timeout, self._on_dictation_timeout)
            self._dictation_timer.daemon = True
            self._dictation_timer.start()
            logger.info(f"Opened {timeout}s dictation window")

        self.dictation_until = time.time() + timeout + 5.0
        threading.Thread(target=_start_dictation_timer, daemon=True).start()

    def close_dictation_window(self, recognized_text: Optional[str] = None):
        """Close the dictation window."""
        if not self.dictation_active:
            return
        if self._dictation_timer:
            try:
                self._dictation_timer.cancel()
            except Exception:
                pass
            self._dictation_timer = None
        self.dictation_active = False
        self.dictation_until = 0.0
        self.on_dictate_callback = None
        if recognized_text:
            logger.info(f"Recognized text: {recognized_text}")
        else:
            logger.info("no speech detected")

    def _on_dictation_timeout(self):
        """Handle dictation window timeout by announcing 'I didn\\'t catch that'."""
        if not self.dictation_active:
            return
        logger.info("Dictation window timed out. Announcing \"I didn't catch that\".")
        self.close_dictation_window(recognized_text=None)
        if self.audio_feedback:
            try:
                if hasattr(self.audio_feedback, 'speak_social'):
                    self.audio_feedback.speak_social("I didn't catch that")
                elif hasattr(self.audio_feedback, 'speak_status'):
                    self.audio_feedback.speak_status("I didn't catch that")
            except Exception as e:
                logger.error(f"Error speaking dictation timeout message: {e}")

    def open_disambiguation_window(self, on_choice: Callable[[str], None], timeout: float = 8.0):
        """
        Open an 8-second window for object disambiguation without requiring a wake word.
        The next utterance is captured and passed to on_choice.
        """
        self.open_dictation_window(on_dictate=on_choice, timeout=timeout)

    def _default_handle_take_note(self, params: Dict[str, Any]):
        """Default handler for take_note intent with empty text dictation flow."""
        text = params.get('text', '').strip()
        if not text:
            if self.audio_feedback:
                if hasattr(self.audio_feedback, 'speak_social'):
                    self.audio_feedback.speak_social("Okay, speak your note")
                elif hasattr(self.audio_feedback, 'speak_status'):
                    self.audio_feedback.speak_status("Okay, speak your note")

            def _on_dictate(dictated_text: str):
                dictated_text = (dictated_text or '').strip()
                if dictated_text:
                    try:
                        import speech_notes
                        speech_notes.take_note(dictated_text)
                    except Exception as e:
                        logger.error(f"Error saving dictated note: {e}")
                    if self.audio_feedback:
                        msg = f"Saved. You said: {dictated_text}"
                        if hasattr(self.audio_feedback, 'speak_social'):
                            self.audio_feedback.speak_social(msg)
                        elif hasattr(self.audio_feedback, 'speak_status'):
                            self.audio_feedback.speak_status(msg)

            self.open_dictation_window(_on_dictate, timeout=8.0)
            return

        # Inline text provided ("take note buy milk")
        try:
            import speech_notes
            speech_notes.take_note(text)
        except Exception as e:
            logger.error(f"Error saving note: {e}")
        if self.audio_feedback:
            msg = f"Saved. You said: {text}"
            if hasattr(self.audio_feedback, 'speak_social'):
                self.audio_feedback.speak_social(msg)
            elif hasattr(self.audio_feedback, 'speak_status'):
                self.audio_feedback.speak_status(msg)

    def _default_handle_quit(self, params: Dict[str, Any]):
        """Default handler for quit/shutdown intent."""
        require_confirm = False
        try:
            import config
            require_confirm = getattr(config, 'REQUIRE_SHUTDOWN_CONFIRM', False)
        except Exception:
            pass

        if require_confirm:
            if self.audio_feedback and hasattr(self.audio_feedback, 'speak_social'):
                self.audio_feedback.speak_social("Say yes to confirm")
            self.open_yes_no_window(
                on_yes=self.stop,
                on_no=lambda: self.audio_feedback.speak_social("Shutdown cancelled") if self.audio_feedback and hasattr(self.audio_feedback, 'speak_social') else None,
                timeout=10.0
            )
        else:
            if self.audio_feedback and hasattr(self.audio_feedback, 'speak_social'):
                self.audio_feedback.speak_social("Shutting down")
            self.stop()

    def _on_wake_word_detected(self):
        """Immediately stop current speech and flush non-emergency items from audio queue."""
        self.last_wake_word_time = time.time()
        if self.audio_feedback:
            try:
                if hasattr(self.audio_feedback, 'flush_non_emergency'):
                    self.audio_feedback.flush_non_emergency()
                elif hasattr(self.audio_feedback, 'stop'):
                    self.audio_feedback.stop()
            except Exception as e:
                logger.error(f"Error flushing audio on wake word: {e}")

    def is_quiet_mode(self) -> bool:
        """
        Return True if quiet mode is active:
        - A dictation window is open
        - A yes/no confirmation window is open
        - Within 3 seconds after a wake word is detected
        """
        now = time.time()
        if getattr(self, 'dictation_active', False):
            return True
        if getattr(self, 'yes_no_active', False):
            return True
        if (now - getattr(self, 'last_wake_word_time', 0.0)) < 3.0:
            return True
        return False

    def _get_tts_active_status(self) -> tuple[bool, str]:
        """
        Check if audio feedback is currently speaking or in 0.25s post-speech mute.
        Returns (is_active: bool, reason: str).
        Audio is dropped only while TTS is actually speaking or within 0.25s post-speech mute.
        """
        if not self.audio_feedback:
            return False, ""

        now = time.time()
        is_spk = bool(getattr(self.audio_feedback, 'is_speaking', False))
        is_ply = bool(getattr(self.audio_feedback, 'is_playing', False))

        if is_spk or is_ply:
            # Guard against stuck TTS flag (e.g. hang longer than 10s)
            start_t = getattr(self.audio_feedback, 'current_speech_start_time', 0.0)
            if start_t > 0 and (now - start_t) > 10.0:
                self.audio_feedback.is_speaking = False
                self.audio_feedback.is_playing = False
                self.audio_feedback.current_speech_start_time = 0.0
                return False, ""
            return True, f"TTS speaking (is_speaking={is_spk}, is_playing={is_ply})"

        last_end = getattr(self.audio_feedback, 'last_speech_end_time', 0.0)
        if last_end > 0:
            elapsed = now - last_end
            if 0 <= elapsed < 0.25:
                return True, f"post-speech mute ({elapsed:.2f}s since speech ended)"

        return False, ""

    def _is_tts_active(self) -> bool:
        """Check if audio feedback is currently speaking or in 0.25s echo suppression window."""
        active, _ = self._get_tts_active_status()
        return active

    def _recognize_audio(self, recognizer: sr.Recognizer, audio: sr.AudioData, window_open: bool = False) -> Optional[str]:
        """
        Recognize audio using Cactus Whistle and/or Google STT according to config.STT_ENGINE.
        Modes:
          - whistle: Whistle only, fully offline.
          - google: Google STT only (current behaviour).
          - auto: Whistle first, fallback to Google only if Whistle returns empty AND internet is reachable.
        Logs every recognition as: STT[engine=... ms=...]: 'text'.
        Rejects filler outputs ("thank you", "you", ".") on non-speech unless wake word or open window is active.
        """
        stt_mode = getattr(config, 'STT_ENGINE', 'auto').lower().strip()
        text = None
        whistle_returned_empty = False

        # 0. Hybrid STT (Google first when online, fallback to Whistle only on network or API errors)
        if stt_mode == "hybrid":
            is_online = stt_engine.is_internet_reachable() if stt_engine else True
            fallback_to_whistle = False
            fallback_reason = None
            google_empty = False

            if not is_online:
                fallback_to_whistle = True
                fallback_reason = "Internet unreachable"
            elif hasattr(recognizer, 'recognize_google'):
                t0 = time.perf_counter()
                try:
                    g_text = recognizer.recognize_google(audio)
                    g_ms = int((time.perf_counter() - t0) * 1000)
                    logger.info(f"STT[engine=google ms={g_ms}]: '{g_text}'")
                    if g_text and g_text.strip():
                        text = normalise_text(g_text)
                    else:
                        google_empty = True
                except sr.UnknownValueError:
                    g_ms = int((time.perf_counter() - t0) * 1000)
                    logger.info(f"STT[engine=google ms={g_ms}]: ''")
                    google_empty = True
                except sr.RequestError as e:
                    g_ms = int((time.perf_counter() - t0) * 1000)
                    logger.info(f"STT[engine=google ms={g_ms}]: ''")
                    fallback_to_whistle = True
                    fallback_reason = f"RequestError ({e})"
                except (socket.timeout, TimeoutError) as e:
                    g_ms = int((time.perf_counter() - t0) * 1000)
                    logger.info(f"STT[engine=google ms={g_ms}]: ''")
                    fallback_to_whistle = True
                    fallback_reason = f"Timeout ({e})"
                except Exception as e:
                    g_ms = int((time.perf_counter() - t0) * 1000)
                    logger.info(f"STT[engine=google ms={g_ms}]: ''")
                    err_str = str(e).lower()
                    if "unknownvalue" in err_str:
                        google_empty = True
                    elif any(w in err_str for w in ["network", "timeout", "connection", "request", "socket", "http"]):
                        fallback_to_whistle = True
                        fallback_reason = f"Network/API error ({e})"
                    else:
                        fallback_to_whistle = True
                        fallback_reason = f"API error ({e})"
            else:
                fallback_to_whistle = True
                fallback_reason = "Google recognizer unavailable"

            if google_empty:
                # An empty Google result (no speech recognised) is NOT a failure. Return '' and ignore it.
                return ""

            if fallback_to_whistle and stt_engine and stt_engine.is_available():
                logger.info(f"Hybrid STT fallback to Whistle: {fallback_reason}")
                try:
                    w_text, w_ms = stt_engine.transcribe_with_timing(audio)
                    logger.info(f"STT[engine=whistle ms={w_ms}]: '{w_text}'")
                    if w_text and not stt_engine.is_filler(w_text):
                        text = normalise_text(w_text)
                except Exception as e:
                    logger.warning(f"Whistle STT error: {e}")

            return text or ""

        # 1. Whistle STT (active if mode is 'whistle' or 'auto' and Whistle is available)
        if stt_mode in ("whistle", "auto") and stt_engine and stt_engine.is_available():
            try:
                w_text, w_ms = stt_engine.transcribe_with_timing(audio)
            except Exception as e:
                logger.warning(f"Whistle STT error: {e}")
                w_text, w_ms = "", 0

            # Log Whistle recognition
            logger.info(f"STT[engine=whistle ms={w_ms}]: '{w_text}'")

            if not w_text:
                whistle_returned_empty = True
            else:
                norm_w = normalise_text(w_text)
                if not norm_w:
                    logger.debug(f"Ignoring Whistle punctuation-only output: '{w_text}'")
                    w_text = ""
                    whistle_returned_empty = True
                else:
                    has_wake = False
                    if hasattr(self, 'parser') and hasattr(self.parser, 'strip_wake_word'):
                        has_wake, _ = self.parser.strip_wake_word(norm_w)
                    is_wake_active = bool(getattr(self, 'wake_active', False))

                    if stt_engine.should_ignore_whistle_output(
                        w_text,
                        is_window_active=window_open,
                        is_wake_active=is_wake_active,
                        has_wake_word=has_wake
                    ):
                        logger.debug(f"Ignoring Whistle filler output on non-speech: '{w_text}'")
                        w_text = ""
                    else:
                        text = norm_w

        elif stt_mode == "whistle" and (not stt_engine or not stt_engine.is_available()):
            logger.warning("STT_ENGINE is set to 'whistle' but cactus-needle is not available. Falling back to Google STT.")

        # 2. Google STT
        if not text:
            should_try_google = False
            if stt_mode == "google":
                should_try_google = True
            elif not stt_engine or not stt_engine.is_available():
                should_try_google = True
            elif stt_mode == "auto":
                if whistle_returned_empty and stt_engine.is_internet_reachable():
                    should_try_google = True

            if should_try_google and hasattr(recognizer, 'recognize_google'):
                t0 = time.perf_counter()
                try:
                    g_text = recognizer.recognize_google(audio)
                    g_ms = int((time.perf_counter() - t0) * 1000)
                    logger.info(f"STT[engine=google ms={g_ms}]: '{g_text}'")
                    if self.is_offline:
                        self.is_offline = False
                        self.offline_announced = False
                        logger.info("Voice recognition back online")
                    text = g_text
                except sr.UnknownValueError:
                    if window_open:
                        logger.info("No speech detected while window is open (UnknownValueError)")
                    else:
                        logger.debug("Speech recognition: UnknownValueError (unrecognized sound)")
                except sr.RequestError as e:
                    if window_open:
                        logger.info(f"Speech recognition error while window is open: {e}")
                    now = time.time()
                    if (now - self.last_offline_time) >= 30.0:
                        self.last_offline_time = now
                        if not self.offline_announced:
                            self.offline_announced = True
                            logger.warning("Voice recognition is offline")
                            if self.audio_feedback and hasattr(self.audio_feedback, 'speak_status'):
                                self.audio_feedback.speak_status("Voice recognition is offline")
                except Exception as e:
                    if window_open:
                        logger.info(f"Speech recognition error while window is open: {e}")
                    logger.error(f"Error during audio callback recognition: {e}")

        return text

    def _bg_audio_callback(self, recognizer: sr.Recognizer, audio: sr.AudioData, _sync: bool = False):
        """Continuous background callback invoked when speech is detected."""
        if not self.enabled or not self.running:
            return

        window_open = bool(self.dictation_active or self.yes_no_active)
        if window_open:
            logger.info("Audio chunk captured while window is open")

        # Echo suppression check: ignore audio captured while TTS speaks or 0.25s after
        tts_active, tts_reason = self._get_tts_active_status()
        if tts_active:
            if window_open:
                logger.info(f"Audio chunk dropped: {tts_reason}")
            else:
                logger.debug("Echo suppression: dropped audio (%s)", tts_reason)
            return

        def _recognize_worker():
            text = self._recognize_audio(recognizer, audio, window_open=window_open)
            if window_open:
                if text:
                    logger.info(f"Recognition result while window is open: '{text}'")
                else:
                    logger.info("No speech detected while window is open")
            if text:
                self.process_text(text)

        if _sync:
            _recognize_worker()
        else:
            threading.Thread(target=_recognize_worker, daemon=True).start()

    def process_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Process recognized speech text.
        Handles yes/no window, wake word alone follow-up, and same-sentence wake commands.
        """
        if not text:
            return None

        clean = normalise_text(text)
        if not clean:
            return None

        now = time.time()
        window_open = bool(self.dictation_active or self.yes_no_active)
        awaiting = bool(self.awaiting_command_until and now <= self.awaiting_command_until)
        has_wake, remainder = self.parser.strip_wake_word(clean)

        if has_wake or window_open or awaiting:
            logger.info(f"Processing voice input: '{clean}'")
        else:
            logger.debug(f"Processing voice input: '{clean}'")

        # 0. Dictation Window (10-second window, next utterance without wake word is note text)
        if self.dictation_active:
            if now > self.dictation_until:
                self._on_dictation_timeout()
            else:
                has_wake, remainder = self.parser.strip_wake_word(clean)
                note_text = remainder if (has_wake and remainder) else clean
                cb = self.on_dictate_callback
                self.close_dictation_window(recognized_text=note_text)
                if cb:
                    try:
                        cb(note_text)
                    except Exception as e:
                        logger.error(f"Error in on_dictate callback: {e}")
                return {'intent': 'dictated_note', 'parameters': {'text': note_text}}

        # 1. Yes/No Confirmation Window (10-second window, wake word not needed)
        if self.yes_no_active:
            if now > self.yes_no_until:
                self._on_yes_no_timeout()
            else:
                has_wake, remainder = self.parser.strip_wake_word(clean)
                target = remainder if (has_wake and remainder) else clean

                # Check if words in target or clean match YES_WORDS or NO_WORDS
                words = set(re.findall(r'\b\w+\b', target.lower())) | set(re.findall(r'\b\w+\b', clean.lower()))
                has_yes = bool(words & YES_WORDS)
                has_no = bool(words & NO_WORDS)

                if has_yes and has_no:
                    logger.info(f"Yes/No window: Both YES and NO detected in '{clean}'; ignoring and continuing to listen")
                    return None
                elif has_yes:
                    logger.info(f"Yes/No window: YES confirmed from '{clean}'")
                    cb = self.on_yes_callback
                    self.close_yes_no_window()
                    if cb:
                        try:
                            cb()
                        except Exception as e:
                            logger.error(f"Error in on_yes callback: {e}")
                    return {'intent': 'yes', 'parameters': {}}
                elif has_no:
                    logger.info(f"Yes/No window: NO rejected from '{clean}'")
                    cb = self.on_no_callback
                    self.close_yes_no_window()
                    if cb:
                        try:
                            cb()
                        except Exception as e:
                            logger.error(f"Error in on_no callback: {e}")
                    return {'intent': 'no', 'parameters': {}}
                elif has_wake and remainder:
                    # User spoke a wake phrase followed by a recognized command other than yes/no
                    cmd = self.parser.parse(remainder)
                    if cmd and cmd.get('intent') not in ['unknown', 'wake_word_only', 'yes', 'no']:
                        self._on_wake_word_detected()
                        logger.info(f"Yes/No window: cancelled by recognized command: '{clean}' -> {cmd['intent']}")
                        self.close_yes_no_window()
                        self._execute_command(cmd)
                        return cmd
                    else:
                        logger.info(f"Yes/No window: Neither YES nor NO detected in '{clean}'; continuing to listen")
                        return None
                else:
                    logger.info(f"Yes/No window: Neither YES nor NO detected in '{clean}'; continuing to listen")
                    return None

        # 2. Awaiting command follow-up (user said wake word alone within the last 6s)
        if self.awaiting_command_until and now <= self.awaiting_command_until:
            self.awaiting_command_until = 0.0
            has_wake, remainder = self.parser.strip_wake_word(clean)
            if has_wake:
                self._on_wake_word_detected()
            target = remainder if (has_wake and remainder) else clean
            cmd = self.parser.parse(target)
            self._execute_command(cmd)
            return cmd

        # 3. Normal mode: utterance must begin with the wake word
        has_wake, remainder = self.parser.strip_wake_word(clean)
        if not has_wake:
            logger.debug(f"Ignored utterance (no wake phrase): '{clean}'")
            return None

        # Wake word detected: stop current speech and flush non-emergency queue immediately
        self._on_wake_word_detected()

        if not remainder:
            # User said ONLY the wake word!
            logger.info("Wake word detected alone! Playing ready beep and opening 6s window...")
            if self.beep_func:
                try:
                    self.beep_func()
                except Exception as e:
                    logger.warning(f"Error emitting ready beep: {e}")
            self.awaiting_command_until = time.time() + 6.0
            return {'intent': 'wake_word_only', 'parameters': {}}

        # Command is in the same sentence (e.g. "hey vision find my mobile")
        cmd = self.parser.parse(remainder)
        self._execute_command(cmd)
        return cmd

    def trigger_voice_input(self):
        """Trigger one-shot voice capture in a worker thread."""
        def _worker():
            if self.voice_capture and self.voice_capture.microphone:
                text = self.voice_capture.listen_and_recognize(timeout=5)
                if text:
                    self.process_text(text)
        threading.Thread(target=_worker, daemon=True).start()

    def _on_button_press(self):
        """Callback for manual button trigger."""
        self.trigger_voice_input()

    def _execute_command(self, command: Dict[str, Any]):
        """Execute a parsed command by calling registered handlers."""
        intent = command.get('intent')
        parameters = command.get('parameters', {})

        if intent in ['unknown', 'wake_word_only']:
            return

        if intent in self.command_callbacks:
            try:
                self.command_callbacks[intent](parameters)
            except Exception as e:
                logger.error(f"Error executing handler for '{intent}': {e}")
        else:
            logger.warning(f"No handler registered for intent: '{intent}'")

    def get_help_text(self) -> str:
        """Get help text describing available voice commands."""
        return self.parser.get_help_text()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    vcs = VoiceCommandSystem()

    def handle_remember(params):
        print(f"[OK] REMEMBER: {params}")

    def handle_find(params):
        print(f"[OK] FIND: {params}")

    def handle_navigate(params):
        print("[OK] NAVIGATE activated")

    vcs.register_command_handler('remember', handle_remember)
    vcs.register_command_handler('find', handle_find)
    vcs.register_command_handler('navigate', handle_navigate)

    print("\nVoice Command System Ready!")
    print(vcs.get_help_text())

    # Test parser directly
    test_phrases = [
        "hey vision find my mobile",
        "vision",
        "stop finding",
        "a vision remember this mug",
        "hey visions take note check the door",
        "quit"
    ]
    print("Testing parser with sample phrases:")
    for p in test_phrases:
        parsed = vcs.parser.parse(p)
        print(f"  '{p}' -> {parsed['intent']} {parsed.get('parameters')}")
