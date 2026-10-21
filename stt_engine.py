"""
STT Engine
==========
Speech-to-Text abstraction supporting offline Cactus Whistle (via cactus-needle)
and fallback / online recognition.

Handles:
1. Converting SpeechRecognition AudioData to 16 kHz mono WAV.
2. Keyword biasing with built-in voice commands and names from VisionMateDB.
3. Loading and warming up Whistle once at startup.
4. Identifying and rejecting non-speech filler outputs (e.g. "thank you", "you", ".").
5. Fast internet reachability checking for engine fallback.
"""

import os
import sys
import time
import json
import wave
import socket
import logging
import tempfile
import threading
from typing import List, Optional, Tuple, Any, Union

import numpy as np
import speech_recognition as sr

logger = logging.getLogger(__name__)

# Try importing cactus-needle (Whistle)
try:
    import needle
    NEEDLE_AVAILABLE = True
except Exception as e:
    logger.warning("Failed to import cactus-needle (%s). Whistle STT will not be available; using Google STT only.", e)
    needle = None
    NEEDLE_AVAILABLE = False


# Default command keywords for biasing
# Always includes fixed vocabulary: cell phone, phone, mobile, bottle, chair,
# keys, clock, tv, couch, person, plus all command words and shutdown phrases.
DEFAULT_KEYWORDS = [
    # Fixed object vocabulary
    "cell phone", "phone", "mobile", "bottle", "chair", "keys", "clock", "tv", "couch", "person",
    # Wake phrases
    "vision", "hey vision", "a vision", "have a vision", "have vision", "ek vision", "eh vision",
    "hey visions", "a visions", "virginia", "hey virginia",
    # Command words
    "remember", "remember person", "remember me", "remember this person", "remember this",
    "find", "stop finding",
    "forget", "delete", "remove", "update", "re-remember", "forget face", "rename face",
    "shutdown", "shut down", "quit", "exit", "close", "stop everything",
    "enroll face", "enroll person",
    "take note", "read notes", "delete note", "clear notes",
    "navigate", "navigate to", "list", "help",
    "yes", "yeah", "yep", "ya", "yup", "sure", "ok", "okay", "correct", "right", "please",
    "no", "nope", "nah", "cancel", "wrong"
]

# Filler outputs produced by Whistle on background noise or non-speech
FILLER_OUTPUTS = {
    "thank you", "thanks", "you", ".", "..", "...", "!", "?",
    "thank you.", "thank you!", "thanks.", "thanks!", "you.",
    "bye", "bye.", "goodbye", "goodbye."
}

def _is_whistle_cached() -> bool:
    """Return True if Whistle model weights are already cached in HF hub cache."""
    try:
        from huggingface_hub import constants
        hub_cache = constants.HF_HUB_CACHE
        if not os.path.exists(hub_cache):
            return False
        for item in os.listdir(hub_cache):
            item_lower = item.lower()
            if "whistle" in item_lower or "needle" in item_lower:
                return True
    except Exception:
        pass
    return False

# If the model is already cached, set HF_HUB_OFFLINE so it never calls the network
if _is_whistle_cached():
    os.environ["HF_HUB_OFFLINE"] = "1"

_MODEL_WARMED_UP = False
_NEEDLE_LOCK = threading.Lock()
_LAST_NET_CHECK = 0.0
_LAST_NET_RESULT = True


def is_available() -> bool:
    """Return True if Cactus Whistle (needle) is installed and available."""
    return bool(NEEDLE_AVAILABLE and needle is not None)


def is_internet_reachable(timeout: float = 0.5, cache_seconds: float = 10.0) -> bool:
    """Check if internet connectivity is available with a short timeout and 10s cache."""
    global _LAST_NET_CHECK, _LAST_NET_RESULT
    now = time.time()
    if (now - _LAST_NET_CHECK) < cache_seconds:
        return _LAST_NET_RESULT

    try:
        sock = socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        sock.close()
        _LAST_NET_RESULT = True
    except (socket.timeout, OSError):
        _LAST_NET_RESULT = False

    _LAST_NET_CHECK = now
    return _LAST_NET_RESULT


def get_db_names(db: Optional[Any] = None) -> List[str]:
    """
    Retrieve enrolled face names and remembered object names and aliases from the database.
    Remembered object names are only added if they are real class names or aliases.
    """
    names: List[str] = []
    should_close = False
    
    real_classes = {
        'person', 'chair', 'bottle', 'couch', 'tv', 'keys', 'cell phone', 'clock',
        'phone', 'mobile'
    }
    known_aliases = {
        'phone', 'mobile', 'cellphone', 'cell', 'smartphone',
        'house keys', 'car keys', 'water bottle', 'sofa', 'wrist watch', 'watch', 'television'
    }
    try:
        import config as cfg
        if hasattr(cfg, 'TARGET_CLASSES'):
            for tc in cfg.TARGET_CLASSES:
                real_classes.add(str(tc).lower().strip())
    except Exception:
        pass

    try:
        if db is None:
            from database import VisionMateDB
            db = VisionMateDB()
            should_close = True

        if hasattr(db, 'get_all_faces'):
            for face in db.get_all_faces():
                if len(face) > 1 and face[1]:
                    c_name = str(face[1]).strip()
                    if c_name:
                        names.append(c_name)

        if hasattr(db, 'get_all_remembered_objects'):
            for obj in db.get_all_remembered_objects():
                # obj is (id, custom_name, yolo_class)
                c_name = str(obj[1]).strip() if len(obj) > 1 and obj[1] else ""
                y_cls = str(obj[2]).strip() if len(obj) > 2 and obj[2] else ""

                # Add all enrolled object names and yolo classes to Whistle keywords
                if c_name:
                    names.append(c_name)
                if y_cls:
                    names.append(y_cls)

        # Retrieve aliases from remembered_objects table
        if hasattr(db, 'cursor') and db.cursor:
            try:
                db.cursor.execute("SELECT visual_features FROM remembered_objects")
                for row in db.cursor.fetchall():
                    if row and row[0]:
                        try:
                            feat = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                            if isinstance(feat, dict) and 'aliases' in feat:
                                for a in feat['aliases']:
                                    a_str = str(a).strip()
                                    if a_str:
                                        names.append(a_str)
                        except Exception:
                            pass
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Could not load DB names for STT keywords: {e}")
    finally:
        if should_close and db is not None:
            try:
                db.close()
            except Exception:
                pass

    return list(dict.fromkeys(names))


_CURRENT_KEYWORDS: Optional[List[str]] = None


def get_keywords(db: Optional[Any] = None, force_refresh: bool = False) -> List[str]:
    """
    Get full list of keywords for biasing:
    built-in commands plus all names and aliases in the DB.
    """
    global _CURRENT_KEYWORDS
    if _CURRENT_KEYWORDS is not None and not force_refresh and db is None:
        return list(_CURRENT_KEYWORDS)

    keywords = list(DEFAULT_KEYWORDS)
    try:
        import config as cfg
        hints = getattr(cfg, 'FACE_NAME_HINTS', ["Bharath", "Dhanush"])
        if hints:
            for hint in hints:
                h_str = str(hint).strip()
                if h_str and h_str not in keywords:
                    keywords.append(h_str)
    except Exception:
        pass

    for name in get_db_names(db):
        if name and name not in keywords:
            keywords.append(name)
    _CURRENT_KEYWORDS = keywords
    return list(_CURRENT_KEYWORDS)


def refresh_keywords(db: Optional[Any] = None) -> List[str]:
    """
    Refresh the remembered object names and aliases in Whistle keywords list at runtime.
    Called after each enrollment.
    """
    kw = get_keywords(db=db, force_refresh=True)
    logger.info(f"Refreshed Whistle keywords list ({len(kw)} keywords)")
    return kw


def warm_up():
    """Load and warm up the Whistle model once at startup with a silent clip."""
    global _MODEL_WARMED_UP
    if not is_available():
        return
    if _MODEL_WARMED_UP:
        return

    with _NEEDLE_LOCK:
        if _MODEL_WARMED_UP:
            return
        try:
            silent_clip = np.zeros(16000, dtype=np.float32)
            kw = get_keywords()
            needle.transcribe(silent_clip, language="en", keywords=kw)
            _MODEL_WARMED_UP = True
            os.environ["HF_HUB_OFFLINE"] = "1"
            logger.info("Cactus Whistle model warmed up successfully. HF_HUB_OFFLINE set to 1.")
        except Exception as e:
            logger.warning(f"Error warming up Whistle model: {e}")


def is_filler(text: str) -> bool:
    """Return True if text is a known non-speech filler output."""
    if not text:
        return True
    cleaned = text.strip().lower()
    if cleaned in FILLER_OUTPUTS:
        return True
    stripped = cleaned.strip(".?!,;: -_")
    if not stripped:
        return True
    if stripped in {"thank you", "thanks", "you", "bye", "goodbye"}:
        return True
    return False


def should_ignore_whistle_output(
    text: str,
    is_window_active: bool = False,
    is_wake_active: bool = False,
    has_wake_word: bool = False
) -> bool:
    """
    Reject Whistle outputs that are filler on non-speech or background speech:
    - Ignore results like "thank you", "you", "." unless a wake word or an open window is active.
    - Ignore results longer than 12 words when no wake word is present and no window is open (background speech).
    """
    if not text:
        return True
    if is_window_active or is_wake_active or has_wake_word:
        return False

    # Ignore results longer than 12 words when no wake word is present and no window is open (background speech)
    words = text.strip().split()
    if len(words) > 12:
        return True

    return is_filler(text)


def _audio_to_wav_path(audio_data: Any) -> Tuple[str, bool]:
    """
    Convert SpeechRecognition AudioData, raw bytes, or numpy array to a 16 kHz mono WAV file path.
    Returns (path, is_temporary).
    """
    if isinstance(audio_data, str) and os.path.isfile(audio_data):
        return audio_data, False

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        temp_path = tf.name

    try:
        if hasattr(audio_data, 'get_wav_data'):
            wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
            with open(temp_path, "wb") as f:
                f.write(wav_bytes)
            return temp_path, True

        if isinstance(audio_data, (bytes, bytearray)):
            if audio_data.startswith(b'RIFF'):
                with open(temp_path, "wb") as f:
                    f.write(audio_data)
            else:
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_data)
            return temp_path, True

        if isinstance(audio_data, np.ndarray):
            if np.issubdtype(audio_data.dtype, np.floating):
                pcm_bytes = (audio_data * 32767.0).astype(np.int16).tobytes()
            else:
                pcm_bytes = audio_data.astype(np.int16).tobytes()
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm_bytes)
            return temp_path, True

    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise e

    return temp_path, True


def transcribe_with_timing(
    audio_data: Any,
    keywords: Optional[List[str]] = None,
    language: str = "en",
    db: Optional[Any] = None
) -> Tuple[str, int]:
    """
    Transcribe audio data using Cactus Whistle and measure duration.
    Returns (text, duration_ms).
    """
    if not is_available():
        logger.warning("Whistle STT requested but cactus-needle is not available.")
        return "", 0

    if not _MODEL_WARMED_UP:
        warm_up()

    if keywords is None:
        keywords = get_keywords(db)
    elif not isinstance(keywords, list):
        keywords = list(keywords)

    temp_path, is_temp = _audio_to_wav_path(audio_data)
    try:
        t0 = time.perf_counter()
        with _NEEDLE_LOCK:
            res = needle.transcribe(temp_path, language=language, keywords=keywords)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        text = (res.get("text") or "").strip()
        return text, elapsed_ms
    except Exception as e:
        logger.warning(f"Error during Whistle transcription: {e}")
        return "", 0
    finally:
        if is_temp and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def transcribe(
    audio_data: Any,
    keywords: Optional[List[str]] = None,
    language: str = "en",
    db: Optional[Any] = None
) -> str:
    """
    Transcribe audio data using Cactus Whistle (offline STT).
    
    Args:
        audio_data: speech_recognition.AudioData, WAV file path, raw PCM bytes, or numpy array.
        keywords: Optional list of keywords for biasing. If None, uses DEFAULT_KEYWORDS + DB names.
        language: Language code (default "en").
        db: Optional VisionMateDB instance.
        
    Returns:
        Transcribed text string.
    """
    text, _ = transcribe_with_timing(audio_data, keywords=keywords, language=language, db=db)
    return text


# Module-level warm up when imported if needle is present
if is_available():
    try:
        warm_up()
    except Exception as _e:
        logger.warning(f"Initial Whistle warm-up deferred: {_e}")
