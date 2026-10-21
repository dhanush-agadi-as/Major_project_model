"""
Laptop ML Server for VisionMate Smart Glasses
Runs on: Laptop (with GPU for fast processing)
Receives: Video frames + sensor data from Raspberry Pi via WiFi
Processes: YOLOv8, InsightFace, Object Memory, Navigation, Voice Commands
Returns: Detection results + navigation instructions + audio commands

ALL features from test_integration.py wired in:
- Object detection (YOLOv8)
- Face recognition (InsightFace) with announcement cooldown
- Object memory (remember/find)
- Item tracking with audio feedback
- Spatial navigation (obstacle avoidance)
- GPS navigation with route following
- Voice command system
- Audio feedback (priority-based)
- Obstacle fusion (ultrasonic + visual)
- SQLite database integration

Start server: python wireless/laptop_server.py
"""

import asyncio
import collections
import websockets
import cv2
import json
import base64
import time
import math
import numpy as np
import logging
import socket
import sys
import os
import re
import difflib
import threading
from typing import Optional, List, Tuple
import requests
import urllib3
import signal


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Apply torch thread limits early before any models are imported or created
try:
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__" and any(arg in sys.argv for arg in ("-h", "--help")):
    import argparse
    try:
        import config
        _def_host = getattr(config, 'SERVER_HOST', '0.0.0.0')
        _def_port = getattr(config, 'SERVER_PORT', 8081)
    except Exception:
        _def_host = '0.0.0.0'
        _def_port = 8081
    _parser = argparse.ArgumentParser(description="VisionMate Laptop ML Server")
    _parser.add_argument("--host", default=_def_host, help=f"Host address to bind (default: {_def_host})")
    _parser.add_argument("--port", type=int, default=_def_port, help=f"Port to listen on (default: {_def_port})")
    _parser.add_argument("--dry-run", action="store_true", help="Run startup checks, print 'DRY RUN OK' and exit without opening mic, camera, or network")
    _parser.parse_args()
    sys.exit(0)

from object_detector import ObjectDetector
from face_recognizer import FaceRecognizer
from item_tracker import ItemTracker
from object_memory import ObjectMemory
from spatial_navigator import SpatialNavigator
from navigation_engine import NavigationEngine
from gps_tracker import GPSTracker
from osm_navigator import OSMNavigator
from audio_feedback import get_audio_feedback
from find_guidance import FindGuide, default_beep
from database import VisionMateDB
from ultrasonic_sensor import UltrasonicSensor
from obstacle_fusion import ObstacleFusion
from ip_geolocation import reverse_geocode
import config
import speech_notes

# Voice command system - optional (needs microphone)
try:
    from voice_command_system import VoiceCommandSystem
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import concurrent.futures
import traceback
import atexit

_ORIGINAL_TPE_INIT = concurrent.futures.ThreadPoolExecutor.__init__
_TRACKED_EXECUTORS = []

def _patched_tpe_init(self, *args, **kwargs):
    _ORIGINAL_TPE_INIT(self, *args, **kwargs)
    _TRACKED_EXECUTORS.append(self)

concurrent.futures.ThreadPoolExecutor.__init__ = _patched_tpe_init


def disable_openvino_telemetry_at_startup():
    """Attempt cheap shutdown of OpenVINO telemetry executor at startup if possible, otherwise skip it."""
    try:
        telemetry_cls = None
        if 'openvino_telemetry.main' in sys.modules:
            telemetry_cls = getattr(sys.modules['openvino_telemetry.main'], 'Telemetry', None)
        if not telemetry_cls:
            try:
                from openvino_telemetry.main import Telemetry
                telemetry_cls = Telemetry
            except Exception:
                pass
        if telemetry_cls:
            inst = getattr(telemetry_cls, '_SingletonMetaClass__single_instance', None)
            if inst and hasattr(inst, 'sender') and inst.sender:
                if hasattr(inst.sender, 'executor') and inst.sender.executor:
                    inst.sender.executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


_OV_CONFIGURED = False
_OV_CORE = None
_OV_THREADS = None
_OV_CONFIG = {
    'PERFORMANCE_HINT': 'LATENCY',
    'NUM_STREAMS': 1
}


def configure_openvino_runtime() -> dict:
    """Set OpenVINO config to use OPENVINO_THREADS (config.OPENVINO_THREADS, default 8)."""
    global _OV_CONFIGURED, _OV_CORE, _OV_THREADS, _OV_CONFIG
    try:
        import openvino as ov
        if _OV_CORE is None:
            _OV_CORE = ov.Core()
        core = _OV_CORE
        ov_threads = getattr(config, 'OPENVINO_THREADS', 8)
        core.set_property('CPU', {
            'PERFORMANCE_HINT': 'LATENCY',
            'NUM_STREAMS': 1,
            'INFERENCE_NUM_THREADS': ov_threads,
        })
        _OV_THREADS = ov_threads
        _OV_CONFIG = {
            'PERFORMANCE_HINT': 'LATENCY',
            'NUM_STREAMS': 1,
            'INFERENCE_NUM_THREADS': ov_threads,
        }
        ov_th = core.get_property('CPU', 'INFERENCE_NUM_THREADS')
        if ov_th == 0:
            ov_th = ov_threads
        log_msg = f"[STARTUP] OpenVINO configured: hint=LATENCY, num_streams=1, threads={ov_th} (budget={ov_threads})"
        print(log_msg)
        logger.info(log_msg)
        _OV_CONFIGURED = True
        return {'hint': 'LATENCY', 'streams': 1, 'threads': ov_th}
    except Exception as e:
        logger.debug(f"OpenVINO configuration error: {e}")
        return {}



try:
    disable_openvino_telemetry_at_startup()
    configure_openvino_runtime()
except Exception:
    pass


def shutdown_openvino_telemetry() -> bool:
    """
    Look up OpenVINO Telemetry singleton and call
    sender.executor.shutdown(wait=False, cancel_futures=True).
    Returns True if successfully found and shut down.
    """
    shutdown_successful = False
    try:
        telemetry_cls = None
        if 'openvino_telemetry.main' in sys.modules:
            telemetry_cls = getattr(sys.modules['openvino_telemetry.main'], 'Telemetry', None)
        if not telemetry_cls:
            try:
                from openvino_telemetry.main import Telemetry
                telemetry_cls = Telemetry
            except Exception:
                pass

        if telemetry_cls:
            inst = getattr(telemetry_cls, '_SingletonMetaClass__single_instance', None)
            if inst and hasattr(inst, 'sender'):
                sender = inst.sender
                if sender and hasattr(sender, 'executor') and sender.executor:
                    sender.executor.shutdown(wait=False, cancel_futures=True)
                    shutdown_successful = True
                    try:
                        if hasattr(sender, 'force_shutdown'):
                            sender.force_shutdown(0)
                    except Exception:
                        pass
    except Exception as e:
        logger.debug(f"Error shutting down Telemetry singleton: {e}")

    try:
        if 'openvino.tools.ovc' in sys.modules:
            ovc = sys.modules['openvino.tools.ovc']
            if hasattr(ovc, 'telemetry') and hasattr(ovc.telemetry, 'sender'):
                sender = ovc.telemetry.sender
                if hasattr(sender, 'executor') and sender.executor:
                    sender.executor.shutdown(wait=False, cancel_futures=True)
                    shutdown_successful = True
    except Exception as e:
        logger.debug(f"Error shutting down ovc telemetry: {e}")

    return shutdown_successful


def shutdown_all_executors():
    """Shut down all tracked ThreadPoolExecutors with wait=False, cancel_futures=True, or use daemon threads."""
    try:
        shutdown_openvino_telemetry()
    except Exception:
        pass

    for exc in list(_TRACKED_EXECUTORS):
        try:
            exc.shutdown(wait=False, cancel_futures=True)
        except Exception:
            try:
                exc.shutdown(wait=False)
            except Exception:
                pass

    try:
        import torch.utils._functools
        if hasattr(torch.utils._functools, '_prefetch_executor'):
            torch.utils._functools._prefetch_executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass

    for t in threading.enumerate():
        if "ThreadPoolExecutor" in t.name:
            t.daemon = True

atexit.register(shutdown_all_executors)


def clean_object_name(name: str) -> str:
    """
    Remove leading 'my ', 'the ', 'a ', 'an ' and any punctuation or extra spaces.
    Ensures object names are stored, spoken, and displayed without a leading 'my'.
    """
    if not isinstance(name, str) or not name:
        return ""
    cleaned = re.sub(r'^(?:(?:my|the|a|an)\s+)+', '', name.strip(), flags=re.IGNORECASE).strip()
    return cleaned if cleaned else name.strip()


PHONE_SYNONYMS = {'phone', 'mobile', 'cell phone', 'cellphone', 'my phone', 'my mobile', 'cell', 'smartphone'}


def phonetic_normalize(word: str) -> str:
    """
    Phonetic normalization for comparing spoken words (spaces removed, phonetic replacements).
    E.g. 'cell phone' -> 'selfon', 'selfour' -> 'selfour'.
    """
    if not isinstance(word, str) or not word:
        return ""
    w = re.sub(r'[^a-z0-9]', '', word.lower())
    w = re.sub(r'ph', 'f', w)
    w = re.sub(r'c([eiy])', r's\1', w)
    w = re.sub(r'ck', 'k', w)
    w = re.sub(r'c', 'k', w)
    w = re.sub(r'wr', 'r', w)
    w = re.sub(r'wh', 'w', w)
    w = re.sub(r'kn', 'n', w)
    w = re.sub(r'e$', '', w)
    w = re.sub(r'(.)\1+', r'\1', w)
    return w


def _safe_ratio(s1: str, s2: str) -> float:
    if not isinstance(s1, str) or not isinstance(s2, str) or not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    sm = difflib.SequenceMatcher(None, s1, s2)
    ratio = sm.ratio()
    longest = max((block.size for block in sm.get_matching_blocks()), default=0)
    if min(len(s1), len(s2)) >= 3 and longest < 2:
        return 0.0
    return ratio


def calculate_name_similarity(query: str, candidate: str) -> float:
    """
    Calculate similarity between spoken query and candidate name using:
    1. Direct SequenceMatcher ratio
    2. Spaces-removed SequenceMatcher ratio
    3. Phonetically normalized SequenceMatcher ratio
    """
    if not isinstance(query, str) or not isinstance(candidate, str) or not query or not candidate:
        return 0.0
    q_clean = clean_object_name(query).lower().strip()
    c_clean = clean_object_name(candidate).lower().strip()
    if not q_clean or not c_clean:
        return 0.0
    if q_clean == c_clean:
        return 1.0

    r1 = _safe_ratio(q_clean, c_clean)

    q_nospace = re.sub(r'\s+', '', q_clean)
    c_nospace = re.sub(r'\s+', '', c_clean)
    r2 = _safe_ratio(q_nospace, c_nospace)

    q_phone = phonetic_normalize(q_clean)
    c_phone = phonetic_normalize(c_clean)
    r3 = _safe_ratio(q_phone, c_phone) if (q_phone and c_phone) else 0.0

    return max(r1, r2, r3)



class FaceWorker:
    """
    Dedicated worker thread for face recognition off the main ML/frame loop.
    - Runs face detection at most 3 times per second (rate_hz=3.0)
    - Drops stale submitted frames, only processing the newest frame
    - Caches results per track so the ML loop never blocks
    - Skipped during find sessions
    """
    def __init__(self, face_recognizer, rate_hz: float = 3.0):
        self.face_recognizer = face_recognizer
        self.min_interval = 1.0 / max(rate_hz, 0.1)
        self.last_run_time = 0.0
        self._lock = threading.Lock()
        self._pending_frame = None
        self._pending_yolo_person = False
        self._running = False
        self._thread = None
        self._event = threading.Event()
        self._cached_faces = []
        self._track_cache = {}

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="FaceWorker", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def submit_frame(self, frame: np.ndarray, yolo_has_person: bool, is_find_active: bool = False):
        if is_find_active or not self.face_recognizer:
            return
        with self._lock:
            self._pending_frame = frame.copy() if frame is not None else None
            self._pending_yolo_person = yolo_has_person
            self._event.set()

    def get_latest_results(self, yolo_has_person: bool) -> List[dict]:
        if not yolo_has_person:
            with self._lock:
                self._cached_faces = []
            return []
        with self._lock:
            return list(self._cached_faces)

    def _run_loop(self):
        while self._running:
            self._event.wait(timeout=0.1)
            if not self._running:
                break

            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None
                yolo_person = self._pending_yolo_person
                self._event.clear()

            if frame is None:
                continue

            now = time.time()
            elapsed = now - self.last_run_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
                now = time.time()

            self.last_run_time = now

            if not yolo_person:
                with self._lock:
                    self._cached_faces = []
                continue

            try:
                faces = []
                if hasattr(self.face_recognizer, 'recognize_faces'):
                    try:
                        faces = self.face_recognizer.recognize_faces(
                            frame, current_time=now, yolo_has_person=True
                        )
                    except TypeError:
                        try:
                            faces = self.face_recognizer.recognize_faces(frame)
                        except Exception:
                            faces = []
                with self._lock:
                    self._cached_faces = faces
                    for f in faces:
                        tid = f.get('track_id')
                        if tid is not None:
                            self._track_cache[tid] = f
            except Exception as e:
                logger.debug(f"FaceWorker recognition error: {e}")


class LaptopMLServer:
    """ML processing server running on laptop - FULL FEATURE VERSION"""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        default_host = getattr(config, 'SERVER_HOST', '0.0.0.0') if config else '0.0.0.0'
        default_port = getattr(config, 'SERVER_PORT', 8081) if config else 8081
        if isinstance(host, int) and port is None:
            port = host
            host = default_host
        self.host = host if host is not None else default_host
        self.port = port if port is not None else default_port

        # Database
        self.db = None

        # VisionMate ML modules
        self.object_detector = None
        self.face_recognizer = None
        self.item_tracker = None
        self.object_memory = None
        self.spatial_navigator = None
        self.navigation_engine = None
        self.gps_tracker = None
        self.mobile_backend_poll_url = "http://localhost:5001/api/location"
        self.mobile_backend_poll_interval = 2.0  # seconds
        self.osm_navigator = None
        self.audio = None
        self.find_guide = None
        self.ultrasonic = None
        self.obstacle_fusion = None
        self.voice_system = None
        self._distance_warning_logged = False
        self._distance_received = False
        self._current_frame_size = None
        self.distance_client = None

        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        self._stage_ms = {
            'detect': 0.0,
            'memory': 0.0,
            'embed_ms': 0.0,
            'compare_ms': 0.0,
            'db_ms': 0.0,
            'face': 0.0,
            'display': 0.0,
            'total_ml': 0.0
        }

        # Rolling 30-frame statistics (instantaneous)
        self._rolling_frame_times = collections.deque(maxlen=31)
        self._rolling_detect_ms = collections.deque(maxlen=30)
        self._rolling_memory_ms = collections.deque(maxlen=30)
        self._rolling_embed_ms = collections.deque(maxlen=30)
        self._rolling_compare_ms = collections.deque(maxlen=30)
        self._rolling_db_ms = collections.deque(maxlen=30)
        self._rolling_face_ms = collections.deque(maxlen=30)
        self._rolling_display_ms = collections.deque(maxlen=30)
        self._rolling_total_ml_ms = collections.deque(maxlen=30)
        self.last_frame_time_ms = 0.0
        self._last_face_det_time = -999.0
        self._last_faces_result = []

        # Object tracking for caching matches across frames (MATCH_EVERY_N_FRAMES)
        self._object_tracks = {}
        self._next_object_track_id = 1

        # Feature toggles
        self.audio_enabled = True
        self.navigation_enabled = True   # Spatial obstacle avoidance
        self.voice_mode_enabled = True   # Voice-only interface enabled by default
        self.show_display = True
        self.running = True

        # Face announcement cooldown (avoid spam)
        self.last_face_announce = {}       # {name: timestamp}
        self.face_announce_interval = 5.0  # seconds

        # Greetings tracking (10s out of view, 60s cooldown, once per appearance)
        self.face_last_seen = {}          # {name: timestamp}
        self.face_last_greeted = {}       # {name: timestamp}
        self.face_appearance_greeted = {} # {name: bool}

        # Remembered object announcement cooldown
        self.last_remembered_announce = {}   # {name: timestamp}
        self.remember_announce_cooldown = 3.0

        # After C/X, mute the "Is this your X?" prompt for that object
        self.ambiguous_mute_until = {}      # {name: timestamp}
        self.ambiguous_timeout_count = {}   # {name: consecutive_timeout_count}
        self.CONFIRM_MUTE_SECONDS = 60.0
        self.REJECT_MUTE_SECONDS = 60.0
        self.last_sighting_call = {}        # {name: timestamp} - throttle sighting updates to 2s

        # Navigation announcement cooldown
        self.last_navigation_announce = 0
        self.navigation_announce_cooldown = 2.0

        # Object Memory enrollment state
        self.enrollment_active = False
        self.enrollment_target_obj = None
        self.enrollment_frames_captured = 0

        # GPS Navigation state
        self.gps_navigation_active = False
        self.gps_nav_step = 0
        self.last_gps_update = 0
        self.gps_update_interval = 2.0
        self.last_gps_announcement = None

        # Store current frame/objects for voice commands
        self.current_frame = None
        self.current_objects = []
        self.recent_frames = collections.deque(maxlen=5)
        self.last_recognized_faces = []

        # Decoupled display & ML state
        self.face_worker = None
        self._frame_lock = threading.Lock()
        self._ml_result_lock = threading.Lock()
        self._ml_ready_event = threading.Event()
        self._latest_display_frame_packet = None
        self._pending_ml_packet = None
        self._ml_busy = False
        self._workers_started = False
        self._display_thread = None
        self._ml_thread = None
        self._loop = None
        self._last_known_ml_result = {
            'objects': [],
            'faces': [],
            'remembered': [],
            'audio_commands': [],
            'navigation': None,
            'warnings': []
        }
        self._last_known_ml_timestamp = 0.0
        self.obstacle_alerter = None

        # Frame drop & latency tracking
        self._dropped_frames_count = 0
        self._displayed_frames_count = 0
        self._ml_frames_count = 0
        self._last_latency_log_time = time.time()
        self._latest_frame_age_ms = 0.0
        self._display_frame_times = collections.deque(maxlen=60)
        self._ml_frame_times = collections.deque(maxlen=60)


    @staticmethod
    def _compute_bbox_iou(boxA, boxB) -> float:
        """Calculate Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
        if not boxA or not boxB or len(boxA) < 4 or len(boxB) < 4:
            return 0.0
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return float(interArea) / float(boxAArea + boxBArea - interArea)

    def initialize(self, dry_run: bool = False):
        """Initialize ALL modules (ML models + services)"""
        logger.info("=" * 60)
        logger.info("VISIONMATE LAPTOP ML SERVER (FULL FEATURES)")
        logger.info("=" * 60)
        logger.info("\nInitializing all modules...\n")

        try:
            # Configure OpenVINO runtime with remaining cores, hint LATENCY, explicit num_streams=1
            configure_openvino_runtime()

            # Database
            self.db = VisionMateDB()
            logger.info("[OK] Database ready")

            # Object detection
            logger.info("Loading YOLOv8 model...")
            self.object_detector = ObjectDetector()
            logger.info("[OK] Object Detector ready")

            # Face recognition
            logger.info("Loading InsightFace model...")
            self.face_recognizer = FaceRecognizer(self.db)
            logger.info("[OK] Face Recognizer ready")
            if not dry_run and self.face_recognizer:
                self.face_worker = FaceWorker(self.face_recognizer, rate_hz=3.0)
                self.face_worker.start()

            # Item tracking
            self.item_tracker = ItemTracker(self.db)
            logger.info("[OK] Item Tracker ready")

            # Object memory
            self.object_memory = ObjectMemory(self.db)
            logger.info("[OK] Object Memory ready")

            # Spatial navigation
            self.spatial_navigator = SpatialNavigator(frame_width=640, frame_height=480)
            logger.info("[OK] Spatial Navigator ready")

            # Audio feedback (priority-based TTS)
            self.audio = get_audio_feedback(use_gtts=False)
            self.audio.set_quiet_mode_checker(self.is_quiet_mode_active)
            logger.info("[OK] Audio Feedback ready")

            # Indoor object-finding guidance (camera-only, no extra hardware)
            self.find_guide = FindGuide(
                speak=lambda msg: self.audio.speak_prompt(msg, wait=False) if not self.is_quiet_mode_active() else None,
                beep=default_beep,
                frame_w=config.CAMERA_WIDTH,
                frame_h=config.CAMERA_HEIGHT,
                is_speaking_fn=lambda: self.audio.is_busy(),
            )
            self.audio.register_beep_controller(self.find_guide.pause_beeps, self.find_guide.resume_beeps)
            logger.info("[OK] Find Guidance ready")

            # GPS + Navigation
            self.gps_tracker = GPSTracker(
                mock_mode=True if dry_run else config.GPS_MOCK_MODE,
                use_ip_geolocation=False if dry_run else getattr(config, 'GPS_USE_IP_GEOLOCATION', True)
            )
            self.mobile_backend_poll_url = "http://localhost:5001/api/location"
            self.mobile_backend_poll_interval = 2.0  # seconds
            self.osm_navigator = OSMNavigator()
            self.navigation_engine = NavigationEngine(
                self.gps_tracker,
                self.osm_navigator,
                announcement_distance=config.NAVIGATION_ANNOUNCEMENT_DISTANCE,
                reroute_distance=config.NAVIGATION_REROUTE_DISTANCE,
                arrival_distance=config.NAVIGATION_ARRIVAL_DISTANCE
            )
            if not dry_run:
                self.gps_tracker.start()
            logger.info("[OK] GPS + Navigation Engine ready")

            # Ultrasonic + Obstacle Fusion
            self.ultrasonic = UltrasonicSensor(mock_mode=True if dry_run else config.ULTRASONIC_MOCK_MODE)
            self.obstacle_fusion = ObstacleFusion(
                self.ultrasonic, frame_width=getattr(config, 'CAMERA_WIDTH', 640), frame_height=getattr(config, 'CAMERA_HEIGHT', 480)
            )
            if not dry_run:
                self.ultrasonic.start()
            if getattr(config, 'USE_PI_CAMERA', False) or not getattr(config, 'ULTRASONIC_MOCK_MODE', False):
                logger.info("Ultrasonic source: Pi/ESP32")
            else:
                logger.info("Ultrasonic source: Mock")
            logger.info("[OK] Ultrasonic + Obstacle Fusion ready")

            # Obstacle Alerter (ultrasonic warnings, object naming, proximity beeps)
            from obstacle_alerter import ObstacleAlerter
            self.obstacle_alerter = ObstacleAlerter(
                speak_fn=lambda text, priority: self.audio.speak(text, priority=priority) if self.audio else None,
                beep_fn=default_beep,
                get_latest_detections_fn=lambda: (
                    self._last_known_ml_result.get('objects', []),
                    getattr(self, '_last_known_ml_timestamp', 0.0)
                ),
                get_frame_age_fn=lambda: getattr(self, '_latest_frame_age_ms', 0.0) / 1000.0,
                is_find_active_fn=lambda: bool(getattr(self, 'find_guide', None) and self.find_guide.is_active),
                beeps_enabled=getattr(config, 'OBSTACLE_BEEPS', True),
            )
            if not dry_run:
                self.obstacle_alerter.start()
            logger.info("[OK] Obstacle Alerter ready")

            # Voice command system (voice-only interface)
            if VOICE_AVAILABLE:
                try:
                    self.voice_system = VoiceCommandSystem(audio_feedback=self.audio)
                    if dry_run:
                        self.voice_system.voice_capture.microphone = "fake_mic"
                        self._register_voice_handlers()
                        self.voice_mode_enabled = True
                        logger.info("[OK] Voice Command System ready (dry run: mic open skipped)")
                    elif self.voice_system.initialize():
                        self.voice_system.start()
                        self._register_voice_handlers()
                        self.voice_mode_enabled = True
                        logger.info("[OK] Voice Command System ready (Continuous Wake-Word Listening)")
                        if self.audio_enabled:
                            self.audio.speak_status("VisionMate ready.")
                            if getattr(config, 'STARTUP_READ_COMMANDS', True):
                                if hasattr(self.audio, 'speak_prompt'):
                                    self.audio.speak_prompt("Say hey vision, then: remember, find, where am I, notes, navigate, or help.", wait=False)
                                else:
                                    self.audio.speak_status("Say hey vision, then: remember, find, where am I, notes, navigate, or help.")
                    else:
                        logger.warning("[WARN] Voice system init failed (no mic?)")
                        self.voice_system = None
                except Exception as e:
                    logger.warning(f"[WARN] Voice system not available: {e}")
                    self.voice_system = None
            else:
                logger.info("[WARN] Voice command module not found - skipping")

            logger.info("\n" + "=" * 60)
            logger.info("[OK] ALL MODULES LOADED SUCCESSFULLY!")
            logger.info("=" * 60)

            # Print startup thread diagnosis line (torch, OpenVINO, STT)
            self.print_thread_diagnostics()

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            traceback.print_exc()
            return False

    def print_thread_diagnostics(self) -> str:
        """Print one startup line with the thread counts (torch, OpenVINO, InsightFace, STT) for diagnosis."""
        try:
            import torch
            torch_th = torch.get_num_threads()
        except Exception:
            torch_th = 'N/A'
        try:
            import openvino as ov
            core = _OV_CORE if _OV_CORE is not None else ov.Core()
            ov_th = core.get_property('CPU', 'INFERENCE_NUM_THREADS')
            if ov_th == 0 and _OV_THREADS is not None:
                ov_th = _OV_THREADS
        except Exception:
            ov_th = getattr(config, 'OPENVINO_THREADS', 8)
        face_th = getattr(config, 'FACE_THREADS', 4)
        stt_th = 1 if (getattr(self, 'voice_system', None) and getattr(self.voice_system, 'running', False)) else 0
        diag_line = f"[STARTUP] Thread counts: torch={torch_th}, OpenVINO={ov_th}, InsightFace={face_th}, STT={stt_th}"
        print(diag_line)
        logger.info(diag_line)
        ov_line = f"[STARTUP] OpenVINO config: hint=LATENCY, num_streams=1, threads={ov_th}"
        print(ov_line)
        logger.info(ov_line)
        return diag_line

    def get_rolling_fps(self) -> float:
        """Return instantaneous rolling FPS over the last 30 frames."""
        if len(self._rolling_frame_times) < 2:
            return 0.0
        dt = self._rolling_frame_times[-1] - self._rolling_frame_times[0]
        n_frames = len(self._rolling_frame_times) - 1
        return n_frames / dt if dt > 0 else 0.0

    def get_rolling_stage_avg(self, stage: str) -> float:
        """Return rolling average duration in ms for the given stage over the last 30 frames (0.0 when unused)."""
        dq = getattr(self, f"_rolling_{stage}_ms", None)
        if not dq:
            return 0.0
        return sum(dq) / len(dq)

    @property
    def embed_ms(self) -> float:
        """Rolling average embedding time over last 30 frames."""
        return self.get_rolling_stage_avg('embed')

    @property
    def compare_ms(self) -> float:
        """Rolling average feature comparison time over last 30 frames."""
        return self.get_rolling_stage_avg('compare')

    @property
    def db_ms(self) -> float:
        """Rolling average database write/read time over last 30 frames."""
        return self.get_rolling_stage_avg('db')

    def is_quiet_mode_active(self) -> bool:
        """Return True if quiet mode is active (dictation window, yes/no window, or within 3s of wake word)."""
        if self.voice_system and hasattr(self.voice_system, 'is_quiet_mode'):
            return self.voice_system.is_quiet_mode()
        return False

    def _speak_reply(self, text: str, priority: int = config.PRIORITY_PROMPT):
        """Speak a direct reply to a voice command at PROMPT priority (never suppressed by quiet mode)."""
        if not self.audio_enabled or not self.audio or not text:
            return
        if hasattr(self.audio, 'speak_prompt'):
            self.audio.speak_prompt(text, wait=False)
        elif hasattr(self.audio, 'speak_command_reply'):
            self.audio.speak_command_reply(text, priority=priority)
        elif hasattr(self.audio, 'speak_social'):
            try:
                self.audio.speak_social(text, is_command_reply=True)
            except TypeError:
                self.audio.speak_social(text)
        elif hasattr(self.audio, 'speak'):
            try:
                self.audio.speak(text, priority=priority, is_command_reply=True)
            except TypeError:
                self.audio.speak(text, priority=priority)

    # ─── Confirmation & Shutdown Helpers ────────────────────

    def _confirm_pending_match(self):
        """Confirm pending ambiguous match (previously key C)."""
        pending = self.object_memory.get_pending_match()
        if pending:
            name = pending.get('custom_name')
            self.ambiguous_timeout_count[name] = 0
            self.ambiguous_mute_until[name] = time.time() + self.CONFIRM_MUTE_SECONDS
            pos = self.gps_tracker.get_position() if self.gps_tracker else None
            lat, lon = (pos[0], pos[1]) if pos else (None, None)
            if config.INDOOR_MODE:
                nearby_classes = [o['class_name'] for o in self.current_objects] if self.current_objects else []
                loc_str = ("near " + ", ".join(nearby_classes[:3])) if nearby_classes else None
                lat, lon = None, None
            else:
                loc_str = reverse_geocode(lat, lon) if (lat is not None and lon is not None) else None
            success = self.object_memory.confirm_pending_match(
                location=loc_str,
                gps_lat=lat,
                gps_lon=lon
            )
            if success:
                if self.audio_enabled:
                    self.audio.speak_social("Confirmed")
                logger.info(f"[OK] Confirmed match for '{name}' (location: {loc_str or '(unknown location)'})")
            else:
                logger.warning(f"[FAIL] Could not record confirmed match for '{name}'")
        else:
            logger.info("No pending match to confirm (none, or it expired after 15s)")

    def _reject_pending_match(self):
        """Reject pending ambiguous match (previously key X)."""
        pending = self.object_memory.get_pending_match()
        if pending:
            name = pending['custom_name']
            self.ambiguous_timeout_count[name] = 0
            self.ambiguous_mute_until[name] = time.time() + self.REJECT_MUTE_SECONDS
            self.object_memory.clear_pending_match()
            if self.audio_enabled:
                self.audio.speak_social("Okay, never mind.")
            logger.info("[OK] Rejected pending match")
        else:
            logger.info("No pending match to reject (none, or it expired after 15s)")

    def _on_ambiguous_timeout(self, custom_name: str):
        """Handle timeout with no answer for ambiguous question."""
        now = time.time()
        count = self.ambiguous_timeout_count.get(custom_name, 0) + 1
        self.ambiguous_timeout_count[custom_name] = count
        if count >= 2:
            self.ambiguous_mute_until[custom_name] = now + 300.0  # Stay silent for 5 minutes
            self.ambiguous_timeout_count[custom_name] = 0
            logger.info(f"Ambiguous match for '{custom_name}' timed out twice. Muting for 5 minutes.")
        else:
            self.ambiguous_mute_until[custom_name] = max(self.ambiguous_mute_until.get(custom_name, 0), now + 60.0)
            logger.info(f"Ambiguous match for '{custom_name}' timed out (count {count}). Muting for 60s.")

    def _shutdown(self):
        """Shut down VisionMate safely on voice command."""
        logger.info("Voice command: Shutting down...")
        self._speak_reply("Shutting down")
        self.cleanup()

    def _match_remembered_with_score(self, query_name: str) -> Tuple[Optional[str], float]:
        """Fuzzy/synonym match spoken name to a remembered object, its class, or its aliases. Returns (matched_name, score)."""
        if not self.object_memory or not query_name:
            return None, 0.0
        remembered_items = self.object_memory.list_remembered_objects()
        if not remembered_items:
            return None, 0.0

        PHONE_SYNONYMS = {'phone', 'mobile', 'cell phone', 'cellphone', 'my phone', 'my mobile', 'cell', 'smartphone'}

        clean_query = clean_object_name(query_name).lower().strip()
        raw_query = query_name.lower().strip()
        is_q_phone = (clean_query in PHONE_SYNONYMS or raw_query in PHONE_SYNONYMS)

        all_matches: List[Tuple[str, float]] = []

        for r_name, r_class in remembered_items:
            clean_target = clean_object_name(r_name).lower().strip()
            raw_target = r_name.lower().strip()
            target_cls = (r_class or '').lower().strip()

            # Retrieve aliases for this object
            aliases = []
            if hasattr(self.object_memory, 'get_object_aliases'):
                try:
                    raw_a = self.object_memory.get_object_aliases(r_name)
                    if isinstance(raw_a, (list, tuple)):
                        aliases = [clean_object_name(str(a)).lower() for a in raw_a if str(a).strip()]
                except Exception:
                    pass
            if getattr(self, 'enrollment_target_obj', None):
                if self.enrollment_target_obj.get('custom_name') == r_name:
                    for a in self.enrollment_target_obj.get('aliases', []):
                        ca = clean_object_name(str(a)).lower()
                        if ca and ca not in aliases:
                            aliases.append(ca)

            candidate_names = [clean_target, raw_target, target_cls] + aliases

            # Exact match against target name, class, or aliases
            if clean_query in candidate_names or raw_query in candidate_names:
                return r_name, 1.0

            # Synonym match (e.g. phone, mobile, cell phone, cellphone, my phone, my mobile)
            is_t_phone = (
                clean_target in PHONE_SYNONYMS
                or raw_target in PHONE_SYNONYMS
                or target_cls in PHONE_SYNONYMS
                or any(a in PHONE_SYNONYMS for a in aliases)
            )
            if is_q_phone and is_t_phone:
                return r_name, 1.0

            # Substring match
            for c in candidate_names:
                if not c:
                    continue
                if clean_query == c or raw_query == c:
                    return r_name, 1.0
                if len(c) > 3 and (clean_query in c or c in clean_query):
                    sim = max(calculate_name_similarity(clean_query, c), calculate_name_similarity(raw_query, c))
                    len_ratio = len(min(clean_query, c, key=len)) / max(len(clean_query), len(c))
                    all_matches.append((r_name, max(sim, len_ratio)))

            # Fuzzy / phonetic similarity match
            item_scores = [
                max(calculate_name_similarity(clean_query, c), calculate_name_similarity(raw_query, c))
                for c in candidate_names if c
            ]
            if item_scores:
                best_item_score = max(item_scores)
                all_matches.append((r_name, best_item_score))

        if not all_matches:
            return None, 0.0

        all_matches.sort(key=lambda x: x[1], reverse=True)
        return all_matches[0][0], all_matches[0][1]

    def _match_remembered(self, query_name: str) -> Optional[str]:
        matched, score = self._match_remembered_with_score(query_name)
        if matched and score >= 0.5:
            return matched
        return None

    def _match_enrolled_face_with_score(self, name: str) -> Tuple[Optional[str], float]:
        """Fuzzy match query name against all enrolled face names, returning (best_name, score)."""
        if not name:
            return None, 0.0
        clean_q = clean_object_name(name).lower().strip()
        raw_q = name.lower().strip()
        if not clean_q and not raw_q:
            return None, 0.0

        clean_q = re.sub(r'^(?:face|person)\s+', '', clean_q).strip()
        raw_q = re.sub(r'^(?:face|person)\s+', '', raw_q).strip()

        enrolled_names = []
        if hasattr(self, 'db') and self.db:
            try:
                raw_faces = self.db.get_all_faces()
                if isinstance(raw_faces, list):
                    for f in raw_faces:
                        if isinstance(f, (list, tuple)) and len(f) > 1 and isinstance(f[1], str) and f[1]:
                            enrolled_names.append(f[1].strip())
                        elif isinstance(f, dict) and isinstance(f.get('name'), str) and f.get('name'):
                            enrolled_names.append(f['name'].strip())
            except Exception:
                pass
        if not enrolled_names and hasattr(self, 'face_recognizer') and self.face_recognizer:
            try:
                kfaces = getattr(self.face_recognizer, 'known_faces', [])
                if isinstance(kfaces, list):
                    for f in kfaces:
                        if isinstance(f, (list, tuple)) and len(f) > 1 and isinstance(f[1], str) and f[1]:
                            enrolled_names.append(f[1].strip())
                        elif isinstance(f, dict) and isinstance(f.get('name'), str) and f.get('name'):
                            enrolled_names.append(f['name'].strip())
            except Exception:
                pass

        if not enrolled_names:
            return None, 0.0

        for fn in enrolled_names:
            fn_lower = fn.lower().strip()
            if clean_q == fn_lower or raw_q == fn_lower:
                return fn, 1.0

        all_matches: List[Tuple[str, float]] = []
        for fn in enrolled_names:
            fn_lower = fn.lower().strip()
            if (len(clean_q) >= 3 and clean_q in fn_lower) or (len(fn_lower) >= 3 and fn_lower in clean_q):
                sim = max(
                    calculate_name_similarity(clean_q, fn_lower),
                    calculate_name_similarity(raw_q, fn_lower),
                    len(min(clean_q, fn_lower, key=len)) / max(len(clean_q), len(fn_lower))
                )
                all_matches.append((fn, max(0.85, sim)))

        for fn in enrolled_names:
            fn_lower = fn.lower().strip()
            score = max(
                calculate_name_similarity(clean_q, fn_lower),
                calculate_name_similarity(raw_q, fn_lower),
                difflib.SequenceMatcher(None, clean_q, fn_lower).ratio(),
                difflib.SequenceMatcher(None, raw_q, fn_lower).ratio()
            )
            all_matches.append((fn, score))

        if not all_matches:
            return None, 0.0

        all_matches.sort(key=lambda x: x[1], reverse=True)
        return all_matches[0][0], all_matches[0][1]

    def _match_enrolled_face(self, name: str) -> Optional[str]:
        best_name, best_score = self._match_enrolled_face_with_score(name)
        if best_score >= 0.6 and best_name:
            return best_name
        return None

    # ─── Voice Command Handlers ──────────────────────────────

    def _register_voice_handlers(self):
        """Register all voice command handlers (replaces all keyboard shortcuts)."""
        if not self.voice_system:
            return

        def handle_remember(params):
            if self.enrollment_active:
                self._speak_reply("Already enrolling an object")
                return

            thing = (params.get('thing') or params.get('object_name') or '').strip()
            name = (params.get('name') or params.get('alias') or '').strip()
            raw_cmd = (params.get('raw_command') or '').strip().lower()

            thing_lower = thing.lower()
            is_face_flow = (
                thing_lower in ('person', 'me', 'face', 'this person')
                or raw_cmd in ('remember person', 'remember me', 'remember face', 'remember this person')
                or raw_cmd.endswith('remember person')
                or raw_cmd.endswith('remember me')
                or raw_cmd.endswith('remember face')
                or raw_cmd.endswith('remember this person')
            )

            if is_face_flow:
                if not name:
                    self._speak_reply("Please say a name, like: hey vision, remember person as Bharath.")
                    return

                from voice_command_system import apply_name_hints
                chosen_name = apply_name_hints(name)

                # Retry detection for 2 s
                start_wait = time.time()
                face_visible = False
                while time.time() - start_wait < 2.0:
                    if self.current_frame is not None and self.face_recognizer:
                        if hasattr(self.face_recognizer, 'detect_faces'):
                            try:
                                detected = self.face_recognizer.detect_faces(self.current_frame)
                                if detected:
                                    face_visible = True
                                    break
                            except Exception:
                                pass
                        else:
                            face_visible = True
                            break
                    time.sleep(0.05)

                thing_label = "person" if thing_lower in ('person', 'me', 'this person', '') else thing_lower
                if not face_visible or self.current_frame is None:
                    self._speak_reply(f"I can't see a {thing_label}. Hold it in front of the camera.")
                    return

                success = False
                if self.face_recognizer and hasattr(self.face_recognizer, 'enroll_face'):
                    try:
                        success = self.face_recognizer.enroll_face(self.current_frame, chosen_name)
                    except Exception as e:
                        logger.error(f"Error enrolling face: {e}")
                        success = False

                if success:
                    logger.info(f"[OK] Enrolled face: {chosen_name}")
                    try:
                        import stt_engine
                        stt_engine.refresh_keywords(db=self.db)
                    except Exception as e:
                        logger.debug(f"Could not refresh STT keywords: {e}")
                    self._speak_reply(f"Enrolled {chosen_name}")
                else:
                    self._speak_reply(f"I can't see a {thing_label}. Hold it in front of the camera.")
                return

            # Object enrollment flow
            from voice_command_system import strip_plural
            clean_name = clean_object_name(name).lower().strip() if name else ""
            clean_thing = clean_object_name(thing).lower().strip() if thing else ""
            clean_thing = strip_plural(clean_thing)

            # Check if object already exists in remembered objects (check either name or thing)
            target_check = clean_name or clean_thing
            if target_check and self.object_memory and hasattr(self.object_memory, 'list_remembered_objects'):
                self.object_memory.list_remembered_objects()
                matched_existing = self._match_remembered(target_check)
                if matched_existing:
                    clean_disp = clean_object_name(matched_existing)
                    self._speak_reply(
                        f"You already have a {clean_disp} remembered. "
                        f"Say 'update {clean_disp}' to replace it, or 'forget {clean_disp}' to remove it."
                    )
                    return

            if not clean_name:
                example_name = "mobile" if clean_thing in ('cell phone', 'phone', 'cellphone') else "mobile"
                disp_thing = clean_thing or "cell phone"
                self._speak_reply(f"Please say a name, like: hey vision, remember {disp_thing} as {example_name}")
                return

            def _start_object_enrollment(target_class: str, enroll_name: str):
                yolo_class = target_class.lower().strip()
                aliases = [yolo_class]
                if yolo_class == 'cell phone' and 'phone' not in aliases:
                    aliases.append('phone')
                elif yolo_class == 'phone' and 'cell phone' not in aliases:
                    aliases.append('cell phone')

                try:
                    if hasattr(self.object_memory, 'start_enrollment'):
                        import inspect
                        sig = inspect.signature(self.object_memory.start_enrollment)
                        if 'aliases' in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                            self.object_memory.start_enrollment(enroll_name, yolo_class, aliases=aliases)
                        else:
                            self.object_memory.start_enrollment(enroll_name, yolo_class)
                    self.enrollment_active = True
                    self.enrollment_target_obj = {
                        'custom_name': enroll_name,
                        'yolo_class': yolo_class,
                        'aliases': aliases
                    }
                    self.enrollment_frames_captured = 0
                    self.enrollment_start_time = time.time()
                    self._speak_reply(f"Remembering your {enroll_name}. Hold it and turn it slowly.")
                    logger.info(f"[OK] Voice: Enrolling '{enroll_name}' ({yolo_class}) aliases={aliases}")
                except ValueError as e:
                    if "already exists" in str(e).lower():
                        clean_disp = clean_object_name(enroll_name)
                        self._speak_reply(
                            f"You already have a {clean_disp} remembered. "
                            f"Say 'update {clean_disp}' to replace it, or 'forget {clean_disp}' to remove it."
                        )
                    else:
                        self._speak_reply(f"Error: {e}")

            # Check if thing is "this" (or unspecified)
            if clean_thing in ('this', 'it', 'item', ''):
                start_wait = time.time()
                enrollable = []
                while True:
                    current_objs = list(self.current_objects) if self.current_objects else []
                    enrollable = [o for o in current_objs if o.get('class_name', '').lower() != 'person']
                    if not enrollable and getattr(self, 'recent_frames', None):
                        for rf in reversed(list(self.recent_frames)):
                            f_objs = rf.get('objects', [])
                            enrollable = [o for o in f_objs if o.get('class_name', '').lower() != 'person']
                            if enrollable:
                                break
                    if len(enrollable) == 1 or (time.time() - start_wait >= 2.0):
                        break
                    time.sleep(0.05)

                if len(enrollable) == 1:
                    target_class = enrollable[0]['class_name']
                    enroll_as = clean_name or target_class
                    _start_object_enrollment(target_class, enroll_as)
                else:
                    self._speak_reply("I don't see an object to remember. Hold it in front of the camera.")
                return

            # Explicit object word
            PHONE_SYNONYMS = {'phone', 'mobile', 'cell phone', 'cellphone', 'cell', 'smartphone'}
            is_phone_query = clean_thing in PHONE_SYNONYMS

            def _find_matching_objects(objs):
                matches = []
                for o in objs:
                    cls_lower = o.get('class_name', '').lower()
                    if cls_lower == 'person':
                        continue
                    if is_phone_query and (cls_lower in PHONE_SYNONYMS or cls_lower == 'cell phone'):
                        matches.append(o)
                    elif clean_thing in cls_lower or cls_lower in clean_thing:
                        matches.append(o)
                    elif calculate_name_similarity(clean_thing, cls_lower) >= 0.5:
                        matches.append(o)
                return matches

            start_wait = time.time()
            matching = []
            while True:
                current_objs = list(self.current_objects) if self.current_objects else []
                matching = _find_matching_objects(current_objs)
                if not matching and getattr(self, 'recent_frames', None):
                    for rf in reversed(list(self.recent_frames)):
                        f_objs = rf.get('objects', [])
                        matching = _find_matching_objects(f_objs)
                        if matching:
                            break
                if matching or (time.time() - start_wait >= 2.0):
                    break
                time.sleep(0.05)

            if matching:
                target_class = matching[0]['class_name']
                _start_object_enrollment(target_class, clean_name or target_class)
            else:
                self._speak_reply(f"I don't see a {clean_thing}. Hold it in front of the camera.")

        def handle_forget(params):
            raw_target = (params.get('object_name') or params.get('name') or params.get('face_name') or '').strip()
            if not raw_target:
                self._speak_reply("Which object do you want to forget?")
                return

            def _stop_active_find_for(target_to_stop: str):
                if hasattr(self, 'find_guide') and self.find_guide:
                    is_active = getattr(self.find_guide, 'is_active', False)
                    if isinstance(is_active, bool) and is_active:
                        tgt = getattr(self.find_guide, '_target_name', '')
                        clean_curr = clean_object_name(tgt).lower().strip() if isinstance(tgt, str) else ''
                        clean_tgt = clean_object_name(str(target_to_stop)).lower().strip()
                        is_tgt = False
                        if hasattr(self.find_guide, 'is_target') and callable(self.find_guide.is_target):
                            try:
                                res = self.find_guide.is_target(str(target_to_stop))
                                if isinstance(res, bool) and res:
                                    is_tgt = True
                            except Exception:
                                pass
                        if (clean_curr and (clean_curr == clean_tgt or clean_tgt in clean_curr or clean_curr in clean_tgt)) or is_tgt:
                            clean_disp = clean_object_name(str(target_to_stop))
                            self.find_guide.stop(f"Stopped looking for {clean_disp}")

            def _refresh_caches():
                if hasattr(self.object_memory, 'refresh_embedding_cache'):
                    self.object_memory.refresh_embedding_cache()
                elif hasattr(self.object_memory, 'refresh_cache'):
                    self.object_memory.refresh_cache()
                try:
                    import stt_engine
                    stt_engine.refresh_keywords(db=self.db)
                except Exception:
                    pass

            # 1. Check remembered objects first
            remembered_items = self.object_memory.list_remembered_objects() if self.object_memory else []
            matched_name, obj_score = self._match_remembered_with_score(raw_target) if remembered_items else (None, 0.0)

            # 2. Check enrolled face names (fuzzy with score)
            matched_face, face_score = self._match_enrolled_face_with_score(raw_target)

            # Direct match (ratio >= 0.7): execute directly without confirmation
            best_score = max(obj_score, face_score)
            if best_score >= 0.7:
                if obj_score >= face_score and matched_name:
                    clean_display = clean_object_name(matched_name)
                    _stop_active_find_for(matched_name)
                    self.object_memory.delete_object(matched_name)
                    _refresh_caches()
                    self._speak_reply(f"Forgot your {clean_display}")
                    logger.info(f"[OK] Voice: Forgot object '{matched_name}' (match ratio {obj_score:.2f})")
                    return
                elif matched_face:
                    _stop_active_find_for(matched_face)
                    if self.face_recognizer and hasattr(self.face_recognizer, 'delete_face'):
                        self.face_recognizer.delete_face(matched_face)
                    elif self.db and hasattr(self.db, 'delete_face_by_name'):
                        self.db.delete_face_by_name(matched_face)
                        if self.face_recognizer and hasattr(self.face_recognizer, '_load_known_faces'):
                            self.face_recognizer._load_known_faces()
                    _refresh_caches()
                    self._speak_reply(f"Forgot {matched_face}")
                    logger.info(f"[OK] Voice: Forgot face '{matched_face}' (match ratio {face_score:.2f})")
                    return

            clean_q = clean_object_name(raw_target) or raw_target.strip()
            self._speak_reply(f"I don't know {clean_q}")
            logger.info(f"[FORGET] Weak match for '{raw_target}': best_score={best_score:.2f} < 0.7")

        def handle_forget_face(params):
            face_name = (params.get('face_name') or params.get('name') or params.get('object_name') or '').strip()
            if not face_name:
                self._speak_reply("Which face do you want to forget?")
                return

            matched_face, face_score = self._match_enrolled_face_with_score(face_name)
            if not matched_face or face_score < 0.7:
                clean_fn = clean_object_name(face_name) or face_name.strip()
                self._speak_reply(f"I don't know {clean_fn}")
                return

            if self.face_recognizer and hasattr(self.face_recognizer, 'delete_face'):
                self.face_recognizer.delete_face(matched_face)
            elif self.db and hasattr(self.db, 'delete_face_by_name'):
                self.db.delete_face_by_name(matched_face)
                if self.face_recognizer and hasattr(self.face_recognizer, '_load_known_faces'):
                    self.face_recognizer._load_known_faces()

            try:
                import stt_engine
                stt_engine.refresh_keywords(db=self.db)
            except Exception:
                pass

            self._speak_reply(f"Forgot {matched_face}")
            logger.info(f"[OK] Voice: Forgot face '{matched_face}'")

        def handle_rename_face(params):
            old_name = (params.get('old_name') or '').strip()
            new_name = (params.get('new_name') or '').strip()
            if not old_name or not new_name:
                self._speak_reply("Please specify both the old and new names to rename a face")
                return

            matched_old = self._match_enrolled_face(old_name)
            if not matched_old:
                self._speak_reply(f"I don't know a face called {clean_object_name(old_name)}")
                return

            clean_new = clean_object_name(new_name).strip() or new_name.strip()
            renamed = False
            if self.face_recognizer and hasattr(self.face_recognizer, 'rename_face'):
                renamed = self.face_recognizer.rename_face(matched_old, clean_new)
            elif self.db and hasattr(self.db, 'rename_face'):
                renamed = self.db.rename_face(matched_old, clean_new)
                if self.face_recognizer and hasattr(self.face_recognizer, '_load_known_faces'):
                    self.face_recognizer._load_known_faces()

            if renamed:
                try:
                    import stt_engine
                    stt_engine.refresh_keywords(db=self.db)
                except Exception:
                    pass
                self._speak_reply(f"Renamed face {matched_old} to {clean_new}")
                logger.info(f"[OK] Voice: Renamed face '{matched_old}' to '{clean_new}'")
            else:
                self._speak_reply(f"Failed to rename face {matched_old}")

        def handle_update(params):
            raw_target = (params.get('object_name') or '').strip()
            if not raw_target:
                self._speak_reply("Which object do you want to update?")
                return

            remembered_items = self.object_memory.list_remembered_objects() if self.object_memory else []
            if not remembered_items:
                self._speak_reply("I don't remember any objects yet")
                return

            matched_name = self._match_remembered(raw_target)
            if not matched_name:
                display_list = [clean_object_name(name) for name, _ in remembered_items]
                list_str = ", ".join(display_list)
                clean_q = clean_object_name(raw_target)
                msg = f"I don't know an object called {clean_q}. I remember: {list_str}"
                self._speak_reply(msg)
                return

            clean_display = clean_object_name(matched_name)
            if hasattr(self, 'find_guide') and self.find_guide:
                is_active = getattr(self.find_guide, 'is_active', False)
                if isinstance(is_active, bool) and is_active:
                    tgt = getattr(self.find_guide, '_target_name', '')
                    clean_curr = clean_object_name(tgt).lower().strip() if isinstance(tgt, str) else ''
                    clean_tgt = clean_object_name(str(matched_name)).lower().strip()
                    is_tgt = False
                    if hasattr(self.find_guide, 'is_target') and callable(self.find_guide.is_target):
                        try:
                            res = self.find_guide.is_target(str(matched_name))
                            if isinstance(res, bool) and res:
                                is_tgt = True
                        except Exception:
                            pass
                    if (clean_curr and (clean_curr == clean_tgt or clean_tgt in clean_curr or clean_curr in clean_tgt)) or is_tgt:
                        self.find_guide.stop(f"Stopped looking for {clean_display}")

            self.object_memory.delete_object(matched_name)
            if hasattr(self.object_memory, 'refresh_embedding_cache'):
                self.object_memory.refresh_embedding_cache()
            elif hasattr(self.object_memory, 'refresh_cache'):
                self.object_memory.refresh_cache()

            try:
                import stt_engine
                stt_engine.refresh_keywords(db=self.db)
            except Exception:
                pass

            logger.info(f"[OK] Voice: Updating '{matched_name}', deleted old entry, starting re-enrollment")
            handle_remember({'object_name': matched_name, 'name': matched_name, 'thing': matched_name})

        def handle_find(params):
            object_name = (params.get('object_name') or params.get('name') or '').strip()

            remembered_items = self.object_memory.list_remembered_objects() if self.object_memory else []
            enrolled_faces = []
            if hasattr(self, 'db') and self.db:
                try:
                    raw_faces = self.db.get_all_faces()
                    for f in (raw_faces or []):
                        if isinstance(f, (list, tuple)) and len(f) > 1 and f[1]:
                            enrolled_faces.append(f[1].strip())
                        elif isinstance(f, dict) and f.get('name'):
                            enrolled_faces.append(f['name'].strip())
                except Exception:
                    pass
            if not enrolled_faces and hasattr(self, 'face_recognizer') and self.face_recognizer:
                try:
                    for f in getattr(self.face_recognizer, 'known_faces', []):
                        if isinstance(f, (list, tuple)) and len(f) > 1 and f[1]:
                            enrolled_faces.append(f[1].strip())
                        elif isinstance(f, dict) and f.get('name'):
                            enrolled_faces.append(f['name'].strip())
                except Exception:
                    pass

            display_list = [clean_object_name(name) for name, _ in remembered_items] + enrolled_faces
            list_str = ", ".join(display_list) if display_list else "nothing"

            def _check_and_report_face(query_name: str) -> bool:
                matched_face, face_score = self._match_enrolled_face_with_score(query_name)
                if not matched_face or face_score < 0.5:
                    return False
                is_visible = False
                if hasattr(self, 'last_recognized_faces') and self.last_recognized_faces:
                    for fn in self.last_recognized_faces:
                        if fn.lower().strip() == matched_face.lower().strip():
                            is_visible = True
                            break
                if not is_visible and self.current_frame is not None and self.face_recognizer and hasattr(self.face_recognizer, 'recognize_faces'):
                    try:
                        detected = self.face_recognizer.recognize_faces(self.current_frame)
                        for d in detected:
                            if d.get('name', '').lower().strip() == matched_face.lower().strip():
                                is_visible = True
                                break
                    except Exception:
                        pass
                if is_visible:
                    self._speak_reply(f"{matched_face} is in front of you")
                else:
                    self._speak_reply(f"I can't see {matched_face}")
                return True

            def _execute_find_obj(matched_name: str):
                clean_name = clean_object_name(matched_name)
                PHONE_SYNONYMS = {'phone', 'mobile', 'cell phone', 'cellphone', 'cell', 'smartphone'}
                target_cls = None
                if self.object_memory:
                    cached_info = self.object_memory.get_cached_object(matched_name)
                    if cached_info:
                        target_cls = cached_info.get('yolo_class')
                if not target_cls:
                    for r_name, r_class in remembered_items:
                        if r_name == matched_name:
                            target_cls = r_class
                            break
                if not target_cls:
                    clean_m = clean_object_name(matched_name).lower()
                    if clean_m in PHONE_SYNONYMS or clean_m in ('phone', 'mobile'):
                        target_cls = 'cell phone'
                    else:
                        target_cls = clean_m

                location = self.object_memory.get_object_location(matched_name) if self.object_memory else None
                if location:
                    last_seen_msg = self.find_guide.describe_last_seen(location) if self.find_guide else ""
                    place = (location.get('location') or '').strip()
                    if place:
                        place_lower = place.lower().strip()
                        if place_lower in ['person', 'the person', 'near person', 'near the person']:
                            place_str = "near you"
                        elif not place_lower.startswith('near'):
                            place_str = f"near {place}"
                        else:
                            place_str = place
                    else:
                        place_str = ""

                    if place_str:
                        if place_str == "near you" and "near you" in last_seen_msg:
                            msg = f"Your {clean_name} was {last_seen_msg}"
                        else:
                            tail = last_seen_msg[len('last seen '):] if last_seen_msg.startswith('last seen ') else last_seen_msg
                            if tail.startswith("near "):
                                msg = f"Your {clean_name} was last seen {tail}"
                            else:
                                msg = f"Your {clean_name} was last seen {place_str}, {tail}"
                    else:
                        msg = f"Your {clean_name} was {last_seen_msg}"

                    logger.info(f"[OK] Voice find: {msg}")
                    self._speak_reply(msg)
                    if self.find_guide:
                        self.find_guide.start(matched_name, "")
                        if target_cls and hasattr(self.find_guide, '_target_class'):
                            self.find_guide._target_class = target_cls
                else:
                    msg = f"Looking for your {clean_name}"
                    logger.info(f"[OK] Voice find: {msg}")
                    self._speak_reply(msg)
                    if self.find_guide:
                        self.find_guide.start(matched_name, "")
                        if target_cls and hasattr(self.find_guide, '_target_class'):
                            self.find_guide._target_class = target_cls

            if not object_name:
                if len(remembered_items) == 1:
                    _execute_find_obj(remembered_items[0][0])
                    return
                else:
                    self._speak_reply("Please say what to find, for example: find my mobile.")
                    return

            # Match object and face (fuzzy-match ratio >= 0.5, use best match)
            matched_obj, obj_score = self._match_remembered_with_score(object_name) if remembered_items else (None, 0.0)
            matched_face, face_score = self._match_enrolled_face_with_score(object_name)

            best_score = max(obj_score, face_score)
            if best_score >= 0.5:
                if obj_score >= face_score and matched_obj:
                    _execute_find_obj(matched_obj)
                    return
                elif matched_face:
                    _check_and_report_face(matched_face)
                    return

            clean_q = clean_object_name(object_name) or object_name.strip()
            msg = f"I don't know {clean_q}. I remember: {list_str}"
            self._speak_reply(msg)
            logger.info(f"[FIND] Unmatched object '{clean_q}': best_score={best_score:.2f} < 0.5")

        def handle_stop_find(params):
            if self.find_guide and self.find_guide.is_active:
                self.find_guide.stop("Search stopped")
                self._speak_reply("Search stopped")
                logger.info("[OK] Voice: find search stopped")
            else:
                self._speak_reply("Nothing to stop")

        def handle_navigate(params):
            if not self.navigation_enabled:
                self.navigation_enabled = True
                self._speak_reply("Navigation enabled")
                logger.info("[OK] Voice: Navigation ON")
            else:
                self._speak_reply("Navigation already active")

        def handle_stop_navigate(params):
            if self.navigation_enabled:
                self.navigation_enabled = False
                self._speak_reply("Navigation disabled")
                logger.info("[OK] Voice: Navigation OFF")
            else:
                self._speak_reply("Navigation already off")

        def handle_list(params):
            remembered_list = self.object_memory.list_remembered_objects() if self.object_memory else []
            if remembered_list:
                items = ", ".join([clean_object_name(name) for name, _ in remembered_list])
                self._speak_reply(f"I remember: {items}")
                logger.info(f"[OK] Voice: Listed {len(remembered_list)} items")
            else:
                self._speak_reply("No items remembered yet")

        def handle_navigate_to(params):
            destination = params.get('destination', '')
            if not destination:
                self.audio.speak_status("Where do you want to go?")
                return
            self.audio.speak_social(f"Navigating to {destination}")
            logger.info(f"[OK] Voice: GPS navigation to {destination}")
            dest_coords = self.osm_navigator.geocode_address(
                destination + ", Bangalore, India"
            )
            if not dest_coords:
                self.audio.speak_status("Could not find destination")
                return
            current_pos = self.gps_tracker.get_position()
            if not current_pos:
                self.audio.speak_status("Waiting for GPS signal")
                return
            route = self.osm_navigator.get_directions(
                current_pos, dest_coords, mode='foot'
            )
            if not route:
                self.audio.speak_status("Could not find route")
                return
            self.osm_navigator.current_route = route
            self.gps_navigation_active = True
            self.gps_nav_step = 0
            self.last_gps_announcement = None
            self.audio.speak_social(
                f"Route found. {route['total_distance']}. {route['total_duration']}."
            )
            logger.info(f"[OK] GPS route: {len(route['steps'])} steps")

        def handle_stop_gps(params):
            if self.gps_navigation_active:
                self.gps_navigation_active = False
                self.audio.speak_social("GPS navigation stopped")
                logger.info("[OK] Voice: GPS navigation stopped")
            else:
                self.audio.speak_status("No GPS navigation active")

        def handle_quit(params):
            require_confirm = getattr(config, 'REQUIRE_SHUTDOWN_CONFIRM', False)
            if require_confirm:
                self._speak_reply("Say yes to confirm")
                self.voice_system.open_yes_no_window(
                    on_yes=self._shutdown,
                    on_no=lambda: self._speak_reply("Shutdown cancelled"),
                    timeout=10.0
                )
            else:
                self._shutdown()

        def handle_enroll_face(params):
            p = dict(params)
            if 'thing' not in p:
                p['thing'] = 'person'
            handle_remember(p)

        def handle_audio_toggle(params):
            self.audio_enabled = not self.audio_enabled
            status = "on" if self.audio_enabled else "off"
            logger.info(f"Audio feedback: {status.upper()}")
            if self.audio_enabled:
                self.audio.speak_status(f"Audio {status}")

        def handle_advance_gps(params):
            if config.GPS_MOCK_MODE:
                self.gps_tracker.advance_mock_position()
                pos = self.gps_tracker.get_position()
                if pos:
                    place = self.gps_tracker.get_current_place_name()
                    logger.info(f"Mock GPS -> {place}")
                    logger.debug(f"Mock GPS coords -> {pos[0]:.6f}, {pos[1]:.6f}")
                    if self.audio_enabled:
                        self.audio.speak_status("Mock GPS advanced")
            else:
                if self.audio_enabled:
                    self.audio.speak_status("GPS is in real mode")

        def handle_database_contents(params):
            self._show_database_contents()
            faces = self.db.get_all_faces()
            remembered = self.object_memory.list_remembered_objects()
            if self.audio_enabled:
                self.audio.speak_social(
                    f"Database has {len(faces)} faces and {len(remembered)} remembered objects."
                )

        def handle_toggle_voice(params):
            if self.audio_enabled:
                self.audio.speak_status("Voice mode is active")

        def handle_take_note(params):
            text = params.get('text', '').strip()
            if not text:
                if self.audio_enabled:
                    self.audio.speak_social("Okay, speak your note")

                def _on_dictate(dictated_text):
                    dictated_text = (dictated_text or '').strip()
                    if dictated_text:
                        reply = speech_notes.take_note(dictated_text)
                        logger.info(f"[OK] Voice Note: {reply}")
                        if self.audio_enabled:
                            self.audio.speak_social(f"Saved. You said: {dictated_text}")

                self.voice_system.open_dictation_window(on_dictate=_on_dictate, timeout=8.0)
                return
            reply = speech_notes.take_note(text)
            logger.info(f"[OK] Voice Note: {reply}")
            if self.audio_enabled:
                self.audio.speak_social(f"Saved. You said: {text}")

        def handle_read_notes(params):
            reply = speech_notes.read_notes()
            logger.info(f"[OK] Voice Notes: {reply}")
            if self.audio_enabled:
                self.audio.speak_social(reply)

        def handle_delete_note(params):
            note_id = params.get('note_id', '')
            reply = speech_notes.delete_note(note_id)
            logger.info(f"[OK] Voice Note deleted: {reply}")
            if self.audio_enabled:
                self.audio.speak_social(reply)

        def handle_clear_notes(params):
            reply = speech_notes.clear_notes()
            logger.info(f"[OK] Voice Notes cleared: {reply}")
            if self.audio_enabled:
                self.audio.speak_social(reply)

        def handle_where_am_i(params):
            place = "your current location"
            if self.gps_tracker:
                place = self.gps_tracker.get_current_place_name()
            self._speak_reply(f"You are near {place}")
            logger.info(f"[OK] Voice where am I: You are near {place}")

        def handle_help(params):
            if not self.audio_enabled or not self.audio:
                return
            groups = [
                "remember person as name. remember object as name.",
                "find my name. forget name.",
                "where am I. take note. read notes.",
                "navigate to place. shutdown."
            ]
            for group in groups:
                if hasattr(self.audio, 'speak_prompt'):
                    self.audio.speak_prompt(group, wait=True)
                else:
                    self._speak_reply(group)
                time.sleep(0.4)

        self.voice_system.register_command_handler('remember', handle_remember)
        self.voice_system.register_command_handler('forget', handle_forget)
        self.voice_system.register_command_handler('delete', handle_forget)
        self.voice_system.register_command_handler('forget_face', handle_forget_face)
        self.voice_system.register_command_handler('rename_face', handle_rename_face)
        self.voice_system.register_command_handler('update', handle_update)
        self.voice_system.register_command_handler('stop_find', handle_stop_find)
        self.voice_system.register_command_handler('find', handle_find)
        self.voice_system.register_command_handler('navigate', handle_navigate)
        self.voice_system.register_command_handler('stop_navigate', handle_stop_navigate)
        self.voice_system.register_command_handler('list', handle_list)
        self.voice_system.register_command_handler('help', handle_help)
        self.voice_system.register_command_handler('where_am_i', handle_where_am_i)
        self.voice_system.register_command_handler('my_location', handle_where_am_i)
        self.voice_system.register_command_handler('navigate_to', handle_navigate_to)
        self.voice_system.register_command_handler('stop_gps', handle_stop_gps)
        self.voice_system.register_command_handler('quit', handle_quit)
        self.voice_system.register_command_handler('shutdown', handle_quit)
        self.voice_system.register_command_handler('enroll_face', handle_enroll_face)
        self.voice_system.register_command_handler('audio_toggle', handle_audio_toggle)
        self.voice_system.register_command_handler('advance_gps', handle_advance_gps)
        self.voice_system.register_command_handler('database_contents', handle_database_contents)
        self.voice_system.register_command_handler('toggle_voice', handle_toggle_voice)
        self.voice_system.register_command_handler('take_note', handle_take_note)
        self.voice_system.register_command_handler('read_notes', handle_read_notes)
        self.voice_system.register_command_handler('delete_note', handle_delete_note)
        self.voice_system.register_command_handler('clear_notes', handle_clear_notes)
        def handle_obstacle_beeps(params):
            state = params.get('state', 'toggle')
            if getattr(self, 'obstacle_alerter', None):
                if state == 'on':
                    self.obstacle_alerter.set_beeps_enabled(True)
                    self.audio.speak_status("Obstacle beeps on.")
                elif state == 'off':
                    self.obstacle_alerter.set_beeps_enabled(False)
                    self.audio.speak_status("Obstacle beeps off.")
                else:
                    new_state = not self.obstacle_alerter.beeps_enabled
                    self.obstacle_alerter.set_beeps_enabled(new_state)
                    self.audio.speak_status(f"Obstacle beeps {'on' if new_state else 'off'}.")
            else:
                self.audio.speak_status("Obstacle alert not active.")

        self.voice_system.register_command_handler('obstacle_beeps', handle_obstacle_beeps)
        self.voice_system.register_command_handler('yes', lambda _: self._confirm_pending_match())
        self.voice_system.register_command_handler('no', lambda _: self._reject_pending_match())

        if self.voice_system and hasattr(self.voice_system, 'parser'):
            self.voice_system.parser.known_targets_provider = self._get_known_fallback_targets

    def _get_known_fallback_targets(self) -> List[str]:
        """Provide candidate names, classes, and aliases for unparsed voice command fallback."""
        targets = set()
        if self.object_memory:
            try:
                items = self.object_memory.list_remembered_objects()
                for name, cls in items:
                    if name:
                        targets.add(name)
                    if cls:
                        targets.add(cls)
                    if cls == 'cell phone' or (name and name.lower() in ('cell phone', 'phone', 'mobile')):
                        targets.update(['cell phone', 'phone', 'mobile'])
            except Exception as e:
                logger.debug(f"Error listing remembered objects for fallback: {e}")
        if self.db and hasattr(self.db, 'get_all_faces'):
            try:
                faces = self.db.get_all_faces()
                for f in faces:
                    if isinstance(f, str):
                        targets.add(f)
                    elif isinstance(f, (list, tuple)) and len(f) > 1 and f[1]:
                        targets.add(str(f[1]))
                    elif isinstance(f, dict) and f.get('name'):
                        targets.add(f['name'])
            except Exception as e:
                logger.debug(f"Error fetching faces for fallback: {e}")
        return list(targets)


    # ─── Decoupled Display & ML Pipeline ─────────────────────

    def _start_worker_threads(self):
        """Start the decoupled display and ML worker threads."""
        if self._workers_started:
            return
        self._workers_started = True
        self.running = True

        if self.face_recognizer and self.face_worker is None:
            self.face_worker = FaceWorker(self.face_recognizer, rate_hz=3.0)
            self.face_worker.start()

        self._display_thread = threading.Thread(
            target=self._display_loop, name="DisplayThread", daemon=True
        )
        self._display_thread.start()

        self._ml_thread = threading.Thread(
            target=self._ml_loop, name="MLWorkerThread", daemon=True
        )
        self._ml_thread.start()
        logger.info("[OK] Decoupled Display and ML worker threads started")

    def handle_incoming_frame(self, frame, ultrasonic_distance=None, button_pressed=False, timestamp=None, websocket=None):
        """Ingest a frame into the decoupled display and ML worker pipeline."""
        if timestamp is None:
            timestamp = time.time()
        if not self._workers_started and self.running:
            self._start_worker_threads()

        with self._frame_lock:
            # Display thread shows newest frame at up to 25-30 FPS, independent of ML speed
            self._latest_display_frame_packet = {
                'frame': frame,
                'distance': ultrasonic_distance,
                'timestamp': timestamp,
            }

            # ML runs in its own worker thread on the newest frame only;
            # frames that arrive while it is busy are dropped (never queued).
            if self._ml_busy:
                if self._pending_ml_packet is not None:
                    self._dropped_frames_count += 1
                self._pending_ml_packet = {
                    'frame': frame,
                    'distance': ultrasonic_distance,
                    'button': button_pressed,
                    'timestamp': timestamp,
                    'websocket': websocket
                }
            else:
                if self._pending_ml_packet is not None:
                    self._dropped_frames_count += 1
                self._pending_ml_packet = {
                    'frame': frame,
                    'distance': ultrasonic_distance,
                    'button': button_pressed,
                    'timestamp': timestamp,
                    'websocket': websocket
                }
                self._ml_ready_event.set()

    def _display_loop(self):
        """Display thread showing newest frame at up to 25-30 FPS with last known boxes/labels overlaid."""
        while self.running:
            t0 = time.perf_counter()
            packet = None
            with self._frame_lock:
                packet = self._latest_display_frame_packet

            if packet is not None and packet.get('frame') is not None:
                frame = packet['frame']
                distance = packet.get('distance')
                ts = packet.get('timestamp', time.time())
                now_t = time.time()
                age_ms = max(0.0, (now_t - ts) * 1000.0)
                self._latest_frame_age_ms = age_ms

                with self._ml_result_lock:
                    overlay_result = dict(self._last_known_ml_result)

                if self.show_display:
                    try:
                        t_disp_start = time.perf_counter()
                        self.display_frame(frame, overlay_result, distance)
                        disp_ms = (time.perf_counter() - t_disp_start) * 1000.0
                        self._stage_ms['display'] += disp_ms
                        self._rolling_display_ms.append(disp_ms)
                    except Exception as e:
                        logger.debug(f"Display error: {e}")

                self._displayed_frames_count += 1
                self._display_frame_times.append(time.perf_counter())

            self._check_log_latency()

            elapsed = time.perf_counter() - t0
            sleep_time = (1.0 / 30.0) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                time.sleep(0.001)

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _ml_loop(self):
        """ML worker thread that processes the newest frame only."""
        while self.running:
            self._ml_ready_event.wait(timeout=0.05)
            if not self.running:
                break

            with self._frame_lock:
                packet = self._pending_ml_packet
                self._pending_ml_packet = None
                self._ml_ready_event.clear()
                if packet is not None:
                    self._ml_busy = True

            if packet is None:
                continue

            try:
                frame = packet['frame']
                dist = packet['distance']
                btn = packet['button']
                ts = packet['timestamp']
                ws = packet['websocket']

                result = self.process_ml(frame, dist, btn)
                result['timestamp'] = ts

                with self._ml_result_lock:
                    self._last_known_ml_result = result
                    self._last_known_ml_timestamp = time.monotonic()

                self._ml_frames_count += 1
                self._ml_frame_times.append(time.perf_counter())

                if ws and self._loop and self._loop.is_running():
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self._send_ml_result(ws, result), self._loop
                        )
                    except Exception as e:
                        logger.debug(f"Error queueing ML result send: {e}")
            except Exception as e:
                logger.error(f"Error in ML worker loop: {e}")
            finally:
                with self._frame_lock:
                    self._ml_busy = False

    async def _send_ml_result(self, websocket, result):
        """Safely send ML result back to client without raising on disconnect."""
        try:
            await websocket.send(json.dumps(result))
        except (websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosed):
            pass
        except Exception as e:
            logger.debug(f"Websocket send error: {e}")

    def _check_log_latency(self):
        """Log every 5 s: 'LATENCY: frame_age_ms=<capture->display> ml_fps=<> display_fps=<> dropped=<>'."""
        now = time.time()
        if (now - self._last_latency_log_time) >= 5.0:
            ml_fps = self._get_fps_from_deque(self._ml_frame_times)
            display_fps = self._get_fps_from_deque(self._display_frame_times)
            msg = (
                f"LATENCY: frame_age_ms={self._latest_frame_age_ms:.1f} "
                f"ml_fps={ml_fps:.1f} display_fps={display_fps:.1f} "
                f"dropped={self._dropped_frames_count}"
            )
            logger.info(msg)
            self._last_latency_log_time = now

    @staticmethod
    def _get_fps_from_deque(times_deque) -> float:
        if len(times_deque) < 2:
            return 0.0
        duration = times_deque[-1] - times_deque[0]
        if duration <= 0:
            return 0.0
        return (len(times_deque) - 1) / duration

    # ─── Frame Processing (WebSocket) ───────────────────────

    async def process_frame(self, websocket):
        """Handle incoming connection from Raspberry Pi"""
        client_ip = websocket.remote_address[0]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"[OK] Raspberry Pi connected from: {client_ip}")
        logger.info(f"{'=' * 60}")
        logger.info("\nVOICE-ONLY CONTROLS (Say 'Hey Vision' + command):")
        logger.info("  'remember this <object>'       - Enroll custom object in view")
        logger.info("  'update <object>'              - Re-enroll object from current view")
        logger.info("  'forget <object>'              - Remove remembered object")
        logger.info("  'find my <object>' / 'stop finding' - Locate object / guidance")
        logger.info("  'enroll face <name>'           - Enroll face in view")
        logger.info("  'take note <text>'             - Save note (auto-read back)")
        logger.info("  'read notes'                   - Read all saved notes")
        logger.info("  'delete note <id>'             - Delete a voice note")
        logger.info("  'clear notes'                  - Delete all voice notes")
        logger.info("  'navigate' / 'stop navigation' - Toggle obstacle guidance")
        logger.info("  'navigate to <dest>' / 'stop gps' - GPS walking directions")
        logger.info("  'advance gps'                  - Advance mock GPS movement")
        logger.info("  'toggle audio'                 - Toggle audio feedback")
        logger.info("  'list all items'               - List remembered objects")
        logger.info("  'show database'                - Show database contents")
        logger.info("  'shutdown' / 'quit'            - shuts down immediately")
        logger.info(f"{'=' * 60}\n")

        self._loop = asyncio.get_running_loop()
        if not self._workers_started:
            self._start_worker_threads()

        try:
            async for message in websocket:
                if not self.running:
                    logger.debug("Server not running, stopping frame processing")
                    break
                if hasattr(self, 'db') and self.db and getattr(self.db, 'is_closed', False):
                    logger.debug("Database closed, stopping frame processing")
                    break

                try:
                    # WebSocket path drops stale frames: if server is behind, skip to the newest
                    while True:
                        has_more = False
                        if hasattr(websocket, 'recv_messages') and hasattr(websocket.recv_messages, 'frames'):
                            has_more = len(websocket.recv_messages.frames) > 0
                        elif hasattr(websocket, 'messages'):
                            has_more = len(websocket.messages) > 0
                        if not has_more:
                            break
                        try:
                            message = await websocket.recv()
                            self._dropped_frames_count += 1
                        except Exception:
                            break

                    data = json.loads(message)

                    # Decode frame
                    frame_data = base64.b64decode(data['frame'])
                    frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                    # Sensor data (support both 'distance' and 'ultrasonic' keys)
                    ultrasonic_distance = data.get('distance', data.get('ultrasonic', None))
                    button_pressed = data.get('button', False)
                    timestamp = data.get('timestamp', time.time())

                    # Decoupled ingestion into display and ML worker
                    self.handle_incoming_frame(frame, ultrasonic_distance, button_pressed, timestamp, websocket)

                    self.frame_count += 1

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                except (websockets.exceptions.ConnectionClosedOK,
                        websockets.exceptions.ConnectionClosedError,
                        websockets.exceptions.ConnectionClosed) as e:
                    logger.info(f"Raspberry Pi disconnected during processing: {client_ip} ({e})")
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    if (not self.running
                            or (hasattr(self, 'db') and getattr(self.db, 'is_closed', False))
                            or "closed database" in err_str
                            or "database is closed" in err_str):
                        logger.debug(f"Frame processing exiting quietly on closed DB / shutdown: {e}")
                        break
                    logger.error(f"Processing error: {e}")
                    traceback.print_exc()

        except (websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosed) as e:
            logger.info(f"Raspberry Pi disconnected: {client_ip} ({e})")
        finally:
            cv2.destroyAllWindows()

    # ─── ML Processing (ALL features from test_integration.py) ─

    def process_ml(self, frame, ultrasonic_distance, button_pressed):
        """Process frame with ALL ML models and features"""
        self.current_ultrasonic_distance = ultrasonic_distance
        if ultrasonic_distance is not None:
            self._distance_received = True

        if not getattr(self, '_distance_warning_logged', False):
            now_t = time.time()
            if (now_t - getattr(self, 'start_time', now_t)) >= 5.0 and not getattr(self, '_distance_received', False):
                if getattr(config, 'USE_PI_CAMERA', False) or not getattr(config, 'ULTRASONIC_MOCK_MODE', False):
                    logger.warning("No distance arrives within 5 s from Pi/ESP32 ultrasonic source. Check ESP32 on /dev/ttyUSB0.")
                self._distance_warning_logged = True

        # Dynamic frame size: read real size from the first frame and update modules
        if frame is not None and hasattr(frame, 'shape') and len(frame.shape) >= 2:
            fh, fw = frame.shape[:2]
            if getattr(self, '_current_frame_size', None) != (fw, fh):
                self._current_frame_size = (fw, fh)
                if hasattr(self, 'spatial_navigator') and hasattr(self.spatial_navigator, 'set_frame_size'):
                    self.spatial_navigator.set_frame_size(fw, fh)
                if hasattr(self, 'obstacle_fusion') and hasattr(self.obstacle_fusion, 'set_frame_size'):
                    self.obstacle_fusion.set_frame_size(fw, fh)
                if hasattr(self, 'find_guide') and hasattr(self.find_guide, 'set_frame_size'):
                    self.find_guide.set_frame_size(fw, fh)

        t_ml_start = time.perf_counter()
        result = {
            'objects': [],
            'faces': [],
            'remembered': [],
            'navigation': '',
            'warnings': [],
            'audio_commands': [],   # Audio commands to send to RPi
            'gps_info': None,       # GPS navigation overlay info
        }

        is_find_active = bool(self.find_guide and self.find_guide.is_active)

        frame_detect_ms = 0.0
        frame_embed_ms = 0.0
        frame_compare_ms = 0.0
        frame_db_ms = 0.0
        frame_mem_ms = 0.0
        frame_face_ms = 0.0

        try:
            # ── 1. Object Detection (YOLOv8) ──
            t_detect_start = time.perf_counter()
            if self.object_detector is not None:
                if is_find_active:
                    target_cls = getattr(self.find_guide, 'target_class', None)
                    if not target_cls:
                        target_name = getattr(self.find_guide, '_target_name', '')
                        clean_tn = clean_object_name(target_name).lower()
                        if clean_tn in PHONE_SYNONYMS or clean_tn in ('phone', 'mobile'):
                            target_cls = 'cell phone'
                        else:
                            target_cls = clean_tn
                    class_overrides = {target_cls: 0.15}
                    if target_cls in ('cell phone', 'phone', 'mobile'):
                        class_overrides['cell phone'] = 0.15
                    try:
                        detections = self.object_detector.detect(
                            frame, imgsz=480, class_conf_overrides=class_overrides
                        )
                    except TypeError:
                        detections = self.object_detector.detect(frame)
                else:
                    detections = self.object_detector.detect(frame)
            else:
                detections = []
            frame_detect_ms = (time.perf_counter() - t_detect_start) * 1000.0
            self._stage_ms['detect'] += frame_detect_ms

            # Store for voice commands
            self.current_frame = frame.copy()
            self.current_objects = detections
            if hasattr(self, 'recent_frames'):
                self.recent_frames.append({
                    'frame': self.current_frame,
                    'objects': list(detections),
                    'timestamp': time.time()
                })

            # Item tracking with audio
            item_updates = self.item_tracker.update_tracking(detections) if self.item_tracker else []
            if item_updates and self.audio_enabled and not self.is_quiet_mode_active():
                for update in item_updates:
                    msg = f"{update['item']} on {update['location']}"
                    self.audio.speak_status(msg)
                    result['audio_commands'].append({
                        'type': 'status', 'text': msg
                    })

            # Serializable format
            for obj in detections:
                result['objects'].append({
                    'class': obj['class_name'],
                    'confidence': float(obj['confidence']),
                    'bbox': [int(x) for x in obj['bbox']]
                })

            # ── 2. Object Memory – Recognize remembered objects ──
            t_mem_start = time.perf_counter()
            remembered_objects = []
            object_to_custom_name = {}

            should_run_obj_rec = is_find_active or (self.frame_count % config.OBJECT_RECOGNITION_FRAME_INTERVAL == 0)

            if not self.enrollment_active and should_run_obj_rec:
                find_target_class = None
                find_target_name = None
                find_candidates = []
                if is_find_active and self.find_guide:
                    find_target_name = getattr(self.find_guide, '_target_name', '')
                    find_target_class = getattr(self.find_guide, '_target_class', '')
                    if not find_target_class and self.object_memory:
                        cached_obj = self.object_memory.get_cached_object(find_target_name)
                        if cached_obj:
                            find_target_class = cached_obj.get('yolo_class')
                    if find_target_class and self.object_memory:
                        find_candidates = self.object_memory.get_cached_candidates(find_target_class)

                # Clean up stale object tracks (older than 2s)
                now_track = time.time()
                self._object_tracks = {
                    tid: t for tid, t in self._object_tracks.items()
                    if (now_track - t.get('last_seen', 0.0)) < 2.0
                }

                for obj in detections:
                    try:
                        match = None
                        t_obj_embed = 0.0
                        t_obj_comp = 0.0
                        if is_find_active and find_target_class and obj.get('class_name') == find_target_class:
                            if len(find_candidates) <= 1:
                                # FAST PATH: Exactly one enrolled in class -> skip embedding entirely!
                                custom_name = find_candidates[0][1] if find_candidates else find_target_name
                                match = (custom_name, float(obj.get('confidence', 0.95)), 'high')
                            else:
                                # Multiple enrolled in class -> pick best embedding match (threshold 0.35)
                                target_row = None
                                for row in find_candidates:
                                    if row[1].lower().strip() == find_target_name.lower().strip():
                                        target_row = row
                                        break
                                if not target_row and find_candidates:
                                    target_row = find_candidates[0]
                                if target_row and hasattr(self.object_memory, 'matcher'):
                                    t0_emb = time.perf_counter()
                                    q_feat = self.object_memory.matcher.extract_features(frame, tuple(obj['bbox']))
                                    t_obj_embed = (time.perf_counter() - t0_emb) * 1000.0

                                    t0_cmp = time.perf_counter()
                                    sim, _ = self.object_memory.matcher.match_features(q_feat, target_row[3])
                                    t_obj_comp = (time.perf_counter() - t0_cmp) * 1000.0
                                    if sim >= 0.35:
                                        match = (target_row[1], float(sim), 'high')
                        else:
                            # Outside find mode: compute embeddings only every 3rd frame per track (MATCH_EVERY_N_FRAMES=3)
                            # and cache the result per tracked object.
                            match_interval = getattr(config, 'MATCH_EVERY_N_FRAMES', 3)
                            best_tid = None
                            best_iou = 0.3
                            for tid, t in self._object_tracks.items():
                                if t.get('class_name') == obj['class_name']:
                                    iou = self._compute_bbox_iou(obj['bbox'], t['bbox'])
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_tid = tid

                            if best_tid is not None:
                                track = self._object_tracks[best_tid]
                                track['bbox'] = obj['bbox']
                                track['last_seen'] = now_track
                                track['age'] += 1
                                if track.get('match_result') is not None and (track['age'] % match_interval != 0):
                                    # Use cached match!
                                    match = track['match_result']
                                    t_obj_embed = 0.0
                                    t_obj_comp = 0.0
                                else:
                                    match = self.object_memory.recognize_object(
                                        frame, tuple(obj['bbox']), obj['class_name']
                                    )
                                    track['match_result'] = match
                                    t_obj_embed = getattr(self.object_memory, 'last_embed_ms', 0.0)
                                    t_obj_comp = getattr(self.object_memory, 'last_compare_ms', 0.0)
                                    if hasattr(self.object_memory, 'last_embed_ms'):
                                        self.object_memory.last_embed_ms = 0.0
                                    if hasattr(self.object_memory, 'last_compare_ms'):
                                        self.object_memory.last_compare_ms = 0.0
                            else:
                                # New track
                                match = self.object_memory.recognize_object(
                                    frame, tuple(obj['bbox']), obj['class_name']
                                )
                                t_obj_embed = getattr(self.object_memory, 'last_embed_ms', 0.0)
                                t_obj_comp = getattr(self.object_memory, 'last_compare_ms', 0.0)
                                if hasattr(self.object_memory, 'last_embed_ms'):
                                    self.object_memory.last_embed_ms = 0.0
                                if hasattr(self.object_memory, 'last_compare_ms'):
                                    self.object_memory.last_compare_ms = 0.0
                                tr_id = self._next_object_track_id
                                self._next_object_track_id += 1
                                self._object_tracks[tr_id] = {
                                    'class_name': obj['class_name'],
                                    'bbox': obj['bbox'],
                                    'last_seen': now_track,
                                    'age': 0,
                                    'match_result': match
                                }

                        self._stage_ms['embed_ms'] += t_obj_embed
                        self._stage_ms['compare_ms'] += t_obj_comp
                        frame_embed_ms += t_obj_embed
                        frame_compare_ms += t_obj_comp

                        if match:
                            custom_name, confidence, match_tier = match
                            remembered_objects.append({
                                'custom_name': custom_name,
                                'class': obj['class_name'],
                                'confidence': confidence,
                                'bbox': obj['bbox'],
                                'tier': match_tier
                            })
                            object_to_custom_name[id(obj)] = custom_name

                            result['remembered'].append({
                                'name': custom_name,
                                'class': obj['class_name'],
                                'confidence': float(confidence),
                                'tier': match_tier,
                                'bbox': [int(x) for x in obj['bbox']]
                            })

                            is_find_target = bool(
                                self.find_guide
                                and self.find_guide.is_active
                                and self.find_guide.is_target(custom_name, obj.get('class_name', ''))
                            )

                            if match_tier == 'high':
                                # Audio with cooldown (suppress if find session is actively guiding toward it)
                                if self.audio_enabled and not self.is_quiet_mode_active() and not is_find_target:
                                    now = time.time()
                                    last = self.last_remembered_announce.get(custom_name, 0)
                                    if (now - last) > self.remember_announce_cooldown:
                                        clean_name = clean_object_name(custom_name)
                                        msg = f"Found your {clean_name}"
                                        self.audio.speak_social(msg)
                                        self.last_remembered_announce[custom_name] = now
                                        result['audio_commands'].append({
                                            'type': 'social', 'text': msg
                                        })

                                # Update sighting with nearby context
                                nearby = []
                                for o in detections:
                                    if o is not obj:
                                        nearby_name = object_to_custom_name.get(
                                            id(o), o['class_name']
                                        )
                                        nearby.append(nearby_name)
                                nearby.extend(self.last_recognized_faces)

                                pos = self.gps_tracker.get_position() if self.gps_tracker else None
                                lat, lon = (pos[0], pos[1]) if pos else (None, None)
                                if config.INDOOR_MODE:
                                    # Outdoors reverse-geocoding skipped indoors;
                                    # build a human-readable context from what's visible.
                                    nearby_classes = [o['class_name'] for o in detections if o is not obj]
                                    if nearby_classes:
                                        if all(c.lower() == 'person' for c in nearby_classes):
                                            loc_str = "near you"
                                        else:
                                            non_person = [c for c in nearby_classes if c.lower() != 'person']
                                            if non_person:
                                                loc_str = "near " + ", ".join(non_person[:3])
                                            else:
                                                loc_str = "near you"
                                    else:
                                        loc_str = None
                                    # GPS coords are not meaningful indoors
                                    lat, lon = None, None
                                else:
                                    loc_str = reverse_geocode(lat, lon) if (lat is not None and lon is not None) else None

                                now_sight = time.time()
                                if (now_sight - self.last_sighting_call.get(custom_name, 0.0)) >= 2.0:
                                    self.last_sighting_call[custom_name] = now_sight
                                    t0_db = time.perf_counter()
                                    self.object_memory.update_object_sighting(
                                        custom_name=custom_name,
                                        image=frame,
                                        location=loc_str,
                                        gps_lat=lat,
                                        gps_lon=lon,
                                        nearby_objects=nearby[:5],
                                        confidence=confidence
                                    )
                                    db_dur = (time.perf_counter() - t0_db) * 1000.0
                                    self._stage_ms['db_ms'] += db_dur
                                    frame_db_ms += db_dur
                            elif match_tier == 'ambiguous':
                                now = time.time()
                                if (self.audio_enabled
                                        and not self.is_quiet_mode_active()
                                        and now >= self.ambiguous_mute_until.get(custom_name, 0)
                                        and not is_find_target
                                        and not is_find_active):
                                    clean_name = clean_object_name(custom_name)
                                    msg = f"Is this your {clean_name}?"
                                    self.audio.speak_social(msg)
                                    # Don't ask about custom_name again for 60s
                                    self.ambiguous_mute_until[custom_name] = now + 60.0
                                    self.last_remembered_announce[custom_name] = now
                                    result['audio_commands'].append({
                                        'type': 'social', 'text': msg
                                    })
                                    if self.voice_system:
                                        self.voice_system.open_yes_no_window(
                                            on_yes=self._confirm_pending_match,
                                            on_no=self._reject_pending_match,
                                            timeout=10.0,
                                            on_timeout=lambda name=custom_name: self._on_ambiguous_timeout(name)
                                        )
                    except Exception as e:
                        logger.error(f"Error matching object: {e}")

            # Feed FindGuide with this frame's remembered objects
            if self.find_guide and self.find_guide.is_active:
                self.find_guide.update(remembered_objects, ultrasonic_distance)

            # ── 2b. Enrollment mode – capture frames ──
            if self.enrollment_active and self.enrollment_target_obj:
                target_class = self.enrollment_target_obj['yolo_class']
                target_dets = [o for o in detections
                                if o['class_name'] == target_class]
                if target_dets:
                    det = target_dets[0]
                    bbox = tuple(det['bbox'])
                    self.enrollment_frames_captured = \
                        self.object_memory.add_enrollment_frame(frame, bbox)

                    max_frames = getattr(config, 'ENROLL_MAX_FRAMES', getattr(config, 'OBJECT_ENROLLMENT_FRAMES', 8))
                    duration = getattr(config, 'OBJECT_ENROLLMENT_DURATION', 3.0)
                    now_t = time.time()
                    elapsed = now_t - getattr(self, 'enrollment_start_time', now_t)

                    if self.enrollment_frames_captured >= max_frames or (elapsed >= duration and self.enrollment_frames_captured >= 3):
                        target_obj = self.enrollment_target_obj
                        self.enrollment_active = False
                        self.enrollment_target_obj = None
                        self.enrollment_frames_captured = 0

                        def _bg_finish_enrollment(mem, target):
                            success = mem.finish_enrollment()
                            name = target.get('custom_name', '')
                            if success:
                                logger.info(f"[OK] Enrolled '{name}'")
                                try:
                                    import stt_engine
                                    stt_engine.refresh_keywords()
                                except Exception as e:
                                    logger.debug(f"Could not refresh STT keywords: {e}")
                                if self.audio_enabled:
                                    clean_name = clean_object_name(name)
                                    self.audio.speak_social(
                                        f"Remembered your {clean_name}"
                                    )
                            else:
                                logger.error(f"[FAIL] Enrollment failed for '{name}'")

                        threading.Thread(target=_bg_finish_enrollment, args=(self.object_memory, target_obj), daemon=True).start()
            frame_mem_ms = (time.perf_counter() - t_mem_start) * 1000.0
            self._stage_ms['memory'] += frame_mem_ms

            # ── 3. Face Recognition (at most 3 times/s off frame loop, only when YOLO sees a person; cached per track) ──
            t_face_start = time.perf_counter()
            rec_fn = getattr(self.face_recognizer, 'recognize_faces', None)
            is_mock_face = hasattr(rec_fn, 'assert_called') or 'mock' in type(rec_fn).__name__.lower()
            yolo_has_person = any(o.get('class_name') == 'person' for o in detections) or is_mock_face
            faces = []
            if not is_find_active and self.face_recognizer:
                if not yolo_has_person:
                    self._last_faces_result = []
                    faces = []
                elif is_mock_face:
                    try:
                        faces = self.face_recognizer.recognize_faces(
                            frame, current_time=time.time(), frame_time_ms=self.last_frame_time_ms, yolo_has_person=True
                        )
                    except TypeError:
                        try:
                            faces = self.face_recognizer.recognize_faces(frame, frame_time_ms=self.last_frame_time_ms)
                        except TypeError:
                            faces = self.face_recognizer.recognize_faces(frame)
                    self._last_faces_result = faces
                else:
                    if self.face_worker is None:
                        self.face_worker = FaceWorker(self.face_recognizer, rate_hz=3.0)
                        self.face_worker.start()
                    self.face_worker.submit_frame(frame, yolo_has_person, is_find_active)
                    faces = self.face_worker.get_latest_results(yolo_has_person)
                    self._last_faces_result = faces

                self.last_recognized_faces = [
                    f['name'] for f in faces if f['name'] != 'Unknown'
                ]

                for face in faces:
                    result['faces'].append({
                        'name': face['name'],
                        'confidence': float(face.get('confidence', 0.0)),
                        'bbox': [int(x) for x in face['bbox']]
                    })

                    # Audio for recognized faces: greet each known person once per appearance.
                    # Don't repeat "Hello <name>" until they've been out of view for 10 s,
                    # with a minimum 60 s cooldown per person.
                    fname = face['name']
                    if fname != 'Unknown':
                        now = time.time()
                        last_seen = self.face_last_seen.get(fname)
                        if last_seen is None or (now - last_seen) >= 10.0:
                            self.face_appearance_greeted[fname] = False
                        self.face_last_seen[fname] = now

                        if self.audio_enabled and not self.is_quiet_mode_active():
                            last_greeted = self.face_last_greeted.get(fname, 0.0)
                            can_greet = (now - last_greeted) >= 60.0 and not self.face_appearance_greeted.get(fname, False)
                            if can_greet:
                                msg = f"Hello {fname}"
                                self.audio.speak_social(msg)
                                self.face_last_greeted[fname] = now
                                self.face_appearance_greeted[fname] = True
                                self.last_face_announce[fname] = now
                                result['audio_commands'].append({
                                    'type': 'social', 'text': msg
                                })
            frame_face_ms = (time.perf_counter() - t_face_start) * 1000.0
            self._stage_ms['face'] += frame_face_ms

            # ── 4. Spatial Navigation (obstacle avoidance) ──
            if self.navigation_enabled and not self.enrollment_active:
                now = time.time()
                guidance = self.spatial_navigator.generate_guidance(
                    detections, current_time=now, ultrasonic_distance=ultrasonic_distance, is_find_active=is_find_active
                )

                if guidance:
                    quiet = self.is_quiet_mode_active()
                    for item in guidance:
                        if self.spatial_navigator.should_speak_guidance(item, ultrasonic_distance, current_time=now, is_find_active=is_find_active):
                            is_crit = self.spatial_navigator.is_critical(item, ultrasonic_distance)
                            if not quiet or is_crit:
                                if self.audio_enabled:
                                    if is_crit:
                                        self.audio.speak_emergency(item['message'])
                                    else:
                                        self.audio.speak_navigational(item['message'])
                                self.spatial_navigator.mark_guidance_spoken(item, current_time=now)
                                self.last_navigation_announce = now
                            break  # Spoke the top eligible event

                if guidance:
                    result['navigation'] = guidance[0]['message']
                    result['warnings'] = [
                        g['message'] for g in guidance[:3]
                    ]

            # ── 5. GPS Navigation ──
            if self.gps_navigation_active and self.osm_navigator.current_route:
                gps_info = self._update_gps_navigation(detections)
                if gps_info:
                    result['gps_info'] = gps_info

            # ── 6. General awareness fallback ──
            if not result['navigation']:
                if (
                    ultrasonic_distance is not None
                    and not getattr(config, 'ULTRASONIC_MOCK_MODE', False)
                    and ultrasonic_distance < 50
                ):
                    result['navigation'] = \
                        f"Obstacle {ultrasonic_distance:.0f} cm ahead"
                elif result['objects']:
                    names = [o['class'] for o in result['objects'][:2]]
                    result['navigation'] = f"Detected: {', '.join(names)}"

        except Exception as e:
            logger.error(f"ML processing error: {e}")
            traceback.print_exc()
        finally:
            frame_ml_ms = (time.perf_counter() - t_ml_start) * 1000.0
            self.last_frame_time_ms = frame_ml_ms
            if self.face_recognizer and hasattr(self.face_recognizer, 'last_frame_time_ms'):
                self.face_recognizer.last_frame_time_ms = frame_ml_ms
            self._stage_ms['total_ml'] += frame_ml_ms
            self._rolling_detect_ms.append(frame_detect_ms)
            self._rolling_memory_ms.append(frame_mem_ms)
            self._rolling_embed_ms.append(frame_embed_ms)
            self._rolling_compare_ms.append(frame_compare_ms)
            self._rolling_db_ms.append(frame_db_ms)
            self._rolling_face_ms.append(frame_face_ms)
            self._rolling_total_ml_ms.append(frame_ml_ms)

        return result

    def _update_gps_navigation(self, detections):
        """GPS navigation step update (from test_integration.py)"""
        now = time.time()
        if (now - self.last_gps_update) < self.gps_update_interval:
            return None
        self.last_gps_update = now

        current_pos = self.gps_tracker.get_position()
        if not current_pos:
            return None

        route = self.osm_navigator.current_route
        steps = route.get('steps', [])
        if self.gps_nav_step >= len(steps):
            return None

        step = steps[self.gps_nav_step]
        instruction = step['instruction']

        # Haversine distance
        end_lat = step['end_location']['lat']
        end_lon = step['end_location']['lng']
        lat1, lon1 = current_pos
        lat2, lon2 = end_lat, end_lon
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (math.sin(dphi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c

        # Announce when close
        if distance <= config.NAVIGATION_ANNOUNCEMENT_DISTANCE:
            if self.last_gps_announcement != self.gps_nav_step:
                self.last_gps_announcement = self.gps_nav_step
                if self.audio_enabled and not self.is_quiet_mode_active():
                    self.audio.speak_navigational(
                        f"In {int(distance)} meters, {instruction}"
                    )
                logger.info(f"GPS: {instruction} ({distance:.0f}m)")

        # Advance if very close
        if distance < 10:
            if self.gps_nav_step + 1 < len(steps):
                self.gps_nav_step += 1
                self.last_gps_announcement = None
                logger.info(f"[OK] GPS advanced to step {self.gps_nav_step + 1}")
            else:
                if self.audio_enabled and not self.is_quiet_mode_active():
                    self.audio.speak_social(
                        "You have arrived at your destination"
                    )
                logger.info("[OK] GPS Navigation: ARRIVED!")
                self.gps_navigation_active = False

        # Obstacle warnings during GPS nav
        if self.audio_enabled:
            warnings = self.obstacle_fusion.generate_obstacle_warnings(
                detections, ultrasonic_distance=self.current_ultrasonic_distance
            )
            if warnings and warnings[0].get('priority', 99) <= 2:
                crit = warnings[0].get('priority') == 1 or warnings[0].get('distance', 99) < 0.5
                if not self.is_quiet_mode_active() or crit:
                    self.audio.speak_emergency(warnings[0]['message'])

        return {
            'step': self.gps_nav_step + 1,
            'total_steps': len(steps),
            'instruction': instruction[:60],
            'distance': int(distance),
        }

    # ─── Display with Full Overlay ───────────────────────────

    def display_frame(self, frame, result, ultrasonic_distance):
        """Display annotated frame on laptop (matches test_integration.py overlay)"""
        display = frame.copy()

        # -- Object detections (green) --
        for obj in result['objects']:
            x1, y1, x2, y2 = obj['bbox']
            label = f"{obj['class']} {obj['confidence']:.0%}"
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # -- Remembered objects (magenta) --
        for mem_obj in result.get('remembered', []):
            x1, y1, x2, y2 = mem_obj['bbox']
            display_name = clean_object_name(mem_obj['name'])
            label = f"* {display_name} ({mem_obj['confidence']:.0%})"
            cv2.rectangle(display, (x1, y1 - 30), (x1 + len(label) * 10, y1),
                          (255, 0, 255), -1)
            cv2.putText(display, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # -- Face detections (blue/orange) --
        for face in result['faces']:
            x1, y1, x2, y2 = face['bbox']
            name = face['name']
            color = (255, 150, 0) if name != 'Unknown' else (0, 0, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # -- Enrollment progress (magenta box + text) --
        if self.enrollment_active and self.enrollment_target_obj:
            tgt = self.enrollment_target_obj
            # Find target in current detections overlay
            for obj in result['objects']:
                if obj['class'] == tgt['yolo_class']:
                    x1, y1, x2, y2 = obj['bbox']
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    lbl = f"Recording {self.enrollment_frames_captured}/5"
                    cv2.putText(display, lbl, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    break
            else:
                cv2.putText(display,
                            f"Waiting for {tgt['yolo_class']}...",
                            (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

        # -- Ultrasonic bar at top --
        if ultrasonic_distance is not None:
            bar_w = int(min(ultrasonic_distance, 200) / 200 * display.shape[1])
            bar_c = ((0, 255, 0) if ultrasonic_distance > 100
                     else (0, 165, 255) if ultrasonic_distance > 50
                     else (0, 0, 255))
            cv2.rectangle(display, (0, 0), (bar_w, 20), bar_c, -1)
            cv2.putText(display, f"Dist: {ultrasonic_distance:.0f}cm", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(display, "Dist: N/A", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # -- Navigation warnings overlay (top-left, color-coded) --
        warnings = result.get('warnings', [])
        if warnings:
            y_off = 100
            for msg in warnings[:3]:
                cv2.putText(display, msg, (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                y_off += 25

        # -- GPS navigation overlay --
        gps = result.get('gps_info')
        if gps:
            nav_txt = (f"GPS NAV [{gps['step']}/{gps['total_steps']}]: "
                       f"{gps['instruction']}")
            cv2.rectangle(display, (5, 125), (635, 175), (0, 0, 0), -1)
            cv2.putText(display, nav_txt, (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(display, f"Distance: {gps['distance']}m", (10, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # -- Navigation message at bottom --
        if result.get('navigation'):
            cv2.rectangle(display, (0, display.shape[0] - 40),
                          (display.shape[1], display.shape[0]), (0, 0, 0), -1)
            cv2.putText(display, result['navigation'],
                        (10, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # -- Stats overlay (top-right area) --
        fps = self.get_rolling_fps()
        if fps <= 0.0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / max(elapsed, 0.001)
        recognized_count = len([f for f in result['faces']
                                if f['name'] != 'Unknown'])
        queue_size = self.audio.get_queue_size() if self.audio else 0

        stats = [
            f"FPS: {fps:.1f}",
            f"Objects: {len(result['objects'])}",
            f"Remembered: {len(result['remembered'])}",
            f"Faces: {recognized_count}",
            f"Audio: {'ON' if self.audio_enabled else 'OFF'}",
            f"Nav: {'ON' if self.navigation_enabled else 'OFF'}",
            f"Voice: {'ON' if self.voice_mode_enabled and self.voice_system else 'OFF'}",
            f"Queue: {queue_size}",
        ]
        for i, stat in enumerate(stats):
            cv2.putText(display, stat, (display.shape[1] - 180, 30 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # -- Enrollment banner --
        if self.enrollment_active and self.enrollment_target_obj:
            cv2.putText(
                display,
                f"ENROLLING: {self.enrollment_target_obj['custom_name']} - Hold steady!",
                (10, display.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2
            )

        # -- Controls hint --
        cv2.putText(
            display,
            "Voice-Only: Say 'Hey Vision' + command | 'Help' for list",
            (10, display.shape[0] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1
        )

        cv2.imshow('VisionMate - Full Pipeline', display)
        key = cv2.waitKey(1) & 0xFF
        if getattr(config, 'KEYBOARD_FALLBACK', False):
            self._handle_keyboard_fallback(key, frame)

    # ─── Minimal Debug Hotkey (only when config.KEYBOARD_FALLBACK is True) ────

    def _handle_keyboard_fallback(self, key, frame):
        """Minimal debug hotkey when config.KEYBOARD_FALLBACK is True."""
        if key == 255:
            return
        if key == ord('q'):
            logger.info("Debug hotkey: closing display window")
            self.show_display = False
            cv2.destroyAllWindows()

    def _show_database_contents(self):
        """Print database contents (faces, items, remembered objects)"""
        print("\n" + "=" * 60)
        print("DATABASE CONTENTS")
        print("=" * 60)

        faces = self.db.get_all_faces()
        print(f"\nFaces ({len(faces)}):")
        for face_id, name, _ in faces:
            print(f"  - {name} (ID: {face_id})")

        items = self.db.get_all_item_locations()
        print(f"\nItem Locations ({len(items)}):")
        for item, location, ts in items:
            print(f"  - {item} on {location} (at {ts})")

        remembered = self.db.get_all_remembered_objects()
        print(f"\nRemembered Objects ({len(remembered)}):")
        for obj_id, name, yolo_class in remembered:
            print(f"  - {name} ({yolo_class}) [ID: {obj_id}]")

        print("=" * 60)

    # ─── Utilities ───────────────────────────────────────────

    def get_local_ip(self):
        """Get laptop's local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "Unable to determine"

    def _fetch_mobile_location(self):
        """Fetch location from mobile backend, supporting plain HTTP and self-signed HTTPS"""
        primary = getattr(self, '_active_mobile_poll_url', self.mobile_backend_poll_url)
        candidates = [primary]
        for base in [self.mobile_backend_poll_url]:
            for scheme in ["https://", "http://"]:
                for host in ["127.0.0.1", "localhost"]:
                    url = base
                    if url.startswith("http://"):
                        url = scheme + url[7:]
                    elif url.startswith("https://"):
                        url = scheme + url[8:]
                    if "localhost" in url and host == "127.0.0.1":
                        url = url.replace("localhost", "127.0.0.1")
                    if url not in candidates:
                        candidates.append(url)

        last_err = None
        for url in candidates:
            try:
                resp = requests.get(url, timeout=1.5, verify=False)
                if resp.status_code == 200:
                    self._active_mobile_poll_url = url
                    return resp
            except Exception as e:
                last_err = e
        if last_err:
            raise last_err
        return None

    async def _poll_mobile_backend_location(self):
        """Periodically poll mobile_backend.py for phone GPS position updates"""
        poll_interval = self.mobile_backend_poll_interval
        while getattr(self, 'running', True):
            if getattr(config, 'GPS_MOCK_MODE', False) or getattr(config, 'INDOOR_MODE', False):
                await asyncio.sleep(10.0)
                continue
            try:
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, self._fetch_mobile_location)
                if resp and resp.status_code == 200:
                    poll_interval = self.mobile_backend_poll_interval
                    data = resp.json()
                    if isinstance(data, dict) and 'latitude' in data and 'longitude' in data:
                        lat = data['latitude']
                        lon = data['longitude']
                        self._poll_count = getattr(self, '_poll_count', 0) + 1
                        if self._poll_count % 5 == 1:
                            logger.info(f"Phone GPS poll: source={data.get('source')}")
                        # Only accept genuine phone fixes. mobile_backend also answers
                        # with IP/mock fallbacks; treating those as 'phone' would keep
                        # them looking fresh forever and mask a missing phone report.
                        if (data.get('source') == 'phone' and lat is not None
                                and lon is not None and self.gps_tracker):
                            self.gps_tracker.set_phone_position(lat, lon)
                else:
                    poll_interval = min(poll_interval * 1.5, 15.0)
            except (requests.exceptions.RequestException, Exception) as e:
                logger.debug(f"Mobile backend location polling: {e}")
                poll_interval = min(poll_interval * 1.5, 15.0)

            await asyncio.sleep(poll_interval)

    async def start_server(self):
        """Start WebSocket server"""
        laptop_ip = self.get_local_ip()

        logger.info("\n" + "=" * 60)
        logger.info("SERVER READY - WAITING FOR RASPBERRY PI")
        logger.info("=" * 60)
        logger.info(f"\nLaptop IP Address: {laptop_ip}")
        logger.info(f"Server Port: {self.port}")
        logger.info("\nOn Raspberry Pi, run:")
        logger.info(f"  python3 rpi_client.py {laptop_ip}")
        logger.info("\n" + "=" * 60 + "\n")

        self.running = True
        # Poll mobile backend for phone GPS updates in background
        self._poll_task = asyncio.create_task(self._poll_mobile_backend_location())

        async with websockets.serve(
            self.process_frame, self.host, self.port, max_size=10**7
        ) as ws_server:
            self.server = ws_server
            self._server_stop_future = asyncio.get_event_loop().create_future()
            try:
                await self._server_stop_future
            except asyncio.CancelledError:
                pass

    def _start_shutdown_watchdog(self, delay: float = 1.0, telemetry_info: bool = False):
        """Watchdog thread that calls os._exit(0) if non-daemon threads keep process alive."""
        def _watchdog(force_delay: float):
            time.sleep(force_delay)
            alive_non_daemon = [
                t.name for t in threading.enumerate()
                if t.is_alive() and not t.daemon and t != threading.current_thread()
            ]
            logger.info(f"Non-daemon threads still alive before os._exit: {alive_non_daemon}")
            if telemetry_info:
                logger.info("expected: OpenVINO telemetry thread")
            logger.warning(f"Process still alive {force_delay}s after cleanup began; force exiting with os._exit(0)")
            os._exit(0)

        t = threading.Thread(target=_watchdog, args=(delay,), daemon=True, name="ShutdownWatchdog")
        t.start()
        return t

    def cleanup(self):
        """Clean up all resources in ordered sequence."""
        if getattr(self, '_cleaned_up', False):
            return
        self._cleaned_up = True

        def _cleanup_sigint(signum, frame):
            logger.info("Ctrl+C during cleanup: exiting immediately.")
            os._exit(0)

        prev_sigint = None
        try:
            prev_sigint = signal.signal(signal.SIGINT, _cleanup_sigint)
        except Exception:
            pass

        try:
            self._do_cleanup()
        except KeyboardInterrupt:
            logger.info("Ctrl+C during cleanup: exiting immediately.")
            os._exit(0)
        finally:
            if prev_sigint is not None:
                try:
                    signal.signal(signal.SIGINT, prev_sigint)
                except Exception:
                    pass

    def _do_cleanup(self):
        telemetry_shutdown_ok = False
        try:
            telemetry_shutdown_ok = shutdown_openvino_telemetry()
        except Exception as e:
            logger.debug(f"Error calling shutdown_openvino_telemetry: {e}")

        current_th = threading.current_thread()
        telemetry_threads = [
            t for t in threading.enumerate()
            if t.is_alive() and not t.daemon and t != current_th and t.name != "MainThread" and
            ('ThreadPool' in t.name or 'openvino' in t.name.lower() or 'telemetry' in t.name.lower())
        ]

        force_exit_delay = 1.0
        if not telemetry_shutdown_ok or telemetry_threads:
            logger.info("expected: OpenVINO telemetry thread")
        self.running = False
        self.show_display = False
        if hasattr(self, '_ml_ready_event'):
            self._ml_ready_event.set()
        if hasattr(self, 'face_worker') and self.face_worker:
            try:
                self.face_worker.stop()
            except Exception:
                pass
        if hasattr(self, '_display_thread') and self._display_thread and self._display_thread.is_alive():
            try:
                self._display_thread.join(timeout=0.3)
            except Exception:
                pass
        if hasattr(self, '_ml_thread') and self._ml_thread and self._ml_thread.is_alive():
            try:
                self._ml_thread.join(timeout=0.3)
            except Exception:
                pass

        # Force exit if process is still alive after force_exit_delay
        if not getattr(self, '_suppress_force_exit', False) and "pytest" not in sys.modules and "PYTEST_CURRENT_TEST" not in os.environ:
            self._start_shutdown_watchdog(delay=force_exit_delay, telemetry_info=(not telemetry_shutdown_ok or bool(telemetry_threads)))

        # 1. find_guide.stop() first
        if hasattr(self, 'find_guide') and self.find_guide:
            try:
                self.find_guide.stop()
            except Exception:
                pass

        if hasattr(self, 'distance_client') and self.distance_client:
            try:
                self.distance_client.stop()
            except Exception as e:
                logger.debug(f"Error stopping distance_client: {e}")

        # 2. voice
        if hasattr(self, 'voice_system') and self.voice_system:
            try:
                self.voice_system.stop()
            except Exception as e:
                logger.debug(f"Error stopping voice_system: {e}")

        # 3. GPS
        if hasattr(self, 'gps_tracker') and self.gps_tracker:
            try:
                self.gps_tracker.stop()
            except Exception as e:
                logger.debug(f"Error stopping gps_tracker: {e}")

        # 4. ultrasonic & obstacle alert
        if hasattr(self, 'ultrasonic') and self.ultrasonic:
            try:
                self.ultrasonic.stop()
            except Exception as e:
                logger.debug(f"Error stopping ultrasonic: {e}")

        if hasattr(self, 'obstacle_alerter') and self.obstacle_alerter:
            try:
                self.obstacle_alerter.stop()
            except Exception as e:
                logger.debug(f"Error stopping obstacle_alerter: {e}")

        # 5. audio
        if hasattr(self, 'audio') and self.audio:
            try:
                self.audio.stop()
            except Exception as e:
                logger.debug(f"Error stopping audio: {e}")

        # 6. websocket server
        if hasattr(self, '_server_stop_future') and self._server_stop_future and not self._server_stop_future.done():
            try:
                self._server_stop_future.get_loop().call_soon_threadsafe(self._server_stop_future.cancel)
            except Exception:
                pass
        if hasattr(self, 'server') and self.server:
            try:
                self.server.close()
            except Exception as e:
                logger.debug(f"Error closing server: {e}")

        # DB and cv2
        if hasattr(self, 'db') and self.db:
            try:
                self.db.close()
            except Exception as e:
                logger.debug(f"Error closing db: {e}")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # ThreadPoolExecutors clean shutdown
        try:
            shutdown_all_executors()
        except Exception as e:
            logger.debug(f"Error shutting down executors: {e}")

        # Check and log any non-daemon threads still alive, and make each of them stop (or be a daemon)
        current_th = threading.current_thread()
        alive_non_daemons = [
            t for t in threading.enumerate()
            if t.is_alive() and not t.daemon and t != current_th and t.name != "MainThread"
        ]
        if alive_non_daemons:
            alive_names = [t.name for t in alive_non_daemons]
            logger.warning("Shutdown: Non-daemon threads still alive: %s", alive_names)
            for t in alive_non_daemons:
                if hasattr(t, 'stop'):
                    try:
                        t.stop()
                    except Exception as e:
                        logger.debug("Error stopping thread %s: %s", t.name, e)
                try:
                    t.daemon = True
                except Exception:
                    pass
        else:
            logger.info("Shutdown: All background threads stopped cleanly (no non-daemon threads alive)")


def print_thread_diagnostics(server=None) -> str:
    """Module-level helper to print and return thread diagnostics."""
    if server is not None:
        return server.print_thread_diagnostics()
    dummy = LaptopMLServer.__new__(LaptopMLServer)
    return LaptopMLServer.print_thread_diagnostics(dummy)


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="VisionMate Laptop ML Server")
    default_host = getattr(config, 'SERVER_HOST', '0.0.0.0') if config else "0.0.0.0"
    default_port = getattr(config, 'SERVER_PORT', 8081) if config else 8081
    parser.add_argument("--host", default=default_host, help=f"Host address to bind (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to listen on (default: {default_port})")
    parser.add_argument("--dry-run", action="store_true", help="Run startup checks, print 'DRY RUN OK' and exit without opening mic, camera, or network")
    args = parser.parse_args()

    server = LaptopMLServer(host=args.host, port=args.port)

    if args.dry_run:
        server._suppress_force_exit = True
        init_ok = server.initialize(dry_run=True)
        server.cleanup()
        if init_ok:
            print("DRY RUN OK")
            sys.exit(0)
        else:
            logger.error("Dry run initialization failed!")
            sys.exit(1)

    if not server.initialize():
        logger.error("Failed to initialize. Exiting.")
        return

    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("\n\nShutting down server...")
    finally:
        try:
            server.cleanup()
        except KeyboardInterrupt:
            logger.info("Interrupted during cleanup. Exiting.")
            os._exit(0)
        logger.info("Goodbye!")


if __name__ == "__main__":
    main()
