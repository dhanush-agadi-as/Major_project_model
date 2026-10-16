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
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add parent directory to path to import VisionMate modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Voice command system - optional (needs microphone)
try:
    from voice_command_system import VoiceCommandSystem
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaptopMLServer:
    """ML processing server running on laptop - FULL FEATURE VERSION"""

    def __init__(self, port=8081):
        self.port = port
        self.host = "0.0.0.0"

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

        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        self._stage_ms = {'detect': 0.0, 'memory': 0.0, 'face': 0.0, 'display': 0.0, 'total_ml': 0.0}

        # Feature toggles
        self.audio_enabled = True
        self.navigation_enabled = True   # Spatial obstacle avoidance
        self.voice_mode_enabled = False
        self.show_display = True

        # Face announcement cooldown (avoid spam)
        self.last_face_announce = {}       # {name: timestamp}
        self.face_announce_interval = 5.0  # seconds

        # Remembered object announcement cooldown
        self.last_remembered_announce = {}   # {name: timestamp}
        self.remember_announce_cooldown = 3.0

        # After C/X, mute the "Is this your X?" prompt for that object
        self.ambiguous_mute_until = {}      # {name: timestamp}
        self.CONFIRM_MUTE_SECONDS = 30.0
        self.REJECT_MUTE_SECONDS = 60.0

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
        self.last_recognized_faces = []

    def initialize(self):
        """Initialize ALL modules (ML models + services)"""
        logger.info("=" * 60)
        logger.info("VISIONMATE LAPTOP ML SERVER (FULL FEATURES)")
        logger.info("=" * 60)
        logger.info("\nInitializing all modules...\n")

        try:
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
            logger.info("[OK] Audio Feedback ready")

            # Indoor object-finding guidance (camera-only, no extra hardware)
            self.find_guide = FindGuide(
                speak=self.audio.speak_navigational,
                beep=default_beep,
            )
            logger.info("[OK] Find Guidance ready")

            # GPS + Navigation
            self.gps_tracker = GPSTracker(mock_mode=config.GPS_MOCK_MODE)
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
            self.gps_tracker.start()
            logger.info("[OK] GPS + Navigation Engine ready")

            # Ultrasonic + Obstacle Fusion
            self.ultrasonic = UltrasonicSensor(mock_mode=config.ULTRASONIC_MOCK_MODE)
            self.obstacle_fusion = ObstacleFusion(
                self.ultrasonic, frame_width=640, frame_height=480
            )
            self.ultrasonic.start()
            logger.info("[OK] Ultrasonic + Obstacle Fusion ready")

            # Voice command system (optional)
            if VOICE_AVAILABLE:
                try:
                    self.voice_system = VoiceCommandSystem()
                    if self.voice_system.initialize():
                        self.voice_system.start()
                        self._register_voice_handlers()
                        self.voice_mode_enabled = True
                        logger.info("[OK] Voice Command System ready")
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
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ─── Voice Command Handlers ──────────────────────────────

    def _register_voice_handlers(self):
        """Register all voice command handlers (matches test_integration.py)"""
        if not self.voice_system:
            return

        def handle_remember(params):
            if self.enrollment_active:
                self.audio.speak_status("Already enrolling an object")
                return
            object_name = params.get('object_name', '')
            if not object_name:
                self.audio.speak_status("Please specify an object name")
                return
            matching = [o for o in self.current_objects
                        if object_name.lower() in o['class_name'].lower()]
            if matching:
                obj = matching[0]
                custom_name = f"my{object_name}"
                try:
                    self.object_memory.start_enrollment(custom_name, obj['class_name'])
                    self.enrollment_active = True
                    self.enrollment_target_obj = {
                        'custom_name': custom_name,
                        'yolo_class': obj['class_name']
                    }
                    self.audio.speak_social(
                        f"Starting enrollment for {custom_name}. Keep it steady."
                    )
                    logger.info(f"[OK] Voice: Enrolling '{custom_name}'")
                except ValueError as e:
                    self.audio.speak_status(f"Error: {e}")
            else:
                self.audio.speak_status(f"No {object_name} detected in view")

        def handle_find(params):
            object_name = params.get('object_name', '')
            if not object_name:
                self.audio.speak_status("Please specify what to find")
                return

            # Check whether this object is in long-term memory
            remembered_names = [n.lower() for n, _ in
                                 self.object_memory.list_remembered_objects()]
            if object_name.lower() in remembered_names:
                location = self.object_memory.get_object_location(object_name)
                if location:
                    # Build a human-friendly last-seen message
                    last_seen_msg = self.find_guide.describe_last_seen(location)
                    place = (location.get('location') or '').strip()
                    if place and not place.lower().startswith('near'):
                        # Outdoor / place-name context: avoid repeating "last seen"
                        tail = last_seen_msg[len('last seen '):] if last_seen_msg.startswith('last seen ') else last_seen_msg
                        msg = f"Your {object_name} was last seen near {place}, {tail}"
                    else:
                        # Indoor context ("near person") or no place: last_seen_msg already says it
                        msg = f"Your {object_name} was {last_seen_msg}"
                    logger.info(f"[OK] Voice find: {msg}")
                    # Start live camera guidance (speaks msg immediately)
                    self.find_guide.start(object_name, msg)
                else:
                    # Remembered but no sighting record yet
                    msg = f"I remember your {object_name} but haven't spotted it yet"
                    self.audio.speak_navigational(msg)
                    self.find_guide.start(object_name, msg)
            else:
                # Fall back to item_tracker (rule-based room positions)
                loc = self.item_tracker.query_item_location(object_name)
                if loc:
                    msg = f"Your {object_name} on the {loc}"
                    self.audio.speak_navigational(msg)
                    logger.info(f"[OK] Voice: {msg}")
                else:
                    self.audio.speak_status(
                        f"I don't have {object_name} remembered. "
                        f"Say: remember this {object_name}, first"
                    )


        def handle_navigate(params):
            if not self.navigation_enabled:
                self.navigation_enabled = True
                self.audio.speak_social("Navigation enabled")
                logger.info("[OK] Voice: Navigation ON")
            else:
                self.audio.speak_status("Navigation already active")

        def handle_stop_navigate(params):
            if self.navigation_enabled:
                self.navigation_enabled = False
                self.audio.speak_social("Navigation disabled")
                logger.info("[OK] Voice: Navigation OFF")
            else:
                self.audio.speak_status("Navigation already off")

        def handle_list(params):
            remembered_list = self.object_memory.list_remembered_objects()
            if remembered_list:
                items = ", ".join([name for name, _ in remembered_list])
                self.audio.speak_social(f"I remember: {items}")
                logger.info(f"[OK] Voice: Listed {len(remembered_list)} items")
            else:
                self.audio.speak_status("No items remembered yet")

        def handle_help(params):
            self.audio.speak_social(
                "Say: remember this phone, find my keys, stop searching, "
                "navigate to destination, stop navigation, or list all items"
            )

        def handle_stop_find(params):
            if self.find_guide and self.find_guide.is_active:
                self.find_guide.stop("Search stopped")
                logger.info("[OK] Voice: find search stopped")
            else:
                self.audio.speak_status("Nothing to stop")

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

        self.voice_system.register_command_handler('remember', handle_remember)
        self.voice_system.register_command_handler('stop_find', handle_stop_find)
        self.voice_system.register_command_handler('find', handle_find)
        self.voice_system.register_command_handler('navigate', handle_navigate)
        self.voice_system.register_command_handler('stop_navigate', handle_stop_navigate)
        self.voice_system.register_command_handler('list', handle_list)
        self.voice_system.register_command_handler('help', handle_help)
        self.voice_system.register_command_handler('navigate_to', handle_navigate_to)
        self.voice_system.register_command_handler('stop_gps', handle_stop_gps)

    # ─── Frame Processing (WebSocket) ───────────────────────

    async def process_frame(self, websocket):
        """Handle incoming connection from Raspberry Pi"""
        client_ip = websocket.remote_address[0]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"[OK] Raspberry Pi connected from: {client_ip}")
        logger.info(f"{'=' * 60}")
        logger.info(f"\nKEYBOARD CONTROLS (click display window first):")
        logger.info(f"  Q = Quit display")
        logger.info(f"  E = Enroll face")
        logger.info(f"  R = Remember object (enroll custom object)")
        logger.info(f"  L = List remembered objects")
        logger.info(f"  F = Find item/object location")
        logger.info(f"  C = Confirm pending match")
        logger.info(f"  X = Reject pending match")
        logger.info(f"  A = Toggle audio feedback")
        logger.info(f"  N = Toggle navigation guidance (obstacle avoidance)")
        logger.info(f"  G = GPS Navigation (navigate to destination)")
        logger.info(f"  M = Advance mock GPS (simulate movement)")
        logger.info(f"  V = Toggle voice mode")
        logger.info(f"  SPACE = Voice command (button simulation)")
        logger.info(f"  D = Show database contents")
        logger.info(f"{'=' * 60}\n")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)

                    # Decode frame
                    frame_data = base64.b64decode(data['frame'])
                    frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                    # Sensor data
                    ultrasonic_distance = data.get('ultrasonic', 999.0)
                    button_pressed = data.get('button', False)
                    timestamp = data.get('timestamp', time.time())

                    # Process with ALL ML models
                    result = self.process_ml(frame, ultrasonic_distance, button_pressed)
                    result['timestamp'] = timestamp

                    # Send results back to RPi for audio
                    await websocket.send(json.dumps(result))

                    # Display on laptop
                    if self.show_display:
                        t_disp_start = time.perf_counter()
                        self.display_frame(frame, result, ultrasonic_distance)
                        self._stage_ms['display'] += (time.perf_counter() - t_disp_start) * 1000.0

                    # Stats every 30 frames
                    self.frame_count += 1
                    if self.frame_count % 30 == 0:
                        elapsed = time.time() - self.start_time
                        fps = self.frame_count / elapsed
                        latency = (time.time() - timestamp) * 1000
                        logger.info(
                            f"FPS: {fps:.1f} | Lat: {latency:.0f}ms | "
                            f"Obj: {len(result['objects'])} | "
                            f"Face: {len(result['faces'])} | "
                            f"Mem: {len(result['remembered'])} | "
                            f"Audio: {'ON' if self.audio_enabled else 'OFF'}"
                        )
                        logger.info(
                            f"STAGES avg ms/frame (last 30): "
                            f"detect={self._stage_ms['detect'] / 30:.1f} "
                            f"memory={self._stage_ms['memory'] / 30:.1f} "
                            f"face={self._stage_ms['face'] / 30:.1f} "
                            f"display={self._stage_ms['display'] / 30:.1f} "
                            f"total_ml={self._stage_ms['total_ml'] / 30:.1f}"
                        )
                        for k in self._stage_ms:
                            self._stage_ms[k] = 0.0

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    import traceback
                    traceback.print_exc()

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Raspberry Pi disconnected: {client_ip}")
        finally:
            cv2.destroyAllWindows()

    # ─── ML Processing (ALL features from test_integration.py) ─

    def process_ml(self, frame, ultrasonic_distance, button_pressed):
        """Process frame with ALL ML models and features"""
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

        try:
            # ── 1. Object Detection (YOLOv8) ──
            t_detect_start = time.perf_counter()
            detections = self.object_detector.detect(frame)
            self._stage_ms['detect'] += (time.perf_counter() - t_detect_start) * 1000.0

            # Store for voice commands
            self.current_frame = frame.copy()
            self.current_objects = detections

            # Item tracking with audio
            item_updates = self.item_tracker.update_tracking(detections)
            if item_updates and self.audio_enabled:
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

            if not self.enrollment_active:
                for obj in detections:
                    try:
                        match = self.object_memory.recognize_object(
                            frame, tuple(obj['bbox']), obj['class_name']
                        )
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

                            if match_tier == 'high':
                                # Audio with cooldown
                                if self.audio_enabled:
                                    now = time.time()
                                    last = self.last_remembered_announce.get(custom_name, 0)
                                    if (now - last) > self.remember_announce_cooldown:
                                        msg = f"Found your {custom_name}"
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
                                    loc_str = ("near " + ", ".join(nearby_classes[:3])) if nearby_classes else None
                                    # GPS coords are not meaningful indoors
                                    lat, lon = None, None
                                else:
                                    loc_str = reverse_geocode(lat, lon) if (lat is not None and lon is not None) else None

                                self.object_memory.update_object_sighting(
                                    custom_name=custom_name,
                                    image=frame,
                                    location=loc_str,
                                    gps_lat=lat,
                                    gps_lon=lon,
                                    nearby_objects=nearby[:5],
                                    confidence=confidence
                                )
                            elif match_tier == 'ambiguous':
                                # Audio with cooldown for ambiguous
                                if (self.audio_enabled
                                        and time.time() >= self.ambiguous_mute_until.get(custom_name, 0)
                                        and not (self.find_guide and self.find_guide.is_active)):
                                    now = time.time()
                                    last = self.last_remembered_announce.get(custom_name, 0)
                                    if (now - last) > self.remember_announce_cooldown:
                                        msg = f"Is this your {custom_name}?"
                                        self.audio.speak_social(msg)
                                        self.last_remembered_announce[custom_name] = now
                                        result['audio_commands'].append({
                                            'type': 'social', 'text': msg
                                        })
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

                    if self.enrollment_frames_captured >= 5:
                        success = self.object_memory.finish_enrollment()
                        name = self.enrollment_target_obj['custom_name']
                        if success:
                            logger.info(f"[OK] Enrolled '{name}'")
                            if self.audio_enabled:
                                self.audio.speak_social(
                                    f"Remembered your {name}"
                                )
                        else:
                            logger.error(f"[FAIL] Enrollment failed for '{name}'")
                        self.enrollment_active = False
                        self.enrollment_target_obj = None
                        self.enrollment_frames_captured = 0
            self._stage_ms['memory'] += (time.perf_counter() - t_mem_start) * 1000.0

            # ── 3. Face Recognition (every FACE_RECOGNITION_FRAME_INTERVAL frames) ──
            t_face_start = time.perf_counter()
            if self.frame_count % config.FACE_RECOGNITION_FRAME_INTERVAL == 0:
                faces = self.face_recognizer.recognize_faces(frame)

                self.last_recognized_faces = [
                    f['name'] for f in faces if f['name'] != 'Unknown'
                ]

                for face in faces:
                    result['faces'].append({
                        'name': face['name'],
                        'confidence': float(face.get('confidence', 0.0)),
                        'bbox': [int(x) for x in face['bbox']]
                    })

                    # Audio for recognized faces with cooldown
                    if self.audio_enabled and face['name'] != 'Unknown':
                        now = time.time()
                        last = self.last_face_announce.get(face['name'], 0)
                        if (now - last) > self.face_announce_interval:
                            msg = f"Hello {face['name']}"
                            self.audio.speak_social(msg)
                            self.last_face_announce[face['name']] = now
                            result['audio_commands'].append({
                                'type': 'social', 'text': msg
                            })
            self._stage_ms['face'] += (time.perf_counter() - t_face_start) * 1000.0

            # ── 4. Spatial Navigation (obstacle avoidance) ──
            if self.navigation_enabled and not self.enrollment_active:
                guidance = self.spatial_navigator.generate_guidance(detections)

                now = time.time()
                if guidance and (now - self.last_navigation_announce) > \
                        self.navigation_announce_cooldown:
                    top = guidance[0]
                    if self.audio_enabled:
                        if top.get('priority') == 1:
                            self.audio.speak_emergency(top['message'])
                        else:
                            self.audio.speak_navigational(top['message'])
                    self.last_navigation_announce = now

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
                if ultrasonic_distance < 50:
                    result['navigation'] = \
                        f"Obstacle {ultrasonic_distance:.0f} cm ahead"
                elif result['objects']:
                    names = [o['class'] for o in result['objects'][:2]]
                    result['navigation'] = f"Detected: {', '.join(names)}"

        except Exception as e:
            logger.error(f"ML processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._stage_ms['total_ml'] += (time.perf_counter() - t_ml_start) * 1000.0

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
                if self.audio_enabled:
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
                if self.audio_enabled:
                    self.audio.speak_social(
                        "You have arrived at your destination"
                    )
                logger.info("[OK] GPS Navigation: ARRIVED!")
                self.gps_navigation_active = False

        # Obstacle warnings during GPS nav
        if self.audio_enabled:
            warnings = self.obstacle_fusion.generate_obstacle_warnings(detections)
            if warnings and warnings[0].get('priority', 99) <= 2:
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
            label = f"* {mem_obj['name']} ({mem_obj['confidence']:.0%})"
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
        bar_w = int(min(ultrasonic_distance, 200) / 200 * display.shape[1])
        bar_c = ((0, 255, 0) if ultrasonic_distance > 100
                 else (0, 165, 255) if ultrasonic_distance > 50
                 else (0, 0, 255))
        cv2.rectangle(display, (0, 0), (bar_w, 20), bar_c, -1)
        cv2.putText(display, f"Dist: {ultrasonic_distance:.0f}cm", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
            "Q:Quit E:Face R:Remember L:List F:Find C:Confirm X:Reject A:Audio N:Nav G:GPS M:Mock V:Voice SPACE:Talk D:DB",
            (10, display.shape[0] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1
        )

        cv2.imshow('VisionMate - Full Pipeline', display)
        key = cv2.waitKey(1) & 0xFF
        self._handle_keyboard(key, frame)

    # ─── Keyboard Controls (ALL from test_integration.py) ────

    def _handle_keyboard(self, key, frame):
        """Handle all keyboard controls"""
        if key == 255:
            return  # No key pressed

        if key == ord('q'):
            self.show_display = False
            cv2.destroyAllWindows()

        elif key == ord('e'):
            # Enroll face
            name = input("\nEnter name for face enrollment: ").strip()
            if name:
                success = self.face_recognizer.enroll_face(frame, name)
                if success:
                    logger.info(f"[OK] Enrolled face: {name}")
                    if self.audio_enabled:
                        self.audio.speak_social(f"Enrolled {name}")
                else:
                    logger.warning("No face detected in frame")

        elif key == ord('r'):
            # Remember object
            if not self.enrollment_active:
                if self.current_objects:
                    print("\nDetected objects:")
                    for i, obj in enumerate(self.current_objects):
                        print(f"  {i + 1}. {obj['class_name']}")
                    try:
                        idx = int(input("Select object number: ").strip()) - 1
                        if 0 <= idx < len(self.current_objects):
                            obj = self.current_objects[idx]
                            custom_name = input(
                                "Enter custom name (e.g., 'house keys'): "
                            ).strip()
                            if custom_name:
                                try:
                                    self.object_memory.start_enrollment(
                                        custom_name, obj['class_name']
                                    )
                                    self.enrollment_active = True
                                    self.enrollment_target_obj = {
                                        'custom_name': custom_name,
                                        'yolo_class': obj['class_name']
                                    }
                                    self.enrollment_frames_captured = 0
                                    logger.info(
                                        f"[OK] Enrolling '{custom_name}' - "
                                        f"keep in view for 5 frames..."
                                    )
                                except ValueError as e:
                                    print(f"[FAIL] {e}")
                    except (ValueError, IndexError):
                        print("[FAIL] Invalid selection")
                else:
                    print("[FAIL] No objects detected")

        elif key == ord('l'):
            # List remembered objects
            remembered_list = self.object_memory.list_remembered_objects()
            print(f"\nRemembered Objects ({len(remembered_list)}):")
            if remembered_list:
                for name, yolo_class in remembered_list:
                    print(f"  - {name} ({yolo_class})")
            else:
                print("  (none)")

        elif key == ord('f'):
            # Find item/object
            item = input("\nEnter item/object to find: ").strip()
            if item:
                location = self.object_memory.get_object_location(item)
                if location:
                    nearby = (location['context'].get('nearby_objects', [])
                              if location['context'] else [])
                    nearby_str = f" near {', '.join(nearby)}" if nearby else ""
                    msg = f"Your {item} last seen{nearby_str}"
                    print(f"[OK] {msg}")
                    if location.get('location'):
                        print(f"  Place: {location['location']}")
                    print(f"  Time: {location['timestamp']}")
                    if location.get('gps_lat') is not None and location.get('gps_lon') is not None:
                        print(f"  GPS: {location['gps_lat']:.6f}, {location['gps_lon']:.6f}")
                    else:
                        print("  GPS: not recorded for this sighting")
                    if self.audio_enabled:
                        self.audio.speak_navigational(msg)
                else:
                    loc = self.item_tracker.query_item_location(item)
                    if loc:
                        msg = f"Your {item} are on the {loc}"
                        print(f"[OK] {msg}")
                        if self.audio_enabled:
                            self.audio.speak_navigational(msg)
                    else:
                        print(f"[FAIL] Haven't seen {item}")

        elif key == ord('c'):
            # Confirm pending ambiguous match (expires after 15s)
            pending = self.object_memory.get_pending_match()
            if pending:
                name = pending.get('custom_name')
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
                    self.ambiguous_mute_until[name] = time.time() + self.CONFIRM_MUTE_SECONDS
                    if self.audio_enabled:
                        self.audio.speak_social("Confirmed")
                    logger.info(f"[OK] Confirmed match for '{name}' (location: {loc_str or '(unknown location)'})")
                else:
                    logger.warning(f"[FAIL] Could not record confirmed match for '{name}'")
            else:
                logger.info("No pending match to confirm (none, or it expired after 15s)")

        elif key == ord('x'):
            # Reject pending ambiguous match
            pending = self.object_memory.get_pending_match()
            if pending:
                self.ambiguous_mute_until[pending['custom_name']] = time.time() + self.REJECT_MUTE_SECONDS
                self.object_memory.clear_pending_match()
                if self.audio_enabled:
                    self.audio.speak_social("Okay, never mind.")
                logger.info("[OK] Rejected pending match")
            else:
                logger.info("No pending match to reject (none, or it expired after 15s)")

        elif key == ord('a'):
            self.audio_enabled = not self.audio_enabled
            logger.info(f"Audio: {'ON' if self.audio_enabled else 'OFF'}")

        elif key == ord('n'):
            self.navigation_enabled = not self.navigation_enabled
            logger.info(
                f"Navigation: {'ON' if self.navigation_enabled else 'OFF'}"
            )

        elif key == ord('g'):
            # GPS Navigation
            if not self.gps_navigation_active:
                destination = input(
                    "\nEnter destination (e.g., 'Majestic'): "
                ).strip()
                if destination:
                    logger.info(f"Starting GPS navigation to {destination}...")
                    if self.audio_enabled:
                        self.audio.speak_social(f"Navigating to {destination}")
                    dest_coords = self.osm_navigator.geocode_address(
                        destination + ", Bangalore, India"
                    )
                    if not dest_coords:
                        print("[FAIL] Could not find destination")
                        if self.audio_enabled:
                            self.audio.speak_status(
                                "Could not find destination"
                            )
                    else:
                        current_pos = self.gps_tracker.get_position()
                        if not current_pos:
                            print("[FAIL] Waiting for GPS signal")
                        else:
                            route = self.osm_navigator.get_directions(
                                current_pos, dest_coords, mode='foot'
                            )
                            if route:
                                self.osm_navigator.current_route = route
                                self.gps_navigation_active = True
                                self.gps_nav_step = 0
                                self.last_gps_announcement = None
                                print(
                                    f"[OK] Route: {route['total_distance']}, "
                                    f"{route['total_duration']}, "
                                    f"{len(route['steps'])} steps"
                                )
                                if self.audio_enabled:
                                    self.audio.speak_social(
                                        f"Route found. "
                                        f"{route['total_distance']}. "
                                        f"{route['total_duration']}."
                                    )
                            else:
                                print("[FAIL] Could not find route")
            else:
                self.gps_navigation_active = False
                logger.info("GPS navigation stopped")
                if self.audio_enabled:
                    self.audio.speak_status("GPS navigation stopped")

        elif key == ord('m'):
            # Advance mock GPS
            if config.GPS_MOCK_MODE:
                self.gps_tracker.advance_mock_position()
                pos = self.gps_tracker.get_position()
                if pos:
                    print(f"Mock GPS -> {pos[0]:.6f}, {pos[1]:.6f}")
            else:
                print("[WARN] GPS is in REAL mode")

        elif key == ord('v'):
            # Toggle voice
            if self.voice_system:
                self.voice_mode_enabled = not self.voice_mode_enabled
                status = "ON" if self.voice_mode_enabled else "OFF"
                logger.info(f"Voice mode: {status}")
                if self.audio_enabled:
                    self.audio.speak_status(f"Voice mode {status}")
            else:
                print("[WARN] Voice system not available")

        elif key == 32:  # SPACE
            # Trigger voice input
            if self.voice_system and self.voice_mode_enabled:
                logger.info("Listening for voice command...")
                if self.audio_enabled:
                    self.audio.speak_status("Listening")
                self.voice_system.trigger_voice_input()
            elif not self.voice_system:
                print("[WARN] Voice system not available")
            else:
                print("[WARN] Voice mode OFF - press V to enable")

        elif key == ord('d'):
            # Show database contents
            self._show_database_contents()

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
        while True:
            try:
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, self._fetch_mobile_location)
                if resp and resp.status_code == 200:
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
            except (requests.exceptions.RequestException, Exception) as e:
                logger.debug(f"Mobile backend location polling: {e}")

            await asyncio.sleep(self.mobile_backend_poll_interval)

    async def start_server(self):
        """Start WebSocket server"""
        laptop_ip = self.get_local_ip()

        logger.info("\n" + "=" * 60)
        logger.info("SERVER READY - WAITING FOR RASPBERRY PI")
        logger.info("=" * 60)
        logger.info(f"\nLaptop IP Address: {laptop_ip}")
        logger.info(f"Server Port: {self.port}")
        logger.info(f"\nOn Raspberry Pi, run:")
        logger.info(f"  python3 rpi_client.py {laptop_ip}")
        logger.info("\n" + "=" * 60 + "\n")

        # Poll mobile backend for phone GPS updates in background
        asyncio.create_task(self._poll_mobile_backend_location())

        async with websockets.serve(
            self.process_frame, self.host, self.port, max_size=10**7
        ):
            await asyncio.Future()

    def cleanup(self):
        """Clean up all resources"""
        if self.gps_tracker:
            self.gps_tracker.stop()
        if self.ultrasonic:
            self.ultrasonic.stop()
        if self.voice_system:
            self.voice_system.stop()
        if self.db:
            self.db.close()
        cv2.destroyAllWindows()


def main():
    """Main entry point"""
    server = LaptopMLServer(port=8081)

    if not server.initialize():
        logger.error("Failed to initialize. Exiting.")
        return

    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("\n\nShutting down server...")
    finally:
        server.cleanup()
        logger.info("Goodbye!")


if __name__ == "__main__":
    main()
