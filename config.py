"""
VisionMate Configuration File
Contains all system parameters and settings
"""
import os


from dotenv import load_dotenv

load_dotenv()

# Thread limits
OPENVINO_THREADS = 8
FACE_THREADS = 4
os.environ.setdefault("OMP_NUM_THREADS", str(FACE_THREADS))
os.environ.setdefault("ONNXRUNTIME_INTRA_OP_NUM_THREADS", str(FACE_THREADS))

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


# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "data", "visionmate.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEST_DATA_DIR = os.path.join(BASE_DIR, "test", "test_data")

# Camera Configuration
USE_PI_CAMERA = True
PI_HOST = "raspberrypi.local"
PI_STREAM_URL = f"http://{PI_HOST}:5000/video"
SENSORS_URL = f"http://{PI_HOST}:5000/sensors"
DISTANCE_URL = f"http://{PI_HOST}:5000/distance"
MOTION_URL = f"http://{PI_HOST}:5000/motion"
PI_MOTION_URL = MOTION_URL
OBSTACLE_WARN_CM = 100
OBSTACLE_STOP_CM = 40
OBSTACLE_BEEPS = True

_cam_env = os.environ.get("VISIONMATE_CAMERA")
if _cam_env is not None:
    _cam_env = _cam_env.strip()
    if _cam_env.isdigit():
        CAMERA_SOURCE = int(_cam_env)
    else:
        CAMERA_SOURCE = _cam_env
else:
    CAMERA_SOURCE = PI_STREAM_URL if USE_PI_CAMERA else 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

ULTRASONIC_MOCK_MODE = not USE_PI_CAMERA

# Pi camera (OV5647, ~53.5 degrees horizontal view) differs from the webcam.
# estimate from OV5647 spec; re-calibrate with tools/calibrate_focal.py
FOCAL_PX = 636.0

# YOLOv8 Configuration
YOLO_MODEL = os.path.join(BASE_DIR, "yolov8n.pt")  # Nano model for speed
YOLO_OPENVINO_MODEL = os.path.join(BASE_DIR, "yolov8n_openvino_model")  # Path to OpenVINO-exported model, used if USE_OPENVINO is True
USE_OPENVINO = True  # Try Intel OpenVINO acceleration (uses idle iGPU/NPU) instead of plain PyTorch CPU inference
OPENVINO_DEVICE = "CPU"  # Target OpenVINO device: 'GPU', 'NPU', 'CPU', or 'AUTO'
YOLO_CONFIDENCE = 0.5
YOLO_IOU_THRESHOLD = 0.45
YOLO_IMGSZ = 320  # Lowered further from 416 for CPU speed - trade-off is slightly reduced accuracy on small/distant objects
YOLO_FRAME_SKIP = 3  # Run YOLO object detection every Nth frame instead of every frame (1 = every frame, 2 = every other frame)
TARGET_CLASSES = ['person', 'chair', 'bottle', 'couch', 'tv', 'keys', 'cell phone', 'clock']  # 'couch' is COCO's 'sofa', 'clock' includes watches

# Face Recognition Configuration
FACE_RECOGNITION_THRESHOLD = 0.65  # Euclidean distance threshold (higher = more lenient)
FACE_SIMILARITY_THRESHOLD = 0.60   # Cosine similarity threshold (higher = stricter)
FACE_MODEL = "buffalo_l"  # InsightFace model
# Increased to (320, 320) to better handle partial faces and occlusions at the cost of slight latency.
FACE_DET_SIZE = (320, 320)
FACE_ENROLLMENT_FRAMES = 5  # Capture multiple frames across angles for robustness
FACE_RECOGNITION_FRAME_INTERVAL = 14  # Increased further - noticeably slower to detect new faces appearing
FACE_NAME_HINTS = ["Bharath", "Dhanush"]  # Editable name hints for voice enrollment disambiguation/correction


# Item Tracking Configuration
ITEM_TRACKING_TIMEOUT = 3.0  # seconds
IOU_THRESHOLD = 0.3  # Minimum IoU to consider "inside"

# Object Memory Configuration (Remember Objects Feature)
# Replaced single threshold with three-tier confidence zone
OBJECT_MATCH_HIGH_CONFIDENCE = 0.75  # Auto-confirm match and update location
SINGLE_OBJECT_MATCH_MIN = 0.35  # Treat match as high-confidence if exactly one remembered object exists for this class
OBJECT_MATCH_AMBIGUOUS_LOW = 0.50  # Ask user to confirm
ENROLL_MAX_FRAMES = 8  # Maximum frames captured during enrollment
ENROLL_AUGMENT_PER_FRAME = 2  # Augmentations generated per frame crop (e.g. +-15 deg rotation)
OBJECT_ENROLLMENT_FRAMES = ENROLL_MAX_FRAMES  # Alias for backward compatibility
OBJECT_ENROLLMENT_DURATION = 3.0  # Duration in seconds to capture multi-pose enrollment frames
MATCH_EVERY_N_FRAMES = 3  # Compute object recognition embeddings only every 3rd frame per track outside find mode
OBJECT_MATCH_COOLDOWN = 2.0  # seconds between announcing same object
USE_DEEP_FEATURES = True  # Use deep learning embeddings (MobileNetV2) for much better recognition
OBJECT_RECOGNITION_FRAME_INTERVAL = 3  # Only run remembered-object recognition every N frames (matches face recognition's existing cadence)
OBJECT_CANDIDATES_CACHE_TTL = 5.0  # Seconds to cache remembered-object DB reads before re-querying the database

# Find Guidance Beep Profile (distance-dependent beep timing, duration, and pitch)
FIND_BEEP_PROFILE = {
    "far": {
        "min_dist": 200.0,
        "duration_ms": 60,
        "interval_s": 1.2,
        "freq_hz": 800,
    },
    "mid": {
        "min_dist": 100.0,
        "duration_ms": 90,
        "interval_s": 0.7,
        "freq_hz": 1000,
    },
    "near": {
        "min_dist": 60.0,
        "duration_ms": 140,
        "interval_s": 0.4,
        "freq_hz": 1200,
    },
    "reach": {  # within arm's reach (< 60 cm)
        "min_dist": 0.0,
        "duration_ms": 220,
        "interval_s": 0.25,
        "freq_hz": 1400,
    },
    "arrival": {
        "duration_ms": 600,
        "freq_hz": 1400,
    },
}

# Traffic Signal Configuration
TRAFFIC_HSV_RED_LOWER = [0, 120, 70]
TRAFFIC_HSV_RED_UPPER = [10, 255, 255]
TRAFFIC_HSV_GREEN_LOWER = [40, 40, 40]
TRAFFIC_HSV_GREEN_UPPER = [80, 255, 255]
TRAFFIC_MIN_AREA = 100  # Minimum pixel area for signal detection

# Audio Feedback Priority (1 = highest)
PRIORITY_EMERGENCY = 1
PRIORITY_PROMPT = 2
PRIORITY_SOCIAL = 3
PRIORITY_NAVIGATIONAL = 4
PRIORITY_STATUS = 5

STARTUP_READ_COMMANDS = True

# Audio Output Target
CLIENT_SPEAKS = False  # set True on the real Raspberry Pi where the speaker is on the glasses; False when client and server run on the same laptop


# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8081
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False

# Google Maps API Configuration
# SECURITY NOTE: Keep this key private! Add config.py to .gitignore before pushing to GitHub
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# Navigation Configuration
NAVIGATION_ANNOUNCEMENT_DISTANCE = 50  # meters - announce turns when this close
NAVIGATION_REROUTE_DISTANCE = 30  # meters - reroute if user is this far off track
NAVIGATION_ARRIVAL_DISTANCE = 20  # meters - consider arrived when within this distance
NAVIGATION_UPDATE_INTERVAL = 5  # seconds - how often to check position

# GPS Configuration
GPS_MOCK_MODE = False  # Set to False when real GPS is connected
PHONE_GPS_URL = "https://127.0.0.1:5001/api/location"
PHONE_GPS_MAX_AGE_S = 30
GPS_PORT = "/dev/ttyUSB0"  # Serial port for GPS module (adjust for your system)
GPS_BAUDRATE = 9600
GPS_USE_IP_GEOLOCATION = True  # Set True to use free IP-based geolocation instead of mock/serial GPS
IP_GEOLOCATION_CACHE_SECONDS = 300  # How long to cache a lookup before re-querying ip-api.com
GPS_USE_WIFI_GEOLOCATION = False  # WiFi layer removed due to MLS shutdown
GPS_MANUAL_OVERRIDE_ENABLED = False  # Set True to pin a fixed location for indoor/bench testing (ignores all network lookups)
GPS_MANUAL_OVERRIDE_LAT = 12.926516  # Latitude for manual override (default: RR Nagar, Bangalore -- change as needed)
GPS_MANUAL_OVERRIDE_LON = 77.526422  # Longitude for manual override

INDOOR_MODE = True  # When True, skip outdoor reverse-geocoding for sightings and build location text
                    # from nearby detected objects instead. Set False to restore outdoor
                    # GPS/reverse-geocoding behavior.

# Ultrasonic Sensor Configuration
ULTRASONIC_MOCK_MODE = not USE_PI_CAMERA  # Real ultrasonic via Pi/ESP32 when USE_PI_CAMERA is True
ULTRASONIC_PORT = "COM3"  # Serial port for ESP32 (Windows: COM3, Linux: /dev/ttyUSB1)
ULTRASONIC_BAUDRATE = 115200
ULTRASONIC_MAX_DISTANCE = 400  # cm - maximum reliable distance
ULTRASONIC_OBSTACLE_THRESHOLD = 100  # cm - warn if obstacle closer than this

# Performance
TARGET_FPS = 30
MAX_FRAME_QUEUE = 10

# Voice & Interface Configuration
KEYBOARD_FALLBACK = False  # Set True for minimal debug hotkeys. Default is fully voice-only.
AUTO_READ_NOTES = True     # Automatically read back newly saved voice notes.
REQUIRE_SHUTDOWN_CONFIRM = False  # Set True to keep old yes/no confirmation flow before shutdown
WAKE_WORDS = [
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
]
WAKE_WORD_TIMEOUT = 6.0    # Seconds to wait for command after wake word
YES_NO_TIMEOUT = 6.0       # Seconds to wait for yes/no confirmation window

# Speech-to-Text Engine: "whistle" (offline Cactus Whistle), "google" (online Google STT), "auto" (Whistle first, fallback to Google if empty and online), "hybrid" (Google first when online, fallback to Whistle on error or offline)
STT_ENGINE = "hybrid"

# Emergency Alert & Fall Detection Configuration
EMERGENCY_USER_NAME = os.environ.get("VISIONMATE_USER_NAME", "Bharath")
EMERGENCY_DRY_RUN = True  # Safety default: log what would be sent, send nothing
FIREBASE_SERVICE_ACCOUNT_PATH = os.environ.get(
    "FIREBASE_SERVICE_ACCOUNT_PATH",
    os.path.join(BASE_DIR, "firebase_service_account.json")
)
EMERGENCY_CONFIG_PATH = os.path.join(BASE_DIR, "emergency_config.json")
PI_MOTION_URL = MOTION_URL
CHECKIN_REPLY_WINDOW_S = 3.0
EMERGENCY_CHECKIN_TIMEOUT_S = 3.0
EMERGENCY_CHECKIN_MAX_ASKS = 3
EMERGENCY_COOLDOWN_S = 60.0
EMERGENCY_AUTO_RATE_LIMIT_S = 300.0  # At most 1 automatic alert per 5 minutes
FALL_TRIGGER_ON_IMPACT = True  # Laptop starts check-in on either fall or impact event


# Debug flag to log ThreadPoolExecutor creation stack once at startup



