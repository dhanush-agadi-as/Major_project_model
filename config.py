"""
VisionMate Configuration File
Contains all system parameters and settings
"""
import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "data", "visionmate.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEST_DATA_DIR = os.path.join(BASE_DIR, "test", "test_data")

# Camera Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# YOLOv8 Configuration
YOLO_MODEL = "yolov8n.pt"  # Nano model for speed
YOLO_OPENVINO_MODEL = "yolov8n_openvino_model/"  # Path to OpenVINO-exported model, used if USE_OPENVINO is True
USE_OPENVINO = True  # Try Intel OpenVINO acceleration (uses idle iGPU/NPU) instead of plain PyTorch CPU inference
OPENVINO_DEVICE = "GPU"  # Target OpenVINO device: 'GPU', 'NPU', 'CPU', or 'AUTO'
YOLO_CONFIDENCE = 0.5
YOLO_IOU_THRESHOLD = 0.45
YOLO_IMGSZ = 320  # Lowered further from 416 for CPU speed - trade-off is slightly reduced accuracy on small/distant objects
YOLO_FRAME_SKIP = 3  # Run YOLO object detection every Nth frame instead of every frame (1 = every frame, 2 = every other frame)
TARGET_CLASSES = ['person', 'chair', 'bottle', 'couch', 'tv', 'keys', 'cell phone', 'clock']  # 'couch' is COCO's 'sofa', 'clock' includes watches

# Face Recognition Configuration
FACE_RECOGNITION_THRESHOLD = 0.65  # Euclidean distance threshold (higher = more lenient)
FACE_MODEL = "buffalo_l"  # InsightFace model
# Lower values (even (160, 160)) can be tried for more speed at the cost of missing small/distant faces.
FACE_DET_SIZE = (128, 128)  # Lowered further - getting close to the practical floor for reliable face detection, verify accuracy carefully after this change
FACE_RECOGNITION_FRAME_INTERVAL = 14  # Increased further - noticeably slower to detect new faces appearing

# Item Tracking Configuration
ITEM_TRACKING_TIMEOUT = 3.0  # seconds
IOU_THRESHOLD = 0.3  # Minimum IoU to consider "inside"

# Object Memory Configuration (Remember Objects Feature)
OBJECT_MEMORY_MATCH_THRESHOLD = 0.62  # Middle ground based on real similarity data: true matches range 0.4-0.71, this is a compromise between false positives and false negatives
OBJECT_ENROLLMENT_FRAMES = 10  # Increased from 5 - more frames averaged into the embedding should make it more robust to lighting/angle variation, which should raise genuine match confidence
OBJECT_MATCH_COOLDOWN = 2.0  # seconds between announcing same object
USE_DEEP_FEATURES = True  # Use deep learning embeddings (MobileNetV2) for much better recognition
OBJECT_RECOGNITION_FRAME_INTERVAL = 3  # Only run remembered-object recognition every N frames (matches face recognition's existing cadence)
OBJECT_CANDIDATES_CACHE_TTL = 5.0  # Seconds to cache remembered-object DB reads before re-querying the database

# Traffic Signal Configuration
TRAFFIC_HSV_RED_LOWER = [0, 120, 70]
TRAFFIC_HSV_RED_UPPER = [10, 255, 255]
TRAFFIC_HSV_GREEN_LOWER = [40, 40, 40]
TRAFFIC_HSV_GREEN_UPPER = [80, 255, 255]
TRAFFIC_MIN_AREA = 100  # Minimum pixel area for signal detection

# Audio Feedback Priority (1 = highest)
PRIORITY_EMERGENCY = 1
PRIORITY_SOCIAL = 2
PRIORITY_NAVIGATIONAL = 3
PRIORITY_STATUS = 4

# Server Configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False

# Google Maps API Configuration
# SECURITY NOTE: Keep this key private! Add config.py to .gitignore before pushing to GitHub
GOOGLE_MAPS_API_KEY = "AIzaSyBqXAmgcRTvtbuW7vxE_6Jv0aKfQ4yu4FM"

# Navigation Configuration
NAVIGATION_ANNOUNCEMENT_DISTANCE = 50  # meters - announce turns when this close
NAVIGATION_REROUTE_DISTANCE = 30  # meters - reroute if user is this far off track
NAVIGATION_ARRIVAL_DISTANCE = 20  # meters - consider arrived when within this distance
NAVIGATION_UPDATE_INTERVAL = 5  # seconds - how often to check position

# GPS Configuration
GPS_MOCK_MODE = True  # Set to False when real GPS is connected
GPS_PORT = "/dev/ttyUSB0"  # Serial port for GPS module (adjust for your system)
GPS_BAUDRATE = 9600

# Ultrasonic Sensor Configuration
ULTRASONIC_MOCK_MODE = True  # Set to False when ESP32 is connected
ULTRASONIC_PORT = "COM3"  # Serial port for ESP32 (Windows: COM3, Linux: /dev/ttyUSB1)
ULTRASONIC_BAUDRATE = 115200
ULTRASONIC_MAX_DISTANCE = 400  # cm - maximum reliable distance
ULTRASONIC_OBSTACLE_THRESHOLD = 100  # cm - warn if obstacle closer than this

# Performance
TARGET_FPS = 30
MAX_FRAME_QUEUE = 10
