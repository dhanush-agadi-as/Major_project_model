# VisionMate — AI-Powered Assistive Smart Glasses

Real-time computer vision + ML pipeline for visually impaired users.

VisionMate is an assistive system built around a Raspberry Pi with a camera module (the "glasses") wirelessly connected to a laptop running all ML models. A Flask backend also exposes a mobile-facing API for a companion caretaker app.

## System Architecture

```
┌──────────────────┐         WiFi (WebSocket)          ┌──────────────────────┐
│   Raspberry Pi     │ ────────────────────────────────► │   Laptop ML Server    │
│   (Smart Glasses)  │  Sends: camera frames,             │   (Port 8080)         │
│                     │  ultrasonic data, button press     │                       │
│  • Pi Camera v2      │ ◄──────────────────────────────── │  • YOLOv8 detection   │
│  • Ultrasonic         │  Returns: detections, faces,      │  • InsightFace        │
│  • Button/GPIO         │  nav instructions, audio cmds     │  • Object Memory      │
│  • Speaker              │                                   │  • Spatial Navigator  │
└──────────────────┘                                     │  • GPS Navigation     │
                                                          │  • Voice Commands     │
                                                          │  • Audio Feedback     │
┌──────────────────┐         WiFi (HTTP REST)            │                       │
│   Mobile Backend  │ ◄──────────────────────────────────  ├──────────────────────┤
│   (Flask, 5001)   │  Serves: location, notifications,    │  Flask API Server     │
│                     │  webcam snapshots                   │  (Port 5000)          │
└──────────────────┘                                     │                       │
                                                          │  • /process_frame     │
                                                          │  • /enroll_face       │
                                                          │  • /query_item        │
                                                          └──────────────────────┘
```

## Features

### ML Pipeline (Laptop Server — `wireless/laptop_server.py`)

| Feature | Description |
|---|---|
| Object Detection | YOLOv8n real-time detection, filtered to a configurable set of target classes |
| Face Recognition | InsightFace (buffalo_l) enrollment + recognition with cooldown announcements |
| Object Memory (Remember/Find) | "Remember this phone" → enrolls object visually via deep CNN features. "Find my phone" → reports where it was last seen |
| Item Tracking | Passive IoU-based tracking of detected objects across frames |
| Spatial Navigation | Obstacle avoidance guidance from visual detections |
| GPS Navigation | Turn-by-turn walking directions via OpenStreetMap (mock GPS mode available for testing) |
| Obstacle Fusion | Combines ultrasonic sensor + visual detections for proximity warnings |
| Voice Commands | Speech recognition for hands-free control (requires a working microphone/PyAudio) |
| Audio Feedback | Priority-based TTS (Emergency > Social > Navigational > Status) via pyttsx3 |

### Mobile Backend (`mobile_backend.py`, Port 5001)

| Feature | Description |
|---|---|
| Live Location | Serves GPS position of the glasses (mock or real) |
| Camera Feed | Live webcam stream via snapshot/MJPEG polling |
| Notifications | Queue text notifications for a connected client to poll |

## Project Structure

```
Major_project_model/
├── wireless/
│   ├── laptop_server.py          # Main ML server (WebSocket, port 8080)
│   ├── laptop_server_backup.py   # Backup/previous version of the server
│   └── rpi_client.py             # Raspberry Pi camera client
│
├── mobile_backend.py             # Flask REST API for mobile app (port 5001)
├── server.py                     # Standalone Flask API server (port 5000)
│
├── object_detector.py            # YOLOv8 wrapper
├── face_recognizer.py            # InsightFace wrapper
├── object_memory.py              # Remember/Find object system
├── deep_feature_extractor.py     # CNN feature extraction for object memory
├── visual_matcher.py             # Visual matching for remembered objects
├── item_tracker.py               # IoU-based passive item tracking
├── spatial_navigator.py          # Obstacle avoidance from detections
├── navigation_engine.py          # Navigation logic
├── gps_tracker.py                # GPS (real serial or mock mode)
├── osm_navigator.py              # OpenStreetMap routing
├── google_maps_navigator.py      # Google Maps integration
├── transit_guide.py              # Public transit guidance
├── obstacle_fusion.py            # Ultrasonic + visual fusion
├── voice_command_system.py       # Speech recognition commands
├── audio_feedback.py             # Priority-based TTS
├── traffic_signal_detector.py    # HSV-based traffic light detection
├── ultrasonic_sensor.py          # HC-SR04 / ESP32 sensor interface
├── camera_reader.py              # Threaded webcam capture
├── database.py                   # SQLite database manager
├── config.py                     # All configuration settings
│
├── hardware/                     # RPi/ESP32 wiring guide and firmware
├── test/                         # Test scripts
│   └── test_integration.py       # Full pipeline test (laptop webcam)
│
├── requirements.txt               # Core Python dependencies
├── requirements_navigation.txt    # Extra dependencies for navigation features
├── .env                           # Local secrets (NOT committed — see Configuration)
└── yolov8n.pt                     # YOLOv8 nano weights (gitignored)
```

> Note: this repo does not currently include a mobile app frontend — the mobile-facing pieces are the `mobile_backend.py` / `server.py` REST APIs only.

## Setup & Installation

### Prerequisites

- Python 3.10+
- Conda (recommended) or venv
- Webcam on the laptop
- GPU (optional — OpenVINO acceleration supported, significantly faster inference)
- Raspberry Pi 4/5 with Pi Camera (for wireless mode)

### 1. ML Backend Setup (Laptop)

```bash
# Create conda environment
conda create -p <path-to-your-env> python=3.10 -y
conda activate <path-to-your-env>

# Install dependencies
pip install -r requirements.txt

# First run auto-downloads YOLOv8 (~6MB) and InsightFace (~200MB)
```

### 2. Configuration — API keys

Google Maps navigation requires an API key. This is loaded from a local `.env` file (never commit this file):

```
GOOGLE_MAPS_API_KEY=your_key_here
```

`config.py` loads this automatically via `python-dotenv` at startup. If no key is set, `GOOGLE_MAPS_API_KEY` will be empty and Google Maps-based navigation features will not work — everything else (object detection, face recognition, object memory, OSM-based navigation) works without it.

### 3. Raspberry Pi Setup (wireless mode)

```bash
ssh <user>@<RPI_IP>

# Install picamera2, websockets, pyserial on the Pi
pip install picamera2 websockets pyserial

# Copy rpi_client.py to the Pi
scp wireless/rpi_client.py <user>@<RPI_IP>:~/VisionMate/
```

## Running the System

### Mode 1: Laptop Only (Testing with Webcam)

```bash
conda activate <your-env>
python test/test_integration.py
```

Keyboard controls on the display window:

| Key | Action |
|---|---|
| Q | Quit |
| E | Enroll a face |
| R | Remember object (enroll custom object) |
| L | List all remembered objects |
| F | Find item/object location |
| A | Toggle audio feedback |
| N | Toggle navigation guidance (obstacle avoidance) |
| G | GPS navigation to a destination |
| M | Advance mock GPS position |
| V | Toggle voice command mode |

### Mode 2: Wireless (RPi → Laptop)

**Terminal 1 — Laptop ML Server:**
```bash
conda activate <your-env>
python wireless/laptop_server.py
```

**Terminal 2 — RPi Client (SSH into the Pi):**
```bash
python rpi_client.py
```

The RPi streams camera frames over WebSocket (port 8080). The laptop processes everything and sends results back.

### Mode 3: Mobile Backend

```bash
conda activate <your-env>
python mobile_backend.py
```
Runs the Flask REST API on port 5001 for a caretaker-facing client to poll.

## API Endpoints

### Mobile Backend (Port 5001)

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Server status, uptime, webcam/GPS info |
| `/api/location` | GET | Current GPS coordinates |
| `/api/video-feed` | GET | MJPEG webcam stream |
| `/api/snapshot` | GET | Single JPEG frame from webcam |
| `/api/notifications` | GET | Poll for pending notifications |

### Flask API Server (Port 5000 — `server.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/process_frame` | POST | Full ML inference on a frame |
| `/enroll_face` | POST | Enroll a new face |
| `/query_item` | GET | Find item location |
| `/get_all_faces` | GET | List enrolled faces |

## Hardware Setup

### Raspberry Pi Wiring

- Pi Camera v2 → CSI port
- HC-SR04 Ultrasonic → GPIO (Trig: pin 23, Echo: pin 24)
- Push Button → GPIO pin 17
- Speaker → 3.5mm audio jack

See `hardware/WIRING_GUIDE.md` for full details.

### Network

- RPi and laptop must be on the same WiFi network
- Laptop ML server: port 8080 (WebSocket)
- Mobile backend: port 5001 (HTTP)
- Flask API server: port 5000 (HTTP)

Firewall rules (Windows — run as admin):
```powershell
netsh advfirewall firewall add rule name="VisionMate ML" dir=in action=allow protocol=TCP localport=8080
netsh advfirewall firewall add rule name="VisionMate Mobile" dir=in action=allow protocol=TCP localport=5001
```

## Configuration Reference

All settings live in `config.py`. Key values (verified against current code):

| Setting | Default | Description |
|---|---|---|
| `YOLO_MODEL` | `yolov8n.pt` | YOLOv8 model variant |
| `YOLO_CONFIDENCE` | `0.5` | Detection confidence threshold |
| `TARGET_CLASSES` | `person, chair, bottle, couch, tv, keys, cell phone, clock` | Object classes YOLO filters to. Note: `keys` is not a real COCO class and will never actually be detected — this entry has no effect |
| `FACE_RECOGNITION_THRESHOLD` | `0.65` | Face match Euclidean distance threshold (higher = more lenient) |
| `OBJECT_MEMORY_MATCH_THRESHOLD` | `0.62` | Similarity threshold for remembered-object recognition |
| `GPS_MOCK_MODE` | `True` | Use simulated GPS path instead of real hardware |
| `ULTRASONIC_MOCK_MODE` | `True` | Use simulated ultrasonic readings instead of real hardware |
| `FLASK_PORT` | `5000` | Port for `server.py` |
| `GOOGLE_MAPS_API_KEY` | *(from `.env`)* | Loaded via `python-dotenv`; empty if unset |

## Performance

| Component | CPU | GPU (OpenVINO) |
|---|---|---|
| YOLOv8n | ~40ms | ~10ms |
| InsightFace | ~60ms | ~15ms |
| Object Memory | ~5ms | ~5ms |
| Spatial Nav | ~2ms | ~2ms |

Wireless mode (RPi → Laptop) is network-bound and typically runs at a lower effective FPS than local webcam testing.

## Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8n (Ultralytics), optional OpenVINO acceleration |
| Face Recognition | InsightFace (buffalo_l) |
| Object Memory | MobileNetV2 CNN feature extraction + cosine similarity |
| Navigation | OpenStreetMap (OSRM) + optional Google Maps |
| GPS | Serial NMEA or mock mode |
| Voice Commands | SpeechRecognition + Google STT (requires PyAudio/working microphone) |
| Audio Feedback | pyttsx3 |
| RPi ↔ Laptop | WebSocket (asyncio) |
| Mobile Backend | Flask + flask-cors |
| Database | SQLite |

## Team

Built by Team NPC's for VisionMate.

## License

MIT License — free for educational and hackathon use.
