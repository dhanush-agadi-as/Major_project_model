# VisionMate — AI-Powered Assistive Smart Glasses

**Real-time computer vision + ML pipeline for visually impaired users**

VisionMate is a complete assistive system built around a Raspberry Pi with a camera module (the "glasses") wirelessly connected to a laptop running all ML models. A companion mobile app lets a caretaker monitor the user's location, receive alerts, and view the live camera feed.

---

## System Architecture

```
┌──────────────────┐         WiFi (WebSocket)        ┌──────────────────────┐
│   Raspberry Pi   │ ──────────────────────────────►  │   Laptop ML Server   │
│   (Smart Glasses)│  Sends: camera frames,           │   (Port 8080)        │
│                  │  ultrasonic data, button press    │                      │
│  • Pi Camera v2  │ ◄──────────────────────────────  │  • YOLOv8 detection  │
│  • Ultrasonic    │  Returns: detections, faces,     │  • InsightFace       │
│  • Button/GPIO   │  nav instructions, audio cmds    │  • Object Memory     │
│  • Speaker       │                                  │  • Spatial Navigator  │
└──────────────────┘                                  │  • GPS Navigation    │
                                                      │  • Voice Commands    │
                                                      │  • Audio Feedback    │
┌──────────────────┐         WiFi (HTTP REST)         │                      │
│   Mobile App     │ ◄──────────────────────────────  ├──────────────────────┤
│   (Expo / RN)    │  Polls: location, notifications  │  Mobile Backend      │
│                  │  Streams: webcam snapshots        │  (Port 5001)         │
│  • Live Map      │                                  │                      │
│  • Camera Feed   │  Terminal CLI → Notifications     │  • Flask REST API    │
│  • Notifications │                                  │  • MJPEG / Snapshots │
└──────────────────┘                                  └──────────────────────┘
```

---

## Features

### ML Pipeline (Laptop Server — `wireless/laptop_server.py`)
| Feature | Description |
|---------|-------------|
| **Object Detection** | YOLOv8n real-time detection of 80+ object classes |
| **Face Recognition** | InsightFace enrollment + recognition with cooldown announcements |
| **Object Memory (Remember/Find)** | "Remember this phone" → enrolls object visually. "Find my phone" → tells you where it was last seen and what was near it |
| **Item Tracking** | Passive IoU-based tracking — "your keys are on the table" |
| **Spatial Navigation** | Obstacle avoidance guidance from visual detections |
| **GPS Navigation** | Turn-by-turn walking directions via OpenStreetMap |
| **Obstacle Fusion** | Combines ultrasonic sensor + visual detections for warnings |
| **Voice Commands** | Speech recognition for hands-free control |
| **Audio Feedback** | Priority-based TTS (Emergency > Social > Navigation > Status) |

### Mobile App Backend (`mobile_backend.py`)
| Feature | Description |
|---------|-------------|
| **Live Location** | Map shows GPS position of the glasses (mock or real) |
| **Camera Feed** | Live webcam stream via snapshot polling (~10 FPS) |
| **Notifications** | Type a message in the terminal → red banner appears on caretaker's phone with vibration |

### Companion Mobile App (`HAL-102-Frontend/`)
| Screen | Description |
|--------|-------------|
| **Onboarding** | Enter caretaker name, phone, and VisionMate user's name |
| **Home** | Dashboard with quick actions + in-app notification banners |
| **Map** | Live location of the glasses on a map (polls every 3s) |
| **Camera Feed** | Live video from the laptop webcam (snapshot polling) |

---

## Project Structure

```
Model_for_VisionMate/
├── wireless/
│   ├── laptop_server.py          # Main ML server (WebSocket, port 8080)
│   └── rpi_client.py             # Raspberry Pi camera client
│
├── mobile_backend.py             # Flask REST API for mobile app (port 5001)
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
├── ultrasonic_sensor.py          # HC-SR04 sensor interface
├── database.py                   # SQLite database manager
├── config.py                     # All configuration settings
├── server.py                     # Standalone Flask API server
│
├── hardware/                     # RPi GPIO and hardware configs
├── test/                         # Test scripts
│   ├── test_integration.py       # Full pipeline test (laptop webcam)
│   └── ...
│
├── requirements.txt              # Python dependencies
└── yolov8n.pt                    # YOLOv8 nano weights
```

---

## Setup & Installation

### Prerequisites
- **Python 3.10+**
- **Conda** (recommended) or venv
- **Webcam** on the laptop
- **GPU** (optional, significantly faster inference)
- **Raspberry Pi 4/5** with Pi Camera (for wireless mode)
- **Node.js 18+** (for mobile app)

### 1. ML Backend Setup (Laptop)

```bash
# Create conda environment
conda create -p D:\conda_envs\visionmate python=3.10 -y
conda activate D:\conda_envs\visionmate

# Install dependencies
pip install -r requirements.txt

# First run auto-downloads YOLOv8 (~6MB) and InsightFace (~200MB)
```

### 2. Raspberry Pi Setup

```bash
ssh pimo@<RPI_IP>

# Install picamera2, websockets, pyserial on the Pi
pip install picamera2 websockets pyserial

# Copy rpi_client.py to the Pi
scp wireless/rpi_client.py pimo@<RPI_IP>:~/VisionMate/
```

### 3. Mobile App Setup

```bash
cd HAL-102-Frontend
npm install

# Update src/config.ts with your laptop's IP address
# Then start Expo
npx expo start
```

Install **Expo Go** on the caretaker's phone & scan the QR code.

---

## Running the System

### Mode 1: Laptop Only (Testing with Webcam)

```bash
conda activate D:\conda_envs\visionmate
python test/test_integration.py
```

Keyboard controls on the display window:

| Key | Action |
|-----|--------|
| **R** | Remember object (enroll) |
| **F** | Find object (where is it?) |
| **L** | List all remembered objects |
| **E** | Enroll a face |
| **N** | Toggle obstacle avoidance |
| **G** | GPS navigation to a destination |
| **M** | Advance mock GPS position |
| **A** | Toggle audio feedback |
| **V** | Toggle voice command mode |
| **SPACE** | Trigger voice command |
| **D** | Show database contents |
| **Q** | Quit |

### Mode 2: Wireless (RPi → Laptop)

**Terminal 1 — Laptop ML Server:**
```bash
conda activate D:\conda_envs\visionmate
python wireless/laptop_server.py
```

**Terminal 2 — RPi Client** (SSH into the Pi):
```bash
python rpi_client.py
```

The RPi streams camera frames over WebSocket. The laptop processes everything and sends results back. Same keyboard controls apply on the display window.

### Mode 3: Mobile App + Backend

**Terminal 1 — Mobile Backend:**
```bash
conda activate D:\conda_envs\visionmate
python mobile_backend.py
```

**Terminal 2 — Expo Dev Server:**
```bash
cd HAL-102-Frontend
npx expo start
```

**On the caretaker's phone:** Open Expo Go → scan the QR code.

**Sending notifications:** Type any message in the mobile_backend terminal:
```
📢 Notify > Help needed at the crossing!
  ✓ Notification queued: "Help needed at the crossing!"
```

A red banner with vibration appears on the phone within ~2 seconds.

**Backend terminal commands:**

| Command | Action |
|---------|--------|
| `<any text>` | Send as notification to the app |
| `/status` | Show server status |
| `/location` | Show current GPS coordinates |
| `/advance` | Move the GPS marker on the map |
| `quit` | Stop the server |

---

## Voice Commands

When voice mode is active (press **V** then **SPACE** on the display window):

| Command | What it does |
|---------|-------------|
| "Remember this phone" | Enrolls the visible object with a custom name |
| "Find my keys" | Reports where the object was last seen |
| "Navigate" | Enables obstacle avoidance guidance |
| "Stop navigation" | Disables obstacle avoidance |
| "Navigate to Majestic" | Starts GPS turn-by-turn walking directions |
| "List all items" | Lists everything you've remembered |
| "Help" | Reads out available commands |

---

## API Endpoints

### Mobile Backend (Port 5001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Server status, uptime, webcam/GPS info |
| `/api/location` | GET | Current GPS coordinates |
| `/api/video-feed` | GET | MJPEG webcam stream |
| `/api/snapshot` | GET | Single JPEG frame from webcam |
| `/api/notifications` | GET | Poll for pending notifications |

### Flask API Server (Port 5000 — `server.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/process_frame` | POST | Full ML inference on a frame |
| `/enroll_face` | POST | Enroll a new face |
| `/query_item` | GET | Find item location |
| `/get_all_faces` | GET | List enrolled faces |

---

## Hardware Setup

### Raspberry Pi Wiring
- **Pi Camera v2** → CSI port
- **HC-SR04 Ultrasonic** → GPIO (Trig: pin 23, Echo: pin 24)
- **Push Button** → GPIO pin 17
- **Speaker** → 3.5mm audio jack

### Network
- RPi and laptop must be on the **same WiFi network**
- Laptop ML server: port **8080** (WebSocket)
- Mobile backend: port **5001** (HTTP)
- Expo dev server: port **8081**

Firewall rules (Windows — run as admin):
```powershell
netsh advfirewall firewall add rule name="VisionMate ML" dir=in action=allow protocol=TCP localport=8080
netsh advfirewall firewall add rule name="VisionMate Mobile" dir=in action=allow protocol=TCP localport=5001
netsh advfirewall firewall add rule name="Expo Metro" dir=in action=allow protocol=TCP localport=8081
```

---

## Configuration

All settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `YOLO_MODEL` | `yolov8n.pt` | YOLOv8 model variant |
| `YOLO_CONFIDENCE` | `0.5` | Detection confidence threshold |
| `FACE_RECOGNITION_THRESHOLD` | `0.6` | Face match distance threshold |
| `GPS_MOCK_MODE` | `True` | Use simulated GPS path |
| `AUDIO_ENABLED` | `True` | Enable/disable TTS |

Mobile app config in `HAL-102-Frontend/src/config.ts`:

| Setting | Default | Description |
|---------|---------|-------------|
| `BACKEND_IP` | `10.199.231.39` | Laptop's WiFi IP address |
| `BACKEND_PORT` | `5001` | Mobile backend port |
| `LOCATION_POLL_INTERVAL` | `3000` | GPS poll rate (ms) |
| `NOTIFICATION_POLL_INTERVAL` | `2000` | Notification poll rate (ms) |

---

## Performance

| Component | CPU | GPU |
|-----------|-----|-----|
| YOLOv8n | ~40ms | ~10ms |
| InsightFace | ~60ms | ~15ms |
| Object Memory | ~5ms | ~5ms |
| Spatial Nav | ~2ms | ~2ms |
| **Total Pipeline** | ~110ms (9 FPS) | ~30ms (33 FPS) |
| **Wireless (RPi→Laptop)** | ~4 FPS | ~4 FPS (network-bound) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Object Detection | YOLOv8n (Ultralytics) |
| Face Recognition | InsightFace (buffalo_l) |
| Object Memory | CNN feature extraction + cosine similarity |
| Navigation | OpenStreetMap (OSRM) |
| GPS | Serial NMEA or mock mode |
| Voice Commands | SpeechRecognition + Google STT |
| Audio Feedback | pyttsx3 / gTTS |
| RPi ↔ Laptop | WebSocket (asyncio) |
| Mobile Backend | Flask + flask-cors |
| Mobile App | Expo SDK 54, React Native, TypeScript |
| Database | SQLite |

---

## Team

Built by **Team NPC's** for VisionMate

---

## License

MIT License — Free for educational and hackathon use
