# VisionMate External Navigation System

## Overview
Complete GPS-based navigation system with Google Maps integration, transit support, and real-time obstacle detection for VisionMate smart glasses.

## Features
- 🗺️ **Google Maps Integration**: Route planning with Directions API and Geocoding API
- 🚌 **Multi-Modal Transit**: Bus, train, walking, and driving directions
- 📍 **GPS Tracking**: Real-time position tracking (USB or GPIO GPS modules)
- 🔊 **Turn-by-Turn Voice Guidance**: Audio announcements at configurable distances
- 📏 **Ultrasonic Distance Sensing**: Precise obstacle distance measurement via ESP32
- 👁️ **Vision Fusion**: Combines camera object detection with ultrasonic ranging
- 🧪 **Mock Mode**: Full testing without hardware (simulated GPS + ultrasonic)

## System Architecture

```
Voice Command: "Navigate to Majestic"
         ↓
   Google Maps API
   (geocoding + directions)
         ↓
    GPS Tracking
   (position updates)
         ↓
  Navigation Engine
  (distance calculation)
         ↓
   Transit Guide ←→ Obstacle Fusion
   (bus stops)      (camera + ultrasonic)
         ↓
   Audio Feedback
   (turn-by-turn voice)
```

## Module Overview

### 1. **google_maps_navigator.py**
- Geocodes addresses to GPS coordinates
- Gets directions with transit/walking/driving modes
- Parses routes into turn-by-turn steps
- Extracts bus line, stops, and transfer info

**Key Methods:**
```python
geocode_address(address) → (lat, lon)
get_directions(origin, destination, mode) → route_dict
get_current_instruction() → "Take Bus 201 from MG Road"
advance_step() → move to next turn
```

### 2. **gps_tracker.py**
- Tracks current GPS position
- Mock mode: Pre-defined Bangalore path (Indiranagar → Majestic)
- Real mode: Reads NMEA sentences from GPS module via serial
- Supports GPGGA and GPRMC formats

**Key Methods:**
```python
get_position() → (lat, lon)
advance_mock_position() → simulate movement (testing)
set_mock_path(path) → custom test routes
```

**Mock Path (10 GPS points):**
```
Start: 12.9716, 77.6412 (Indiranagar)
  ...  (8 intermediate points)
End:   12.9767, 77.5715 (Majestic)
```

### 3. **ultrasonic_sensor.py**
- Reads distance measurements from ESP32
- Mock mode: Realistic random distances (20-300cm with noise)
- Real mode: Serial protocol "DIST:123\n"
- 4-level warning system

**Warning Levels:**
- **Emergency**: < 30 cm (immediate danger)
- **Warning**: < 60 cm (caution needed)
- **Notice**: < 100 cm (be aware)
- **Clear**: > 100 cm (safe)

**Key Methods:**
```python
get_distance() → distance in cm
is_obstacle_close() → bool
get_obstacle_status() → {'level': 'warning', 'distance': 45}
```

### 4. **navigation_engine.py**
- Core turn-by-turn navigation logic
- Calculates distances using Haversine formula
- Determines when to announce turns (50m threshold)
- Detects off-route (30m deviation)
- Arrival detection (20m threshold)

**Key Methods:**
```python
start_navigation(origin, destination, mode) → bool
update() → {'status': 'navigating', 'instruction': '...', 'announce': True}
get_progress() → current step, total steps, distance, duration
```

**Navigation Cycle:**
```
1. Get GPS position
2. Calculate distance to next waypoint
3. If < 50m and not announced → speak instruction
4. If < 10m → advance to next step
5. If > 30m off route → trigger reroute
6. If < 20m to destination → arrived!
```

### 5. **transit_guide.py**
- Handles bus/train specific guidance
- Detects arrival at bus stops (50m radius)
- Tracks stops passed
- Announces "Get off in N stops"

**Key Methods:**
```python
start_transit_step(transit_step) → load bus/train info
get_boarding_instruction() → "Board Bus 201 at MG Road"
mark_boarded() → user on vehicle
mark_stop_passed() → increment counter
get_exit_warning() → "Get off in 3 stops"
```

**Transit Flow:**
```
1. User approaches bus stop → "At Indiranagar Bus Stop"
2. User boards → "Board Bus 201 towards Majestic"
3. Passing stops → count stops
4. 3 stops before exit → "Get off in 3 stops at Majestic"
5. Next stop → "Prepare to exit at Majestic Bus Station"
```

### 6. **obstacle_fusion.py**
- Combines YOLO object detection with ultrasonic distance
- Replaces bbox-based distance estimates with precise measurements
- Ultrasonic FOV: 15° (narrow beam ahead)
- Camera FOV: 60° (wider view)

**How It Works:**
```
1. YOLO detects: "Person at (320, 240)" (bbox center)
2. Check if bbox center within ultrasonic FOV (±7.5° from center)
3. If yes → use ultrasonic distance (e.g., 85 cm)
4. If no → estimate from bbox size (e.g., ~2.0 m)
5. Generate warning: "Person 0.85 meters ahead"
```

**Key Methods:**
```python
get_object_with_distance(objects) → enhanced objects with precise_distance
get_critical_obstacles(objects) → obstacles < 1.5m
generate_obstacle_warnings(objects) → priority-sorted warnings
```

## Configuration (config.py)

```python
# Google Maps API
GOOGLE_MAPS_API_KEY = "Your api keyw"

# Navigation Parameters
NAVIGATION_ANNOUNCEMENT_DISTANCE = 50  # meters before turn to announce
NAVIGATION_REROUTE_DISTANCE = 30       # meters off-route to trigger reroute
NAVIGATION_ARRIVAL_DISTANCE = 20       # meters from destination = arrived

# GPS Configuration
GPS_MOCK_MODE = True                   # True = simulated, False = real hardware
GPS_SERIAL_PORT = "/dev/ttyUSB0"       # Linux GPS module port (or COM3 on Windows)
GPS_BAUDRATE = 9600                    # GPS module baud rate

# Ultrasonic Sensor Configuration
ULTRASONIC_MOCK_MODE = True            # True = simulated, False = ESP32
ULTRASONIC_SERIAL_PORT = "COM3"        # ESP32 serial port
ULTRASONIC_BAUDRATE = 115200           # ESP32 baud rate
ULTRASONIC_MAX_DISTANCE = 400          # cm
ULTRASONIC_OBSTACLE_THRESHOLD = 100    # cm to consider "close"
```

## Testing Without Hardware

All modules support **mock mode** for immediate testing on your PC:

### Run Navigation Demo
```bash
python demo_navigation.py
```

**Demo Controls:**
- **N** - Start navigation to Majestic
- **M** - Advance mock GPS position (simulate walking)
- **S** - Show navigation status
- **O** - Check obstacles
- **Q** - Quit

**Mock Mode Features:**
- GPS: Follows predefined Bangalore path
- Ultrasonic: Generates realistic distances (20-300 cm)
- Maps API: Real Google Maps responses
- Audio: Full voice guidance

### Test Individual Modules

```bash
# Test Google Maps API
python google_maps_navigator.py

# Test GPS tracking
python gps_tracker.py

# Test ultrasonic sensor
python ultrasonic_sensor.py

# Test navigation engine
python navigation_engine.py

# Test transit guide
python transit_guide.py

# Test obstacle fusion
python obstacle_fusion.py
```

## Hardware Integration

### Required Hardware

1. **GPS Module** ($10-20)
   - Option A: USB GPS (plug-and-play)
   - Option B: UART GPS (TX/RX connection)
   - Recommended: G-Mouse USB GPS or Adafruit Ultimate GPS

2. **ESP32 + Ultrasonic Sensor** (you already have this)
   - HC-SR04 or similar ultrasonic sensor
   - ESP32 development board
   - Wire connections: VCC, GND, TRIG, ECHO

### GPS Module Setup

> [!NOTE]
> **Indoor vs Outdoor Navigation**: GPS and reverse-geocoded addresses are for OUTDOOR navigation only. Indoors, set `INDOOR_MODE = True` in `config.py` (already the default) so object-finding guidance relies on the camera and nearby-object context instead of GPS, which has no satellite signal indoors.

**Windows:**
1. Plug in USB GPS
2. Check Device Manager for COM port (e.g., COM4)
3. Update config.py: `GPS_SERIAL_PORT = "COM4"`
4. Set `GPS_MOCK_MODE = False`

**Linux:**
1. Plug in USB GPS
2. Check port: `ls /dev/ttyUSB*` or `ls /dev/ttyACM*`
3. Update config.py: `GPS_SERIAL_PORT = "/dev/ttyUSB0"`
4. Add user to dialout group: `sudo usermod -a -G dialout $USER`
5. Set `GPS_MOCK_MODE = False`

**Verify GPS:**
```python
from gps_tracker import GPSTracker
gps = GPSTracker(mock_mode=False)
gps.start()
time.sleep(5)
pos = gps.get_position()
print(f"GPS: {pos}")  # Should show your actual location
```

### ESP32 Ultrasonic Setup

**Arduino Code (upload to ESP32):**
```cpp
#define TRIG_PIN 5
#define ECHO_PIN 18

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
}

void loop() {
  // Trigger ultrasonic pulse
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  // Read echo
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);  // 30ms timeout
  long distance = duration * 0.034 / 2;  // cm
  
  // Send to Python via serial
  Serial.print("DIST:");
  Serial.println(distance);
  
  delay(100);  // 10 Hz update rate
}
```

**Python Setup:**
1. Upload above code to ESP32
2. Check COM port in Device Manager (e.g., COM3)
3. Update config.py: `ULTRASONIC_SERIAL_PORT = "COM3"`
4. Set `ULTRASONIC_MOCK_MODE = False`

**Verify Ultrasonic:**
```python
from ultrasonic_sensor import UltrasonicSensor
sensor = UltrasonicSensor(mock_mode=False)
sensor.start()
time.sleep(1)
distance = sensor.get_distance()
print(f"Distance: {distance} cm")  # Should show real distance
```

## Usage Examples

### Example 1: Basic Navigation
```python
from gps_tracker import GPSTracker
from google_maps_navigator import GoogleMapsNavigator
from navigation_engine import NavigationEngine
import config

# Initialize
gps = GPSTracker(mock_mode=True)
gps.start()
maps = GoogleMapsNavigator(config.GOOGLE_MAPS_API_KEY)
engine = NavigationEngine(gps, maps)

# Start navigation
engine.start_navigation("Indiranagar, Bangalore", "Majestic, Bangalore", mode='transit')

# Navigation loop
while True:
    status = engine.update()
    
    if status['announce']:
        print(f"🔊 {status['instruction']}")
    
    if status['status'] == 'arrived':
        print("Arrived!")
        break
    
    gps.advance_mock_position()  # Simulate movement
    time.sleep(2)
```

### Example 2: Obstacle Detection During Navigation
```python
from ultrasonic_sensor import UltrasonicSensor
from obstacle_fusion import ObstacleFusion
from object_detector import ObjectDetector
import cv2

# Initialize
detector = ObjectDetector()
sensor = UltrasonicSensor(mock_mode=True)
sensor.start()
fusion = ObstacleFusion(sensor)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    objects = detector.detect(frame)
    
    # Get warnings
    warnings = fusion.generate_obstacle_warnings(objects)
    
    for warn in warnings:
        if warn['priority'] == 1:  # EMERGENCY
            print(f"🚨 {warn['message']}")
    
    cv2.imshow('Navigation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Example 3: Full System with Voice Commands
```python
# Add to test_integration.py or demo_voice_remember_find.py

def handle_voice_navigate_to(params):
    """Handle 'Navigate to [place]' voice command"""
    destination = params.get('destination')
    
    if not destination:
        audio.speak("Where do you want to go?")
        return
    
    audio.speak(f"Navigating to {destination}")
    
    # Start navigation
    success = nav_engine.start_navigation(
        gps.get_position(),
        destination,
        mode='transit'
    )
    
    if success:
        progress = nav_engine.get_progress()
        audio.speak(f"Route found. {progress['total_distance']}. {progress['total_duration']}")
    else:
        audio.speak("Could not find route")

# Register voice handler
voice_handlers['navigate to'] = handle_voice_navigate_to
```

## Google Maps API Setup

### Enable APIs (Google Cloud Console)
1. Go to: https://console.cloud.google.com/
2. Create project: "VisionMate"
3. Enable APIs:
   - ✅ Directions API
   - ✅ Geocoding API
   - ✅ Places API (optional)
4. Create API key
5. Add key to config.py

### API Pricing (Free Tier)
- **$200 free credit/month**
- Directions API: $5 per 1000 requests
- Geocoding API: $5 per 1000 requests
- **Free tier = ~40,000 requests/month**

### Security Note
⚠️ **Add config.py to .gitignore before pushing to GitHub!**

Create `.gitignore`:
```
config.py
__pycache__/
*.pyc
.env
```

## Troubleshooting

### GPS Not Working
```
Problem: get_position() returns None
Solutions:
1. Check GPS module has clear view of sky (window or outdoors)
2. Wait 1-2 minutes for GPS fix (cold start)
3. Verify COM port in Device Manager
4. Check baud rate matches GPS module (usually 9600)
5. Test with mock_mode=True first
```

### Ultrasonic Not Working
```
Problem: get_distance() returns None or 0
Solutions:
1. Check ESP32 is powered and code uploaded
2. Verify COM port in Device Manager
3. Check serial baud rate (115200)
4. Test sensor directly: point at wall, should read ~20-50 cm
5. Check wiring: VCC→3.3V, GND→GND, TRIG→GPIO5, ECHO→GPIO18
```

### API Key Errors
```
Problem: "API key not valid" or "REQUEST_DENIED"
Solutions:
1. Verify key in Google Cloud Console
2. Check Directions API and Geocoding API are enabled
3. Check API restrictions (none should be set for testing)
4. Wait 5 minutes after creating key (propagation delay)
```

### Navigation Not Starting
```
Problem: start_navigation() returns False
Solutions:
1. Check GPS has valid position
2. Verify internet connection (Maps API requires internet)
3. Check API key in config.py
4. Try with mock_mode=True to isolate GPS vs API issues
5. Check logs for specific error messages
```

## Performance

- **GPS Update Rate**: 1 Hz (1 position/second)
- **Ultrasonic Update Rate**: 10 Hz (10 readings/second)
- **Navigation Update Rate**: 0.5 Hz (every 2 seconds)
- **Obstacle Check Rate**: 1 Hz (every second)
- **Camera Frame Rate**: ~30 FPS
- **YOLO Inference**: ~30-50 ms/frame (CPU), ~10-15 ms/frame (GPU)

## Next Steps

1. ✅ All 6 navigation modules created
2. ✅ Mock mode testing available
3. ⏳ Integrate voice commands ("Navigate to X")
4. ⏳ Test with real GPS module
5. ⏳ Test with ESP32 ultrasonic
6. ⏳ Full outdoor navigation test

## Files Created

```
google_maps_navigator.py    - Google Maps API integration (250 lines)
gps_tracker.py              - GPS tracking with mock mode (270 lines)
ultrasonic_sensor.py        - ESP32 ultrasonic interface (200 lines)
obstacle_fusion.py          - Camera + ultrasonic fusion (200 lines)
navigation_engine.py        - Turn-by-turn logic (250 lines)
transit_guide.py            - Bus/train guidance (230 lines)
demo_navigation.py          - Full system demo (200 lines)
NAVIGATION_README.md        - This file
```

**Total: ~1600 lines of navigation code**

## Support

For issues or questions:
1. Check logs: `logging.basicConfig(level=logging.DEBUG)`
2. Test each module individually
3. Verify mock mode works before trying real hardware
4. Check Google Maps API quota in Cloud Console

---

**VisionMate Navigation System - Ready for Testing! 🚀**
