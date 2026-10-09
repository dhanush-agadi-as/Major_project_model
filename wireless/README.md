# VisionMate Wireless Smart Glasses Setup Guide

## 🚀 Quick Start (4-Hour Timeline)

### Hour 1: Hardware Setup (Hardware Guy)

#### Step 1.1: Flash Raspberry Pi OS to Pendrive
1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Insert pendrive into laptop
3. Open Raspberry Pi Imager:
   - Choose OS: **Raspberry Pi OS Lite (64-bit)**
   - Choose Storage: Your pendrive
   - Click Settings (gear icon):
     - Set hostname: `visionmate`
     - Enable SSH
     - Set username: `pi`, password: `raspberry`
     - Configure WiFi: Enter your WiFi name and password
   - Click **Write** (takes ~10 minutes)

#### Step 1.2: Boot Raspberry Pi
1. Remove SD card (if any) from Raspberry Pi
2. Insert pendrive into **blue USB 3.0 port**
3. Connect power bank to Raspberry Pi USB-C
4. Wait 2 minutes for boot
5. Find IP address:
   - Check your router's connected devices, or
   - On laptop: `ping visionmate.local`

#### Step 1.3: Upload ESP32 Code
1. Open Arduino IDE on laptop
2. Install ESP32 board support (if not installed):
   - File → Preferences → Additional Board Manager URLs
   - Add: `https://dl.espressif.com/dl/package_esp32_index.json`
   - Tools → Board → Board Manager → Search "ESP32" → Install
3. Open: `hardware/esp32_sensors.ino`
4. Select: Tools → Board → ESP32 Dev Module
5. Connect ESP32 to laptop via USB
6. Click Upload
7. Wait for "Done uploading"

#### Step 1.4: Wire Components

**ESP32 Connections (on breadboard):**
```
ESP32 Pin    →    Component
─────────────────────────────
GPIO 18      →    Ultrasonic TRIG
GPIO 19      →    Ultrasonic ECHO
GPIO 0       →    Push Button (other leg → GND)
5V           →    Ultrasonic VCC
GND          →    Ultrasonic GND, Button GND
USB          →    Raspberry Pi USB port
```

**Raspberry Pi Connections:**
```
RPi Port          →    Component
─────────────────────────────────────
CSI Port          →    Camera ribbon cable
USB Port (blue)   →    Pendrive (boot disk)
USB Port          →    ESP32 (USB cable)
USB-C             →    Power bank
Bluetooth         →    Pair wireless earbuds
```

**Test ultrasonic sensor:**
- Hold hand in front of sensor
- Distance should change on Arduino Serial Monitor

---

### Hour 2: Software Setup

#### Step 2.1: Install Dependencies on Laptop
```bash
# On Windows laptop (your existing environment)
cd D:\Model_for_VisionMate
conda activate D:\conda_envs\visionmate

# Install websockets
pip install websockets

# Verify existing packages
python -c "import cv2, torch; print('✓ Ready')"
```

#### Step 2.2: Install Dependencies on Raspberry Pi
```bash
# SSH into Raspberry Pi from laptop
ssh pi@visionmate.local
# Password: raspberry (or what you set)

# Update system
sudo apt update

# Install Python packages
pip3 install opencv-python websockets pyserial pyttsx3 numpy

# Install camera support
sudo apt install -y python3-picamera2

# Test camera
libcamera-hello
# You should see camera preview for 5 seconds
```

#### Step 2.3: Copy Code to Raspberry Pi
```bash
# On laptop (in PowerShell)
cd D:\Model_for_VisionMate

# Copy RPi client to Raspberry Pi
scp wireless\rpi_client.py pi@visionmate.local:~/

# Verify
ssh pi@visionmate.local "ls -l rpi_client.py"
```

---

### Hour 3: Testing & Integration

#### Step 3.1: Start Laptop Server
```bash
# On laptop
cd D:\Model_for_VisionMate
conda activate D:\conda_envs\visionmate

python wireless\laptop_server.py
```

**Expected output:**
```
VISIONMATE LAPTOP ML SERVER
============================================================
Initializing ML models...
✓ Object Detector ready
✓ Face Recognizer ready
✓ Navigation Engine ready

SERVER READY - WAITING FOR RASPBERRY PI
Laptop IP Address: 192.168.1.XXX  ← COPY THIS IP
Server Port: 8765
```

**Copy the Laptop IP Address shown!**

#### Step 3.2: Start Raspberry Pi Client
```bash
# SSH into Raspberry Pi
ssh pi@visionmate.local

# Run client (replace with your laptop IP)
python3 rpi_client.py 192.168.1.XXX
```

**Expected output:**
```
Raspberry Pi Client
✓ Camera initialized (640x480 @ 30fps)
✓ ESP32 connected on /dev/ttyUSB0
✓ Audio engine initialized
✓ Raspberry Pi Client ready!
Connecting to laptop server: ws://192.168.1.XXX:8765
✓ Connected to laptop ML server!
Starting video streaming...
Streaming @ 28.5 fps, Distance: 67.3cm
```

#### Step 3.3: Test Everything

**On laptop server, you should see:**
```
✓ Raspberry Pi connected from: 192.168.1.YYY
Processing @ 29.3 fps | Latency: 95ms | Distance: 67cm
```

**✓ SUCCESS! Wireless system is working!**

**Test each feature:**
1. **Camera**: Move camera around, server should detect objects
2. **Ultrasonic**: Move hand in front of sensor, distance should change
3. **Button**: Press button, server should receive button press
4. **Audio**: Server sends navigation → RPi speaks via Bluetooth earbuds

---

### Hour 4: Final Assembly & Demo Prep

#### Step 4.1: Physical Assembly (Quick Version)
```
For demo, don't worry about fancy frame yet!

Setup 1: Table Demo
- Tape RPi to small board
- Mount camera pointing forward
- Attach ESP32/breadboard nearby
- Power bank in pocket/on table
- Wear earbuds
- Hold board/wear on head with headstrap

Setup 2: Wearable Proto
- Use existing cap/helmet
- Tape RPi to top
- Camera points forward
- ESP32 on side
- Power bank in pocket (cable up back)
- Earbuds in ears
```

#### Step 4.2: Demo Script Practice
```
Demo Flow:
1. "This is VisionMate - smart glasses for the visually impaired"
2. Show camera view on laptop (detections happening)
3. Walk around: "Person ahead, chair on right"
4. Press button: Voice command demo
5. Find remembered object: "Find my keys"
6. GPS navigation: "Navigate to Majestic"
7. Face recognition: Recognize someone
```

---

## 🐛 Troubleshooting

### Raspberry Pi won't boot from pendrive
- **Solution**: Update bootloader first with SD card, then try pendrive
- Or use MicroSD card instead (works same way)

### ESP32 not found on RPi
- **Check**: `ls /dev/ttyUSB* /dev/ttyACM*`
- **Solution**: Try different USB port, check cable, re-upload code

### Camera not working
- **Check**: `libcamera-hello`
- **Solution**: Enable camera in `sudo raspi-config` → Interface Options → Legacy Camera

### Can't connect to WiFi
- **Check**: Same WiFi network? `ping 192.168.1.1`
- **Solution**: Check WiFi credentials in pendrive boot partition `wpa_supplicant.conf`

### Bluetooth earbuds not pairing
- **Check**: `bluetoothctl`
- **Commands**:
  ```bash
  sudo bluetoothctl
  scan on
  pair XX:XX:XX:XX:XX:XX  # Your earbud MAC
  connect XX:XX:XX:XX:XX:XX
  trust XX:XX:XX:XX:XX:XX
  exit
  ```

### High latency (>200ms)
- **Check**: WiFi signal strength
- **Solution**: Move closer to router, reduce JPEG quality in `rpi_client.py` (line 127)

### Low FPS (<15fps)
- **Check**: Laptop CPU usage
- **Solution**: Close other programs, reduce frame resolution

---

## 📊 Expected Performance

| Metric | Target | Actual (typical) |
|--------|--------|------------------|
| Video FPS | 30 fps | 25-30 fps |
| Processing Latency | <100ms | 80-150ms |
| WiFi Range | 10-30m | 15-25m |
| Battery Life | 3-4 hours | 3-5 hours |
| Object Detection | <50ms | 35-60ms |
| Face Recognition | <100ms | 80-120ms |

---

## 🎯 Demo Checklist

**Before Demo:**
- [ ] Laptop fully charged
- [ ] Power bank fully charged (10,000mAh+)
- [ ] Raspberry Pi boots correctly
- [ ] Camera working (`libcamera-hello`)
- [ ] ESP32 uploading sensor data
- [ ] Bluetooth earbuds paired and connected
- [ ] Laptop server running (models loaded)
- [ ] RPi client connected to server
- [ ] Test: Walk around, hear audio instructions
- [ ] Test: Press button, voice command works
- [ ] Test: Object detection working
- [ ] Test: Face recognition working
- [ ] Prepare demo script (memorize key points)

**During Demo:**
- Keep laptop nearby (in backpack or on table)
- WiFi must be stable (use laptop hotspot if venue WiFi bad)
- Have backup: If wireless fails, use laptop webcam demo
- Explain: "ML processing on laptop, sensors on glasses"

---

## 🔧 File Structure

```
Model_for_VisionMate/
├── wireless/
│   ├── laptop_server.py        ← Run on laptop (ML processing)
│   ├── rpi_client.py           ← Run on Raspberry Pi
│   └── README.md               ← This file
├── hardware/
│   └── esp32_sensors.ino       ← Upload to ESP32
├── [All your existing modules]
└── [Databases, models, etc.]
```

---

## 🚀 Post-Demo Improvements

**Week 2+:**
1. Design 3D printable glasses frame
2. Add GPS module for real navigation
3. Add LiPo battery (smaller than power bank)
4. Optimize ML models for lower latency
5. Add web dashboard (React app)
6. Build custom PCB to reduce wiring

---

## 📞 Quick Commands Reference

**Laptop:**
```bash
# Start server
python wireless\laptop_server.py

# Find laptop IP
ipconfig  # Look for "IPv4 Address"
```

**Raspberry Pi:**
```bash
# Connect via SSH
ssh pi@visionmate.local

# Run client
python3 rpi_client.py <LAPTOP_IP>

# Check camera
libcamera-hello

# Check USB devices
ls /dev/ttyUSB*

# Check WiFi
iwconfig
```

**ESP32:**
```bash
# In Arduino IDE Serial Monitor
# Should see: DIST:123.45,BTN:0
```

---

## 💡 Tips

1. **Use laptop WiFi hotspot** if venue WiFi is unreliable
2. **Lower video quality** (change JPEG quality) if latency too high
3. **Practice demo** at least 3 times before presenting
4. **Have backup**: Run on laptop webcam if wireless fails
5. **Explain clearly**: "Brain is laptop, sensors are on glasses"

---

## ✅ Success Criteria

You'll know it's working when:
- ✓ Laptop shows "Raspberry Pi connected"
- ✓ Laptop displays FPS (25-30 fps)
- ✓ Objects detected in real-time
- ✓ Audio plays through Bluetooth earbuds
- ✓ Navigation instructions speak automatically
- ✓ Button press triggers voice commands
- ✓ Ultrasonic distance changes when moving hand

**ALL DONE! GO ROCK THAT DEMO! 🎉🚀**
