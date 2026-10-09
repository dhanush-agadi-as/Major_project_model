# ⚡ VISIONMATE WIRELESS - QUICK START COMMAND LIST

## 🎯 IMMEDIATE ACTIONS (Next 4 Hours)

---

## 👨‍💻 YOU (Software Guy) - DO NOW

### 1. Test Laptop Server (2 minutes)
```bash
# Activate environment
conda activate D:\conda_envs\visionmate

# Start server
cd D:\Model_for_VisionMate
python wireless\laptop_server.py
```

**Expected Output:**
```
✓ Object Detector ready
✓ Face Recognizer ready
✓ Navigation Engine ready

Laptop IP Address: 192.168.1.XXX  ← WRITE THIS DOWN!
```

**Press Ctrl+C to stop** (just testing it works)

---

## 🔧 HARDWARE GUY - DO NOW

### Download These (30 minutes):

**1. Raspberry Pi Imager:**
- Download: https://www.raspberrypi.com/software/
- Install on laptop

**2. Arduino IDE** (if not installed):
- Download: https://www.arduino.cc/en/software
- Install ESP32 support:
  - File → Preferences → Additional Board URLs:
  - Add: `https://dl.espressif.com/dl/package_esp32_index.json`
  - Tools → Board Manager → Search "ESP32" → Install

---

### Flash Pendrive (15 minutes):

**Open Raspberry Pi Imager:**
1. Choose OS: **Raspberry Pi OS Lite (64-bit)**
2. Choose Storage: **Your pendrive**
3. Click **Settings (gear icon)**:
   - Hostname: `visionmate`
   - Enable SSH: ✓
   - Username: `pi`
   - Password: `raspberry`
   - WiFi SSID: `YOUR_WIFI_NAME`
   - WiFi Password: `YOUR_WIFI_PASSWORD`
4. Click **WRITE**

**Wait 10 minutes for flashing to complete**

---

### Upload ESP32 Code (10 minutes):

**In Arduino IDE:**
1. Open: `D:\Model_for_VisionMate\hardware\esp32_sensors.ino`
2. Tools → Board → **ESP32 Dev Module**
3. Tools → Port → **COM3** (or your ESP32 port)
4. Click **Upload** (arrow button)
5. Wait for "Done uploading"

**Test in Serial Monitor (Tools → Serial Monitor, 115200 baud):**
```
DIST:123.45,BTN:0
DIST:125.67,BTN:0
```
**✓ Working if you see distance values!**

---

### Wire Everything (30 minutes):

**Follow:** `hardware\WIRING_GUIDE.md`

**Quick checklist:**
- [ ] ESP32 + Ultrasonic sensor on breadboard
- [ ] Push button on breadboard  
- [ ] Camera ribbon cable to RPi
- [ ] ESP32 USB cable to RPi
- [ ] Pendrive in RPi USB 3.0 (blue port)
- [ ] Power bank to RPi USB-C

**Boot Raspberry Pi:**
1. Connect power bank
2. Wait 2 minutes
3. Check: Green LED blinking = booting
4. Find IP: Check router or ping from laptop:
   ```bash
   ping visionmate.local
   ```

---

## 🤝 TOGETHER - INTEGRATION (Hour 3-4)

### Step 1: Copy Code to Raspberry Pi

**From laptop:**
```bash
# Copy client code to RPi
scp wireless\rpi_client.py pi@visionmate.local:~/
# Password: raspberry
```

### Step 2: Install Python Packages on RPi

**SSH into RPi:**
```bash
ssh pi@visionmate.local
# Password: raspberry

# Install packages
pip3 install opencv-python websockets pyserial pyttsx3 numpy

# Test camera
libcamera-hello
# Should show 5-second preview
```

### Step 3: Start Everything

**Terminal 1 (Laptop):**
```bash
cd D:\Model_for_VisionMate
conda activate D:\conda_envs\visionmate
python wireless\laptop_server.py

# Note: Your laptop IP will be shown (e.g., 192.168.1.100)
```

**Terminal 2 (Raspberry Pi via SSH):**
```bash
ssh pi@visionmate.local

# Replace XXX with your laptop IP
python3 rpi_client.py 192.168.1.XXX
```

### Step 4: VERIFY IT WORKS! ✅

**On laptop server, you should see:**
```
✓ Raspberry Pi connected from: 192.168.1.YYY
Processing @ 28.5 fps | Latency: 95ms
```

**On RPi client, you should see:**
```
✓ Connected to laptop ML server!
Streaming @ 28.5 fps, Distance: 67.3cm
```

**🎉 SUCCESS IF BOTH SHOWING FPS!**

---

## 🚨 EMERGENCY TROUBLESHOOTING

### Can't connect to Raspberry Pi
```bash
# Find all devices on network
nmap -sn 192.168.1.0/24
# Or check router's connected devices list
```

### Camera not working on RPi
```bash
# Enable legacy camera
sudo raspi-config
# Navigate: Interface Options → Legacy Camera → Enable
sudo reboot
```

### ESP32 code won't upload
- Try different USB port
- Check cable (must be data cable, not charge-only)
- Hold BOOT button on ESP32 while uploading

### Bluetooth earbuds won't pair
```bash
sudo bluetoothctl
power on
scan on
# Find your earbud MAC (XX:XX:XX:XX:XX:XX)
pair XX:XX:XX:XX:XX:XX
connect XX:XX:XX:XX:XX:XX
trust XX:XX:XX:XX:XX:XX
exit
```

---

## 📋 DEMO PREP CHECKLIST (Hour 4)

**Before Showing:**
- [ ] Laptop server running (models loaded)
- [ ] RPi client connected and streaming
- [ ] Both showing ~30 fps
- [ ] Audio working (hear through earbuds)
- [ ] Camera detects objects (test with phone, keys, person)
- [ ] Face recognition works (test with your face)
- [ ] Ultrasonic sensor working (wave hand, distance changes)
- [ ] Button press detected (press button, check serial/logs)

**Demo Script:**
1. "Smart glasses for visually impaired"
2. Walk around: "Person ahead, chair right"
3. Object memory: "Remember phone" → "Find phone"
4. Face recognition: Recognize someone
5. GPS navigation: "Navigate to Majestic"

---

## 📞 CONTACTS

**If stuck, check:**
1. `wireless\README.md` - Detailed setup guide
2. `hardware\WIRING_GUIDE.md` - Wiring diagrams
3. Test individually:
   - Test laptop server alone
   - Test RPi camera alone (`libcamera-hello`)
   - Test ESP32 alone (Serial Monitor)
   - Then combine

---

## ⏰ TIME TRACKING

- [x] Hour 1: Hardware setup (Flash pendrive, wire components)
- [ ] Hour 2: Software setup (Install packages, copy code)
- [ ] Hour 3: Integration testing (Connect and verify)
- [ ] Hour 4: Demo prep and practice

**YOU'VE GOT THIS! 🚀**

**Current Status:**
✅ All code files created
✅ Websockets installed on laptop
✅ Documentation ready

**Next Step:** Hardware guy flashes pendrive and wires components!
