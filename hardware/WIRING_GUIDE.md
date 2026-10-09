# Hardware Wiring Guide for VisionMate Smart Glasses

## Component List
- ✅ Raspberry Pi 4B (8GB)
- ✅ RPi Camera Module 3 Wide
- ✅ ESP32 Dev Board
- ✅ HC-SR04 Ultrasonic Sensor
- ✅ Push Button (tactile switch)
- ✅ SPH0645 Microphone (optional for demo)
- ✅ Breadboard (small)
- ✅ Jumper wires (M-F, F-F)
- ✅ Power bank (10,000mAh+)
- ✅ Wireless earbuds (Bluetooth)
- ✅ USB cables

---

## Wiring Diagram

```
                    POWER BANK (10000mAh)
                          |
                    [USB-C Cable]
                          |
                          ↓
        ╔════════════════════════════════════╗
        ║    RASPBERRY PI 4B (8GB)           ║
        ║                                    ║
        ║  [CSI Port]──→ Camera Module      ║
        ║                                    ║
        ║  [USB Port 1]──→ ESP32 ─────────┐ ║
        ║                                  │ ║
        ║  [WiFi]──→ Laptop (wireless)    │ ║
        ║                                  │ ║
        ║  [Bluetooth]──→ Earbuds         │ ║
        ╚═════════════════════════════════│══╝
                                          │
                                          ↓
                    ╔═══════════════════════════╗
                    ║  ESP32 DEV BOARD          ║
                    ║  (on breadboard)          ║
                    ║                           ║
                    ║  GPIO 18 ──→ TRIG ┐      ║
                    ║  GPIO 19 ──→ ECHO │ HC-  ║
                    ║  5V      ──→ VCC  │ SR04 ║
                    ║  GND     ──→ GND  ┘      ║
                    ║                           ║
                    ║  GPIO 0  ──→ Button ──GND ║
                    ╚═══════════════════════════╝
```

---

## Step-by-Step Wiring

### 1. ESP32 + Ultrasonic Sensor (on Breadboard)

**Materials:**
- ESP32 board
- HC-SR04 ultrasonic sensor
- 4x jumper wires (F-F recommended)

**Connections:**
```
ESP32 Pin          Wire Color      HC-SR04 Pin
─────────────────────────────────────────────
GPIO 18            Yellow          → TRIG
GPIO 19            Green           → ECHO
5V (or 3V3)        Red             → VCC
GND                Black           → GND
```

**How to wire:**
1. Place ESP32 on breadboard (USB port facing out)
2. Place HC-SR04 sensor on breadboard (facing forward)
3. Connect wires between ESP32 and HC-SR04 pins
4. Make sure connections are firm

**Test:**
- Connect ESP32 to laptop via USB
- Open Arduino Serial Monitor (115200 baud)
- Should see: `DIST:123.45,BTN:0`
- Wave hand in front of sensor → distance changes

---

### 2. ESP32 + Push Button

**Materials:**
- Push button (tactile switch, 4-pin)
- 2x jumper wires (M-M)

**Connections:**
```
ESP32 Pin          Wire Color      Button
──────────────────────────────────────────
GPIO 0             Yellow          → Leg 1
GND                Black           → Leg 2
```

**Button Orientation:**
```
    [1]   [2]     ← Top view of button
     │     │
    [3]   [4]
    
Connect: Pin 1 → GPIO 0
         Pin 2 → GND
(Diagonal pins: 1-4 and 2-3 are connected when pressed)
```

**Test:**
- Press button
- Serial Monitor should show: `DIST:123.45,BTN:1`
- Release button → `BTN:0`

---

### 3. Raspberry Pi + Camera Module

**Materials:**
- RPi Camera Module 3 Wide
- Ribbon cable (comes with camera)

**Connections:**
1. Find CSI port on Raspberry Pi (between HDMI ports)
2. Pull up black plastic clip gently
3. Insert ribbon cable (blue side facing away from HDMI)
4. Push down black clip to lock
5. Connect other end to camera module

**Test:**
```bash
# On Raspberry Pi
libcamera-hello
# Should show camera preview for 5 seconds
```

---

### 4. Raspberry Pi + ESP32

**Materials:**
- USB Micro-B to USB-A cable
- ESP32 (already wired from steps 1-2)

**Connections:**
1. ESP32 Micro-USB port → RPi USB port (any USB port)
2. That's it!

**Test:**
```bash
# On Raspberry Pi
ls /dev/ttyUSB*
# Should show: /dev/ttyUSB0 or /dev/ttyACM0
```

---

### 5. Raspberry Pi + Power Bank

**Materials:**
- USB-C to USB-A cable
- Power bank (10,000mAh or more)

**Connections:**
1. Power bank USB-A port → USB-C cable → RPi USB-C port
2. Turn on power bank
3. RPi should boot (red LED on, green LED blinking)

**Power requirements:**
- Raspberry Pi 4B: 5V 3A (15W)
- Use power bank with 3A output minimum
- Battery life: 10,000mAh ÷ 3000mA = ~3.3 hours

---

### 6. Bluetooth Earbuds Pairing

**One-time setup:**

1. Put earbuds in pairing mode (usually: hold button 5 sec)
2. On Raspberry Pi:
```bash
sudo bluetoothctl
power on
scan on
# Wait for your earbud MAC address to appear
# Example: XX:XX:XX:XX:XX:XX

pair XX:XX:XX:XX:XX:XX
connect XX:XX:XX:XX:XX:XX
trust XX:XX:XX:XX:XX:XX
exit
```

3. Set as audio output:
```bash
# Check audio devices
pactl list short sinks

# Set Bluetooth as default (if needed)
pactl set-default-sink <bluetooth_sink_name>
```

**Test:**
```bash
# Test audio
espeak "Hello world"
# Should hear from earbuds
```

---

## Complete Assembly Checklist

### Breadboard Setup
- [ ] ESP32 board inserted in breadboard
- [ ] Ultrasonic sensor on breadboard
- [ ] Push button on breadboard
- [ ] 6 wires connecting everything
- [ ] All connections firm (wiggle test)

### Raspberry Pi Connections
- [ ] Camera ribbon cable connected (CSI port)
- [ ] ESP32 USB cable connected
- [ ] Power bank USB-C cable connected
- [ ] Pendrive inserted (USB 3.0 blue port)
- [ ] No loose connections

### Wireless Connections
- [ ] Bluetooth earbuds paired
- [ ] WiFi configured (check with phone hotspot first)
- [ ] Laptop on same WiFi network

### Power On Test
1. [ ] Connect power bank → RPi boots (red LED on)
2. [ ] Wait 2 minutes for boot
3. [ ] Ping: `ping visionmate.local` (from laptop)
4. [ ] SSH: `ssh pi@visionmate.local` (password: raspberry)
5. [ ] Check camera: `libcamera-hello`
6. [ ] Check ESP32: `ls /dev/ttyUSB*`
7. [ ] Check earbuds: `bluetoothctl info XX:XX:XX:XX:XX:XX`

---

## Physical Mounting (Quick Demo Version)

### Option 1: Table Demo
```
┌─────────────────────────┐
│   Small board/cardboard │
│                         │
│  [Camera]  [RPi]  [ESP32]│
│    ↓        ↓      ↓    │
│   tape    tape   tape   │
└─────────────────────────┘
     ↓
  Power bank
  (on table)
```

### Option 2: Head-Mounted (Quick)
```
   Existing cap/helmet/headband
        ___________
       /           \
      |  🎥 [RPi]  |  ← Tape components on top
       \___________/
            |
        [Cable down back]
            |
        Power bank
        (in pocket)
```

**Tips:**
- Use strong tape (duct tape or double-sided foam tape)
- Camera pointing forward (test angle first)
- ESP32/breadboard on side (not blocking camera)
- Keep wires organized (zip ties or tape)
- Power bank cable down back to pocket (comfortable)
- Earbuds in ears (Bluetooth, no wires!)

---

## Common Wiring Mistakes

❌ **Wrong:** Camera ribbon backwards (blue side wrong direction)  
✅ **Right:** Blue side faces away from HDMI ports

❌ **Wrong:** Button not working (no pull-up resistor)  
✅ **Right:** Use GPIO 0 (has internal pull-up in code)

❌ **Wrong:** Ultrasonic TRIG/ECHO swapped  
✅ **Right:** GPIO 18 = TRIG (output), GPIO 19 = ECHO (input)

❌ **Wrong:** Power bank can't power RPi (voltage drop)  
✅ **Right:** Use quality cable, 3A output power bank

❌ **Wrong:** ESP32 not detected on RPi  
✅ **Right:** Try different USB port, check cable

---

## Troubleshooting Hardware

### Camera not working
1. Check ribbon cable connection (both ends)
2. Check cable orientation (blue side correct?)
3. Enable camera: `sudo raspi-config` → Interface → Legacy Camera
4. Reboot: `sudo reboot`

### Ultrasonic readings stuck at 999
1. Check wiring (TRIG/ECHO correct?)
2. Check power (5V or 3.3V to VCC)
3. Check sensor (try different distance)
4. Re-upload ESP32 code

### Button not detected
1. Check wiring (GPIO 0 and GND)
2. Check button seating in breadboard
3. Try pressing harder
4. Test with multimeter (continuity)

### Bluetooth earbuds disconnecting
1. Keep RPi close to earbuds (<5 meters)
2. Re-pair earbuds
3. Check battery (earbuds and power bank)
4. Use `bluetoothctl connect XX:XX:XX:XX:XX:XX`

### Power bank not charging RPi
1. Check cable (must be USB-C)
2. Check power bank output (need 3A)
3. Try different cable
4. Check power bank charge level

---

## Safety Notes

⚠️ **Important:**
- Don't short circuit pins (always double-check wiring)
- Don't exceed 3.3V on GPIO pins (use 5V only on 5V pins)
- Don't pull camera ribbon cable while RPi is on
- Don't use damaged cables
- Keep electronics away from water
- Don't leave power bank charging overnight unattended

---

## Tools You'll Need

- [ ] Laptop (for SSH, Arduino IDE)
- [ ] USB cable (ESP32 programming)
- [ ] Small screwdriver (optional, for breadboard)
- [ ] Tape (mounting components)
- [ ] Scissors (cutting tape)
- [ ] Multimeter (optional, for debugging)
- [ ] Phone (hotspot WiFi backup)

---

## Pin Reference Card (Print This!)

**ESP32 Pins:**
```
GPIO 18 → Ultrasonic TRIG
GPIO 19 → Ultrasonic ECHO
GPIO 0  → Push Button
5V      → Ultrasonic VCC
GND     → Ultrasonic GND, Button GND
USB     → Raspberry Pi (data + power)
```

**Raspberry Pi Ports:**
```
CSI Port       → Camera ribbon cable
USB 3.0 (blue) → Pendrive (boot disk)
USB 2.0        → ESP32 (any USB port)
USB-C          → Power bank (15W, 3A)
GPIO Pins      → (not used in quick demo)
WiFi           → Built-in (connect to network)
Bluetooth      → Built-in (pair earbuds)
```

**HC-SR04 Pins:**
```
VCC  → 5V (red wire)
TRIG → GPIO 18 (yellow wire)
ECHO → GPIO 19 (green wire)
GND  → GND (black wire)
```

---

**DONE! Hardware wiring complete. Now go to wireless/README.md for software setup!** 🎉
