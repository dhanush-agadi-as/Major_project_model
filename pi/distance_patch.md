# Distance Endpoint Patch for `pi_hub.py` on Raspberry Pi

This patch adds the `GET /distance` endpoint to `pi_hub.py` without modifying the `/video` MJPEG streaming endpoint.

---

## 1. Code to Add to `pi_hub.py`

### A. Imports & Global State
Ensure `threading`, `time`, `serial`, and `jsonify` are imported:

```python
import time
import threading
import serial
from flask import Flask, Response, jsonify
```

Add the thread-safe ESP32 distance tracker:

```python
# ─── ESP32 Serial & Distance State ──────────────────────────────────────────
latest_distance = None
latest_distance_time = 0.0
distance_lock = threading.Lock()

def esp32_serial_reader():
    """Background thread reading ESP32 distance over USB serial (/dev/ttyUSB0)."""
    global latest_distance, latest_distance_time
    ser = None
    for port in ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1']:
        try:
            ser = serial.Serial(port, 115200, timeout=0.1)
            print(f"[OK] Connected to ESP32 on {port}")
            break
        except Exception as e:
            continue

    if ser is None:
        print("[WARNING] Could not connect to ESP32 on USB serial")

    while True:
        if ser is None or not ser.is_open:
            time.sleep(1.0)
            for port in ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1']:
                try:
                    ser = serial.Serial(port, 115200, timeout=0.1)
                    print(f"[OK] Reconnected to ESP32 on {port}")
                    break
                except Exception:
                    continue
            continue

        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith("DIST:"):
                # Example: "DIST:123.45,BTN:0"
                parts = line.split(',')
                dist_str = parts[0].split(':')[1]
                val = float(dist_str)
                with distance_lock:
                    latest_distance = val
                    latest_distance_time = time.time()
        except Exception:
            time.sleep(0.05)

# Start reader thread at module startup if not already reading serial
esp32_thread = threading.Thread(target=esp32_serial_reader, daemon=True, name="ESP32-Reader")
esp32_thread.start()
```

*(Note: If `pi_hub.py` already has an ESP32 serial reader loop, simply update `latest_distance` and `latest_distance_time` under `distance_lock` when `line.startswith("DIST:")` is parsed).*

---

### B. Add `GET /distance` Route (Do NOT modify `/video`)
Add this endpoint to the Flask app:

```python
@app.route('/distance', methods=['GET'])
def get_distance():
    """
    Returns latest ESP32 ultrasonic distance:
    {"distance_cm": <float or null>, "age_ms": <int>, "ok": true}
    null when no valid reading for 500 ms.
    """
    now = time.time()
    with distance_lock:
        dist = latest_distance
        t = latest_distance_time

    if t > 0:
        age_ms = int((now - t) * 1000)
    else:
        age_ms = 999999

    # Return null if no reading or if older than 500 ms
    valid_distance = dist if (age_ms <= 500 and dist is not None) else None

    return jsonify({
        "distance_cm": valid_distance,
        "age_ms": age_ms,
        "ok": True
    })
```

---

## 2. How to Apply this to the Raspberry Pi

### Option 1: Copy the complete `pi/pi_hub.py` from this repository
From your laptop terminal / PowerShell:
```bash
scp pi/pi_hub.py pi@raspberrypi.local:~/pi_hub.py
```

### Option 2: Edit `pi_hub.py` directly on the Pi
1. SSH into the Pi:
   ```bash
   ssh pi@raspberrypi.local
   ```
2. Open `pi_hub.py`:
   ```bash
   nano ~/pi_hub.py
   ```
3. Paste the `esp32_serial_reader` and `@app.route('/distance')` blocks shown above into `pi_hub.py`.
4. Save (`Ctrl+O`, `Enter`) and exit (`Ctrl+X`).
5. Restart `pi_hub.py`:
   ```bash
   python3 ~/pi_hub.py
   ```

### Option 3: Verify the Endpoint
On your laptop or in a browser, open:
```
http://raspberrypi.local:5000/distance
```
You should see:
```json
{"age_ms": 42, "distance_cm": 68.4, "ok": true}
```
