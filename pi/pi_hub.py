#!/usr/bin/env python3
"""
pi_hub.py
=========
Raspberry Pi Hub for VisionMate Smart Glasses:
- Serves MJPEG video stream at http://<PI_HOST>:5000/video (640x480, OV5647 via picamera2)
- Connects to ESP32 over USB serial (/dev/ttyUSB0)
- Serves ultrasonic distance at http://<PI_HOST>:5000/distance
"""

import time
import threading
import logging
import cv2
from flask import Flask, Response, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pi_hub")

app = Flask(__name__)

# ─── Global State ─────────────────────────────────────────────────────────────
latest_raw_frame = None
latest_jpeg_frame = None
frame_lock = threading.Lock()

active_clients = 0
clients_lock = threading.Lock()
last_encode_time = 0.0

latest_distance = None
latest_distance_time = 0.0
distance_lock = threading.Lock()

# ─── Camera Setup (picamera2 / OpenCV fallback) ────────────────────────────────
picam2 = None
try:
    from picamera2 import Picamera2
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    logger.info("[OK] picamera2 OV5647 camera started (640x480)")
except Exception as e:
    logger.warning("picamera2 not available or failed (%s); falling back to OpenCV VideoCapture", e)
    picam2 = None


def camera_capture_loop():
    """Background thread continuously grabbing frames for the MJPEG stream without blocking."""
    global latest_raw_frame, latest_jpeg_frame, last_encode_time
    cap = None
    if picam2 is None:
        try:
            from camera_source import open_camera
            cap = open_camera(0, width=640, height=480)
        except Exception:
            device_index = 0
            cap = cv2.VideoCapture(device_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        try:
            if picam2 is not None:
                frame_rgb = picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            elif cap is not None and cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    time.sleep(0.01)
                    continue
            else:
                time.sleep(0.05)
                continue

            with frame_lock:
                latest_raw_frame = frame_bgr

            # Encode ONLY when a client is connected, capped at 20 FPS (0.05s)
            with clients_lock:
                has_clients = active_clients > 0

            if has_clients:
                now = time.monotonic()
                if (now - last_encode_time) >= 0.05:  # Cap at 20 FPS
                    _, jpeg = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    with frame_lock:
                        latest_jpeg_frame = jpeg.tobytes()
                    last_encode_time = now
        except Exception as e:
            logger.debug("Camera capture error: %s", e)
            time.sleep(0.01)


# ─── ESP32 Serial Reader ──────────────────────────────────────────────────────
def esp32_serial_reader():
    """Background thread reading ESP32 distance over USB serial."""
    global latest_distance, latest_distance_time
    import serial

    ser = None
    for port in ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1']:
        try:
            ser = serial.Serial(port, 115200, timeout=0.1)
            logger.info("Connected to ESP32 on %s", port)
            break
        except Exception:
            continue

    if ser is None:
        logger.warning("ESP32 not found on USB serial ports. Retrying in background...")

    while True:
        if ser is None or not ser.is_open:
            time.sleep(1.0)
            for port in ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1']:
                try:
                    ser = serial.Serial(port, 115200, timeout=0.1)
                    logger.info("Connected to ESP32 on %s", port)
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


# ─── Flask Routes ─────────────────────────────────────────────────────────────
def generate_mjpeg():
    """Generator yielding MJPEG frames capped at 20 FPS."""
    global active_clients
    with clients_lock:
        active_clients += 1
    try:
        while True:
            with frame_lock:
                frame_bytes = latest_jpeg_frame
            if frame_bytes is None:
                time.sleep(0.01)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)  # Cap at 20 FPS
    finally:
        with clients_lock:
            active_clients = max(0, active_clients - 1)


@app.route('/video')
def video_feed():
    """MJPEG streaming endpoint for VisionMate."""
    return Response(
        generate_mjpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )



@app.route('/sensors', methods=['GET'])
def get_sensors():
    """
    Returns latest sensor readings:
    {"ultrasonic_distance_cm": <float|null>, "age_ms": <int>, "ok": <bool>}
    Set distance to null when no valid line arrived for 1.5 s (never keep the old value).
    """
    now = time.time()
    with distance_lock:
        dist = latest_distance
        t = latest_distance_time

    age_ms = int((now - t) * 1000) if t > 0 else 999999
    ok = (age_ms < 1500)
    ultrasonic_dist = dist if ok else None

    return jsonify({
        "ultrasonic_distance_cm": ultrasonic_dist,
        "age_ms": age_ms,
        "ok": ok
    })


@app.route('/distance', methods=['GET'])
def get_distance():
    """
    Returns latest ultrasonic distance reading:
    {"distance_cm": <float or null>, "age_ms": <int>, "ok": true}
    null when no valid reading for 500 ms.
    """
    now = time.time()
    with distance_lock:
        dist = latest_distance
        t = latest_distance_time

    age_ms = int((now - t) * 1000) if t > 0 else 999999
    valid_distance = dist if (age_ms <= 500 and dist is not None) else None

    return jsonify({
        "distance_cm": valid_distance,
        "age_ms": age_ms,
        "ok": True
    })


@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "pi_hub"})


if __name__ == '__main__':
    # Start background threads
    cam_th = threading.Thread(target=camera_capture_loop, daemon=True, name="CamCapture")
    cam_th.start()

    esp_th = threading.Thread(target=esp32_serial_reader, daemon=True, name="ESP32Reader")
    esp_th.start()

    logger.info("Starting pi_hub on port 5000...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
