import cv2
import serial
import threading
import time
import math
import re
from flask import Flask, Response, jsonify
from picamera2 import Picamera2
from mpu6050 import mpu6050

app = Flask(__name__)

# -----------------------------
# 1. Picamera2 Video Pipeline
# -----------------------------
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# -----------------------------
# 2. Shared Telemetry Data & Lock
# -----------------------------
data_lock = threading.Lock()
telemetry = {
    "ultrasonic_distance_cm": -1.0,
    "mpu6050": {
        "status": "initializing",
        "accel_x": 0.0,
        "accel_y": 0.0,
        "accel_z": 0.0,
        "gyro_x": 0.0,
        "gyro_y": 0.0,
        "gyro_z": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "temp_c": 0.0
    }
}

# -----------------------------
# 3. ESP32 Serial Reader Thread
# -----------------------------
def read_esp32_serial():
    global telemetry
    possible_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0']
    
    # Matches any float like 4.42, 24.5, 150.32
    number_pattern = re.compile(r"[-+]?\d*\.\d+|\d+")

    while True:
        for port in possible_ports:
            try:
                esp = serial.Serial(port, 115200, timeout=1.0)
                # Discard startup garbage from buffer
                time.sleep(0.5)
                esp.reset_input_buffer()
                print(f"[INFO] ESP32 ultrasonic connected on {port}")
                
                while True:
                    raw_bytes = esp.readline()
                    line = raw_bytes.decode('ascii', errors='ignore').strip()
                    if line:
                        # Extract the first valid number found in the string
                        match = number_pattern.search(line)
                        if match:
                            try:
                                val = float(match.group())
                                if val > 0.0:
                                    with data_lock:
                                        telemetry["ultrasonic_distance_cm"] = round(val, 2)
                            except ValueError:
                                pass
            except (serial.SerialException, FileNotFoundError):
                pass
        time.sleep(2.0)

# -----------------------------
# 4. MPU-6050 I2C Reader Thread
# -----------------------------
def read_mpu6050():
    global telemetry
    sensor = None
    
    while sensor is None:
        try:
            sensor = mpu6050(0x68)
            print("[INFO] MPU-6050 connected on I2C address 0x68")
        except Exception:
            print("[WARN] Waiting for MPU-6050 connection...")
            time.sleep(2.0)

    while True:
        try:
            accel = sensor.get_accel_data()
            gyro = sensor.get_gyro_data()
            temp = sensor.get_temp()

            ax, ay, az = accel['x'], accel['y'], accel['z']
            roll = math.atan2(ay, az) * 180.0 / math.pi
            pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az)) * 180.0 / math.pi

            with data_lock:
                telemetry["mpu6050"] = {
                    "status": "online",
                    "accel_x": round(ax, 3),
                    "accel_y": round(ay, 3),
                    "accel_z": round(az, 3),
                    "gyro_x": round(gyro['x'], 3),
                    "gyro_y": round(gyro['y'], 3),
                    "gyro_z": round(gyro['z'], 3),
                    "pitch": round(pitch, 2),
                    "roll": round(roll, 2),
                    "temp_c": round(temp, 1)
                }
        except Exception:
            with data_lock:
                telemetry["mpu6050"]["status"] = "error"
        time.sleep(0.05)

# Start background reader threads
threading.Thread(target=read_esp32_serial, daemon=True).start()
threading.Thread(target=read_mpu6050, daemon=True).start()

# -----------------------------
# 5. HTTP Endpoints
# -----------------------------
def generate_video():
    while True:
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sensors')
def sensors():
    with data_lock:
        data_copy = dict(telemetry)
    return jsonify(data_copy)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)