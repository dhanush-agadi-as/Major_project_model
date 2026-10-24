"""
VisionMate Mobile App Backend
Runs on: Laptop (alongside laptop_server.py)
Port: 5001

Features:
  1. GET  /api/location       → GPS coordinates (from mock GPS tracker)
  2. GET  /api/video-feed     → MJPEG stream from laptop webcam
  3. GET  /api/snapshot       → Single JPEG frame
  4. GET  /api/notifications  → Poll for new notifications
  5. GET  /api/status         → Server status + device info

Terminal CLI: Type a message → sends as notification to the mobile app

Usage:
  conda activate D:\\conda_envs\\visionmate
  python mobile_backend.py
"""

import sys
import os
import time
import json
import socket
import threading
import cv2
from flask import Flask, Response, request, jsonify
from flask_cors import CORS

# Add parent directory for VisionMate modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gps_tracker import GPSTracker
import config

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the mobile app

# ─── State ───────────────────────────────────────────────
gps_tracker = GPSTracker(mock_mode=config.GPS_MOCK_MODE)
notifications_queue = []  # Pending notifications for the app to pick up
notifications_lock = threading.Lock()

# Webcam for video feed
camera = None
camera_lock = threading.Lock()

# Server start time
server_start_time = time.time()


# ─── Webcam Helpers ──────────────────────────────────────

def get_camera():
    """Get or initialize the camera"""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            from camera_source import open_camera
            camera = open_camera(width=640, height=480, fps=30)
            if camera.isOpened():
                print("[OK] Camera opened for video feed")
            else:
                print("[WARN] Could not open camera")
    return camera


def generate_mjpeg_frames():
    """Generator that yields MJPEG frames from laptop webcam"""
    while True:
        cam = get_camera()
        if cam is None or not cam.isOpened():
            # Send a placeholder frame
            placeholder = _create_placeholder_frame("No Camera Available")
            _, buffer = cv2.imencode('.jpg', placeholder)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
            time.sleep(1)
            continue

        with camera_lock:
            ret, frame = cam.read()

        if not ret:
            time.sleep(0.1)
            continue

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
        time.sleep(0.066)  # ~15 fps


def _create_placeholder_frame(text):
    """Create a placeholder frame with text"""
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)
    cv2.putText(frame, text, (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "VisionMate", (220, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
    return frame


# ─── API Endpoints ───────────────────────────────────────

_last_phone_post_time = None
_last_phone_accuracy = None


@app.route('/api/location', methods=['GET'])
def get_location():
    """Return current GPS coordinates (highest-priority available source)"""
    pos = gps_tracker.get_position()
    place_name = gps_tracker.get_current_place_name()
    if pos:
        if gps_tracker.manual_override:
            source = "manual"
        elif gps_tracker._get_fresh_phone_position() is not None:
            source = "phone"
        elif gps_tracker.ip_tracker is not None:
            source = "ip"
        else:
            source = "mock"
        ts = _last_phone_post_time if (source == "phone" and _last_phone_post_time is not None) else time.time()
        resp = {
            "latitude": pos[0],
            "longitude": pos[1],
            "place_name": place_name,
            "source": source,
            "timestamp": ts
        }
        if source == "phone" and _last_phone_accuracy is not None:
            resp["accuracy"] = _last_phone_accuracy
        return jsonify(resp)
    # Default fallback: RR Nagar, Bangalore
    return jsonify({
        "latitude": 12.926516,
        "longitude": 77.526422,
        "place_name": place_name,
        "source": "mock",
        "timestamp": time.time()
    })


@app.route('/api/location', methods=['POST'])
def post_location():
    """
    Accept a GPS position reported by a phone/browser (navigator.geolocation).
    This is the free, precise, software-only location source: open /locate
    on a phone, grant location permission, and it will keep posting here.

    Expected JSON body: {"latitude": float, "longitude": float, "accuracy": float (optional)}
    """
    global _last_phone_post_time, _last_phone_accuracy
    data = request.get_json(silent=True) or {}
    try:
        lat = float(data['latitude'])
        lon = float(data['longitude'])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "latitude and longitude (floats) are required"}), 400

    accuracy = data.get('accuracy')
    try:
        accuracy = float(accuracy) if accuracy is not None else None
    except (TypeError, ValueError):
        accuracy = None

    _last_phone_post_time = time.time()
    _last_phone_accuracy = accuracy
    gps_tracker.set_phone_position(lat, lon, accuracy=accuracy, timestamp=_last_phone_post_time)
    return jsonify({
        "status": "ok",
        "latitude": lat,
        "longitude": lon,
        "accuracy": accuracy,
        "timestamp": _last_phone_post_time
    })


@app.route('/locate')
def locate_page():
    """
    Minimal phone-facing page: open this URL in a phone's browser (on the
    same network as this server) and grant location permission. It will
    keep posting the phone's real GPS position to POST /api/location,
    which becomes the primary position source (Layer 2) for the whole
    system as long as reports stay fresh.
    """
    host_hint = request.host.split(':')[0]
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VisionMate Location Sharing</title>
  <style>
    :root {{
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-muted: #8b949e;
      --ok: #2ea043;
      --err: #cf222e;
      --waiting: #d29922;
      --sending: #58a6ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      text-align: center;
      padding: 1.5em 1em;
      margin: 0;
    }}
    .container {{
      max-width: 480px;
      margin: 0 auto;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5em;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    h2 {{
      margin: 0.2em 0 0.5em 0;
      color: #f0f6fc;
      font-size: 1.4em;
    }}
    p.subtitle {{
      color: var(--text-muted);
      font-size: 0.9em;
      margin-bottom: 1.2em;
    }}
    #status {{
      font-size: 1.05em;
      padding: 0.8em;
      border-radius: 8px;
      margin: 1em 0;
      white-space: pre-line;
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border);
      font-weight: 500;
    }}
    .ok {{ color: var(--ok); border-color: rgba(46,160,67,0.4) !important; background: rgba(46,160,67,0.1) !important; }}
    .err {{ color: var(--err); border-color: rgba(207,34,46,0.4) !important; background: rgba(207,34,46,0.1) !important; }}
    .waiting {{ color: var(--waiting); border-color: rgba(210,153,34,0.4) !important; background: rgba(210,153,34,0.1) !important; }}
    .status-sending {{ color: var(--sending); font-weight: 600; }}
    .info-card {{
      background: rgba(0,0,0,0.25);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.6em 1em;
      text-align: left;
      font-size: 0.95em;
    }}
    .info-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.5em 0;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }}
    .info-row:last-child {{
      border-bottom: none;
    }}
    .info-label {{
      color: var(--text-muted);
      font-weight: 500;
    }}
    .info-value {{
      font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, monospace;
      color: #f0f6fc;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h2>VisionMate Location Sharing</h2>
    <p class="subtitle">Keep this page open to share your precise location with the device.</p>
    <div id="status" class="waiting">Waiting for location... try near a window</div>
    <div class="info-card">
      <div class="info-row"><span class="info-label">Status</span><span class="info-value" id="sendStatus">Idle</span></div>
      <div class="info-row"><span class="info-label">Latitude</span><span class="info-value" id="lat">-</span></div>
      <div class="info-row"><span class="info-label">Longitude</span><span class="info-value" id="lon">-</span></div>
      <div class="info-row"><span class="info-label">Accuracy</span><span class="info-value" id="acc">-</span></div>
      <div class="info-row"><span class="info-label">Time</span><span class="info-value" id="time">-</span></div>
      <div class="info-row"><span class="info-label">Positions Sent</span><span class="info-value" id="sentCount">0</span></div>
    </div>
  </div>
  <script>
    const statusEl = document.getElementById('status');
    const sendStatusEl = document.getElementById('sendStatus');
    const latEl = document.getElementById('lat');
    const lonEl = document.getElementById('lon');
    const accEl = document.getElementById('acc');
    const timeEl = document.getElementById('time');
    const sentCountEl = document.getElementById('sentCount');

    let lowWatchId = null;
    let highWatchId = null;
    let lastPos = null;
    let sentCount = 0;
    let lastSentTime = 0;

    if (window.isSecureContext === false) {{
      statusEl.className = 'err';
      statusEl.textContent = 'Open the https link and accept the certificate';
      sendStatusEl.textContent = 'Insecure context';
    }} else if (!navigator.geolocation) {{
      statusEl.className = 'err';
      statusEl.textContent = 'Geolocation is not supported by this browser.';
      sendStatusEl.textContent = 'Unsupported';
    }} else {{
      function sendPosition(posData) {{
        sendStatusEl.textContent = 'Sending...';
        sendStatusEl.className = 'info-value status-sending';

        fetch('/api/location', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            latitude: posData.latitude,
            longitude: posData.longitude,
            accuracy: posData.accuracy
          }})
        }})
        .then(function(r) {{
          if (!r.ok) {{
            throw new Error('HTTP ' + r.status);
          }}
          return r.json();
        }})
        .then(function() {{
          sentCount++;
          lastSentTime = Date.now();
          const nowStr = new Date().toLocaleTimeString();

          sendStatusEl.textContent = 'Sent OK';
          sendStatusEl.className = 'info-value ok';
          statusEl.className = 'ok';
          statusEl.textContent = 'Sharing location';

          latEl.textContent = Number(posData.latitude).toFixed(6);
          lonEl.textContent = Number(posData.longitude).toFixed(6);
          accEl.textContent = Math.round(posData.accuracy) + ' m';
          timeEl.textContent = nowStr;
          sentCountEl.textContent = sentCount;
        }})
        .catch(function(e) {{
          sendStatusEl.textContent = 'Failed to send: ' + e;
          sendStatusEl.className = 'info-value err';
        }});
      }}

      function onPosition(pos, isHigh) {{
        const coords = pos.coords;
        const latitude = coords.latitude;
        const longitude = coords.longitude;
        const accuracy = coords.accuracy;

        // Use whichever result arrives first and keep improving as better ones come in
        if (lastPos) {{
          if (!isHigh && lastPos.isHigh && accuracy > lastPos.accuracy) {{
            return;
          }}
        }}

        lastPos = {{
          latitude: latitude,
          longitude: longitude,
          accuracy: accuracy,
          isHigh: isHigh
        }};
        sendPosition(lastPos);
      }}

      function onError(err, source) {{
        if (err.code === 1 || err.code === err.PERMISSION_DENIED) {{
          statusEl.className = 'err';
          statusEl.textContent = 'Location permission denied.\\nPlease allow location access for this site in your browser settings (https://{host_hint}:5001/locate) and reload.';
          if (lowWatchId !== null) {{ navigator.geolocation.clearWatch(lowWatchId); lowWatchId = null; }}
          if (highWatchId !== null) {{ navigator.geolocation.clearWatch(highWatchId); highWatchId = null; }}
        }} else if (err.code === 2 || err.code === err.POSITION_UNAVAILABLE) {{
          statusEl.className = 'err';
          statusEl.textContent = 'Location unavailable, turn on phone location';
          setTimeout(function() {{
            if (source === 'low') startLowWatch();
            else startHighWatch();
          }}, 5000);
        }} else if (err.code === 3 || err.code === err.TIMEOUT) {{
          statusEl.className = 'waiting';
          statusEl.textContent = 'Waiting for location... try near a window';
          if (source === 'low') {{
            if (lowWatchId !== null) {{ navigator.geolocation.clearWatch(lowWatchId); lowWatchId = null; }}
            setTimeout(startLowWatch, 1000);
          }} else {{
            if (highWatchId !== null) {{ navigator.geolocation.clearWatch(highWatchId); highWatchId = null; }}
            setTimeout(startHighWatch, 1000);
          }}
        }} else {{
          statusEl.className = 'err';
          statusEl.textContent = 'Location error: ' + (err.message || 'Unknown error');
        }}
      }}

      function startLowWatch() {{
        if (lowWatchId !== null) {{
          navigator.geolocation.clearWatch(lowWatchId);
          lowWatchId = null;
        }}
        lowWatchId = navigator.geolocation.watchPosition(
          function(pos) {{ onPosition(pos, false); }},
          function(err) {{ onError(err, 'low'); }},
          {{ enableHighAccuracy: false, timeout: 30000, maximumAge: 60000 }}
        );
      }}

      function startHighWatch() {{
        if (highWatchId !== null) {{
          navigator.geolocation.clearWatch(highWatchId);
          highWatchId = null;
        }}
        highWatchId = navigator.geolocation.watchPosition(
          function(pos) {{ onPosition(pos, true); }},
          function(err) {{ onError(err, 'high'); }},
          {{ enableHighAccuracy: true, timeout: 60000 }}
        );
      }}

      // Start low-accuracy first, and in parallel request high-accuracy watch
      startLowWatch();
      startHighWatch();

      // Keep-alive: re-send last fix every 5s so server does not treat fix as stale after 30s
      setInterval(function() {{
        if (lastPos && (Date.now() - lastSentTime >= 4500)) {{
          sendPosition(lastPos);
        }}
      }}, 5000);

      if ('wakeLock' in navigator) {{
        navigator.wakeLock.request('screen').catch(function() {{}});
      }}
    }}
  </script>
</body>
</html>"""


@app.route('/api/video-feed')
def video_feed():
    """MJPEG video stream from laptop webcam"""
    return Response(
        generate_mjpeg_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/snapshot')
def snapshot():
    """Single JPEG frame from laptop webcam"""
    cam = get_camera()
    if cam and cam.isOpened():
        with camera_lock:
            ret, frame = cam.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return Response(buffer.tobytes(), mimetype='image/jpeg')

    # Return placeholder
    placeholder = _create_placeholder_frame("No Camera")
    _, buffer = cv2.imencode('.jpg', placeholder)
    return Response(buffer.tobytes(), mimetype='image/jpeg')


@app.route('/api/notifications')
def get_notifications():
    """Poll for new notifications (app calls this every 2-3 seconds)"""
    with notifications_lock:
        notifs = list(notifications_queue)
        notifications_queue.clear()
    return jsonify({"notifications": notifs})


@app.route('/api/status')
def get_status():
    """Server status and info"""
    uptime = time.time() - server_start_time
    cam = get_camera()
    cam_ok = cam is not None and cam.isOpened()
    pos = gps_tracker.get_position()
    place_name = gps_tracker.get_current_place_name()

    return jsonify({
        "status": "running",
        "uptime_seconds": int(uptime),
        "webcam_available": cam_ok,
        "gps_position": {"lat": pos[0], "lng": pos[1]} if pos else None,
        "place_name": place_name,
        "gps_mock_mode": config.GPS_MOCK_MODE,
        "pending_notifications": len(notifications_queue),
    })


# ─── Notification Helpers ────────────────────────────────

def add_notification(title, message, priority="high"):
    """Add a notification to the queue (picked up by app polling)"""
    with notifications_lock:
        notifications_queue.append({
            "id": int(time.time() * 1000),
            "title": title,
            "message": message,
            "priority": priority,
            "timestamp": time.time()
        })


# ─── Terminal CLI ────────────────────────────────────────

def notification_cli(local_ip):
    """Interactive terminal for sending notifications to the app"""
    print("\n" + "=" * 60)
    print("NOTIFICATION SENDER")
    print("=" * 60)
    print("Type a message and press Enter to send to the app")
    print("Commands:")
    print("  <message>     -> Send emergency notification")
    print("  /status       -> Show server status")
    print("  /location     -> Show current GPS position")
    print("  /advance      -> Advance mock GPS position")
    print("  quit          -> Exit server")
    print("=" * 60 + "\n")

    while True:
        try:
            msg = input("[ALERT] Notify > ").strip()
            if not msg:
                continue

            if msg.lower() == 'quit':
                print("Shutting down...")
                os._exit(0)

            elif msg == '/status':
                cam = get_camera()
                cam_ok = cam is not None and cam.isOpened()
                pos = gps_tracker.get_position()
                uptime = time.time() - server_start_time
                print(f"  Uptime: {int(uptime)}s")
                print(f"  Webcam: {'OK' if cam_ok else 'NOT AVAILABLE'}")
                print(f"  GPS: {pos}")
                print(f"  Pending notifications: {len(notifications_queue)}")

            elif msg == '/location':
                pos = gps_tracker.get_position()
                print(f"  GPS Position: {pos}")

            elif msg == '/advance':
                if config.GPS_MOCK_MODE:
                    gps_tracker.advance_mock_position()
                    pos = gps_tracker.get_position()
                    print(f"  GPS advanced -> {pos[0]:.6f}, {pos[1]:.6f}")
                else:
                    print("  [WARN] GPS is in real mode, cannot advance mock")

            else:
                # Send as emergency notification
                add_notification(
                    title="[ALERT] VisionMate Alert",
                    message=msg,
                    priority="high"
                )
                print(f"  [OK] Notification queued: \"{msg}\"")
                print(f"    (App will pick it up on next poll)")

        except (EOFError, KeyboardInterrupt):
            print("\nShutting down...")
            break


# ─── Main ────────────────────────────────────────────────

def get_local_ip():
    """Get laptop's local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    local_ip = get_local_ip()

    # Start GPS tracker
    gps_tracker.start()

    print("=" * 60)
    print("VISIONMATE MOBILE APP BACKEND")
    print("=" * 60)
    print(f"\n  Server: https://{local_ip}:5001")
    print(f"\n  Endpoints:")
    print(f"    GET  /api/location       -> GPS coordinates")
    print(f"    GET  /api/video-feed     -> MJPEG webcam stream")
    print(f"    GET  /api/snapshot       -> Single JPEG frame")
    print(f"    GET  /api/notifications  -> Poll for notifications")
    print(f"    GET  /api/status         -> Server status")
    print(f"\n  Quick test from browser:")
    print(f"    https://{local_ip}:5001/api/location")
    print(f"    https://{local_ip}:5001/api/video-feed")
    print("=" * 60)

    # Start Flask in background thread
    flask_thread = threading.Thread(
        target=lambda: app.run(
            host='0.0.0.0',
            port=5001,
            ssl_context='adhoc',
            threaded=True,
            use_reloader=False
        ),
        daemon=True
    )
    flask_thread.start()
    print(f"\n[OK] Flask server started on port 5001")

    # Run notification CLI in foreground
    notification_cli(local_ip)

    # Cleanup
    gps_tracker.stop()
    if camera:
        camera.release()
    print("Goodbye!")


if __name__ == '__main__':
    main()
