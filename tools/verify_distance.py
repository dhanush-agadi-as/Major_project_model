"""
Live distance check - shows exactly what find_guidance.py would estimate, without voice or mic.

Run from the project root:   python tools/verify_distance.py --cls "cell phone"
Hold the object at distances you measure with a tape (lens to object): 40, 70, 100, 150 cm.
The window shows the estimate on the box; the terminal prints a 1-second median.
Pass if the estimate is within about 10-15% of the tape measure. Press q to quit.
The thresholds that matter for the spoken callouts are 45 cm (arrival), 70 cm and 150 cm.
"""
import argparse, os, sys, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cv2, config
import find_guidance as fg
from object_detector import ObjectDetector

ap = argparse.ArgumentParser()
ap.add_argument("--cls", default="cell phone")
ap.add_argument("--camera", default=None)
a = ap.parse_args()

cam_src = a.camera if a.camera is not None else getattr(config, 'CAMERA_SOURCE', 0)
size = fg.TYPICAL_SIZES_CM.get(a.cls.lower(), fg._DEFAULT_SIZE_CM)
print(f"FOCAL_PX={fg.FOCAL_PX}  assumed size of '{a.cls}'={size} cm"
      + ("  (DEFAULT size - not in TYPICAL_SIZES_CM)" if a.cls.lower() not in fg.TYPICAL_SIZES_CM else ""))

from camera_source import open_camera
cap = open_camera(cam_src, width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT)
det = ObjectDetector()
win, last_print = [], time.time()

def bucket(d):
    if d > fg._DIST_BUCKET_FAR: return "about 2 meters"
    if d > fg._DIST_BUCKET_MID: return "about 1 meter"
    return "within arm's reach"

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    hits = [d for d in det.detect(frame) if d["class_name"] == a.cls]
    if hits:
        x1, y1, x2, y2 = max(hits, key=lambda d: d["confidence"])["bbox"]
        dist = fg.FOCAL_PX * size / max(max(x2 - x1, y2 - y1), 1)
        win.append(dist); win[:] = win[-15:]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{dist:.0f} cm  {bucket(dist)}", (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("verify distance", frame)
    if time.time() - last_print > 1.0 and win:
        print(f"estimate median: {statistics.median(win):.0f} cm  -> {bucket(statistics.median(win))}")
        last_print = time.time()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release(); cv2.destroyAllWindows()
