"""
Measure FOCAL_PX for YOUR laptop webcam (pinhole model: f = pixels * distance / real_size).

Setup: put a known object (default: cell phone, longest side 15 cm - measure yours!)
at a measured distance (default 50 cm, camera lens to object), long side parallel to the
image plane, then run from the project root:

    python tools/calibrate_focal.py --cls "cell phone" --size-cm 15 --dist-cm 50
    (repeat at 80 cm for a sanity check - both should agree within ~5%)

Paste the printed line into find_guidance.py (FOCAL_PX).
Uses the same CAMERA_WIDTH/HEIGHT as the app so the value matches live behaviour.
"""
import argparse, os, sys, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cv2, config
from object_detector import ObjectDetector

ap = argparse.ArgumentParser()
ap.add_argument("--cls", default="cell phone")
ap.add_argument("--size-cm", type=float, default=15.0, help="true longest side of the object")
ap.add_argument("--dist-cm", type=float, default=50.0, help="measured lens-to-object distance")
ap.add_argument("--samples", type=int, default=40)
ap.add_argument("--camera", type=int, default=0)
a = ap.parse_args()

cap = cv2.VideoCapture(a.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
det = ObjectDetector()
vals, t0 = [], time.time()
print(f"Hold the {a.cls} {a.dist_cm:.0f} cm from the lens. Collecting {a.samples} samples (q to abort)...")
while len(vals) < a.samples and time.time() - t0 < 60:
    ok, frame = cap.read()
    if not ok:
        continue
    hits = [d for d in det.detect(frame) if d["class_name"] == a.cls]
    if hits:
        x1, y1, x2, y2 = max(hits, key=lambda d: d["confidence"])["bbox"]
        px = max(x2 - x1, y2 - y1)
        vals.append(px * a.dist_cm / a.size_cm)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{len(vals)}/{a.samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("calibrate", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release(); cv2.destroyAllWindows()
if len(vals) < 10:
    sys.exit(f"Only {len(vals)} detections - object not seen reliably. Improve lighting / bring it closer.")
med = statistics.median(vals)
print(f"\nsamples={len(vals)}  median={med:.1f}  stdev={statistics.pstdev(vals):.1f}")
print(f"\n>>> FOCAL_PX: float = {med:.1f}\n")
