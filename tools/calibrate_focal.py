"""
Distance Calibration Helper
===========================
Calculates FOCAL_PX for your webcam (pinhole model: FOCAL_PX = px * N / 15).

Usage:
    python tools/calibrate_focal.py 50
    python tools/calibrate_focal.py --distance 50
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import config
from object_detector import ObjectDetector


def main():
    parser = argparse.ArgumentParser(description="Calibrate camera focal length using a cell phone (15 cm).")
    parser.add_argument("dist_cm", nargs="?", type=float, default=None,
                        help="Distance in cm to hold cell phone (e.g. 50)")
    parser.add_argument("--distance", "-d", "--dist-cm", dest="opt_dist_cm", type=float, default=None,
                        help="Distance in cm to hold cell phone (e.g. 50)")
    parser.add_argument("--frames", type=int, default=30,
                        help="Number of detection frames to average (default: 30)")
    parser.add_argument("--camera", default=None,
                        help="Camera device index or stream URL (default: config.CAMERA_SOURCE)")
    args = parser.parse_args()

    dist_cm = args.dist_cm if args.dist_cm is not None else (args.opt_dist_cm if args.opt_dist_cm is not None else 50.0)
    target_frames = args.frames

    cam_src = args.camera if args.camera is not None else getattr(config, 'CAMERA_SOURCE', 0)

    print("\n" + "=" * 60)
    print("DISTANCE CALIBRATION HELPER")
    print("=" * 60)
    print(f"Camera source: {cam_src}")
    print(f"Please hold your cell phone exactly {dist_cm:.0f} cm away from the camera lens.")
    print("Hold it upright, facing the camera directly, until 30 frames are captured.")
    print("Press 'q' at any time to exit.\n")

    frame_w = getattr(config, 'CAMERA_WIDTH', getattr(config, 'FRAME_WIDTH', 640))
    frame_h = getattr(config, 'CAMERA_HEIGHT', getattr(config, 'FRAME_HEIGHT', 480))

    from camera_source import open_camera
    cap = open_camera(cam_src, width=frame_w, height=frame_h)

    if not cap.isOpened():
        sys.exit(f"Error: Unable to open camera source: {cam_src}.")

    det = ObjectDetector()
    px_samples = []
    start_time = time.time()

    actual_w = 640
    actual_h = 480

    try:
        while len(px_samples) < target_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            actual_h, actual_w = frame.shape[:2]

            # Detect cell phone
            hits = [d for d in det.detect(frame) if d.get("class_name") in ("cell phone", "phone")]
            if hits:
                best = max(hits, key=lambda d: d.get("confidence", 0.0))
                x1, y1, x2, y2 = best["bbox"]
                px = float(max(x2 - x1, y2 - y1))
                px_samples.append(px)

                # Visual feedback on frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Cell phone: {px:.1f}px ({len(px_samples)}/{target_frames})",
                    (int(x1), max(20, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            status_text = f"Frames: {len(px_samples)}/{target_frames} at {dist_cm:.0f} cm ({actual_w}x{actual_h})"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Calibrate Focal Length", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nCalibration aborted by user.")
                break

            # Timeout after 90 seconds
            if time.time() - start_time > 90.0:
                print("\nCalibration timed out (no stable detections within 90 seconds).")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(px_samples) < 5:
        sys.exit(f"\nCollected only {len(px_samples)} frames. Please ensure lighting is good and cell phone is clearly visible.")

    avg_px = sum(px_samples) / len(px_samples)
    focal_px = avg_px * dist_cm / 15.0
    focal_640 = focal_px * (640.0 / actual_w) if actual_w > 0 else focal_px

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Captured frames: {len(px_samples)}")
    print(f"Frame resolution: {actual_w}x{actual_h}")
    print(f"Target distance (N): {dist_cm:.1f} cm")
    print(f"Average bbox longest side (px): {avg_px:.2f} pixels")
    print("\nFormula:")
    print("  FOCAL_PX = px * N / 15")
    print(f"  FOCAL_PX (for {actual_w}px width) = {avg_px:.2f} * {dist_cm:.1f} / 15 = {focal_px:.1f}")
    if actual_w != 640:
        print(f"  FOCAL_PX normalized for 640px base: {focal_640:.1f}")
    print("\nYou can paste this into config.py:")
    print(f"FOCAL_PX = {focal_640:.1f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
