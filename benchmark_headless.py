"""
VisionMate headless benchmark — no display window required.
Captures startup log + per-stage timing for BENCH_FRAMES frames.
Mirrors the exact timing logic in test/test_integration.py.

v2: Uses CameraReader (threaded capture) so cap.read() never blocks
    the AI inference pipeline.  cap.read timing now reflects how often
    a *new* frame is available, not the raw I/O wait.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from camera_reader import CameraReader
from object_detector import ObjectDetector
from face_recognizer import FaceRecognizer
from object_memory import ObjectMemory
from database import VisionMateDB
from config import (
    OBJECT_RECOGNITION_FRAME_INTERVAL,
    FACE_RECOGNITION_FRAME_INTERVAL,
    YOLO_FRAME_SKIP,
)

BENCH_FRAMES = 30          # number of *inference* frames to time
REPORT_INTERVAL = 10       # print rolling average every N frames

print("=" * 60)
print("VISIONMATE HEADLESS BENCHMARK (threaded camera)")
print("=" * 60)

# ── init ──────────────────────────────────────────────────────
print("\nInitialising modules...")
db = VisionMateDB()
object_detector = ObjectDetector()
face_recognizer = FaceRecognizer(db)
object_memory   = ObjectMemory(db)
print("[OK] Modules ready\n")

print("Opening webcam (threaded, DirectShow)...")
cam = CameraReader(cam_index=0, width=640, height=480, fps=30,
                   backend=cv2.CAP_DSHOW, warmup_frames=5)
cam.start()

# Wait until the first frame is available (max 3 s)
deadline = time.time() + 3.0
while True:
    frame, fid = cam.read()
    if frame is not None:
        break
    if time.time() > deadline:
        print("[FAIL] No frame from webcam within 3 s - aborting")
        cam.stop()
        sys.exit(1)
    time.sleep(0.01)
print("[OK] Webcam opened (first frame received)\n")

# ── timing state ───────────────────────────────────────────────
timing_stats   = dict(new_frame_wait=0.0, yolo=0.0, obj_memory=0.0, face=0.0)
last_objects   = []
last_faces     = []
last_remembered_objects    = []
last_object_to_custom_name = {}

frame_count  = 0          # inference frames completed
last_fid     = fid        # last seen frame_id from camera thread

wall_start = time.time()

print(f"Benchmarking {BENCH_FRAMES} inference frames "
      f"(YOLO_FRAME_SKIP={YOLO_FRAME_SKIP}, "
      f"FACE_INTERVAL={FACE_RECOGNITION_FRAME_INTERVAL}, "
      f"OBJ_MEM_INTERVAL={OBJECT_RECOGNITION_FRAME_INTERVAL})...\n")

while frame_count < BENCH_FRAMES:
    # ── wait for a NEW frame (non-blocking poll) ─────────────────
    t0 = time.time()
    while True:
        frame, fid = cam.read()
        if fid != last_fid:
            break
        time.sleep(0.001)   # yield briefly — don't pin the CPU
    t1 = time.time()
    last_fid = fid
    frame_count += 1
    timing_stats['new_frame_wait'] += (t1 - t0) * 1000.0

    # ── YOLO (frame-skipped) ────────────────────────────────────
    t_y0 = time.time()
    if frame_count % YOLO_FRAME_SKIP == 0:
        objects = object_detector.detect(frame)
        last_objects = objects
    else:
        objects = last_objects
    t_y1 = time.time()
    timing_stats['yolo'] += (t_y1 - t_y0) * 1000.0

    # ── Object Memory (frame-skipped) ───────────────────────────
    t_m0 = time.time()
    if frame_count % OBJECT_RECOGNITION_FRAME_INTERVAL == 0:
        remembered_objects = []
        object_to_custom_name = {}
        for obj in objects:
            bbox = tuple(obj['bbox'])
            match = object_memory.recognize_object(frame, bbox, obj['class_name'])
            if match:
                custom_name, confidence, match_tier = match
                remembered_objects.append({
                    'custom_name': custom_name,
                    'yolo_class': obj['class_name'],
                    'confidence': confidence,
                    'tier': match_tier,
                    'bbox': obj['bbox']
                })
                object_to_custom_name[id(obj)] = custom_name
        last_remembered_objects    = remembered_objects
        last_object_to_custom_name = object_to_custom_name
    else:
        remembered_objects    = last_remembered_objects
        object_to_custom_name = last_object_to_custom_name
    t_m1 = time.time()
    timing_stats['obj_memory'] += (t_m1 - t_m0) * 1000.0

    # ── Face Recognition (frame-skipped) ────────────────────────
    t_f0 = time.time()
    if frame_count % FACE_RECOGNITION_FRAME_INTERVAL == 0:
        faces = face_recognizer.recognize_faces(frame)
        last_faces = faces
    else:
        faces = last_faces
    t_f1 = time.time()
    timing_stats['face'] += (t_f1 - t_f0) * 1000.0

    # ── rolling print every REPORT_INTERVAL frames ───────────────
    if frame_count % REPORT_INTERVAL == 0:
        n = REPORT_INTERVAL
        elapsed = time.time() - wall_start
        fps = frame_count / elapsed
        print(f"[Frame {frame_count:3d}] "
              f"new_frame_wait: {timing_stats['new_frame_wait']/n:6.1f}ms | "
              f"yolo: {timing_stats['yolo']/n:6.1f}ms | "
              f"face: {timing_stats['face']/n:6.1f}ms | "
              f"obj_memory: {timing_stats['obj_memory']/n:6.1f}ms | "
              f"FPS: {fps:.2f}")
        for k in timing_stats:
            timing_stats[k] = 0.0

cam.stop()
db.close()

total_elapsed = time.time() - wall_start
avg_fps = frame_count / total_elapsed
cam_fps = cam.frames_captured / total_elapsed
print(f"\nBenchmark complete: {frame_count} inference frames in {total_elapsed:.1f}s "
      f"= {avg_fps:.2f} inference-FPS")
print(f"Camera thread captured: {cam.frames_captured} raw frames "
      f"= {cam_fps:.2f} raw-FPS")
print("=" * 60)
