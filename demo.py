"""
Simple demo script for quick testing
Runs a basic demonstration of all VisionMate features
"""
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from object_detector import ObjectDetector
from database import VisionMateDB

print("=" * 60)
print("VISIONMATE QUICK DEMO")
print("=" * 60)

# Initialize
print("\nInitializing AI modules...")
db = VisionMateDB()
detector = ObjectDetector()
print("[OK] Ready!")

# Test with webcam
print("\nOpening webcam (press 'q' to quit)...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[FAIL] No webcam found. Using static test instead.")
    print("\nCreating test image...")
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "VisionMate Demo", (150, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    print("Running object detection...")
    detections = detector.detect(frame)
    print(f"[OK] Detected {len(detections)} objects")
    
    annotated = detector.draw_detections(frame, detections)
    cv2.imshow('VisionMate Demo', annotated)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect objects
        detections = detector.detect(frame)
        
        # Draw results
        display = detector.draw_detections(frame, detections)
        
        # Show info
        cv2.putText(display, f"Objects: {len(detections)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Press Q to quit", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('VisionMate Demo', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[OK] Processed {frame_count} frames")

db.close()
print("\n[OK] Demo complete!")
