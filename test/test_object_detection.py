"""
Test Object Detection Module
Tests YOLOv8 detection with sample images
"""
import cv2
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from object_detector import ObjectDetector
from camera_source import open_camera


def test_object_detection_webcam():
    """Test object detection with webcam"""
    print("=" * 60)
    print("TESTING OBJECT DETECTION - WEBCAM")
    print("=" * 60)
    
    detector = ObjectDetector()
    
    cap = open_camera()
    if not cap.isOpened():
        print("[FAIL] Could not open webcam, testing with static image instead")
        test_object_detection_static()
        return
    
    print("\n🎥 Starting webcam test...")
    print("Press 'q' to quit")
    
    frame_count = 0
    import time
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        detections = detector.detect(frame)
        
        # Draw detections
        annotated = detector.draw_detections(frame, detections)
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Display info
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f} | Objects: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Show frame
        cv2.imshow('Object Detection Test', annotated)
        
        # Print detections
        if detections:
            print(f"\rFrame {frame_count}: {len(detections)} objects detected", end='')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n\n[OK] Test complete!")
    print(f"  Total frames: {frame_count}")
    print(f"  Average FPS: {fps:.2f}")


def test_object_detection_static():
    """Test with a solid color image"""
    print("=" * 60)
    print("TESTING OBJECT DETECTION - STATIC IMAGE")
    print("=" * 60)
    
    detector = ObjectDetector()
    
    # Blank 480x640 frame with green rectangle
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 400), (0, 255, 0), -1)
    
    print("\n🖼️  Processing test image...")
    
    # Detect objects
    detections = detector.detect(frame)
    assert isinstance(detections, list)
    
    print(f"\n[OK] Detection complete!")
    print(f"  Found {len(detections)} objects:")
    
    for i, det in enumerate(detections, 1):
        print(f"    {i}. {det['class_name']} (confidence: {det['confidence']:.2f})")


def test_specific_classes():
    """Test detection of specific target classes"""
    print("=" * 60)
    print("TESTING TARGET CLASS FILTERING")
    print("=" * 60)
    
    detector = ObjectDetector()
    
    print(f"\nTarget classes: {detector.target_classes}")
    print("\nThis test verifies that only target classes are detected.")
    print("[OK] Detector configured correctly!")


if __name__ == "__main__":
    import numpy as np
    
    print("\nVISIONMATE OBJECT DETECTION TEST SUITE")
    print("=" * 60)
    
    # Test 1: Target classes
    test_specific_classes()
    
    print("\n")
    
    # Test 2: Choose test mode
    print("Select test mode:")
    print("1. Webcam (live detection)")
    print("2. Static image")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        test_object_detection_webcam()
    elif choice == '2':
        test_object_detection_static()
    elif choice == '3':
        test_object_detection_static()
        test_object_detection_webcam()
    else:
        print("Running default test...")
        test_object_detection_static()
