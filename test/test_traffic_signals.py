"""
Test Traffic Signal Detection
Tests HSV-based red/green light detection
"""
import cv2
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_signal_detector import TrafficSignalDetector


def create_test_signal(color='red', size=(200, 200)):
    """Create a synthetic traffic signal for testing"""
    # Create black background
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Draw signal light
    center = (200, 200)
    radius = 50
    
    if color == 'red':
        cv2.circle(img, center, radius, (0, 0, 255), -1)
    elif color == 'green':
        cv2.circle(img, center, radius, (0, 255, 0), -1)
    
    return img


def test_synthetic_signals():
    """Test with synthetic red and green signals"""
    print("=" * 60)
    print("TESTING TRAFFIC SIGNAL DETECTION - SYNTHETIC")
    print("=" * 60)
    
    detector = TrafficSignalDetector()
    
    # Test red signal
    print("\n1. Testing RED signal...")
    red_img = create_test_signal('red')
    red_detection = detector.detect_signal(red_img)
    
    if red_detection:
        print(f"   [OK] Detected: {red_detection['signal_type']}")
        print(f"   Confidence: {red_detection['confidence']:.0f}")
        annotated_red = detector.draw_signal(red_img, red_detection)
    else:
        print("   [FAIL] No signal detected")
        annotated_red = red_img
    
    # Test green signal
    print("\n2. Testing GREEN signal...")
    green_img = create_test_signal('green')
    green_detection = detector.detect_signal(green_img)
    
    if green_detection:
        print(f"   [OK] Detected: {green_detection['signal_type']}")
        print(f"   Confidence: {green_detection['confidence']:.0f}")
        annotated_green = detector.draw_signal(green_img, green_detection)
    else:
        print("   [FAIL] No signal detected")
        annotated_green = green_img
    
    # Display results
    combined = np.hstack([annotated_red, annotated_green])
    cv2.imshow('Traffic Signal Test: RED | GREEN', combined)
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n[OK] Synthetic signal test complete!")


def test_webcam_signals():
    """Test traffic signal detection with webcam"""
    print("=" * 60)
    print("TESTING TRAFFIC SIGNAL DETECTION - WEBCAM")
    print("=" * 60)
    
    detector = TrafficSignalDetector()
    
    print("\nOpening webcam...")
    print("Show red or green objects to the camera")
    print("Press 'q' to quit, 'm' to see color masks")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Could not open webcam")
        return
    
    show_masks = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect signal
        detection = detector.detect_signal(frame)
        
        if detection:
            # Draw detection
            display = detector.draw_signal(frame, detection)
            print(f"\r🚦 {detection['signal_type']} light detected!", end='')
        else:
            display = frame.copy()
            cv2.putText(
                display,
                "No signal detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Show instructions
        cv2.putText(
            display,
            "Press Q to quit | M for masks",
            (10, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        cv2.imshow('Traffic Signal Detection', display)
        
        # Show color masks if requested
        if show_masks:
            masks = detector.get_color_masks(frame)
            cv2.imshow('Red Mask', masks['red'])
            cv2.imshow('Green Mask', masks['green'])
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_masks = not show_masks
            if not show_masks:
                cv2.destroyWindow('Red Mask')
                cv2.destroyWindow('Green Mask')
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n\n[OK] Webcam test complete!")


def test_hsv_ranges():
    """Test and visualize HSV color ranges"""
    print("=" * 60)
    print("TESTING HSV COLOR RANGES")
    print("=" * 60)
    
    detector = TrafficSignalDetector()
    
    print("\nConfigured HSV ranges:")
    print(f"  RED:   {detector.red_lower} to {detector.red_upper}")
    print(f"  GREEN: {detector.green_lower} to {detector.green_upper}")
    
    # Create color samples
    red_sample = np.zeros((100, 200, 3), dtype=np.uint8)
    red_sample[:, :] = (0, 0, 255)
    
    green_sample = np.zeros((100, 200, 3), dtype=np.uint8)
    green_sample[:, :] = (0, 255, 0)
    
    # Test detection on pure colors
    red_det = detector.detect_signal(red_sample)
    green_det = detector.detect_signal(green_sample)
    
    print(f"\n  Pure RED detection: {'[OK]' if red_det and red_det['signal_type'] == 'RED' else '[FAIL]'}")
    print(f"  Pure GREEN detection: {'[OK]' if green_det and green_det['signal_type'] == 'GREEN' else '[FAIL]'}")
    
    print("\n[OK] HSV range test complete!")


if __name__ == "__main__":
    print("\nVISIONMATE TRAFFIC SIGNAL TEST SUITE")
    print("=" * 60)
    
    print("\nSelect test mode:")
    print("1. Synthetic signals (recommended for first test)")
    print("2. Webcam detection")
    print("3. HSV range verification")
    print("4. All tests")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        test_synthetic_signals()
    elif choice == '2':
        test_webcam_signals()
    elif choice == '3':
        test_hsv_ranges()
    elif choice == '4':
        test_hsv_ranges()
        print("\n")
        test_synthetic_signals()
        print("\n")
        test_webcam_signals()
    else:
        print("Running synthetic test...")
        test_synthetic_signals()
