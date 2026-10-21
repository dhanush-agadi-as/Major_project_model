"""
Test Item Tracking Module
Tests IoU-based passive item tracking (Find My Keys feature)
"""
import cv2
import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from item_tracker import ItemTracker
from object_detector import ObjectDetector
from database import VisionMateDB
from camera_source import open_camera


def test_iou_calculation():
    """Test IoU calculation with known boxes"""
    print("=" * 60)
    print("TESTING IoU CALCULATION")
    print("=" * 60)
    
    tracker = ItemTracker()
    
    # Test cases
    test_cases = [
        {
            'name': 'Overlapping boxes',
            'box1': [100, 100, 200, 200],
            'box2': [150, 150, 250, 250],
            'expected': 0.14  # Approximate
        },
        {
            'name': 'Identical boxes',
            'box1': [100, 100, 200, 200],
            'box2': [100, 100, 200, 200],
            'expected': 1.0
        },
        {
            'name': 'Non-overlapping boxes',
            'box1': [100, 100, 200, 200],
            'box2': [300, 300, 400, 400],
            'expected': 0.0
        },
        {
            'name': 'Box inside another',
            'box1': [50, 50, 300, 300],
            'box2': [100, 100, 200, 200],
            'expected': 0.16  # Approximate
        }
    ]
    
    print("\nRunning IoU tests...")
    for test in test_cases:
        iou = tracker.calculate_iou(test['box1'], test['box2'])
        status = '[OK]' if abs(iou - test['expected']) < 0.1 else '[FAIL]'
        print(f"  {status} {test['name']}: IoU = {iou:.3f} (expected ~{test['expected']:.2f})")
    
    print("\n[OK] IoU calculation test complete!")


def test_item_tracking_simulation():
    """Test item tracking with simulated detections"""
    print("=" * 60)
    print("TESTING ITEM TRACKING - SIMULATION")
    print("=" * 60)
    
    db = VisionMateDB()
    tracker = ItemTracker(db)
    
    print(f"\nTracking timeout: {tracker.timeout} seconds")
    print("Simulating 'keys' on 'chair' detection...")
    
    # Simulate detections over time
    for i in range(5):
        print(f"\nFrame {i+1}:")
        
        # Simulate detections: keys inside chair
        detections = [
            {
                'class_name': 'keys',
                'bbox': [120, 120, 180, 180],
                'confidence': 0.9
            },
            {
                'class_name': 'chair',
                'bbox': [100, 100, 300, 300],
                'confidence': 0.95
            }
        ]
        
        # Update tracking
        confirmed = tracker.update_tracking(detections)
        
        if confirmed:
            print(f"  [OK] Confirmed location: {confirmed}")
        else:
            print(f"  ⊙ Still tracking...")
        
        time.sleep(1.5)  # Simulate frame delay
    
    # Query location
    location = tracker.query_item_location('keys')
    print(f"\n[OK] Final query result: keys are on {location}")
    
    # Show all tracked items
    all_items = tracker.get_all_tracked_items()
    print(f"\n[OK] All tracked items: {all_items}")
    
    db.close()


def test_item_tracking_live():
    """Test item tracking with live webcam + object detection"""
    print("=" * 60)
    print("TESTING ITEM TRACKING - LIVE")
    print("=" * 60)
    
    db = VisionMateDB()
    detector = ObjectDetector()
    tracker = ItemTracker(db)
    
    print("\nOpening webcam...")
    print("Place small objects on/near larger objects")
    print("Press 'q' to quit, 's' to query item location")
    
    cap = open_camera()
    if not cap.isOpened():
        print("[FAIL] Could not open webcam")
        db.close()
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        detections = detector.detect(frame)
        
        # Update tracking
        confirmed = tracker.update_tracking(detections)
        
        # Draw detections
        display = detector.draw_detections(frame, detections)
        
        # Show tracking state
        tracking_info = f"Tracking: {len(tracker.tracking_state)} pairs"
        cv2.putText(
            display,
            tracking_info,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Show confirmed locations
        if confirmed:
            for item in confirmed:
                msg = f"Confirmed: {item['item']} on {item['location']}"
                print(msg)
                cv2.putText(
                    display,
                    msg,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
        
        cv2.putText(
            display,
            "Press Q to quit | S to query item",
            (10, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        cv2.imshow('Item Tracking Test', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            item = input("\nEnter item to query (e.g., 'keys'): ").strip()
            location = tracker.query_item_location(item)
            if location:
                print(f"[OK] {item} are on {location}")
            else:
                print(f"[FAIL] Haven't seen {item}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n[OK] Live tracking test complete!")
    db.close()


def test_database_item_storage():
    """Test item location database operations"""
    print("=" * 60)
    print("TESTING ITEM LOCATION DATABASE")
    print("=" * 60)
    
    db = VisionMateDB()
    
    # Add test items
    print("\nAdding test items...")
    db.update_item_location('keys', 'chair')
    db.update_item_location('bottle', 'sofa')
    print("  [OK] Added 'keys' on 'chair'")
    print("  [OK] Added 'bottle' on 'sofa'")
    
    # Query items
    print("\nQuerying items...")
    keys_loc = db.get_item_location('keys')
    bottle_loc = db.get_item_location('bottle')
    print(f"  Keys location: {keys_loc}")
    print(f"  Bottle location: {bottle_loc}")
    
    # Get all items
    all_items = db.get_all_item_locations()
    print(f"\nAll stored items: {len(all_items)}")
    for item, loc, timestamp in all_items:
        print(f"  - {item} on {loc} (at {timestamp})")
    
    print("\n[OK] Database test complete!")
    db.close()


if __name__ == "__main__":
    print("\nVISIONMATE ITEM TRACKING TEST SUITE")
    print("=" * 60)
    
    print("\nSelect test mode:")
    print("1. IoU calculation test")
    print("2. Tracking simulation (3 seconds)")
    print("3. Live tracking with webcam")
    print("4. Database operations")
    print("5. All tests")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        test_iou_calculation()
    elif choice == '2':
        test_item_tracking_simulation()
    elif choice == '3':
        test_item_tracking_live()
    elif choice == '4':
        test_database_item_storage()
    elif choice == '5':
        test_iou_calculation()
        print("\n")
        test_database_item_storage()
        print("\n")
        test_item_tracking_simulation()
        print("\n")
        test_item_tracking_live()
    else:
        print("Running IoU test...")
        test_iou_calculation()
