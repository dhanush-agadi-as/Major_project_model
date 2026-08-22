"""
Test script for Object Memory functionality
Tests enrollment, recognition, and sighting tracking
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from object_memory import ObjectMemory
import numpy as np
import cv2

def test_object_memory():
    """Test object memory operations"""
    print("=" * 60)
    print("Testing Object Memory System")
    print("=" * 60)
    
    memory = ObjectMemory(); memory.delete_object("house keys"); memory.delete_object("test object"); memory.delete_object("work badge")
    print("[OK] Object memory initialized\n")
    
    # Test 1: Start enrollment
    print("Test 1: Start enrollment for 'house keys'")
    try:
        memory.start_enrollment("house keys", "keys")
        progress = memory.get_enrollment_progress()
        print(f"  [OK] Enrollment started")
        print(f"    Active: {progress['active']}")
        print(f"    Object: {progress['custom_name']}")
        print(f"    Target: {progress['frames_needed']} frames\n")
    except Exception as e:
        print(f"  [FAIL] Error: {e}\n")
        return False
    
    # Test 2: Add enrollment frames with actual visual features
    print("Test 2: Adding enrollment frames with test images")
    for i in range(5):
        # Create a test image with rich texture for feature detection
        img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        # Add multiple geometric patterns with high contrast
        center_x, center_y = 320 + i*5, 240 + i*5
        
        # Draw key-like shape with lots of edges
        cv2.rectangle(img, (center_x-40, center_y-25), (center_x+40, center_y+25), (210, 160, 90), -1)
        cv2.rectangle(img, (center_x-30, center_y-15), (center_x+30, center_y+15), (180, 130, 60), 2)
        
        # Add circles and lines for keypoints
        for j in range(5):
            cv2.circle(img, (center_x-30+j*15, center_y-10), 3, (255, 255, 255), -1)
            cv2.circle(img, (center_x-30+j*15, center_y+10), 3, (50, 50, 50), -1)
        
        # Add teeth-like pattern (key teeth)
        for k in range(8):
            x_pos = center_x - 35 + k*10
            cv2.rectangle(img, (x_pos, center_y+15), (x_pos+4, center_y+25), (255, 255, 255), -1)
        
        # Add text for additional features
        cv2.putText(img, f"KEY{i}", (center_x-25, center_y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        bbox = (center_x-40, center_y-25, center_x+40, center_y+25)
        num = memory.add_enrollment_frame(img, bbox)
        print(f"  [OK] Frame {num}/{5} captured")
    print()
    
    # Test 3: Finish enrollment
    print("Test 3: Finishing enrollment")
    success = memory.finish_enrollment()
    if success:
        print("  [OK] Object enrolled successfully\n")
    else:
        print("  [FAIL] Enrollment failed\n")
        return False
    
    # Test 4: List remembered objects
    print("Test 4: Listing remembered objects")
    objects = memory.list_remembered_objects()
    print(f"  [OK] Found {len(objects)} objects:")
    for name, yolo_class in objects:
        print(f"    - '{name}' (class: {yolo_class})")
    print()
    
    # Test 5: Try to recognize similar object
    print("Test 5: Testing object recognition")
    test_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    
    # Draw similar key-like pattern
    cv2.rectangle(test_img, (270, 210), (350, 260), (210, 160, 90), -1)
    cv2.rectangle(test_img, (280, 220), (340, 250), (180, 130, 60), 2)
    
    for j in range(5):
        cv2.circle(test_img, (280+j*12, 225), 3, (255, 255, 255), -1)
        cv2.circle(test_img, (280+j*12, 245), 3, (50, 50, 50), -1)
    
    for k in range(8):
        x_pos = 275 + k*9
        cv2.rectangle(test_img, (x_pos, 250), (x_pos+4, 260), (255, 255, 255), -1)
    
    cv2.putText(test_img, "KEY2", (290, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    result = memory.recognize_object(test_img, (270, 210, 350, 260), "keys")
    if result:
        custom_name, confidence, tier = result
        print(f"  [OK] Recognized: '{custom_name}' (confidence: {confidence:.2%})")
    else:
        print(f"  ℹ Not recognized (below threshold or no match)")
    print()
    
    # Test 6: Update sighting
    print("Test 6: Recording object sighting")
    success = memory.update_object_sighting(
        custom_name="house keys",
        location="kitchen counter",
        gps_lat=37.7749,
        gps_lon=-122.4194,
        nearby_objects=["bottle", "phone", "laptop"],
        confidence=0.85
    )
    if success:
        print("  [OK] Sighting recorded\n")
    else:
        print("  [FAIL] Failed to record sighting\n")
        return False
    
    # Test 7: Query location
    print("Test 7: Querying object location")
    location = memory.get_object_location("house keys")
    if location:
        print(f"  [OK] Last seen:")
        print(f"    Location: {location['location']}")
        print(f"    GPS: ({location['gps_lat']}, {location['gps_lon']})")
        print(f"    Nearby: {location['context'].get('nearby_objects', [])}")
        print(f"    Confidence: {location['confidence']:.2%}")
        print(f"    Time: {location['timestamp']}\n")
    else:
        print("  [FAIL] No location found\n")
        return False
    
    # Test 8: Update sighting again (test update logic)
    print("Test 8: Updating sighting location")
    memory.update_object_sighting(
        custom_name="house keys",
        location="bedroom nightstand",
        gps_lat=37.7750,
        gps_lon=-122.4195,
        nearby_objects=["lamp", "book"],
        confidence=0.92
    )
    location2 = memory.get_object_location("house keys")
    print(f"  [OK] New location: {location2['location']}\n")
    
    # Test 9: Enroll second object
    print("Test 9: Enrolling second object 'work badge'")
    memory.start_enrollment("work badge", "person")  # Using 'person' as placeholder
    for i in range(3):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add some texture
        cv2.putText(img, "BADGE", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
        memory.add_enrollment_frame(img, (200, 200, 400, 280))
    memory.finish_enrollment()
    objects = memory.list_remembered_objects()
    print(f"  [OK] Total objects: {len(objects)}")
    for name, yolo_class in objects:
        print(f"    - {name} ({yolo_class})")
    print()
    
    # Test 10: Delete objects
    print("Test 10: Cleaning up - deleting objects")
    memory.delete_object("house keys")
    memory.delete_object("work badge")
    objects = memory.list_remembered_objects()
    print(f"  [OK] Remaining objects: {len(objects)}\n")
    
    # Test 11: Try duplicate enrollment (should fail)
    print("Test 11: Testing duplicate enrollment prevention")
    memory.start_enrollment("test object", "bottle")
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    memory.add_enrollment_frame(img, (100, 100, 200, 200))
    memory.finish_enrollment()
    
    try:
        memory.start_enrollment("test object", "bottle")  # Should raise error
        print("  [FAIL] Duplicate allowed (should have failed)\n")
        return False
    except ValueError as e:
        print(f"  [OK] Duplicate prevented: {e}\n")
    
    # Cleanup
    memory.delete_object("test object")
    
    print("=" * 60)
    print("All tests passed! [OK]")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = test_object_memory()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
