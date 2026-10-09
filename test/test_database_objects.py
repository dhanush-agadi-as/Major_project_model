"""
Test script for remembered objects database functionality
Tests the new tables and methods added for the "Remember Objects" feature
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import VisionMateDB
import numpy as np

def test_remembered_objects():
    """Test remembered objects CRUD operations"""
    print("=" * 60)
    print("Testing Remembered Objects Database")
    print("=" * 60)
    
    # Initialize database
    db = VisionMateDB()
    print("[OK] Database initialized\n")
    
    # Test 1: Add a remembered object
    print("Test 1: Enrolling 'house keys'")
    visual_features = {
        'keypoints': [[100, 150], [200, 250], [300, 350]],
        'descriptors': [0.1, 0.2, 0.3, 0.4, 0.5],
        'color_histogram': [10, 20, 30, 40, 50]
    }
    
    obj_id = db.add_remembered_object(
        custom_name="house keys",
        yolo_class="keys",
        visual_features=visual_features,
        enrollment_frames=5
    )
    
    if obj_id > 0:
        print(f"  [OK] Object enrolled with ID: {obj_id}\n")
    else:
        print("  [FAIL] Failed to enroll (duplicate name?)\n")
        return False
    
    # Test 2: Retrieve the object
    print("Test 2: Retrieving 'house keys'")
    obj = db.get_remembered_object("house keys")
    if obj:
        obj_id, name, yolo_class, features, frames = obj
        print(f"  [OK] Found: {name}")
        print(f"    YOLO Class: {yolo_class}")
        print(f"    Enrollment Frames: {frames}")
        print(f"    Features: {list(features.keys())}\n")
    else:
        print("  [FAIL] Object not found\n")
        return False
    
    # Test 3: Add another object
    print("Test 3: Enrolling 'work wallet'")
    obj_id2 = db.add_remembered_object(
        custom_name="work wallet",
        yolo_class="wallet",
        visual_features={'desc': [1, 2, 3]},
        enrollment_frames=3
    )
    print(f"  [OK] Enrolled with ID: {obj_id2}\n")
    
    # Test 4: Get all remembered objects
    print("Test 4: Listing all remembered objects")
    all_objects = db.get_all_remembered_objects()
    print(f"  [OK] Found {len(all_objects)} objects:")
    for oid, name, yolo_class in all_objects:
        print(f"    - {name} ({yolo_class}) [ID: {oid}]")
    print()
    
    # Test 5: Get objects by YOLO class
    print("Test 5: Getting objects by class 'keys'")
    keys_objects = db.get_remembered_objects_by_class("keys")
    print(f"  [OK] Found {len(keys_objects)} 'keys' objects:")
    for oid, name, features in keys_objects:
        print(f"    - {name} [ID: {oid}]")
    print()
    
    # Test 6: Add a sighting
    print("Test 6: Recording sighting of 'house keys'")
    db.add_object_sighting(
        object_id=obj_id,
        location="kitchen table",
        gps_lat=37.7749,
        gps_lon=-122.4194,
        context={'nearby_objects': ['bottle', 'phone'], 'room': 'kitchen'},
        confidence=0.92
    )
    print("  [OK] Sighting recorded\n")
    
    # Test 7: Retrieve sighting
    print("Test 7: Retrieving latest sighting")
    sighting = db.get_latest_sighting(obj_id)
    if sighting:
        print(f"  [OK] Last seen:")
        print(f"    Location: {sighting['location']}")
        print(f"    GPS: ({sighting['gps_lat']}, {sighting['gps_lon']})")
        print(f"    Confidence: {sighting['confidence']:.2f}")
        print(f"    Context: {sighting['context']}")
        print(f"    Time: {sighting['timestamp']}\n")
    else:
        print("  [FAIL] No sighting found\n")
        return False
    
    # Test 8: Update sighting (test UNIQUE constraint)
    print("Test 8: Updating sighting location")
    db.add_object_sighting(
        object_id=obj_id,
        location="bedroom dresser",
        gps_lat=37.7750,
        gps_lon=-122.4195,
        context={'nearby_objects': ['lamp', 'book'], 'room': 'bedroom'},
        confidence=0.88
    )
    sighting2 = db.get_sighting_by_name("house keys")
    print(f"  [OK] Updated location: {sighting2['location']}\n")
    
    # Test 9: Delete object (should cascade to sightings)
    print("Test 9: Deleting 'work wallet'")
    deleted = db.delete_remembered_object("work wallet")
    if deleted:
        print("  [OK] Object deleted\n")
    else:
        print("  [FAIL] Failed to delete\n")
    
    # Test 10: Verify deletion
    print("Test 10: Verifying deletion")
    all_objects = db.get_all_remembered_objects()
    print(f"  [OK] Remaining objects: {len(all_objects)}")
    for oid, name, yolo_class in all_objects:
        print(f"    - {name} ({yolo_class})")
    print()
    
    # Cleanup
    db.delete_remembered_object("house keys")
    db.close()
    
    print("=" * 60)
    print("All tests passed! [OK]")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = test_remembered_objects()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
