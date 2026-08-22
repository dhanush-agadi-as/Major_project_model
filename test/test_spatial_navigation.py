"""
Test Spatial Navigation Module
Tests zone detection, distance estimation, and guidance generation
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spatial_navigator import SpatialNavigator


def test_zone_detection():
    """Test that objects are correctly assigned to zones"""
    print("=" * 60)
    print("TEST 1: Zone Detection")
    print("=" * 60)
    
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    
    test_cases = [
        # (bbox, expected_zone)
        ((50, 50, 150, 150), 'top-left'),
        ((300, 50, 400, 150), 'top'),
        ((500, 50, 600, 150), 'top-right'),
        ((50, 200, 150, 300), 'left'),
        ((300, 200, 400, 300), 'center'),
        ((500, 200, 600, 300), 'right'),
        ((50, 400, 150, 470), 'bottom-left'),
        ((300, 400, 400, 470), 'bottom'),
        ((500, 400, 600, 470), 'bottom-right'),
    ]
    
    passed = 0
    for bbox, expected_zone in test_cases:
        actual_zone = nav.get_object_zone(bbox)
        status = "[OK]" if actual_zone == expected_zone else "[FAIL]"
        print(f"  {status} Bbox {bbox[:2]} -> {actual_zone} (expected: {expected_zone})")
        if actual_zone == expected_zone:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_distance_estimation():
    """Test distance estimation based on bbox size"""
    print("\n" + "=" * 60)
    print("TEST 2: Distance Estimation")
    print("=" * 60)
    
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    frame_area = 640 * 480
    
    test_cases = [
        # (bbox, expected_distance)
        ((0, 0, 640, 480), 'very-close'),  # 100% of frame
        ((100, 100, 540, 380), 'very-close'),  # ~50% of frame
        ((200, 150, 440, 330), 'medium'),  # ~15% of frame
        ((250, 200, 390, 280), 'far'),  # ~3.6% of frame
        ((300, 240, 340, 260), 'far'),  # <1% of frame
    ]
    
    passed = 0
    for bbox, expected_dist in test_cases:
        actual_dist = nav.estimate_distance(bbox)
        x1, y1, x2, y2 = bbox
        area_pct = ((x2-x1) * (y2-y1)) / frame_area * 100
        status = "[OK]" if actual_dist == expected_dist else "[FAIL]"
        print(f"  {status} {area_pct:.1f}% of frame -> {actual_dist} (expected: {expected_dist})")
        if actual_dist == expected_dist:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_obstacle_classification():
    """Test which objects are marked as obstacles"""
    print("\n" + "=" * 60)
    print("TEST 3: Obstacle Classification")
    print("=" * 60)
    
    nav = SpatialNavigator()
    
    obstacles = ['person', 'chair', 'couch', 'car', 'bicycle']
    non_obstacles = ['cell phone', 'book', 'bottle', 'clock']
    
    print("\n  Obstacles (should be True):")
    passed = 0
    total = 0
    for obj in obstacles:
        is_obstacle = nav.is_obstacle(obj)
        status = "[OK]" if is_obstacle else "[FAIL]"
        print(f"    {status} {obj}: {is_obstacle}")
        if is_obstacle:
            passed += 1
        total += 1
    
    print("\n  Non-obstacles (should be False):")
    for obj in non_obstacles:
        is_obstacle = nav.is_obstacle(obj)
        status = "[OK]" if not is_obstacle else "[FAIL]"
        print(f"    {status} {obj}: {is_obstacle}")
        if not is_obstacle:
            passed += 1
        total += 1
    
    print(f"\nPassed: {passed}/{total}")
    return passed == total


def test_clear_direction():
    """Test clear path detection"""
    print("\n" + "=" * 60)
    print("TEST 4: Clear Direction Detection")
    print("=" * 60)
    
    nav = SpatialNavigator()
    
    test_cases = [
        # (occupied_zones, expected_direction)
        ([], 'straight'),
        (['bottom'], 'left'),  # Object directly ahead, safer to go around
        (['bottom-left'], 'right'),
        (['bottom-right'], 'left'),
        (['bottom', 'bottom-left'], 'right'),
        (['bottom', 'bottom-right'], 'left'),
        (['bottom-left', 'bottom', 'bottom-right'], None),  # All blocked
    ]
    
    passed = 0
    for occupied, expected_dir in test_cases:
        actual_dir = nav.get_clear_direction(occupied)
        status = "[OK]" if actual_dir == expected_dir else "[FAIL]"
        print(f"  {status} Occupied: {occupied or 'none'}")
        print(f"      -> Direction: {actual_dir or 'STOP'} (expected: {expected_dir or 'STOP'})")
        if actual_dir == expected_dir:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_guidance_generation():
    """Test complete guidance message generation"""
    print("\n" + "=" * 60)
    print("TEST 5: Guidance Message Generation")
    print("=" * 60)
    
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    
    # Simulate detections
    detections = [
        {'bbox': (50, 400, 150, 470), 'class_name': 'chair', 'confidence': 0.9},
        {'bbox': (500, 200, 600, 300), 'class_name': 'person', 'confidence': 0.85},
        {'bbox': (300, 50, 400, 150), 'class_name': 'bottle', 'confidence': 0.7},  # Not obstacle
    ]
    
    guidance = nav.generate_guidance(detections)
    
    print(f"\n  Generated {len(guidance)} guidance messages:")
    for i, g in enumerate(guidance, 1):
        print(f"\n  {i}. {g['message']}")
        print(f"     Zone: {g['zone']}, Distance: {g['distance']}, Priority: {g['priority']}")
    
    # Validate
    has_chair_guidance = any('chair' in g['message'].lower() for g in guidance)
    has_person_guidance = any('person' in g['message'].lower() for g in guidance)
    has_direction = any(g['zone'] == 'navigation' for g in guidance)
    no_bottle = not any('bottle' in g['message'].lower() for g in guidance)
    
    print("\n  Validation:")
    print(f"    {'[OK]' if has_chair_guidance else '[FAIL]'} Chair mentioned")
    print(f"    {'[OK]' if has_person_guidance else '[FAIL]'} Person mentioned")
    print(f"    {'[OK]' if has_direction else '[FAIL]'} Direction provided")
    print(f"    {'[OK]' if no_bottle else '[FAIL]'} Non-obstacle (bottle) ignored")
    
    passed = has_chair_guidance and has_person_guidance and has_direction and no_bottle
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_realistic_scenario():
    """Test a realistic walking scenario"""
    print("\n" + "=" * 60)
    print("TEST 6: Realistic Walking Scenario")
    print("=" * 60)
    
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    
    print("\n  Scenario: Walking down a hallway with obstacles")
    print("  - Chair in bottom-left")
    print("  - Person approaching from right")
    print("  - Couch on left side")
    
    detections = [
        {'bbox': (50, 380, 200, 470), 'class_name': 'chair', 'confidence': 0.92},  # Bottom-left, large
        {'bbox': (450, 250, 580, 420), 'class_name': 'person', 'confidence': 0.88},  # Right, medium
        {'bbox': (20, 200, 150, 320), 'class_name': 'couch', 'confidence': 0.85},  # Left, medium
    ]
    
    guidance = nav.generate_guidance(detections)
    
    print(f"\n  Generated guidance (in priority order):")
    for i, g in enumerate(guidance, 1):
        priority_label = {1: 'EMERGENCY', 2: 'SOCIAL', 3: 'NAVIGATIONAL'}
        print(f"    {i}. [{priority_label.get(g['priority'], 'UNKNOWN')}] {g['message']}")
    
    # Expected: Path is blocked (bottom-left AND bottom-right both occupied)
    # Chair is in bottom-left, person is in bottom-right = path blocked
    navigation_guidance = [g for g in guidance if g['zone'] == 'navigation']
    if navigation_guidance:
        direction = navigation_guidance[0]['message']
        print(f"\n  Recommended action: {direction}")
        # With obstacles in both bottom corners, should stop
        success = 'stop' in direction.lower() or 'blocked' in direction.lower()
        print(f"  Explanation: Bottom-left (chair) + Bottom-right (person) = path blocked [OK]")
    else:
        success = False
    
    print(f"\n  Result: {'[OK] PASS' if success else '[FAIL] FAIL'}")
    return success


def run_all_tests():
    """Run all spatial navigation tests"""
    print("\n" + "=" * 60)
    print("SPATIAL NAVIGATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Zone Detection", test_zone_detection),
        ("Distance Estimation", test_distance_estimation),
        ("Obstacle Classification", test_obstacle_classification),
        ("Clear Direction", test_clear_direction),
        ("Guidance Generation", test_guidance_generation),
        ("Realistic Scenario", test_realistic_scenario),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[FAIL] Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed_count = sum(1 for _, p in results if p)
    for name, passed in results:
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + "=" * 60)
    print(f"Overall: {passed_count}/{len(results)} tests passed")
    print("=" * 60)
    
    return passed_count == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
