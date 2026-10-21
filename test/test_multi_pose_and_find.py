"""
Tests for multi-pose object enrollment, class gating, find mode tracking,
unparsed command fallback, and shutdown watchdog reliability.
"""

import time
import threading
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from visual_matcher import VisualMatcher, _rotate_image, _scale_image, _adjust_brightness
from object_memory import ObjectMemory
from find_guidance import FindGuide
from voice_command_system import VoiceCommandParser, VoiceCommandSystem
from wireless.laptop_server import LaptopMLServer


def test_enrollment_stores_at_least_6_embeddings():
    """Requirement 1 & 6: Multi-pose enrollment stores >= 6 embeddings in 'deep_embeddings'."""
    matcher = VisualMatcher(feature_type='DEEP')
    
    # Create synthetic test frame with distinct pattern
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[40:160, 40:160] = [120, 200, 80]
    frame[70:130, 70:130] = [240, 50, 180]
    bbox = (40, 40, 160, 160)
    
    # Extract features across frames (1 frame already generates 1 original + 8 augmentations = 9 embeddings)
    features = matcher.extract_multi_frame_features([frame], [bbox])
    
    assert features.get('feature_type') == 'deep'
    assert 'deep_embeddings' in features
    deep_embeddings = features['deep_embeddings']
    assert isinstance(deep_embeddings, list)
    assert len(deep_embeddings) >= 6, f"Expected >= 6 embeddings, got {len(deep_embeddings)}"
    
    # Backward compatibility: 'deep_embedding' must still exist as a mean vector
    assert 'deep_embedding' in features
    assert isinstance(features['deep_embedding'], list)
    assert len(features['deep_embedding']) == len(deep_embeddings[0])


def test_rotated_20_deg_crop_still_matches():
    """Requirement 1 & 6: A 20 degree rotated crop still matches stored multi-pose embeddings."""
    matcher = VisualMatcher(feature_type='DEEP')
    
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[50:150, 50:150] = [200, 100, 50]
    frame[75:125, 75:125] = [30, 220, 180]
    bbox = (50, 50, 150, 150)
    
    enrolled_features = matcher.extract_multi_frame_features([frame], [bbox])
    
    # Rotate the frame by 20 degrees
    crop = frame[50:150, 50:150]
    rotated_crop = _rotate_image(crop, 20.0)
    
    # Build query frame containing the rotated crop
    query_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    h, w = rotated_crop.shape[:2]
    query_frame[50:50+h, 50:50+w] = rotated_crop
    query_bbox = (50, 50, 50+w, 50+h)
    
    query_features = matcher.extract_features(query_frame, query_bbox)
    
    # Match using top-3 cosine similarities against multi-pose embeddings
    similarity, _ = matcher.match_features(query_features, enrolled_features)
    assert similarity >= 0.50, f"Expected rotated crop to match with similarity >= 0.50, got {similarity:.3f}"


def test_backward_compatibility_single_embedding():
    """Requirement 1: Backward compatibility: if only old mean embedding exists, use it."""
    matcher = VisualMatcher(feature_type='DEEP')
    
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:80, 20:80] = [150, 200, 50]
    bbox = (20, 20, 80, 80)
    
    q_feat = matcher.extract_features(frame, bbox)
    old_enrolled = {
        'feature_type': 'deep',
        'deep_embedding': list(q_feat['deep_embedding'])
        # intentionally no 'deep_embeddings' key
    }
    
    sim, _ = matcher.match_features(q_feat, old_enrolled)
    assert sim >= 0.95, f"Expected ~1.0 for exact self-match with old schema, got {sim}"


def test_one_object_class_gating_high_tier():
    """Requirement 2 & 6: One-object class gating treats similarity >= 0.35 as high tier."""
    db_mock = MagicMock()
    # Exactly one cell phone enrolled
    db_mock.get_remembered_objects_by_class.return_value = [
        (1, "my phone", "cell phone", {'feature_type': 'deep'}, None, "2026-01-01")
    ]
    
    mem = ObjectMemory(db=db_mock, matcher=VisualMatcher(feature_type='DEEP'))
    mem.matcher.extract_features = MagicMock(return_value={'feature_type': 'deep', 'deep_embedding': [0.1]*512})
    
    # Scenario A: Exactly one object enrolled, confidence = 0.38 (>= 0.35)
    mem.matcher.find_best_match = MagicMock(return_value=(1, "my phone", 0.38))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    res = mem.recognize_object(frame, (10, 10, 50, 50), "cell phone")
    assert res is not None
    custom_name, conf, tier = res
    assert custom_name == "my phone"
    assert conf == 0.38
    assert tier == "high", f"Expected high tier for single object with sim >= 0.35, got {tier}"
    
    # Scenario B: Multiple objects of same class enrolled -> existing thresholds (0.75 high, 0.50 ambiguous)
    db_mock.get_remembered_objects_by_class.return_value = [
        (1, "work phone", "cell phone", {'feature_type': 'deep'}, None, "2026-01-01"),
        (2, "home phone", "cell phone", {'feature_type': 'deep'}, None, "2026-01-01")
    ]
    mem._candidates_cache.clear()
    
    # Confidence 0.60: ambiguous
    mem.matcher.find_best_match = MagicMock(return_value=(1, "work phone", 0.60))
    res2 = mem.recognize_object(frame, (10, 10, 50, 50), "cell phone")
    assert res2 is not None
    _, _, tier2 = res2
    assert tier2 == "ambiguous"


def test_find_verb_mishearing_driven_by_salt_phone():
    """Requirement 4 & 6: 'driven by salt phone' with cell phone remembered starts a find."""
    parser = VoiceCommandParser()
    
    # Test verb mishearing in parser
    parsed = parser.parse("hey vision driven by salt phone")
    assert parsed['intent'] == 'find'
    assert parsed['parameters']['object_name'] == 'salt phone'
    
    # Test full server execution with cell phone remembered
    server = LaptopMLServer.__new__(LaptopMLServer)
    server.object_memory = MagicMock()
    server.object_memory.list_remembered_objects.return_value = [("cell phone", "cell phone")]
    server.object_memory.get_object_location.return_value = None
    server.object_memory.get_object_aliases.return_value = ["phone", "mobile"]
    server.item_tracker = None
    server.find_guide = MagicMock()
    server.audio = MagicMock()
    server._speak_reply = MagicMock()
    server._match_enrolled_face = MagicMock(return_value=None)
    
    server.voice_system = VoiceCommandSystem()
    server._register_voice_handlers()
    
    handler = server.voice_system.command_callbacks['find']
    handler(parsed['parameters'])
    
    assert server.find_guide.start.called
    called_name = server.find_guide.start.call_args[0][0]
    assert called_name == "cell phone"
    assert any("cell phone" in str(arg) for arg in server._speak_reply.call_args[0])


def test_unparsed_command_fallback_looking_for_object():
    """Requirement 4: Unparsed text with no verb fuzzy-matches remembered object -> 'Looking for your <object>'."""
    server = LaptopMLServer.__new__(LaptopMLServer)
    server.object_memory = MagicMock()
    server.object_memory.list_remembered_objects.return_value = [("cell phone", "cell phone")]
    server.object_memory.get_object_aliases.return_value = ["phone", "mobile"]
    server.db = MagicMock()
    server.db.get_all_faces.return_value = []
    server.find_guide = MagicMock()
    server.audio = MagicMock()
    server._speak_reply = MagicMock()
    server._match_enrolled_face = MagicMock(return_value=None)
    
    server.voice_system = VoiceCommandSystem()
    server._register_voice_handlers()
    
    # "salt phone" has no verb but fuzzy-matches "cell phone" (ratio >= 0.6)
    parsed = server.voice_system.parser.parse("hey vision salt phone")
    assert parsed['intent'] == 'find'
    assert parsed['parameters'].get('fallback_phrase') is True
    
    handler = server.voice_system.command_callbacks['find']
    handler(parsed['parameters'])
    
    assert server.find_guide.start.called
    assert any("Looking for your cell phone" in str(arg) for arg in server._speak_reply.call_args[0])


def test_find_mode_iou_tracker_and_1_5s_hold():
    """Requirement 3: Tracker uses IoU to keep bbox identity and holds for 1.5s before visible=False."""
    fg = FindGuide(speak=lambda msg: None, beep=lambda f, ms: None, frame_w=640, frame_h=480)
    fg.start("cell phone", "")
    
    # Frame 1: Target detected at [100, 100, 200, 200]
    cand1 = {'custom_name': 'cell phone', 'class': 'cell phone', 'confidence': 0.9, 'bbox': (100, 100, 200, 200), 'tier': 'high'}
    cand2 = {'custom_name': 'cell phone', 'class': 'cell phone', 'confidence': 0.95, 'bbox': (400, 400, 500, 500), 'tier': 'high'}
    
    res1 = fg.update([cand1], ultrasonic_cm=None)
    assert res1.get('visible') is True
    
    # Frame 2: cand1 moves slightly to (105, 105, 205, 205). Even though cand2 has higher confidence, IoU locks cand1!
    cand1_moved = {'custom_name': 'cell phone', 'class': 'cell phone', 'confidence': 0.88, 'bbox': (105, 105, 205, 205), 'tier': 'high'}
    res2 = fg.update([cand1_moved, cand2], ultrasonic_cm=None)
    assert res2.get('visible') is True
    assert fg._last_known_detection['bbox'] == (105, 105, 205, 205)
    
    # Frame 3: Target temporarily drops (e.g. head tilt) -> held for 1.5s
    res3 = fg.update([], ultrasonic_cm=None)
    assert res3.get('visible') is True, "Target should be held visible within 1.5s hysteresis"
    
    # Simulate time passing past 1.5s
    fg._last_detected_time = time.monotonic() - 1.6
    res4 = fg.update([], ultrasonic_cm=None)
    assert res4.get('visible') is False, "Target should become invisible after 1.5s"


def test_shutdown_watchdog_calls_os_exit():
    """Requirement 5: ShutdownWatchdog calls os._exit(0) when a non-daemon thread stays alive."""
    server = LaptopMLServer.__new__(LaptopMLServer)
    
    # Event to keep thread alive during test
    stop_event = threading.Event()
    
    def _non_daemon_worker():
        stop_event.wait(timeout=2.0)
    
    # Create an alive non-daemon thread
    worker_th = threading.Thread(target=_non_daemon_worker, daemon=False, name="DummyNonDaemonThread")
    worker_th.start()
    
    try:
        with patch('os._exit') as mock_exit:
            watchdog_th = server._start_shutdown_watchdog(delay=0.05)
            # Wait for watchdog to trigger
            watchdog_th.join(timeout=0.5)
            
            mock_exit.assert_called_once_with(0)
    finally:
        stop_event.set()
        worker_th.join(timeout=1.0)
