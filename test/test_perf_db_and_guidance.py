"""
Performance, Threading, Embedding Cache, and Guidance Flicker Tests

Tests for:
1. Concurrent find + frame matching raises no cursor errors (thread-safe DB access).
2. Embedding cache refreshes after remember/forget in RAM.
3. Matching 100 embeddings takes under 5 ms (vectorized matrix cosine).
4. Person flickering for 1 frame does not re-trigger 'Person ahead', announces only after >= 5 s absence,
   and suppressed during active find unless distance is under 50 cm.
"""

import sys
import os
import time
import threading
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import VisionMateDB
from object_memory import ObjectMemory
from visual_matcher import VisualMatcher
from spatial_navigator import SpatialNavigator
import config


def test_concurrent_find_and_frame_matching_no_cursor_errors(tmp_path):
    """Test that concurrent DB queries (voice handler, frame thread, find mode) raise no cursor errors."""
    db_path = str(tmp_path / "concurrent_test.db")
    db = VisionMateDB(db_path=db_path)
    mem = ObjectMemory(db=db, matcher=VisualMatcher(feature_type='DEEP'))

    # Seed with objects
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mem.start_enrollment("test phone", "cell phone")
    mem.add_enrollment_frame(img, (10, 10, 90, 90))
    mem.finish_enrollment()

    mem.start_enrollment("water bottle", "bottle")
    mem.add_enrollment_frame(img, (10, 10, 90, 90))
    mem.finish_enrollment()

    errors = []
    stop_event = threading.Event()

    def frame_thread_worker():
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(50):
            if stop_event.is_set():
                break
            try:
                # Frame recognition (using in-RAM cache)
                mem.recognize_object(dummy_frame, (10, 10, 90, 90), "cell phone")
                # Sighting update (DB write)
                mem.update_object_sighting("test phone", confidence=0.9)
                time.sleep(0.002)
            except Exception as e:
                errors.append(f"frame_thread: {e}")

    def voice_find_worker():
        for _ in range(50):
            if stop_event.is_set():
                break
            try:
                # Find queries hitting DB or cache
                cands = db.get_remembered_objects_by_class("cell phone")
                obj = db.get_remembered_object("test phone")
                all_objs = db.get_all_remembered_objects()
                mem.list_remembered_objects()
                time.sleep(0.002)
            except Exception as e:
                errors.append(f"voice_worker: {e}")

    threads = [
        threading.Thread(target=frame_thread_worker),
        threading.Thread(target=voice_find_worker),
        threading.Thread(target=frame_thread_worker),
        threading.Thread(target=voice_find_worker),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    db.close()
    assert len(errors) == 0, f"Encountered cursor/DB errors: {errors}"


def test_embedding_cache_refreshes_after_remember_and_forget(tmp_path):
    """Test that in-RAM embedding cache refreshes upon enrollment and deletion."""
    db_path = str(tmp_path / "cache_test.db")
    db = VisionMateDB(db_path=db_path)
    mem = ObjectMemory(db=db, matcher=VisualMatcher(feature_type='DEEP'))

    # Initially empty
    assert len(mem.get_cached_candidates("bottle")) == 0
    assert mem._all_embeddings_matrix.shape[0] == 0

    # Enroll object
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mem.start_enrollment("blue bottle", "bottle")
    mem.add_enrollment_frame(img, (10, 10, 90, 90))
    success = mem.finish_enrollment()
    assert success is True

    # Cache refreshed
    cands = mem.get_cached_candidates("bottle")
    assert len(cands) == 1
    assert cands[0][1] == "blue bottle"
    assert mem.get_cached_object("blue bottle") is not None
    assert mem._all_embeddings_matrix.shape[0] > 0

    # Forget / delete
    del_ok = mem.delete_object("blue bottle")
    assert del_ok is True

    # Cache refreshed again
    assert len(mem.get_cached_candidates("bottle")) == 0
    assert mem.get_cached_object("blue bottle") is None
    assert mem._all_embeddings_matrix.shape[0] == 0

    db.close()


def test_matching_100_embeddings_under_5ms():
    """Test that vectorized matching against 100 stored embeddings takes < 5 ms."""
    matcher = VisualMatcher(feature_type='DEEP')

    dim = 512
    num_embeddings = 100
    np.random.seed(42)
    matrix = np.random.randn(num_embeddings, dim).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    labels = [f"object_{i // 10}" for i in range(num_embeddings)]
    query = np.random.randn(dim).astype(np.float32)
    query /= np.linalg.norm(query)

    # Warmup
    matcher.match_vectorized(query, matrix, labels)

    # Benchmark 100 iterations
    t0 = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        best_name, confidence = matcher.match_vectorized(query, matrix, labels)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iterations

    assert best_name is not None
    assert elapsed_ms < 5.0, f"Expected matching time < 5 ms, got {elapsed_ms:.3f} ms"


def test_person_flickering_1_frame_does_not_retrigger_announcement():
    """Test that a person flickering for 1 frame does not re-trigger the 'Person ahead' announcement,
    announces only after >= 5 s absence, and is suppressed during active find unless distance is under 50 cm."""
    nav = SpatialNavigator()

    person_det = [{'class_name': 'person', 'bbox': [100, 100, 200, 300], 'confidence': 0.9}]

    # Frame 1: Person appears at t=0.0 -> Should announce
    g1 = nav.generate_guidance(person_det, current_time=0.0)
    assert len(g1) > 0
    p_item = next(item for item in g1 if item['class_name'] == 'person')
    assert nav.should_speak_guidance(p_item, current_time=0.0) is True
    nav.mark_guidance_spoken(p_item, current_time=0.0)

    # Frame 2: 1 missed frame (flicker) at t=0.1
    nav.generate_guidance([], current_time=0.1)

    # Frame 3: Person back at t=0.2 -> MUST NOT re-trigger announcement
    g3 = nav.generate_guidance(person_det, current_time=0.2)
    p_item3 = next(item for item in g3 if item['class_name'] == 'person')
    assert nav.should_speak_guidance(p_item3, current_time=0.2) is False

    # Person leaves at t=1.0
    nav.generate_guidance([], current_time=1.0)

    # Reappears at t=3.0 (< 5s absence) -> MUST NOT re-trigger
    g_early = nav.generate_guidance(person_det, current_time=3.0)
    p_early = next(item for item in g_early if item['class_name'] == 'person')
    assert nav.should_speak_guidance(p_early, current_time=3.0) is False

    # Absent for >= 5s (reappears at t=9.0, elapsed = 8.0s >= 5.0s)
    nav.generate_guidance([], current_time=3.1)
    g_after_5s = nav.generate_guidance(person_det, current_time=9.0)
    p_after_5s = next(item for item in g_after_5s if item['class_name'] == 'person')
    assert nav.should_speak_guidance(p_after_5s, current_time=9.0) is True
    nav.mark_guidance_spoken(p_after_5s, current_time=9.0)

    # During find mode: suppressed unless < 50 cm
    # Person absent for 6s, appears at t=16.0 during active find session:
    nav.generate_guidance([], current_time=10.0)
    g_find = nav.generate_guidance(person_det, current_time=16.0, is_find_active=True)
    p_find = next(item for item in g_find if item['class_name'] == 'person')
    # Normal distance (100 cm) -> suppressed
    assert nav.should_speak_guidance(p_find, ultrasonic_distance=100.0, current_time=16.0, is_find_active=True) is False
    # Under 50 cm (40 cm) -> allowed
    assert nav.should_speak_guidance(p_find, ultrasonic_distance=40.0, current_time=16.0, is_find_active=True) is True
