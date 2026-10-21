"""
Comprehensive verification tests for the 6 core user requirements:
1. deep_feature_extractor / visual_matcher:
   - Stored embedding is 1280D.
   - GAP (mean over H, W) applied to MobileNetV2 feature map, then L2-normalized.
   - Log shows "Deep embedding: 1280D vector".
   - Each enrollment stores 24 vectors of 1280D.
   - Old 62720D rows detected on load, ignored, and user prompted to re-enroll.
   - Freshly enrolled phone matches at >= 85% immediately, compare_ms for 24 vectors is under 5 ms.
2. Dictation windows (face name, notes):
   - Captured through background listener pipeline (_bg_audio_callback).
   - Opened after TTS finishes with 8.0s timeout.
   - Logs "Audio chunk captured while window is open" and STT text.
   - Does not call separate blocking recognizer.listen on the shared mic.
3. Face detection cost:
   - InsightFace det_size is (320, 320).
   - Runs detection at most 5 times/s and only when YOLO sees a person.
   - Reuses last result between runs.
   - Recognition per track at most once per second.
4. "Person ahead" guidance:
   - For a person who stays in view, not repeated within 30 s.
   - Announced again only when count changes, new person arrives, or person gets closer than 1 m.
5. Rolling averages:
   - embed_ms, compare_ms, db_ms are true rolling averages, resetting/rolling to 0 when unused.
6. Integration test suite validation.
"""

import os
import sys
import time
import math
import logging
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from deep_feature_extractor import DeepFeatureExtractor
from visual_matcher import VisualMatcher
from database import VisionMateDB, is_legacy_62720d
from object_memory import ObjectMemory
from voice_command_system import VoiceCommandSystem, VoiceCommandParser
from face_recognizer import FaceRecognizer
from spatial_navigator import SpatialNavigator
from wireless.laptop_server import LaptopMLServer


# ─────────────────────────────────────────────────────────────────────────────
# 1. Deep Feature Extractor & Visual Matcher (1280D, GAP, L2-norm, 24 vectors)
# ─────────────────────────────────────────────────────────────────────────────

def test_deep_feature_extractor_1280d_gap_l2_norm(caplog):
    """Verify deep feature extractor outputs 1280D, L2-normalized vector with exact log."""
    caplog.set_level(logging.DEBUG)
    extractor = DeepFeatureExtractor()

    crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    with caplog.at_level(logging.DEBUG):
        emb = extractor.extract_embedding(crop)

    assert emb is not None, "Embedding extraction failed"
    assert emb.shape == (1280,), f"Expected 1280D embedding, got shape {emb.shape}"
    norm = np.linalg.norm(emb)
    assert math.isclose(norm, 1.0, abs_tol=1e-4), f"Expected L2 norm ~ 1.0, got {norm}"

    # Verify required log message
    assert any("Deep embedding: 1280D vector" in record.message for record in caplog.records), \
        "Expected log 'Deep embedding: 1280D vector' not found"


def test_visual_matcher_enrollment_stores_24_vectors_of_1280d(caplog):
    """Verify VisualMatcher creates exactly 24 vectors of 1280D for enrollment."""
    caplog.set_level(logging.DEBUG)
    matcher = VisualMatcher(feature_type='DEEP', use_deep_features=True)

    dummy_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    bbox = (20, 20, 160, 160)

    with caplog.at_level(logging.DEBUG):
        features = matcher.extract_multi_frame_features([dummy_image], [bbox])

    assert features is not None
    assert features.get('feature_type') == 'deep'
    deep_embs = features.get('deep_embeddings')
    assert deep_embs is not None
    assert len(deep_embs) == 24, f"Expected exactly 24 vectors, got {len(deep_embs)}"
    for idx, vec in enumerate(deep_embs):
        assert len(vec) == 1280, f"Vector {idx} length is {len(vec)}, expected 1280"
        norm = np.linalg.norm(vec)
        assert math.isclose(norm, 1.0, abs_tol=1e-4), f"Vector {idx} L2 norm is {norm}, expected 1.0"


def test_legacy_62720d_detected_and_ignored(tmp_path, caplog):
    """Verify legacy 62720D rows are detected, ignored on DB load, and log asks to re-enroll."""
    caplog.set_level(logging.WARNING)
    db_file = str(tmp_path / "test_legacy.db")
    db = VisionMateDB(db_path=db_file)

    # Insert a dummy legacy 62720D object
    legacy_features = {
        'feature_type': 'deep',
        'embedding_size': 62720,
        'deep_embedding': [0.0] * 62720
    }

    obj_id = db.add_remembered_object(
        custom_name="old_phone",
        yolo_class="cell phone",
        visual_features=legacy_features,
        enrollment_frames=1
    )
    assert obj_id is not None

    # Verify helper identifies it
    assert is_legacy_62720d(legacy_features) is True

    # Loading via get_remembered_object should return None and log warning
    with caplog.at_level(logging.WARNING):
        res = db.get_remembered_object("old_phone")
    assert res is None
    assert any("62720D" in rec.message and "re-enroll" in rec.message for rec in caplog.records)

    # Loading via get_remembered_objects_by_class should ignore it
    by_class = db.get_remembered_objects_by_class("cell phone")
    assert len(by_class) == 0

    # Also test ObjectMemory.refresh_cache ignores it
    matcher = VisualMatcher(feature_type='DEEP', use_deep_features=True)
    mem = ObjectMemory(db=db, matcher=matcher)
    assert "old_phone" not in mem._cached_objects


def test_fresh_enrollment_matches_itself_ge_85_and_under_5ms(tmp_path):
    """Verify a freshly enrolled phone matches itself at >= 85% and compare_ms < 5ms."""
    db_file = str(tmp_path / "test_fresh_match.db")
    db = VisionMateDB(db_path=db_file)
    matcher = VisualMatcher(feature_type='DEEP', use_deep_features=True)
    mem = ObjectMemory(db=db, matcher=matcher)

    # Simulate enrolling a phone
    phone_image = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
    phone_bbox = (50, 40, 180, 260)

    mem.start_enrollment(yolo_class="cell phone", custom_name="my_phone")
    for _ in range(4):
        mem.add_enrollment_frame(phone_image, phone_bbox)
    success = mem.finish_enrollment()
    assert success is True

    # Show the phone immediately
    match = mem.recognize_object(phone_image, phone_bbox, yolo_class="cell phone")
    assert match is not None
    custom_name, confidence, tier = match
    assert custom_name == "my_phone"
    assert confidence >= 0.85, f"Expected similarity >= 0.85, got {confidence:.4f}"

    # Verify 24 vectors of 1280D stored and compare_ms is under 5 ms
    cls_matrix = mem._cached_class_matrices["cell phone"]["matrix"]
    row_labels = mem._cached_class_matrices["cell phone"]["row_to_name"]
    assert cls_matrix.shape == (24, 1280), f"Expected shape (24, 1280), got {cls_matrix.shape}"
    q_emb = cls_matrix[0]
    t_comp0 = time.perf_counter()
    matcher.match_vectorized(q_emb, cls_matrix, row_labels)
    compare_ms = (time.perf_counter() - t_comp0) * 1000.0
    assert compare_ms < 5.0, f"Expected compare_ms < 5 ms, got {compare_ms:.3f} ms"

    # Also check mem.last_compare_ms on repeat recognition
    mem.recognize_object(phone_image, phone_bbox, yolo_class="cell phone")
    assert mem.last_compare_ms < 5.0, f"Expected last_compare_ms < 5 ms, got {mem.last_compare_ms:.3f} ms"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dictation Windows (Background listener pipeline, 8s timeout, log)
# ─────────────────────────────────────────────────────────────────────────────

def test_dictation_window_background_pipeline_8s_timeout_and_logs(caplog):
    """Verify dictation window opens with 8s timeout, receives audio via _bg_audio_callback without blocking mic."""
    caplog.set_level(logging.INFO)

    class DummyAudio:
        def __init__(self):
            self.is_speaking = False
            self.last_speech_end_time = time.time() - 2.0
        def is_tts_active(self):
            return False

    audio = DummyAudio()
    vcs = VoiceCommandSystem(audio_feedback=audio)
    vcs.running = True
    vcs.enabled = True

    captured_notes = []
    def on_note_dictated(text):
        captured_notes.append(text)

    # Open dictation window (default timeout 8.0s)
    vcs.open_dictation_window(on_dictate=on_note_dictated, timeout=8.0)
    assert vcs.dictation_active is True
    assert vcs.dictation_until <= time.time() + 14.0
    assert vcs.dictation_until >= time.time() + 7.5

    # Simulate audio arrival via background listener callback
    dummy_recognizer = MagicMock()
    dummy_audio = MagicMock()

    # Mock recognizer STT result
    with patch.object(vcs, '_recognize_audio', return_value="meeting at 3 pm"), \
         caplog.at_level(logging.INFO):
        vcs._bg_audio_callback(dummy_recognizer, dummy_audio, _sync=True)

    # Verify required logs
    assert any("Audio chunk captured while window is open" in rec.message for rec in caplog.records), \
        "Expected log 'Audio chunk captured while window is open' not found"
    assert any("meeting at 3 pm" in rec.message for rec in caplog.records), \
        "Expected STT text logged"

    # Verify dictation callback received note and window closed
    assert captured_notes == ["meeting at 3 pm"]
    assert vcs.dictation_active is False

    # Verify recognizer.listen was NEVER called on shared mic
    assert dummy_recognizer.listen.called is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Face Detection Cost (det_size=(320, 320), 5 Hz limit, YOLO gating, 1 Hz track)
# ─────────────────────────────────────────────────────────────────────────────

def test_face_detection_det_size_and_yolo_gating():
    """Verify InsightFace det_size is (320, 320), gated on YOLO person and rate-limited to 5 Hz."""
    assert config.FACE_DET_SIZE == (320, 320)

    fr = FaceRecognizer()
    fr.insightface_available = True
    mock_app = MagicMock()
    mock_face = MagicMock()
    mock_face.bbox = np.array([50, 50, 150, 150])
    mock_face.det_score = 0.95
    mock_face.normed_embedding = np.random.randn(512).astype(np.float32)
    mock_app.get.return_value = [mock_face]
    fr.app = mock_app

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. YOLO sees NO person -> detection skipped entirely
    res1 = fr.detect_faces(dummy_frame, yolo_has_person=False, current_time=1.0)
    assert res1 == []
    assert mock_app.get.called is False

    # 2. YOLO sees person -> detection runs at t=1.0
    res2 = fr.detect_faces(dummy_frame, yolo_has_person=True, current_time=1.0)
    assert len(res2) == 1
    assert mock_app.get.call_count == 1

    # 3. Call again at t=1.1 (< 0.2s elapsed -> 5 Hz limit) -> reuses last result without running app.get
    res3 = fr.detect_faces(dummy_frame, yolo_has_person=True, current_time=1.1)
    assert len(res3) == 1
    assert mock_app.get.call_count == 1  # Not incremented!

    # 4. Call again at t=1.25 (>= 0.2s elapsed) -> runs app.get
    res4 = fr.detect_faces(dummy_frame, yolo_has_person=True, current_time=1.25)
    assert len(res4) == 1
    assert mock_app.get.call_count == 2


def test_face_recognition_per_track_at_most_once_per_second():
    """Verify face recognition per track runs at most once per second."""
    fr = FaceRecognizer()
    fr.insightface_available = True
    fr._match_face = MagicMock(return_value=("Bob", 0.35, 0.2))

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    face_dict = {
        'bbox': [100, 100, 200, 200],
        'embedding': np.random.randn(512).astype(np.float32),
        'confidence': 0.95
    }
    fr.detect_faces = MagicMock(return_value=[face_dict])

    # First sight at t=10.0 -> recognized
    res1 = fr.recognize_faces(dummy_frame, current_time=10.0, yolo_has_person=True)
    assert len(res1) == 1
    assert res1[0]['name'] == 'Bob'
    assert fr._match_face.call_count == 1

    # Same face at t=10.3 (<1.0s track recognition interval)
    fr.last_detection_time = 0.0  # allow detection pass
    res2 = fr.recognize_faces(dummy_frame, current_time=10.3, yolo_has_person=True)
    assert len(res2) == 1
    assert res2[0]['name'] == 'Bob'
    assert fr._match_face.call_count == 1  # Track recognition throttled!

    # Same face at t=11.1 (>= 1.0s elapsed) -> recognition runs again
    fr.last_detection_time = 0.0
    res3 = fr.recognize_faces(dummy_frame, current_time=11.1, yolo_has_person=True)
    assert len(res3) == 1
    assert fr._match_face.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. "Person ahead": Repeat Suppression (30s cooldown, triggers on change)
# ─────────────────────────────────────────────────────────────────────────────

def test_person_ahead_not_repeated_within_30s_for_same_person():
    """Verify 'Person ahead' isn't repeated within 30s for a person staying in view.
    Announce again only when count changes, new person arrives, or person gets closer than 1m.
    """
    nav = SpatialNavigator(frame_width=640, frame_height=480)

    # Person at 2m (area_ratio = (200*160)/(640*480) ~ 0.104 -> medium distance ~ 2.0m)
    person_det = [{'bbox': [220, 160, 420, 320], 'class_name': 'person', 'confidence': 0.9}]

    spoken_count = 0

    # Stream 300 frames at 10 fps (30.0 seconds of continuous presence)
    for frame_idx in range(300):
        t = frame_idx * 0.1
        g = nav.generate_guidance(person_det, current_time=t)
        p_item = next(item for item in g if item['class_name'] == 'person')
        if nav.should_speak_guidance(p_item, current_time=t):
            spoken_count += 1
            nav.mark_guidance_spoken(p_item, current_time=t)

    # Exactly 1 announcement in the first 30 seconds
    assert spoken_count == 1, f"Expected 1 announcement in 30s, got {spoken_count}"

    # At t=35s, person still in view at 2m (no change, not <1m, same count) -> STILL NOT REPEATED
    for frame_idx in range(300, 350):
        t = frame_idx * 0.1
        g = nav.generate_guidance(person_det, current_time=t)
        p_item = next(item for item in g if item['class_name'] == 'person')
        assert nav.should_speak_guidance(p_item, current_time=t) is False

    # At t=36s, person gets closer than 1m -> ANNOUNCE!
    close_det = [{'bbox': [100, 40, 540, 440], 'class_name': 'person', 'confidence': 0.9}]
    g_close = nav.generate_guidance(close_det, current_time=36.0)
    p_close = next(item for item in g_close if item['class_name'] == 'person')
    assert nav.should_speak_guidance(p_close, current_time=36.0) is True
    nav.mark_guidance_spoken(p_close, current_time=36.0)

    # While staying under 1m at t=36.1s to 39.0s -> NOT repeated
    for frame_idx in range(361, 390):
        t = frame_idx * 0.1
        g = nav.generate_guidance(close_det, current_time=t)
        p_item = next(item for item in g if item['class_name'] == 'person')
        assert nav.should_speak_guidance(p_item, current_time=t) is False

    # At t=40s, count changes from 1 to 2 -> ANNOUNCE!
    two_persons = [
        {'bbox': [50, 50, 200, 300], 'class_name': 'person', 'confidence': 0.9},
        {'bbox': [350, 50, 500, 300], 'class_name': 'person', 'confidence': 0.9}
    ]
    g_two = nav.generate_guidance(two_persons, current_time=40.0)
    p_two = next(item for item in g_two if item['class_name'] == 'person')
    assert p_two['count'] == 2
    assert nav.should_speak_guidance(p_two, current_time=40.0) is True
    nav.mark_guidance_spoken(p_two, current_time=40.0)

    # Person leaves at t=45.0, absent for 6s (>= 5s), returns at t=51.0 -> ANNOUNCE!
    for frame_idx in range(450, 510):
        nav.generate_guidance([], current_time=frame_idx * 0.1)

    g_return = nav.generate_guidance(person_det, current_time=51.0)
    p_return = next(item for item in g_return if item['class_name'] == 'person')
    assert nav.should_speak_guidance(p_return, current_time=51.0) is True


# ─────────────────────────────────────────────────────────────────────────────
# 5. True Rolling Averages (embed_ms, compare_ms, db_ms reset to 0 when unused)
# ─────────────────────────────────────────────────────────────────────────────

def test_rolling_averages_reset_to_zero_when_unused():
    """Verify embed_ms, compare_ms, db_ms roll down to 0.0 when unused."""
    db = MagicMock()
    matcher = MagicMock()
    mem = ObjectMemory(db=db, matcher=matcher)

    # Initially empty deque -> 0.0
    assert mem.embed_ms == 0.0
    assert mem.compare_ms == 0.0
    assert mem.db_ms == 0.0

    # Feed some non-zero values
    for _ in range(10):
        mem._rolling_embed_ms.append(12.5)
        mem._rolling_compare_ms.append(1.8)
        mem._rolling_db_ms.append(3.2)

    assert mem.embed_ms > 10.0
    assert mem.compare_ms > 1.0
    assert mem.db_ms > 2.0

    # Now unused: feed 30 zeros (deque maxlen=30)
    for _ in range(30):
        mem._rolling_embed_ms.append(0.0)
        mem._rolling_compare_ms.append(0.0)
        mem._rolling_db_ms.append(0.0)

    assert mem.embed_ms == 0.0
    assert mem.compare_ms == 0.0
    assert mem.db_ms == 0.0
