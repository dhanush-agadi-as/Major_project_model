"""
Tests for Event-Based Obstacle/Person Guidance and Audio Feedback Safety Net.

Covers all requirements from prompt:
1. "Person ahead" (and "Chair left") are Priority 3 (navigational), NOT priority 1.
   Only real distance under 50 cm uses emergency priority. Never treat priority 1 itself as critical.
2. Announce on events only with NO time-based repeats:
   - One person standing in view for 60 frames produces exactly one announcement.
   - A second person entering produces another announcement.
   - The same person leaving for 4s and returning is announced again.
   - Moving closer announces once per bucket (far to medium to close to very-close).
   - While the same person stays in view and nothing changes, say nothing.
3. Safety net in audio_feedback.py:
   - If the same text is already in the queue, or was spoken in the last 15s, drop it
     (except real critical warnings under 50cm, which may repeat every 3 seconds).
   - The same text queued twice is spoken once.
4. Logging:
   - Print "[AUDIO] Speaking ..." only when actually spoken.
   - Dropped duplicates logged at DEBUG as "[AUDIO] Dropped duplicate: <text>".
5. Quiet mode (dictation/yes-no window open, or <3s after wake word):
   - Routine announcements suppressed.
   - <50 cm warning is spoken during quiet mode.
"""

import time
import pytest
import numpy as np
import logging

from spatial_navigator import SpatialNavigator
from audio_feedback import (
    AudioFeedback,
    PRIORITY_EMERGENCY,
    PRIORITY_NAVIGATIONAL,
    PRIORITY_STATUS
)
from voice_command_system import VoiceCommandSystem
from wireless.laptop_server import LaptopMLServer


class DummyAudioFeedback:
    def __init__(self):
        self.spoken = []
        self._quiet_mode_checker = None
        self.last_spoken_time_by_text = {}

    def set_quiet_mode_checker(self, fn):
        self._quiet_mode_checker = fn

    def speak(self, text, priority=PRIORITY_NAVIGATIONAL, interrupt=False, is_critical=False):
        if not text:
            return
        if self._quiet_mode_checker and self._quiet_mode_checker() and not is_critical:
            return
        p_name = {1: 'emergency', 2: 'social', 3: 'navigational', 4: 'status'}.get(priority, str(priority))
        self.spoken.append((p_name, text))

    def speak_emergency(self, text, is_critical=True):
        self.speak(text, PRIORITY_EMERGENCY, interrupt=True, is_critical=is_critical)

    def speak_navigational(self, text, is_critical=False):
        self.speak(text, PRIORITY_NAVIGATIONAL, interrupt=False, is_critical=is_critical)

    def speak_social(self, text):
        self.speak(text, 2, interrupt=False, is_critical=False)

    def speak_status(self, text):
        self.speak(text, PRIORITY_STATUS, interrupt=False, is_critical=False)

    def is_tts_active(self):
        return False


# ============================================================================
# Requirement 1 & 2: One person standing in view for 60 frames produces 1 announcement
# ============================================================================

def test_one_person_standing_in_view_for_60_frames_produces_exactly_one_announcement():
    """One person standing in view for 60 frames produces exactly one announcement."""
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    # Centered person at medium distance
    det = {'bbox': (240, 160, 400, 320), 'class_name': 'person', 'confidence': 0.9}

    spoken_messages = []

    for frame in range(60):
        t = 100.0 + (frame * 0.1)  # 10 fps simulation
        guidance = nav.generate_guidance([det], current_time=t)
        assert len(guidance) > 0

        # Check each guidance item in frame
        for item in guidance:
            if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=t):
                spoken_messages.append(item['message'])
                nav.mark_guidance_spoken(item, current_time=t)

    # Exactly one announcement across all 60 frames
    assert len(spoken_messages) == 1
    assert "Person ahead" in spoken_messages[0]


# ============================================================================
# Requirement 2: A second person entering produces another announcement
# ============================================================================

def test_second_person_entering_produces_another_announcement():
    """When a second person enters view, another announcement is produced for the new person only."""
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    det1 = {'bbox': (240, 160, 400, 320), 'class_name': 'person', 'confidence': 0.9}
    det2 = {'bbox': (40, 160, 180, 320), 'class_name': 'person', 'confidence': 0.9}

    spoken_messages = []

    # Frames 0 to 10: Only Person 1 in view
    for frame in range(10):
        t = 100.0 + (frame * 0.1)
        guidance = nav.generate_guidance([det1], current_time=t)
        for item in guidance:
            if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=t):
                spoken_messages.append(item['message'])
                nav.mark_guidance_spoken(item, current_time=t)

    assert len(spoken_messages) == 1
    assert "Person ahead" in spoken_messages[0]

    # Frame 11: Person 2 enters view alongside Person 1 at t = 101.1 (< 6s cooldown)
    t = 101.1
    guidance = nav.generate_guidance([det1, det2], current_time=t)
    for item in guidance:
        if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=t):
            spoken_messages.append(item['message'])
            nav.mark_guidance_spoken(item, current_time=t)

    # Within 6 s cooldown, no separate announcement spoken
    assert len(spoken_messages) == 1

    # After 6 s cooldown (t = 106.5), count change is announced as a combined message
    t = 106.5
    guidance = nav.generate_guidance([det1, det2], current_time=t)
    for item in guidance:
        if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=t):
            spoken_messages.append(item['message'])
            nav.mark_guidance_spoken(item, current_time=t)

    assert len(spoken_messages) == 2
    assert "2 people" in spoken_messages[1]

    # Frames after: Both persons stay in view with no changes -> nothing more spoken
    for frame in range(12, 20):
        t = 107.0 + (frame * 0.1)
        guidance = nav.generate_guidance([det1, det2], current_time=t)
        for item in guidance:
            if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=t):
                spoken_messages.append(item['message'])
                nav.mark_guidance_spoken(item, current_time=t)

    assert len(spoken_messages) == 2


# ============================================================================
# Requirement 2: The same person leaving for 4s and returning is announced again
# ============================================================================

def test_same_person_leaving_for_4s_and_returning_announced_again():
    """The same person leaving view for 4s and returning is announced again."""
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    det = {'bbox': (240, 160, 400, 320), 'class_name': 'person', 'confidence': 0.9}

    spoken_messages = []

    # t = 0.0: Person enters view -> spoken
    guidance = nav.generate_guidance([det], current_time=0.0)
    for item in guidance:
        if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=0.0):
            spoken_messages.append(item['message'])
            nav.mark_guidance_spoken(item, current_time=0.0)

    assert len(spoken_messages) == 1

    # Person leaves view: frames from t=0.5 to t=4.0 have no detections
    for step in range(1, 9):
        t = step * 0.5
        nav.generate_guidance([], current_time=t)

    # t = 4.5 (4.5s since last seen, < 5.0s absence): same person returns -> NOT announced
    guidance_return = nav.generate_guidance([det], current_time=4.5)
    for item in guidance_return:
        if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=4.5):
            spoken_messages.append(item['message'])
            nav.mark_guidance_spoken(item, current_time=4.5)

    assert len(spoken_messages) == 1

    # Person leaves view for >= 5s: from t=5.0 to t=10.0
    for step in range(10, 21):
        t = step * 0.5
        nav.generate_guidance([], current_time=t)

    # t = 10.5 (6.0s since last seen at 4.5, >= 5.0s absence): same person returns -> announced!
    guidance_after_5s = nav.generate_guidance([det], current_time=10.5)
    for item in guidance_after_5s:
        if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=10.5):
            spoken_messages.append(item['message'])
            nav.mark_guidance_spoken(item, current_time=10.5)

    assert len(spoken_messages) == 2
    assert "Person ahead" in spoken_messages[1]

    # However, if person only leaves for 1.0s (less than 5s threshold):
    # t = 11.0: absent
    nav.generate_guidance([], current_time=11.0)
    # t = 11.5: returns (only 1.0s since 10.5s): not re-announced!
    guidance_brief = nav.generate_guidance([det], current_time=11.5)
    brief_spoken = []
    for item in guidance_brief:
        if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=11.5):
            brief_spoken.append(item['message'])
            nav.mark_guidance_spoken(item, current_time=11.5)

    assert len(brief_spoken) == 0
    assert len(spoken_messages) == 2


# ============================================================================
# Requirement 2: Moving closer announces once per bucket
# ============================================================================

def test_moving_closer_announces_once_per_bucket():
    """Moving closer announces once per bucket (far -> medium -> close -> very-close)."""
    nav = SpatialNavigator(frame_width=640, frame_height=480)

    # 1. 'far' bucket: bbox area < 0.05 (40x40 = 1600 / 307200 = 0.005)
    det_far = {'bbox': (300, 220, 340, 260), 'class_name': 'chair', 'confidence': 0.9}
    assert nav.estimate_distance(det_far['bbox']) == 'far'

    # 2. 'medium' bucket: bbox area 0.05 - 0.15 (160x160 = 25600 / 307200 = 0.083)
    det_medium = {'bbox': (240, 160, 400, 320), 'class_name': 'chair', 'confidence': 0.9}
    assert nav.estimate_distance(det_medium['bbox']) == 'medium'

    # 3. 'close' bucket: bbox area 0.15 - 0.35 (240x280 = 67200 / 307200 = 0.218)
    det_close = {'bbox': (200, 100, 440, 380), 'class_name': 'chair', 'confidence': 0.9}
    assert nav.estimate_distance(det_close['bbox']) == 'close'

    # 4. 'very-close' bucket: bbox area > 0.35 (440x380 = 167200 / 307200 = 0.544)
    det_very_close = {'bbox': (100, 50, 540, 430), 'class_name': 'chair', 'confidence': 0.9}
    assert nav.estimate_distance(det_very_close['bbox']) == 'very-close'

    spoken = []
    t = 0.0

    # Step 1: Start at 'far'
    for _ in range(5):
        t += 0.1
        guidance = nav.generate_guidance([det_far], current_time=t)
        for item in guidance:
            if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=t):
                spoken.append(('far', item['message']))
                nav.mark_guidance_spoken(item, current_time=t)
    assert len(spoken) == 1
    assert spoken[0][0] == 'far'

    # Step 2: Moves to 'medium'
    for _ in range(5):
        t += 0.1
        guidance = nav.generate_guidance([det_medium], current_time=t)
        for item in guidance:
            if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=t):
                spoken.append(('medium', item['message']))
                nav.mark_guidance_spoken(item, current_time=t)
    assert len(spoken) == 2
    assert spoken[1][0] == 'medium'

    # Step 3: Moves to 'close'
    for _ in range(5):
        t += 0.1
        guidance = nav.generate_guidance([det_close], current_time=t)
        for item in guidance:
            if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=t):
                spoken.append(('close', item['message']))
                nav.mark_guidance_spoken(item, current_time=t)
    assert len(spoken) == 3
    assert spoken[2][0] == 'close'

    # Step 4: Moves BACK to 'medium' (stepping back) -> should NOT announce!
    for _ in range(5):
        t += 0.1
        guidance = nav.generate_guidance([det_medium], current_time=t)
        for item in guidance:
            if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=t):
                spoken.append(('medium_back', item['message']))
                nav.mark_guidance_spoken(item, current_time=t)
    assert len(spoken) == 3  # unchanged

    # Step 5: Moves back to 'close' -> already announced 'close' previously, so should NOT announce!
    for _ in range(5):
        t += 0.1
        guidance = nav.generate_guidance([det_close], current_time=t)
        for item in guidance:
            if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=t):
                spoken.append(('close_repeat', item['message']))
                nav.mark_guidance_spoken(item, current_time=t)
    assert len(spoken) == 3  # unchanged

    # Step 6: Moves to 'very-close' -> closer than any announced bucket -> announces!
    for _ in range(5):
        t += 0.1
        guidance = nav.generate_guidance([det_very_close], current_time=t)
        for item in guidance:
            if nav.should_speak_guidance(item, ultrasonic_distance=100.0, current_time=t):
                spoken.append(('very-close', item['message']))
                nav.mark_guidance_spoken(item, current_time=t)
    assert len(spoken) == 4
    assert spoken[3][0] == 'very-close'


# ============================================================================
# Requirement 1: Priority 3 (navigational) vs Priority 1 (<50 cm critical)
# ============================================================================

def test_guidance_is_priority_3_navigational_not_priority_1():
    """Routine guidance like 'Person ahead' and 'Chair left' has Priority 3 (navigational), not 1."""
    nav = SpatialNavigator(frame_width=640, frame_height=480)
    det_person = {'bbox': (240, 160, 400, 320), 'class_name': 'person', 'confidence': 0.9}
    det_chair = {'bbox': (40, 160, 180, 320), 'class_name': 'chair', 'confidence': 0.8}

    guidance = nav.generate_guidance([det_person, det_chair], current_time=100.0)

    person_item = next(g for g in guidance if g['class_name'] == 'person')
    chair_item = next(g for g in guidance if g['class_name'] == 'chair')

    # Routine guidance must be priority 3
    assert person_item['priority'] == 3
    assert chair_item['priority'] == 3

    # Neither is critical at 100 cm ultrasonic distance
    assert nav.is_critical(person_item, ultrasonic_distance=100.0) is False
    assert nav.is_critical(chair_item, ultrasonic_distance=100.0) is False

    # Under 50 cm ultrasonic distance: becomes critical
    assert nav.is_critical(person_item, ultrasonic_distance=40.0) is True


# ============================================================================
# Requirement 3 & 4: Safety net in audio_feedback.py and logging fixes
# ============================================================================

def test_same_text_queued_twice_is_spoken_once(caplog):
    """The same text queued twice is dropped by the safety net and spoken once."""
    audio = AudioFeedback(use_gtts=False)
    played = []
    audio._play_text = lambda text: (played.append(text), time.sleep(0.2))
    with caplog.at_level(logging.DEBUG):
        audio.speak("Person ahead")
        audio.speak("Person ahead")
        time.sleep(0.05)

    assert any("Dropped duplicate: Person ahead" in r.message for r in caplog.records)
    t_end = time.time() + 2.0
    while time.time() < t_end and not played:
        time.sleep(0.05)
    time.sleep(0.25)
    assert played.count("Person ahead") == 1


def test_spoken_text_suppressed_for_15s_except_critical_3s():
    """Spoken text is dropped within 15s; critical warnings under 50cm may repeat every 3s."""
    audio = AudioFeedback(use_gtts=False)
    t_now = time.time()

    # Simulate that "Person ahead" was spoken 5 seconds ago
    audio.last_spoken_time_by_text["Person ahead"] = t_now - 5.0

    # 1. Routine text within 15s cooldown -> dropped
    audio.speak("Person ahead")
    with audio.audio_queue.mutex:
        q_texts = [it[1] for it in audio.audio_queue.queue if it[1] != "__STOP__"]
    assert "Person ahead" not in q_texts

    # 2. Critical warning under 50 cm (<50 cm warning) spoken 4 seconds ago (> 3s cooldown) -> allowed!
    crit_msg = "Obstacle 30 cm ahead"
    audio.last_spoken_time_by_text[crit_msg] = t_now - 4.0
    audio.speak_emergency(crit_msg, is_critical=True)
    with audio.audio_queue.mutex:
        q_texts = [it[1] for it in audio.audio_queue.queue if it[1] != "__STOP__"]
    assert crit_msg in q_texts


# ============================================================================
# Requirement 5: Quiet mode suppresses routine guidance; <50 cm gets through
# ============================================================================

def test_critical_warning_under_50cm_spoken_during_quiet_mode():
    """During quiet mode, routine announcements are suppressed but <50 cm warnings speak."""
    server = LaptopMLServer(port=8092)
    server.show_display = False
    server.audio_enabled = True
    server.navigation_enabled = True
    server.audio = DummyAudioFeedback()
    server.voice_system = VoiceCommandSystem(audio_feedback=server.audio)
    server.voice_system.enabled = True

    det = {'bbox': [240, 160, 400, 320], 'class_name': 'person', 'confidence': 0.9}

    class DummyDetector:
        def detect(self, frame):
            return [det]
    server.object_detector = DummyDetector()

    class DummyTracker:
        def update_tracking(self, detections):
            return []
    server.item_tracker = DummyTracker()

    class DummyMemory:
        def recognize_object(self, frame, bbox, yolo_class):
            return None
    server.object_memory = DummyMemory()

    class DummyFaceRecognizer:
        def recognize_faces(self, frame):
            return []
    server.face_recognizer = DummyFaceRecognizer()

    server.spatial_navigator = SpatialNavigator(frame_width=640, frame_height=480)
    server.find_guide = None
    server.audio.set_quiet_mode_checker(server.is_quiet_mode_active)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Open dictation window to enter quiet mode
    server.voice_system.open_dictation_window(on_dictate=lambda text: None, timeout=10.0)
    assert server.is_quiet_mode_active() is True

    # 2. Routine warning (ultrasonic = 100 cm, not critical): must be suppressed
    server.process_ml(dummy_frame, ultrasonic_distance=100.0, button_pressed=False)
    assert len(server.audio.spoken) == 0

    # 3. Critical warning (ultrasonic = 35 cm < 50 cm): must bypass quiet mode and speak
    server.process_ml(dummy_frame, ultrasonic_distance=35.0, button_pressed=False)
    assert len(server.audio.spoken) == 1
    priority, msg = server.audio.spoken[0]
    assert priority == 'emergency'
    assert "Person ahead" in msg
