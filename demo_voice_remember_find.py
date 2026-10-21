"""
Simple Voice Command Demo - Remember & Find Objects
Press ENTER to speak, system executes commands
"""

import os
import sys
import cv2
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from object_detector import ObjectDetector
from database import VisionMateDB
from object_memory import ObjectMemory
from voice_command_system import VoiceCommandSystem
from audio_feedback import get_audio_feedback
from camera_source import open_camera

def main():
    print("=" * 70)
    print("VOICE COMMAND DEMO - Remember & Find Objects")
    print("=" * 70)
    
    # Initialize
    print("\nInitializing...")
    db = VisionMateDB()
    detector = ObjectDetector()
    memory = ObjectMemory(db)
    audio = get_audio_feedback(use_gtts=False)
    
    # Initialize voice system
    voice = VoiceCommandSystem()
    if not voice.initialize():
        print("[FAIL] Voice system failed - check microphone")
        return
    
    voice.start()
    print("[OK] All systems ready!")
    
    # State
    current_frame = None
    current_objects = []
    enrollment_mode = False
    enrollment_target = None
    enrollment_count = 0
    
    # Voice handlers
    def handle_remember(params):
        nonlocal enrollment_mode, enrollment_target, enrollment_count
        
        obj_name = params.get('object_name', '')
        if not obj_name:
            audio.speak_status("Please specify object name")
            return
        
        # Find matching object
        matches = [o for o in current_objects if obj_name in o['class_name'].lower()]
        if not matches:
            audio.speak_status(f"No {obj_name} detected in view")
            print(f"   Objects in view: {[o['class_name'] for o in current_objects]}")
            return
        
        obj = matches[0]
        custom_name = f"my{obj_name}"
        
        try:
            memory.start_enrollment(custom_name, obj['class_name'])
            enrollment_mode = True
            enrollment_target = {'custom_name': custom_name, 'yolo_class': obj['class_name']}
            enrollment_count = 0
            audio.speak_social(f"Starting enrollment for {custom_name}. Keep it in view.")
            print(f"[OK] Enrolling '{custom_name}'...")
        except ValueError as e:
            audio.speak_status(str(e))
            print(f"[FAIL] Error: {e}")
    
    def handle_find(params):
        obj_name = params.get('object_name', '')
        if not obj_name:
            audio.speak_status("Please specify what to find")
            return
        
        location = memory.get_object_location(obj_name)
        if location:
            nearby = location['context'].get('nearby_objects', []) if location['context'] else []
            nearby_str = f" near {', '.join(nearby[:3])}" if nearby else ""
            msg = f"Your {obj_name} last seen{nearby_str}"
            audio.speak_navigational(msg)
            print(f"[OK] {msg}")
        else:
            audio.speak_status(f"Haven't seen {obj_name}")
            print(f"[FAIL] No record of {obj_name}")
    
    def handle_list(params):
        items = memory.list_remembered_objects()
        if items:
            names = ", ".join([name for name, _ in items])
            audio.speak_social(f"I remember: {names}")
            print(f"[OK] Remembered: {names}")
        else:
            audio.speak_status("No items remembered yet")
            print("[FAIL] No items remembered")
    
    voice.register_command_handler('remember', handle_remember)
    voice.register_command_handler('find', handle_find)
    voice.register_command_handler('list', handle_list)
    
    # Open camera source (Pi stream or local webcam)
    cap = open_camera()
    if not cap.isOpened():
        print("[FAIL] Cannot open webcam")
        return
    
    print("\n" + "=" * 70)
    print("READY TO USE!")
    print("=" * 70)
    print("\nControls:")
    print("  ENTER - Trigger voice command")
    print("  Q - Quit")
    print("\nTry saying:")
    print("  'Remember this phone'  - Enrolls phone as 'myphone'")
    print("  'Find my phone'        - Shows last seen location")
    print("  'List all items'       - Shows all remembered objects")
    print("\nMake sure objects are visible in webcam!")
    print("=" * 70)
    
    last_voice_time = 0
    voice_cooldown = 1.0  # Seconds between voice commands
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame = frame.copy()
        display = frame.copy()
        
        # Detect objects
        current_objects = detector.detect(frame)
        display = detector.draw_detections(display, current_objects)
        
        # Handle enrollment
        if enrollment_mode and enrollment_target:
            matches = [o for o in current_objects if o['class_name'] == enrollment_target['yolo_class']]
            if matches:
                obj = matches[0]
                bbox = tuple(obj['bbox'])
                enrollment_count = memory.add_enrollment_frame(frame, bbox)
                
                # Draw indicator
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(display, f"Recording {enrollment_count}/5", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                if enrollment_count >= 5:
                    if memory.finish_enrollment():
                        audio.speak_social(f"Remembered your {enrollment_target['custom_name']}")
                        print(f"[OK] Enrolled '{enrollment_target['custom_name']}'!")
                    enrollment_mode = False
                    enrollment_target = None
        
        # Show status
        status_lines = [
            f"Objects: {len(current_objects)}",
            f"Enrollment: {'ACTIVE' if enrollment_mode else 'OFF'}",
            "Press ENTER to speak",
        ]
        
        for i, line in enumerate(status_lines):
            cv2.putText(display, line, (10, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # List detected objects
        if current_objects:
            y = display.shape[0] - 100
            cv2.putText(display, "Detected:", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for i, obj in enumerate(current_objects[:3]):
                y += 20
                cv2.putText(display, f"  - {obj['class_name']}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Voice Command Demo - Press ENTER to speak', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == 13:  # ENTER key
            current_time = time.time()
            if current_time - last_voice_time > voice_cooldown:
                print("\n[MIC] Listening for command...")
                audio.speak_status("Listening")
                voice.trigger_voice_input()
                last_voice_time = current_time
            else:
                print("[WAIT] Please wait a moment...")
    
    cap.release()
    cv2.destroyAllWindows()
    voice.stop()
    db.close()
    
    print("\n[OK] Demo complete!")


if __name__ == "__main__":
    main()
