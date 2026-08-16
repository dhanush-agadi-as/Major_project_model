"""
Integration Test - Full VisionMate Pipeline
Tests all modules working together
"""
import cv2
import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from object_detector import ObjectDetector
from face_recognizer import FaceRecognizer
# from traffic_signal_detector import TrafficSignalDetector  # DISABLED: Too many false positives
from item_tracker import ItemTracker
from audio_feedback import get_audio_feedback
from database import VisionMateDB
from object_memory import ObjectMemory
from spatial_navigator import SpatialNavigator
from voice_command_system import VoiceCommandSystem
from camera_reader import CameraReader  # Fix 1: threaded webcam capture

# FREE Navigation modules (OpenStreetMap)
from gps_tracker import GPSTracker
from osm_navigator import OSMNavigator
from ultrasonic_sensor import UltrasonicSensor
from obstacle_fusion import ObstacleFusion
from navigation_engine import NavigationEngine
import config
from config import OBJECT_RECOGNITION_FRAME_INTERVAL, FACE_RECOGNITION_FRAME_INTERVAL, YOLO_FRAME_SKIP


def test_full_pipeline_webcam():
    """Test complete AI pipeline with webcam"""
    print("=" * 60)
    print("VISIONMATE FULL INTEGRATION TEST")
    print("=" * 60)
    
    print("\nInitializing all modules...")
    db = VisionMateDB()
    object_detector = ObjectDetector()
    
    # Fix 2: GPU warmup — OpenVINO compiles GPU kernels on the very first
    # inference call (~8 s).  Run it now on a blank frame so the delay
    # happens here (with a clear message) rather than freezing the live
    # feed silently on the first real frame.
    print("Warming up GPU inference (one-time, ~8s)...")
    _ = object_detector.detect(np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8))
    print("Ready.")
    
    face_recognizer = FaceRecognizer(db)
    # traffic_detector = TrafficSignalDetector()  # DISABLED: Too many false positives
    item_tracker = ItemTracker(db)
    object_memory = ObjectMemory(db)
    spatial_nav = SpatialNavigator(frame_width=config.CAMERA_WIDTH, frame_height=config.CAMERA_HEIGHT)
    audio = get_audio_feedback(use_gtts=False)
    
    # Initialize FREE navigation system (OpenStreetMap)
    gps_tracker = GPSTracker(mock_mode=config.GPS_MOCK_MODE)
    osm_nav = OSMNavigator()
    ultrasonic = UltrasonicSensor(mock_mode=config.ULTRASONIC_MOCK_MODE)
    obstacle_fusion = ObstacleFusion(ultrasonic, frame_width=config.CAMERA_WIDTH, frame_height=config.CAMERA_HEIGHT)
    nav_engine = NavigationEngine(
        gps_tracker,
        osm_nav,
        announcement_distance=config.NAVIGATION_ANNOUNCEMENT_DISTANCE,
        reroute_distance=config.NAVIGATION_REROUTE_DISTANCE,
        arrival_distance=config.NAVIGATION_ARRIVAL_DISTANCE
    )
    
    # Start background services
    gps_tracker.start()
    ultrasonic.start()
    
    # Initialize voice command system
    voice_system = VoiceCommandSystem()
    voice_initialized = voice_system.initialize()
    if voice_initialized:
        voice_system.start()
        print("✓ Voice command system initialized!")
    else:
        print("⚠ Voice command system not available (microphone issue)")
        voice_system = None
    
    print("✓ All modules initialized!")
    
    print("\nOpening webcam...")
    print("Controls:")
    print("  Q - Quit")
    print("  E - Enroll face")
    print("  R - Remember object (enroll custom object)")
    print("  L - List remembered objects")
    print("  F - Find item/object location")
    print("  A - Toggle audio feedback")
    print("  N - Toggle navigation guidance (obstacle avoidance)")
    print("  G - GPS Navigation (navigate to destination)")
    print("  M - Advance mock GPS (simulate movement)")
    print("  SPACEBAR - Voice command (button simulation)")
    print("  V - Toggle voice mode")
    if voice_initialized:
        print("\n✓ Voice commands available:")
        print("  - 'Remember this [object]' - Enroll object")
        print("  - 'Find my [object]' - Locate object")
        print("  - 'Navigate to [destination]' - GPS navigation")
        print("  - 'Navigate' / 'Stop navigation' - Control obstacle guidance")
        print("  - 'List all items' - Show remembered objects")
    
    # Fix 1: Use threaded CameraReader instead of blocking cv2.VideoCapture.
    # The reader thread grabs frames continuously in the background; cam.read()
    # returns the latest frame instantly without blocking the AI pipeline.
    cam = CameraReader(cam_index=0, width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT, fps=30,
                       backend=cv2.CAP_DSHOW, warmup_frames=5)
    try:
        cam.start()
    except RuntimeError as e:
        print(f"✗ Could not open webcam: {e}")
        db.close()
        return
    
    # Wait until the first frame is ready (max 3 s)
    deadline = time.time() + 3.0
    while True:
        frame, _fid = cam.read()
        if frame is not None:
            break
        if time.time() > deadline:
            print("✗ No frame from webcam within 3 s")
            cam.stop()
            db.close()
            return
        time.sleep(0.01)
    print("✓ Webcam opened (threaded, first frame received)")
    
    last_cam_fid = _fid  # track which frame_id the inference loop last processed
    
    frame_count = 0
    start_time = time.time()
    audio_enabled = True
    navigation_enabled = True  # Enable spatial navigation by default
    voice_mode_enabled = voice_initialized  # Enable voice if available
    seen_faces = set()  # Track faces already announced
    face_announce_interval = 5.0  # Seconds between re-announcing same person
    last_face_announce = {}  # Track when each person was last announced
    last_recognized_faces = []  # Store last recognized face names for nearby context
    last_faces = []  # Store last recognized faces for persistent drawing across skipped frames
    last_navigation_announce = 0  # Track when navigation was last announced
    navigation_announce_cooldown = 2.0  # Seconds between navigation announcements
    
    # Object memory state
    enrollment_mode = False
    enrollment_target_obj = None
    enrollment_frames_captured = 0
    enrollment_buffer = []
    last_remembered_announce = {}  # Track announcements to avoid spam
    remember_announce_cooldown = 3.0  # Seconds
    last_remembered_objects = []
    last_object_to_custom_name = {}
    last_objects = []  # Store last detected objects to persist across frame skips
    
    # Timing instrumentation stats (in milliseconds)
    timing_stats = {
        'read': 0.0,
        'yolo': 0.0,
        'obj_memory': 0.0,
        'face': 0.0,
        'nav': 0.0,
        'display': 0.0
    }
    
    # Voice command state
    current_frame = None  # Store current frame for voice commands
    current_objects = []  # Store current objects for voice commands
    
    # GPS Navigation state
    gps_navigation_active = False
    gps_nav_step = 0
    last_gps_update = 0
    gps_update_interval = 2.0  # Update every 2 seconds
    last_gps_announcement = None
    
    # Register voice command handlers
    if voice_system:
        def handle_voice_remember(params):
            """Handle 'remember this [object]' command"""
            nonlocal enrollment_mode, enrollment_target_obj
            
            if enrollment_mode:
                audio.speak_status("Already enrolling an object")
                return
                
            object_name = params.get('object_name', '')
            if not object_name:
                audio.speak_status("Please specify an object name")
                return
            
            # Match object name to detected objects
            matching_objects = [o for o in current_objects if object_name.lower() in o['class_name'].lower()]
            
            if matching_objects:
                obj = matching_objects[0]  # Take first match
                custom_name = f"my{object_name}"  # e.g., "myphone"
                
                try:
                    object_memory.start_enrollment(custom_name, obj['class_name'])
                    enrollment_mode = True
                    enrollment_target_obj = {
                        'custom_name': custom_name,
                        'yolo_class': obj['class_name']
                    }
                    audio.speak_social(f"Starting enrollment for {custom_name}. Keep it steady.")
                    print(f"✓ Voice: Enrolling '{custom_name}'")
                except ValueError as e:
                    audio.speak_status(f"Error: {e}")
            else:
                audio.speak_status(f"No {object_name} detected in view")
        
        def handle_voice_find(params):
            """Handle 'find my [object]' command"""
            object_name = params.get('object_name', '')
            if not object_name:
                audio.speak_status("Please specify what to find")
                return
            
            # Try remembered objects
            location = object_memory.get_object_location(object_name)
            if location:
                nearby = location['context'].get('nearby_objects', []) if location['context'] else []
                nearby_str = f" near {', '.join(nearby)}" if nearby else ""
                msg = f"Your {object_name} last seen{nearby_str}"
                audio.speak_navigational(msg)
                print(f"✓ Voice: {msg}")
            else:
                # Try item tracker
                loc = item_tracker.query_item_location(object_name)
                if loc:
                    msg = f"Your {object_name} on the {loc}"
                    audio.speak_navigational(msg)
                    print(f"✓ Voice: {msg}")
                else:
                    audio.speak_status(f"Haven't seen {object_name}")
        
        def handle_voice_navigate(params):
            """Handle 'navigate' command"""
            nonlocal navigation_enabled
            if not navigation_enabled:
                navigation_enabled = True
                audio.speak_social("Navigation enabled")
                print("✓ Voice: Navigation ON")
            else:
                audio.speak_status("Navigation already active")
        
        def handle_voice_stop_navigate(params):
            """Handle 'stop navigation' command"""
            nonlocal navigation_enabled
            if navigation_enabled:
                navigation_enabled = False
                audio.speak_social("Navigation disabled")
                print("✓ Voice: Navigation OFF")
            else:
                audio.speak_status("Navigation already off")
        
        def handle_voice_list(params):
            """Handle 'list all items' command"""
            remembered_list = object_memory.list_remembered_objects()
            if remembered_list:
                items = ", ".join([name for name, _ in remembered_list])
                audio.speak_social(f"I remember: {items}")
                print(f"✓ Voice: Listed {len(remembered_list)} items")
            else:
                audio.speak_status("No items remembered yet")
        
        def handle_voice_help(params):
            """Handle 'help' command"""
            audio.speak_social("Say: remember this phone, find my keys, navigate to destination, stop navigation, or list all items")
        
        def handle_voice_navigate_to(params):
            """Handle 'navigate to [destination]' command for GPS navigation"""
            nonlocal gps_navigation_active, gps_nav_step, last_gps_announcement
            
            destination = params.get('destination', '')
            if not destination:
                audio.speak_status("Where do you want to go?")
                return
            
            audio.speak_social(f"Navigating to {destination}")
            print(f"\n✓ Voice: Starting GPS navigation to {destination}")
            
            # Geocode destination
            dest_coords = osm_nav.geocode_address(destination + ", Bangalore, India")
            if not dest_coords:
                audio.speak_status("Could not find destination")
                print(f"✗ Geocoding failed for {destination}")
                return
            
            # Get current position
            current_pos = gps_tracker.get_position()
            if not current_pos:
                audio.speak_status("Waiting for GPS signal")
                print("✗ No GPS signal")
                return
            
            # Get directions
            route = osm_nav.get_directions(current_pos, dest_coords, mode='foot')
            if not route:
                audio.speak_status("Could not find route")
                print("✗ Routing failed")
                return
            
            # Store route and activate navigation
            osm_nav.current_route = route
            gps_navigation_active = True
            gps_nav_step = 0
            last_gps_announcement = None
            
            # Announce route
            audio.speak_social(f"Route found. {route['total_distance']}. {route['total_duration']}.")
            print(f"✓ GPS Navigation started: {len(route['steps'])} steps")
            print(f"  Distance: {route['total_distance']}")
            print(f"  Duration: {route['total_duration']}")
        
        def handle_voice_stop_gps(params):
            """Handle 'stop GPS navigation' command"""
            nonlocal gps_navigation_active
            if gps_navigation_active:
                gps_navigation_active = False
                audio.speak_social("GPS navigation stopped")
                print("✓ Voice: GPS navigation stopped")
            else:
                audio.speak_status("No GPS navigation active")
        
        # Register all handlers
        voice_system.register_command_handler('remember', handle_voice_remember)
        voice_system.register_command_handler('find', handle_voice_find)
        voice_system.register_command_handler('navigate', handle_voice_navigate)
        voice_system.register_command_handler('stop_navigate', handle_voice_stop_navigate)
        voice_system.register_command_handler('list', handle_voice_list)
        voice_system.register_command_handler('help', handle_voice_help)
    
    while True:
        # Fix 1: non-blocking read from the threaded CameraReader.
        # 'read' timing now measures the time to get the latest frame from the
        # reader thread (usually <1 ms when a new frame is waiting), NOT the raw
        # I/O block time of cap.read() — those are different things.
        t0 = time.time()
        while True:
            frame, cam_fid = cam.read()
            if cam_fid != last_cam_fid:
                break          # a genuinely new frame is ready
            if frame is None:  # shouldn't happen after start(), but be safe
                time.sleep(0.001)
                continue
            time.sleep(0.001)  # yield briefly; don't pin the CPU
        t1 = time.time()
        last_cam_fid = cam_fid
        
        if frame is None:
            break
        
        frame_count += 1
        timing_stats['read'] += (t1 - t0) * 1000.0
        
        # Create display frame
        display = frame.copy()
        
        # 1. Object Detection
        t_yolo_0 = time.time()
        if frame_count % YOLO_FRAME_SKIP == 0:
            objects = object_detector.detect(frame)
            last_objects = objects
        else:
            objects = last_objects
        t_yolo_1 = time.time()
        timing_stats['yolo'] += (t_yolo_1 - t_yolo_0) * 1000.0
        
        display = object_detector.draw_detections(display, objects)
        
        # Store current state for voice commands
        current_frame = frame.copy()
        current_objects = objects
        
        # 1b. Recognize remembered objects
        t_mem_0 = time.time()
        if not enrollment_mode and frame_count % OBJECT_RECOGNITION_FRAME_INTERVAL == 0:
            remembered_objects = []
            object_to_custom_name = {}  # Map object IDs to custom names
            for obj in objects:
                bbox = (obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3])
                match = object_memory.recognize_object(frame, bbox, obj['class_name'])
                if match:
                    custom_name, confidence = match
                    remembered_objects.append({
                        'custom_name': custom_name,
                        'confidence': confidence,
                        'bbox': obj['bbox']
                    })
                    
                    # Map this object to its custom name
                    obj_id = id(obj)
                    object_to_custom_name[obj_id] = custom_name
                    
                    # Audio announcement with cooldown
                    if audio_enabled:
                        current_time = time.time()
                        if custom_name not in last_remembered_announce or \
                           (current_time - last_remembered_announce[custom_name]) > remember_announce_cooldown:
                            audio.speak_social(f"Found your {custom_name}")
                            last_remembered_announce[custom_name] = current_time
                    
                    # Update sighting with custom names for nearby objects and faces
                    nearby = []
                    for o in objects:
                        if o != obj:
                            # Use custom name if recognized, otherwise use YOLO class
                            nearby_name = object_to_custom_name.get(id(o), o['class_name'])
                            nearby.append(nearby_name)
                    
                    # Add recognized face names to nearby list
                    nearby.extend(last_recognized_faces)
                    
                    object_memory.update_object_sighting(
                        custom_name=custom_name,
                        image=frame,
                        nearby_objects=nearby[:5],
                        confidence=confidence
                    )
            last_remembered_objects = remembered_objects
            last_object_to_custom_name = object_to_custom_name
        else:
            remembered_objects = last_remembered_objects
            object_to_custom_name = last_object_to_custom_name
        t_mem_1 = time.time()
        timing_stats['obj_memory'] += (t_mem_1 - t_mem_0) * 1000.0

        # Draw custom labels for remembered objects on display frame
        for rem_obj in remembered_objects:
            custom_name = rem_obj['custom_name']
            confidence = rem_obj['confidence']
            x1, y1, x2, y2 = rem_obj['bbox']
            label = f"{custom_name} ({confidence:.0%})"
            cv2.rectangle(display, (x1, y1-30), (x1+len(label)*10, y1), (255, 0, 255), -1)
            cv2.putText(display, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 2. Face Recognition (every FACE_RECOGNITION_FRAME_INTERVAL frames)
        t_face_0 = time.time()
        if frame_count % FACE_RECOGNITION_FRAME_INTERVAL == 0:
            faces = face_recognizer.recognize_faces(frame)
            last_faces = faces
            
            # Store recognized face names for nearby context
            last_recognized_faces = [face['name'] for face in faces if face['name'] != 'Unknown']
            
            # Audio for recognized faces (with cooldown to avoid spam)
            if audio_enabled:
                current_time = time.time()
                for face in faces:
                    if face['name'] != 'Unknown':
                        # Only announce if not announced recently
                        if face['name'] not in last_face_announce or \
                           (current_time - last_face_announce[face['name']]) > face_announce_interval:
                            audio.speak_social(f"Hello {face['name']}")
                            last_face_announce[face['name']] = current_time
        else:
            faces = last_faces
            
        display = face_recognizer.draw_faces(display, faces)
        t_face_1 = time.time()
        timing_stats['face'] += (t_face_1 - t_face_0) * 1000.0
        
        # 3. Traffic Signal Detection - DISABLED (too many false positives)
        # traffic_signal = traffic_detector.detect_signal(frame)
        # if traffic_signal:
        #     display = traffic_detector.draw_signal(display, traffic_signal)
        #     
        #     # Emergency audio for red light
        #     if audio_enabled and traffic_signal['signal_type'] == 'RED':
        #         audio.speak_emergency("STOP - Red Light")
        
        # 4. Item Tracking
        item_updates = item_tracker.update_tracking(objects)
        if item_updates and audio_enabled:
            for update in item_updates:
                audio.speak_status(f"{update['item']} on {update['location']}")
        
        # 5. Object Enrollment Mode
        if enrollment_mode and enrollment_target_obj:
            # Find the target object in current detections
            target_detections = [o for o in objects if o['class_name'] == enrollment_target_obj['yolo_class']]
            
            if target_detections:
                # Use the first detection
                det = target_detections[0]
                bbox = (det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3])
                
                # Add to enrollment
                enrollment_frames_captured = object_memory.add_enrollment_frame(frame, bbox)
                
                # Visual feedback
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 3)
                label = f"Recording {enrollment_frames_captured}/5"
                cv2.putText(display, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Check if done
                if enrollment_frames_captured >= 5:
                    success = object_memory.finish_enrollment()
                    if success:
                        print(f"\u2713 Enrolled '{enrollment_target_obj['custom_name']}'")
                        if audio_enabled:
                            audio.speak_social(f"Remembered your {enrollment_target_obj['custom_name']}")
                    else:
                        print(f"\u2717 Enrollment failed - try again")
                    enrollment_mode = False
                    enrollment_target_obj = None
                    enrollment_frames_captured = 0
            else:
                # Show message if object not in view
                cv2.putText(display, f"Waiting for {enrollment_target_obj['yolo_class']}...",
                           (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 5. Spatial Navigation for Obstacles
        t_nav_0 = time.time()
        if navigation_enabled and audio_enabled and not enrollment_mode:
            # Generate spatial guidance
            guidance = spatial_nav.generate_guidance(objects)
            
            # Announce highest priority guidance (with cooldown)
            current_time = time.time()
            if guidance and (current_time - last_navigation_announce) > navigation_announce_cooldown:
                # Take the top priority message
                top_guidance = guidance[0]
                
                # Map priority to audio level
                if top_guidance['priority'] == 1:
                    audio.speak_emergency(top_guidance['message'])
                else:
                    audio.speak_navigational(top_guidance['message'])
                
                last_navigation_announce = current_time
            
            # Draw guidance overlay on screen
            if guidance:
                y_offset = 100
                for g in guidance[:3]:  # Show top 3 guidance messages
                    color = (0, 0, 255) if g['priority'] == 1 else (0, 165, 255)  # Red for emergency, orange for normal
                    cv2.putText(display, g['message'], (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 25
        t_nav_1 = time.time()
        timing_stats['nav'] += (t_nav_1 - t_nav_0) * 1000.0
        
        # 6. GPS Navigation Update
        if gps_navigation_active and osm_nav.current_route:
            current_time = time.time()
            
            # Update GPS navigation every 2 seconds
            if (current_time - last_gps_update) >= gps_update_interval:
                last_gps_update = current_time
                
                # Get current position
                current_pos = gps_tracker.get_position()
                if current_pos:
                    # Get current instruction
                    if gps_nav_step < len(osm_nav.current_route['steps']):
                        step = osm_nav.current_route['steps'][gps_nav_step]
                        instruction = step['instruction']
                        
                        # Calculate distance to waypoint
                        end_lat = step['end_location']['lat']
                        end_lon = step['end_location']['lng']
                        
                        import math
                        # Haversine distance calculation
                        R = 6371000  # Earth radius in meters
                        lat1, lon1 = current_pos
                        lat2, lon2 = end_lat, end_lon
                        phi1 = math.radians(lat1)
                        phi2 = math.radians(lat2)
                        delta_phi = math.radians(lat2 - lat1)
                        delta_lambda = math.radians(lon2 - lon1)
                        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
                        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                        distance = R * c
                        
                        # Check if should announce
                        if distance <= config.NAVIGATION_ANNOUNCEMENT_DISTANCE:
                            if last_gps_announcement != gps_nav_step:
                                last_gps_announcement = gps_nav_step
                                if audio_enabled:
                                    audio.speak_navigational(f"In {int(distance)} meters, {instruction}")
                                print(f"\n🔊 GPS: {instruction} ({distance:.0f}m)")
                        
                        # Display navigation info on screen
                        total_steps = len(osm_nav.current_route['steps'])
                        nav_text = f"GPS NAV [{gps_nav_step + 1}/{total_steps}]: {instruction[:60]}"
                        cv2.rectangle(display, (5, 125), (635, 175), (0, 0, 0), -1)  # Black background
                        cv2.putText(display, nav_text, (10, 145),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        cv2.putText(display, f"Distance: {distance:.0f}m", (10, 165),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        
                        # Advance to next step if close
                        if distance < 10:
                            if gps_nav_step + 1 < len(osm_nav.current_route['steps']):
                                gps_nav_step += 1
                                last_gps_announcement = None
                                print(f"✓ Advanced to step {gps_nav_step + 1}")
                            else:
                                # Arrived!
                                if audio_enabled:
                                    audio.speak_social("You have arrived at your destination")
                                print("\n✓ GPS Navigation: ARRIVED!")
                                gps_navigation_active = False
                    
                    # Enhanced obstacle detection during GPS navigation
                    if audio_enabled:
                        # Use obstacle fusion for precise warnings
                        warnings = obstacle_fusion.generate_obstacle_warnings(objects)
                        if warnings:
                            highest = warnings[0]
                            if highest['priority'] <= 2:  # Emergency or High warning
                                # Don't spam - only announce critical obstacles
                                audio.speak_emergency(highest['message'])
        
        # Display stats
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Count recognized faces
        recognized_count = len([f for f in faces if f.get('name') != 'Unknown']) if 'faces' in locals() and faces else 0
        
        stats = [
            f"FPS: {fps:.1f}",
            f"Objects: {len(objects)}",
            f"Remembered: {len(remembered_objects)}",
            f"Faces: {recognized_count}",
            f"Audio: {'ON' if audio_enabled else 'OFF'}",
            f"Navigation: {'ON' if navigation_enabled else 'OFF'}",
            f"Voice: {'ON' if voice_mode_enabled and voice_system else 'OFF'}",
            f"Queue: {audio.get_queue_size()}"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(
                display,
                stat,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Controls
        if enrollment_mode:
            cv2.putText(
                display,
                f"ENROLLING: {enrollment_target_obj['custom_name']} - Hold steady!",
                (10, display.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )
        
        cv2.putText(
            display,
            "Q:Quit | E:Face | R:Remember | L:List | F:Find | A:Audio | N:Nav | V:Voice | SPACE:Talk",
            (10, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )
        
        t_disp_0 = time.time()
        cv2.imshow('VisionMate Integration Test', display)
        key = cv2.waitKey(1) & 0xFF
        t_disp_1 = time.time()
        timing_stats['display'] += (t_disp_1 - t_disp_0) * 1000.0
        
        if frame_count % 10 == 0:
            avg_read = timing_stats['read'] / 10.0
            avg_yolo = timing_stats['yolo'] / 10.0
            avg_mem = timing_stats['obj_memory'] / 10.0
            avg_face = timing_stats['face'] / 10.0
            avg_nav = timing_stats['nav'] / 10.0
            avg_disp = timing_stats['display'] / 10.0
            print(f"[Frame {frame_count}] cap.read: {avg_read:.1f}ms | yolo: {avg_yolo:.1f}ms | face: {avg_face:.1f}ms | obj_memory: {avg_mem:.1f}ms | nav: {avg_nav:.1f}ms | display: {avg_disp:.1f}ms")
            for k in timing_stats:
                timing_stats[k] = 0.0
        
        if key == ord('q'):
            break
        elif key == ord('e'):
            # Enroll face
            name = input("\nEnter name for enrollment: ").strip()
            if name:
                success = face_recognizer.enroll_face(frame, name)
                if success:
                    print(f"✓ Enrolled {name}")
                    if audio_enabled:
                        audio.speak_social(f"Enrolled {name}")
        elif key == ord('r'):
            # Remember object
            if not enrollment_mode:
                # Show available objects
                if objects:
                    print("\nDetected objects:")
                    for i, obj in enumerate(objects):
                        print(f"  {i+1}. {obj['class_name']}")
                    
                    # Get selection
                    try:
                        idx = int(input("Select object number: ").strip()) - 1
                        if 0 <= idx < len(objects):
                            obj = objects[idx]
                            custom_name = input("Enter custom name (e.g., 'house keys'): ").strip()
                            
                            if custom_name:
                                try:
                                    object_memory.start_enrollment(custom_name, obj['class_name'])
                                    enrollment_mode = True
                                    enrollment_target_obj = {
                                        'custom_name': custom_name,
                                        'yolo_class': obj['class_name']
                                    }
                                    print(f"\u2713 Starting enrollment for '{custom_name}'...")
                                    print("  Keep object in view, capturing 5 frames...")
                                except ValueError as e:
                                    print(f"\u2717 {e}")
                    except (ValueError, IndexError):
                        print("\u2717 Invalid selection")
                else:
                    print("\u2717 No objects detected")
        elif key == ord('l'):
            # List remembered objects
            remembered_list = object_memory.list_remembered_objects()
            print(f"\nRemembered Objects ({len(remembered_list)}):")
            if remembered_list:
                for name, yolo_class in remembered_list:
                    print(f"  - {name} ({yolo_class})")
            else:
                print("  (none)")
        elif key == ord('f'):
            # Query object/item
            item = input("\nEnter item/object to find: ").strip()
            if item:
                # Try remembered objects first
                location = object_memory.get_object_location(item)
                if location:
                    nearby = location['context'].get('nearby_objects', []) if location['context'] else []
                    nearby_str = f" near {', '.join(nearby)}" if nearby else ""
                    msg = f"Your {item} last seen{nearby_str}"
                    print(f"✓ {msg}")
                    print(f"  Time: {location['timestamp']}")
                    if audio_enabled:
                        audio.speak_navigational(msg)
                else:
                    # Try regular item tracker
                    loc = item_tracker.query_item_location(item)
                    if loc:
                        msg = f"Your {item} are on the {loc}"
                        print(f"✓ {msg}")
                        if audio_enabled:
                            audio.speak_navigational(msg)
                    else:
                        print(f"✗ Haven't seen {item}")
        elif key == ord('a'):
            audio_enabled = not audio_enabled
            print(f"\n🔊 Audio feedback: {'ON' if audio_enabled else 'OFF'}")
        elif key == ord('n'):
            navigation_enabled = not navigation_enabled
            print(f"\n🧭 Navigation guidance: {'ON' if navigation_enabled else 'OFF'}")
        elif key == ord('g'):
            # Start GPS navigation
            if not gps_navigation_active:
                destination = input("\nEnter destination (e.g., 'Majestic'): ").strip()
                if destination:
                    print(f"\n🗺️ Starting GPS navigation to {destination}...")
                    if audio_enabled:
                        audio.speak_social(f"Navigating to {destination}")
                    
                    # Geocode destination
                    dest_coords = osm_nav.geocode_address(destination + ", Bangalore, India")
                    if not dest_coords:
                        print("✗ Could not find destination")
                        if audio_enabled:
                            audio.speak_status("Could not find destination")
                    else:
                        # Get current position
                        current_pos = gps_tracker.get_position()
                        if not current_pos:
                            print("✗ Waiting for GPS signal")
                            if audio_enabled:
                                audio.speak_status("Waiting for GPS signal")
                        else:
                            # Get directions
                            route = osm_nav.get_directions(current_pos, dest_coords, mode='foot')
                            if route:
                                osm_nav.current_route = route
                                gps_navigation_active = True
                                gps_nav_step = 0
                                last_gps_announcement = None
                                print(f"✓ Route found: {route['total_distance']}, {route['total_duration']}")
                                print(f"  Steps: {len(route['steps'])}")
                                if audio_enabled:
                                    audio.speak_social(f"Route found. {route['total_distance']}. {route['total_duration']}.")
                            else:
                                print("✗ Could not find route")
                                if audio_enabled:
                                    audio.speak_status("Could not find route")
            else:
                # Stop GPS navigation
                gps_navigation_active = False
                print("\n🗺️ GPS navigation stopped")
                if audio_enabled:
                    audio.speak_status("GPS navigation stopped")
        elif key == ord('m'):
            # Advance mock GPS position
            if config.GPS_MOCK_MODE:
                gps_tracker.advance_mock_position()
                new_pos = gps_tracker.get_position()
                print(f"\n📍 Mock GPS advanced to: {new_pos[0]:.6f}, {new_pos[1]:.6f}")
            else:
                print("\n⚠ GPS is in REAL mode, not mock mode")
        elif key == ord('v'):
            # Toggle voice mode
            if voice_system:
                voice_mode_enabled = not voice_mode_enabled
                status = "ON" if voice_mode_enabled else "OFF"
                print(f"\n🎤 Voice mode: {status}")
                if audio_enabled:
                    audio.speak_status(f"Voice mode {status}")
            else:
                print("\n⚠ Voice system not available")
        elif key == 32:  # SPACEBAR
            # Trigger voice input (button simulation)
            if voice_system and voice_mode_enabled:
                print("\n🎤 Listening for voice command...")
                if audio_enabled:
                    audio.speak_status("Listening")
                voice_system.trigger_voice_input()
            else:
                if not voice_system:
                    print("\n⚠ Voice system not available")
                else:
                    print("\n⚠ Voice mode is OFF - press V to enable")
    
    # Fix 1: stop the background reader thread and release the webcam handle.
    cam.stop()
    cv2.destroyAllWindows()
    
    # Cleanup navigation services
    gps_tracker.stop()
    ultrasonic.stop()
    
    # Cleanup voice system
    if voice_system:
        voice_system.stop()
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {fps:.2f}")
    print(f"Total time: {elapsed:.2f}s")
    print("=" * 60)
    
    db.close()


def test_priority_system():
    """Test audio feedback priority system"""
    print("=" * 60)
    print("TESTING AUDIO PRIORITY SYSTEM")
    print("=" * 60)
    
    audio = get_audio_feedback(use_gtts=False)
    
    print("\nTesting priority hierarchy...")
    print("Order: STATUS -> NAVIGATIONAL -> SOCIAL -> EMERGENCY")
    
    # Queue messages in different priorities
    audio.speak_status("Battery low")
    time.sleep(0.5)
    audio.speak_navigational("Chair ahead")
    time.sleep(0.5)
    audio.speak_social("Hello John")
    time.sleep(0.5)
    audio.speak_emergency("STOP - Red Light")  # Should interrupt
    
    print("\nEmergency message should play immediately and interrupt others")
    print("✓ Priority test queued!")
    
    # Wait for audio to finish
    time.sleep(10)


def test_database_integration():
    """Test database with all components"""
    print("=" * 60)
    print("TESTING DATABASE INTEGRATION")
    print("=" * 60)
    
    db = VisionMateDB()
    
    # Check faces
    faces = db.get_all_faces()
    print(f"\n✓ Faces in database: {len(faces)}")
    for face_id, name, _ in faces:
        print(f"  - {name} (ID: {face_id})")
    
    # Check item locations
    items = db.get_all_item_locations()
    print(f"\n✓ Item locations in database: {len(items)}")
    for item, location, timestamp in items:
        print(f"  - {item} on {location} (at {timestamp})")
    
    # Check remembered objects
    remembered = db.get_all_remembered_objects()
    print(f"\n✓ Remembered objects in database: {len(remembered)}")
    for obj_id, name, yolo_class in remembered:
        print(f"  - {name} ({yolo_class}) [ID: {obj_id}]")
    
    db.close()
    print("\n✓ Database integration test complete!")


if __name__ == "__main__":
    print("\nVISIONMATE INTEGRATION TEST SUITE")
    print("=" * 60)
    
    print("\nSelect test mode:")
    print("1. Full pipeline with webcam (recommended)")
    print("2. Audio priority system test")
    print("3. Database integration test")
    print("4. All tests")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        test_full_pipeline_webcam()
    elif choice == '2':
        test_priority_system()
    elif choice == '3':
        test_database_integration()
    elif choice == '4':
        test_database_integration()
        print("\n")
        test_priority_system()
        print("\n")
        test_full_pipeline_webcam()
    else:
        print("Running full pipeline test...")
        test_full_pipeline_webcam()
