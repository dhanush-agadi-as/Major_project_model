"""
Laptop ML Server for VisionMate Smart Glasses
Runs on: Laptop (with GPU for fast processing)
Receives: Video frames + sensor data from Raspberry Pi via WiFi
Processes: YOLOv8, InsightFace, Object Memory, Navigation
Returns: Detection results + navigation instructions

Start server: python wireless/laptop_server.py
Your laptop IP will be displayed - give this to RPi client
"""

import asyncio
import websockets
import cv2
import json
import base64
import time
import numpy as np
import logging
import socket
import sys
import os

# Add parent directory to path to import VisionMate modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from object_detector import ObjectDetector
from face_recognizer import FaceRecognizer
from item_tracker import ItemTracker
from object_memory import ObjectMemory
from spatial_navigator import SpatialNavigator
from navigation_engine import NavigationEngine
from gps_tracker import GPSTracker
from osm_navigator import OSMNavigator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaptopMLServer:
    """ML processing server running on laptop"""
    
    def __init__(self, port=8765):
        self.port = port
        self.host = "0.0.0.0"  # Listen on all interfaces
        
        # VisionMate modules
        self.object_detector = None
        self.face_recognizer = None
        self.item_tracker = None
        self.object_memory = None
        self.spatial_navigator = None
        self.navigation_engine = None
        self.gps_tracker = None
        self.osm_navigator = None
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        
        # Navigation state
        self.navigation_active = True  # Obstacle avoidance ON by default
        self.gps_navigation_active = False
        self.current_route = None
        
        # Display
        self.show_display = True  # Show camera feed on laptop
        self.last_frame = None
        self.last_result = None
        
        # Object Memory enrollment state
        self.enrollment_active = False
        self.enrollment_name = None
        self.enrollment_class = None
        self.enrollment_frames = 0
    def initialize(self):
        """Initialize all ML models"""
        logger.info("="*60)
        logger.info("VISIONMATE LAPTOP ML SERVER")
        logger.info("="*60)
        logger.info("\nInitializing ML models (this may take a minute)...\n")
        
        try:
            # Object detection
            logger.info("Loading YOLOv8 model...")
            self.object_detector = ObjectDetector()
            logger.info("[OK] Object Detector ready")
            
            # Face recognition
            logger.info("Loading InsightFace model...")
            self.face_recognizer = FaceRecognizer()
            logger.info("[OK] Face Recognizer ready")
            
            # Item tracking
            self.item_tracker = ItemTracker()
            logger.info("[OK] Item Tracker ready")
            
            # Object memory
            self.object_memory = ObjectMemory()
            logger.info("[OK] Object Memory ready")
            
            # Spatial navigation
            self.spatial_navigator = SpatialNavigator(frame_width=640, frame_height=480)
            logger.info("[OK] Spatial Navigator ready")
            
            # GPS navigation
            self.gps_tracker = GPSTracker(mock_mode=True)
            self.osm_navigator = OSMNavigator()
            self.navigation_engine = NavigationEngine(self.gps_tracker, self.osm_navigator)
            logger.info("[OK] Navigation Engine ready")
            
            logger.info("\n" + "="*60)
            logger.info("[OK] ALL ML MODELS LOADED SUCCESSFULLY!")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def process_frame(self, websocket):
        """Handle incoming connection from Raspberry Pi"""
        client_ip = websocket.remote_address[0]
        logger.info(f"\n{'='*60}")
        logger.info(f"[OK] Raspberry Pi connected from: {client_ip}")
        logger.info(f"{'='*60}")
        logger.info(f"\nKEYBOARD CONTROLS (click on the display window first):")
        logger.info(f"  Q = Quit display")
        logger.info(f"  R = Start Remember Object enrollment")
        logger.info(f"  F = Register new face")
        logger.info(f"  N = Toggle navigation/obstacle warnings")
        logger.info(f"{'='*60}\n")
        
        try:
            async for message in websocket:
                try:
                    # Parse incoming data
                    data = json.loads(message)
                    
                    # Decode frame
                    frame_data = base64.b64decode(data['frame'])
                    frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    
                    # Get sensor data
                    ultrasonic_distance = data.get('ultrasonic', 999.0)
                    button_pressed = data.get('button', False)
                    timestamp = data.get('timestamp', time.time())
                    
                    # Process frame with ML models
                    result = self.process_ml(frame, ultrasonic_distance, button_pressed)
                    
                    # Add timestamp for latency calculation
                    result['timestamp'] = timestamp
                    
                    # Send results back to RPi
                    await websocket.send(json.dumps(result))
                    
                    # Show display on laptop with annotations
                    if self.show_display:
                        self.display_frame(frame, result, ultrasonic_distance)
                    
                    # Statistics
                    self.frame_count += 1
                    if self.frame_count % 30 == 0:
                        elapsed = time.time() - self.start_time
                        fps = self.frame_count / elapsed
                        latency = (time.time() - timestamp) * 1000
                        obj_count = len(result['objects'])
                        face_count = len(result['faces'])
                        logger.info(f"FPS: {fps:.1f} | Latency: {latency:.0f}ms | Objects: {obj_count} | Faces: {face_count} | Dist: {ultrasonic_distance:.0f}cm")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Raspberry Pi disconnected: {client_ip}")
        finally:
            cv2.destroyAllWindows()
    
    def display_frame(self, frame, result, ultrasonic_distance):
        """Display annotated frame on laptop screen"""
        display = frame.copy()
        
        # Draw object detections (green boxes)
        for obj in result['objects']:
            x1, y1, x2, y2 = obj['bbox']
            label = f"{obj['class']} {obj['confidence']:.0%}"
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw face detections (blue boxes)
        for face in result['faces']:
            x1, y1, x2, y2 = face['bbox']
            name = face['name']
            color = (255, 150, 0) if name != 'Unknown' else (0, 0, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, name, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw recognized objects (yellow boxes)
        for mem_obj in result.get('remembered', []):
            x1, y1, x2, y2 = mem_obj['bbox']
            label = f"★ {mem_obj['name']} ({mem_obj['confidence']:.0%})"
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(display, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw ultrasonic distance bar at top
        bar_width = int(min(ultrasonic_distance, 200) / 200 * display.shape[1])
        bar_color = (0, 255, 0) if ultrasonic_distance > 100 else (0, 165, 255) if ultrasonic_distance > 50 else (0, 0, 255)
        cv2.rectangle(display, (0, 0), (bar_width, 20), bar_color, -1)
        cv2.putText(display, f"Dist: {ultrasonic_distance:.0f}cm", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw navigation message at bottom
        if result.get('navigation'):
            nav_text = result['navigation']
            cv2.rectangle(display, (0, display.shape[0] - 40), (display.shape[1], display.shape[0]), (0, 0, 0), -1)
            cv2.putText(display, nav_text, (10, display.shape[0] - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw enrollment status if active
        if self.enrollment_active:
            cv2.rectangle(display, (0, 25), (display.shape[1], 55), (0, 0, 200), -1)
            cv2.putText(display, f"ENROLLING: {self.enrollment_name} ({self.enrollment_frames}/3 frames) - Press SPACE to capture", 
                       (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw FPS
        elapsed = time.time() - self.start_time
        fps = self.frame_count / max(elapsed, 0.001)
        cv2.putText(display, f"FPS: {fps:.1f}", (display.shape[1] - 100, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw controls hint
        cv2.putText(display, "Q:quit R:remember F:face N:nav", (5, display.shape[0] - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Show the frame
        cv2.imshow('VisionMate - Live Feed', display)
        key = cv2.waitKey(1) & 0xFF
        
        # Handle keyboard controls
        if key == ord('q'):
            self.show_display = False
            cv2.destroyAllWindows()
        elif key == ord('n'):
            self.navigation_active = not self.navigation_active
            logger.info(f"Navigation: {'ON' if self.navigation_active else 'OFF'}")
        elif key == ord('r'):
            # Start object enrollment
            self.start_enrollment_mode()
        elif key == ord('f'):
            # Register face
            self.register_face_mode(frame)
        elif key == ord(' ') and self.enrollment_active:
            # Capture enrollment frame
            self.capture_enrollment_frame(frame)
    
    def start_enrollment_mode(self):
        """Start remember object enrollment via terminal input"""
        name = input("\nEnter name for object (e.g., 'my keys'): ").strip()
        yolo_class = input("YOLO class of object (e.g., 'keys', 'bottle', 'cell phone'): ").strip()
        if name and yolo_class:
            self.object_memory.start_enrollment(name, yolo_class)
            self.enrollment_active = True
            self.enrollment_name = name
            self.enrollment_class = yolo_class
            self.enrollment_frames = 0
            logger.info(f"Enrollment started for '{name}' ({yolo_class})")
            logger.info("Point camera at the object and press SPACE to capture (need 3 frames)")
    
    def capture_enrollment_frame(self, frame):
        """Capture a frame for enrollment"""
        if not self.enrollment_active:
            return
        
        # Find the object in current detections
        detections = self.object_detector.detect(frame)
        target = [d for d in detections if d['class_name'] == self.enrollment_class]
        
        if target:
            bbox = tuple(target[0]['bbox'])
            self.enrollment_frames = self.object_memory.add_enrollment_frame(frame, bbox)
            logger.info(f"Captured frame {self.enrollment_frames}/3 for '{self.enrollment_name}'")
            
            if self.enrollment_frames >= 3:
                success = self.object_memory.finish_enrollment()
                if success:
                    logger.info(f"[OK] Object '{self.enrollment_name}' enrolled successfully!")
                else:
                    logger.error(f"Enrollment failed for '{self.enrollment_name}'")
                self.enrollment_active = False
        else:
            logger.warning(f"No '{self.enrollment_class}' detected in frame. Point camera at the object.")
    
    def register_face_mode(self, frame):
        """Register a new face from current frame"""
        name = input("\nEnter person's name: ").strip()
        if name:
            success = self.face_recognizer.enroll_face(frame, name)
            if success:
                logger.info(f"[OK] Face registered: {name}")
            else:
                logger.warning("No face detected in current frame. Make sure a face is visible.")
    
    def process_ml(self, frame, ultrasonic_distance, button_pressed):
        """
        Process frame with all ML models
        
        Returns:
            dict: {
                'objects': [...],
                'faces': [...],
                'navigation': "instruction text",
                'warnings': [...]
            }
        """
        result = {
            'objects': [],
            'faces': [],
            'remembered': [],
            'navigation': '',
            'warnings': []
        }
        
        try:
            # 1. Object Detection (YOLOv8)
            detections = self.object_detector.detect(frame)
            confirmed_items = self.item_tracker.update_tracking(detections)
            
            # Convert to serializable format
            for obj in detections:
                result['objects'].append({
                    'class': obj['class_name'],
                    'confidence': float(obj['confidence']),
                    'bbox': [int(x) for x in obj['bbox']]
                })
            
            # 2. Object Memory - Recognize remembered objects
            for obj in detections:
                try:
                    match = self.object_memory.recognize_object(
                        frame, tuple(obj['bbox']), obj['class_name']
                    )
                    if match:
                        custom_name, confidence, match_tier = match
                        result['remembered'].append({
                            'name': custom_name,
                            'class': obj['class_name'],
                            'confidence': float(confidence),
                            'bbox': [int(x) for x in obj['bbox']],
                            'tier': match_tier
                        })
                except Exception:
                    pass
            
            # 3. Face Recognition (InsightFace)
            faces = self.face_recognizer.recognize_faces(frame)
            for face in faces:
                result['faces'].append({
                    'name': face['name'],
                    'confidence': float(face.get('confidence', 0.0)),
                    'bbox': [int(x) for x in face['bbox']]
                })
            
            # 3. Spatial Navigation (obstacle avoidance)
            if self.navigation_active:
                guidance = self.spatial_navigator.generate_guidance(detections)
                
                # Get most important warning
                if guidance:
                    result['navigation'] = guidance[0]['message']
                    result['warnings'] = [g['message'] for g in guidance]
            
            # 4. GPS Navigation
            if self.gps_navigation_active and self.navigation_engine.navigation_active:
                nav_update = self.navigation_engine.update()
                if nav_update and nav_update.get('message'):
                    result['navigation'] = nav_update['message']
            
            # 5. Generate general awareness
            if not result['navigation']:
                # Basic obstacle warning
                if ultrasonic_distance < 50:
                    result['navigation'] = f"Obstacle {ultrasonic_distance:.0f} cm ahead"
                elif len(result['objects']) > 0:
                    obj_names = [o['class'] for o in result['objects'][:2]]
                    if obj_names:
                        result['navigation'] = f"Detected: {', '.join(obj_names)}"
            
        except Exception as e:
            logger.error(f"ML processing error: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def get_local_ip(self):
        """Get laptop's local IP address"""
        try:
            # Create a socket to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "Unable to determine"
    
    async def start_server(self):
        """Start WebSocket server"""
        laptop_ip = self.get_local_ip()
        
        logger.info("\n" + "="*60)
        logger.info("SERVER READY - WAITING FOR RASPBERRY PI")
        logger.info("="*60)
        logger.info(f"\nLaptop IP Address: {laptop_ip}")
        logger.info(f"Server Port: {self.port}")
        logger.info(f"\nOn Raspberry Pi, run:")
        logger.info(f"  python3 rpi_client.py {laptop_ip}")
        logger.info("\n" + "="*60 + "\n")
        
        async with websockets.serve(self.process_frame, self.host, self.port, max_size=10**7):
            await asyncio.Future()  # Run forever


def main():
    """Main entry point"""
    # Create server
    server = LaptopMLServer(port=8080)
    
    # Initialize ML models
    if not server.initialize():
        logger.error("Failed to initialize ML models. Exiting.")
        return
    
    # Start navigation (obstacle avoidance enabled by default)
    server.navigation_active = True
    
    try:
        # Start server
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("\n\nShutting down server...")
        logger.info("Goodbye! 👋")


if __name__ == "__main__":
    main()
