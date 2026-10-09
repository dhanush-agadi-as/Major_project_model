"""
Navigation System Demo
Full test of external navigation with GPS, Google Maps, and obstacle detection
Tests all 6 navigation modules working together
"""

import cv2
import logging
import time
import sys

# Import VisionMate core
from object_detector import ObjectDetector
from audio_feedback import AudioFeedback

# Import navigation modules
from gps_tracker import GPSTracker
from google_maps_navigator import GoogleMapsNavigator
from ultrasonic_sensor import UltrasonicSensor
from obstacle_fusion import ObstacleFusion
from navigation_engine import NavigationEngine
from transit_guide import TransitGuide

import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NavigationDemo:
    """Full navigation system demonstration"""
    
    def __init__(self):
        logger.info("Initializing VisionMate Navigation System...")
        
        # Core vision
        self.object_detector = ObjectDetector()
        self.audio = AudioFeedback()
        
        # Navigation components
        self.gps = GPSTracker(mock_mode=config.GPS_MOCK_MODE)
        self.maps = GoogleMapsNavigator(config.GOOGLE_MAPS_API_KEY)
        self.ultrasonic = UltrasonicSensor(mock_mode=config.ULTRASONIC_MOCK_MODE)
        self.obstacle_fusion = ObstacleFusion(self.ultrasonic)
        self.nav_engine = NavigationEngine(
            self.gps, 
            self.maps,
            announcement_distance=config.NAVIGATION_ANNOUNCEMENT_DISTANCE,
            reroute_distance=config.NAVIGATION_REROUTE_DISTANCE,
            arrival_distance=config.NAVIGATION_ARRIVAL_DISTANCE
        )
        self.transit_guide = TransitGuide(self.gps)
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        
        # State
        self.navigating = False
        self.last_nav_update = 0
        self.last_obstacle_check = 0
        
        logger.info("[OK] All systems initialized!")
    
    def start(self):
        """Start all background services"""
        logger.info("Starting services...")
        self.gps.start()
        self.ultrasonic.start()
        logger.info("[OK] Services started!")
    
    def stop(self):
        """Stop all services"""
        logger.info("Stopping services...")
        self.gps.stop()
        self.ultrasonic.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("[OK] Services stopped!")
    
    def start_navigation(self, destination: str):
        """Start navigation to destination"""
        logger.info(f"Starting navigation to: {destination}")
        self.audio.speak(f"Navigating to {destination}")
        
        # Get current position
        current_pos = self.gps.get_position()
        if not current_pos:
            self.audio.speak("Waiting for GPS signal")
            logger.warning("No GPS signal")
            return False
        
        # Start navigation
        success = self.nav_engine.start_navigation(current_pos, destination, mode='transit')
        
        if success:
            self.navigating = True
            progress = self.nav_engine.get_progress()
            self.audio.speak(f"Route found. {progress['total_distance']}. {progress['total_duration']}.")
            logger.info(f"Navigation started: {progress['total_steps']} steps")
            return True
        else:
            self.audio.speak("Could not find route")
            logger.error("Navigation failed to start")
            return False
    
    def update_navigation(self):
        """Update navigation state"""
        current_time = time.time()
        
        # Update navigation every 2 seconds
        if current_time - self.last_nav_update < 2.0:
            return
        
        self.last_nav_update = current_time
        
        # Get navigation update
        status = self.nav_engine.update()
        
        if status['status'] == 'navigating':
            # Check if should announce
            if status['announce']:
                instruction = status['instruction']
                distance = status['distance_to_waypoint']
                self.audio.speak(f"In {int(distance)} meters, {instruction}")
                logger.info(f"[AUDIO] ANNOUNCE: {instruction}")
            
            # Display progress
            print(f"\rStep {status['current_step']}/{status['total_steps']} | {status['distance_to_waypoint']:.0f}m", end='')
        
        elif status['status'] == 'arrived':
            self.audio.speak(status['message'])
            logger.info("[OK] Arrived at destination!")
            self.navigating = False
        
        elif status['status'] == 'off_route':
            self.audio.speak("Off route. Recalculating...")
            logger.warning("User off route")
            # Could implement rerouting here
    
    def check_obstacles(self, frame):
        """Check for obstacles during navigation"""
        current_time = time.time()
        
        # Check obstacles every 1 second
        if current_time - self.last_obstacle_check < 1.0:
            return
        
        self.last_obstacle_check = current_time
        
        # Detect objects
        objects = self.object_detector.detect(frame)
        
        # Get obstacle warnings
        warnings = self.obstacle_fusion.generate_obstacle_warnings(objects)
        
        # Announce critical obstacles
        if warnings:
            highest_priority = warnings[0]
            
            if highest_priority['priority'] == 1:  # EMERGENCY
                self.audio.speak(highest_priority['message'])
                logger.warning(f"[ALERT] OBSTACLE: {highest_priority['message']}")
            elif highest_priority['priority'] == 2:  # HIGH WARNING
                self.audio.speak(highest_priority['message'])
                logger.info(f"[WARN] OBSTACLE: {highest_priority['message']}")
    
    def run(self):
        """Main demo loop"""
        print("\n" + "="*60)
        print("VisionMate Navigation System Demo")
        print("="*60)
        print("\nControls:")
        print("  N - Start navigation to Majestic")
        print("  M - Advance mock GPS position (simulate movement)")
        print("  S - Show navigation status")
        print("  O - Check obstacles")
        print("  Q - Quit")
        print("\nStarting in mock mode (no hardware required)...")
        print("="*60 + "\n")
        
        self.start()
        
        # Wait for GPS
        time.sleep(1)
        pos = self.gps.get_position()
        if pos:
            logger.info(f"GPS Position: {pos[0]:.4f}, {pos[1]:.4f}")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Draw GPS position
                pos = self.gps.get_position()
                if pos:
                    cv2.putText(frame, f"GPS: {pos[0]:.4f}, {pos[1]:.4f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw navigation status
                if self.navigating:
                    progress = self.nav_engine.get_progress()
                    cv2.putText(frame, f"Navigating: Step {progress['current_step']}/{progress['total_steps']}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Update navigation
                    self.update_navigation()
                    
                    # Check obstacles
                    self.check_obstacles(frame)
                
                # Draw ultrasonic distance
                distance = self.ultrasonic.get_distance()
                if distance:
                    cv2.putText(frame, f"Distance: {distance}cm", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow('VisionMate Navigation', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    self.start_navigation("Majestic, Bangalore")
                elif key == ord('m'):
                    # Advance mock GPS position
                    self.gps.advance_mock_position()
                    new_pos = self.gps.get_position()
                    logger.info(f"Mock GPS advanced to: {new_pos[0]:.4f}, {new_pos[1]:.4f}")
                elif key == ord('s'):
                    if self.navigating:
                        progress = self.nav_engine.get_progress()
                        print(f"\n--- Navigation Status ---")
                        print(f"Step: {progress['current_step']}/{progress['total_steps']}")
                        print(f"Total: {progress['total_distance']}, {progress['total_duration']}")
                    else:
                        print("\nNo active navigation")
                elif key == ord('o'):
                    # Manual obstacle check
                    objects = self.object_detector.detect(frame)
                    warnings = self.obstacle_fusion.generate_obstacle_warnings(objects)
                    print(f"\n--- Obstacle Check ---")
                    if warnings:
                        for warn in warnings:
                            print(f"P{warn['priority']}: {warn['message']}")
                    else:
                        print("No obstacles detected")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.stop()
            print("\n[OK] Demo complete!")


if __name__ == "__main__":
    demo = NavigationDemo()
    demo.run()
