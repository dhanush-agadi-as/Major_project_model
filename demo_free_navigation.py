"""
Free Navigation System Demo
Uses OpenStreetMap (no API keys, no payment, unlimited usage)
"""

import cv2
import logging
import time
import sys

# Import VisionMate core
from object_detector import ObjectDetector
from audio_feedback import AudioFeedback

# Import FREE navigation modules
from gps_tracker import GPSTracker
from osm_navigator import OSMNavigator  # FREE alternative
from ultrasonic_sensor import UltrasonicSensor
from obstacle_fusion import ObstacleFusion
from navigation_engine import NavigationEngine

import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FreeNavigationDemo:
    """Navigation system using 100% FREE OpenStreetMap services"""
    
    def __init__(self):
        logger.info("Initializing VisionMate FREE Navigation System...")
        
        # Core vision
        self.object_detector = ObjectDetector()
        self.audio = AudioFeedback()
        
        # FREE Navigation components
        self.gps = GPSTracker(mock_mode=config.GPS_MOCK_MODE)
        self.maps = OSMNavigator()  # No API key needed!
        self.ultrasonic = UltrasonicSensor(mock_mode=config.ULTRASONIC_MOCK_MODE)
        self.obstacle_fusion = ObstacleFusion(self.ultrasonic)
        self.nav_engine = NavigationEngine(
            self.gps, 
            self.maps,
            announcement_distance=config.NAVIGATION_ANNOUNCEMENT_DISTANCE,
            reroute_distance=config.NAVIGATION_REROUTE_DISTANCE,
            arrival_distance=config.NAVIGATION_ARRIVAL_DISTANCE
        )
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        
        # State
        self.navigating = False
        self.last_nav_update = 0
        self.last_obstacle_check = 0
        self.current_step = 0
        
        logger.info("[OK] All systems initialized (100% FREE)!")
    
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
        
        # Geocode destination
        dest_coords = self.maps.geocode_address(destination + ", Bangalore, India")
        if not dest_coords:
            self.audio.speak("Could not find destination")
            logger.error("Geocoding failed")
            return False
        
        # Get current position
        current_pos = self.gps.get_position()
        if not current_pos:
            self.audio.speak("Waiting for GPS signal")
            logger.warning("No GPS signal")
            return False
        
        # Get directions
        route = self.maps.get_directions(current_pos, dest_coords, mode='foot')
        
        if not route:
            self.audio.speak("Could not find route")
            logger.error("Routing failed")
            return False
        
        # Store route
        self.maps.current_route = route
        self.navigating = True
        self.current_step = 0
        self.last_announcement = None
        
        # Announce route
        self.audio.speak(f"Route found. {route['total_distance']}. {route['total_duration']}.")
        logger.info(f"Navigation started: {len(route['steps'])} steps")
        
        return True
    
    def update_navigation(self):
        """Update navigation state"""
        if not self.navigating:
            return
        
        current_time = time.time()
        
        # Update navigation every 2 seconds
        if current_time - self.last_nav_update < 2.0:
            return
        
        self.last_nav_update = current_time
        
        # Get current position
        current_pos = self.gps.get_position()
        if not current_pos:
            return
        
        # Get current instruction
        instruction = self.maps.get_current_instruction(self.current_step)
        if not instruction:
            return
        
        # Calculate distance to next waypoint
        step = self.maps.current_route['steps'][self.current_step]
        end_lat = step['end_location']['lat']
        end_lon = step['end_location']['lng']
        
        distance = self._haversine_distance(
            current_pos[0], current_pos[1],
            end_lat, end_lon
        )
        
        # Check if should announce
        if distance <= config.NAVIGATION_ANNOUNCEMENT_DISTANCE:
            if self.last_announcement != self.current_step:
                self.last_announcement = self.current_step
                self.audio.speak(f"In {int(distance)} meters, {instruction}")
                logger.info(f"[AUDIO] ANNOUNCE: {instruction}")
        
        # Display progress
        total_steps = len(self.maps.current_route['steps'])
        print(f"\rStep {self.current_step + 1}/{total_steps} | {distance:.0f}m | {instruction[:50]}", end='')
        
        # Advance to next step if close
        if distance < 10:
            if self.current_step + 1 < len(self.maps.current_route['steps']):
                self.current_step += 1
                self.last_announcement = None
                logger.info(f"Advanced to step {self.current_step + 1}")
            else:
                # Arrived!
                self.audio.speak("You have arrived at your destination")
                logger.info("[OK] Arrived!")
                self.navigating = False
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between GPS coordinates"""
        import math
        R = 6371000  # Earth radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
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
        print("VisionMate FREE Navigation System")
        print("100% Free - No API Keys - Unlimited Usage")
        print("="*60)
        print("\nControls:")
        print("  N - Start navigation to Majestic")
        print("  M - Advance mock GPS position (simulate movement)")
        print("  S - Show navigation status")
        print("  O - Check obstacles")
        print("  Q - Quit")
        print("\nUsing OpenStreetMap (free alternative)...")
        print("Note: Walking directions only (no bus routes in free version)")
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
                    total_steps = len(self.maps.current_route['steps'])
                    cv2.putText(frame, f"Navigating: Step {self.current_step + 1}/{total_steps}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Update navigation
                    self.update_navigation()
                    
                    # Check obstacles
                    self.check_obstacles(frame)
                else:
                    cv2.putText(frame, "FREE Navigation - Press N to start", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Draw ultrasonic distance
                distance = self.ultrasonic.get_distance()
                if distance:
                    cv2.putText(frame, f"Distance: {distance}cm", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow('VisionMate FREE Navigation', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    self.start_navigation("Majestic")
                elif key == ord('m'):
                    # Advance mock GPS position
                    self.gps.advance_mock_position()
                    new_pos = self.gps.get_position()
                    logger.info(f"Mock GPS advanced to: {new_pos[0]:.4f}, {new_pos[1]:.4f}")
                elif key == ord('s'):
                    if self.navigating:
                        progress = self.maps.get_progress(self.current_step)
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
    demo = FreeNavigationDemo()
    demo.run()
