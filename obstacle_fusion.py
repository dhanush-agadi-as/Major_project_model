"""
Obstacle Detection Fusion
Combines camera object detection with ultrasonic distance sensor
Provides precise obstacle warnings with real distance measurements
"""

import logging
from typing import Dict, List, Optional

import os
import sys

# Ensure config can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import config
except ImportError:
    config = None

logger = logging.getLogger(__name__)

_SENTINEL = object()


class ObstacleFusion:
    """
    Fuses camera-based object detection with ultrasonic distance sensor
    Replaces bbox-based distance estimation with actual measurements
    """
    
    def __init__(self, ultrasonic_sensor, frame_width: int = 640, frame_height: int = 480):
        self.ultrasonic_sensor = ultrasonic_sensor
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sensor_fov = 15  # Ultrasonic sensor field of view (degrees)
        self.camera_fov = 60  # Typical webcam horizontal FOV (degrees)
        
        logger.info("Obstacle Fusion initialized")
        logger.info(f"  Frame size: {frame_width}x{frame_height}")
        logger.info(f"  Ultrasonic FOV: {self.sensor_fov} deg")

    def set_frame_size(self, width: int, height: int):
        """Update frame dimensions dynamically from real camera stream."""
        self.frame_width = width
        self.frame_height = height
    
    def get_object_with_distance(self, objects: List[Dict],
                                 ultrasonic_distance: Optional[float] = _SENTINEL) -> List[Dict]:
        """
        Enhance object detections with precise distance from ultrasonic sensor
        
        Args:
            objects: List of detected objects with bboxes from YOLO
            ultrasonic_distance: Optional distance in cm, or None if unavailable.
                                 If omitted, reads from self.ultrasonic_sensor.
            
        Returns:
            List of objects with added 'precise_distance' field
        """
        is_mock = False
        if self.ultrasonic_sensor is not None and getattr(self.ultrasonic_sensor, 'mock_mode', False):
            is_mock = True
        if config is not None and getattr(config, 'ULTRASONIC_MOCK_MODE', False):
            is_mock = True

        if ultrasonic_distance is _SENTINEL:
            if is_mock or self.ultrasonic_sensor is None:
                ultrasonic_distance = None
            else:
                ultrasonic_distance = self.ultrasonic_sensor.get_distance()
        elif is_mock:
            # Mock mode must never generate obstacle warnings; treat as no ultrasonic reading
            ultrasonic_distance = None

        enhanced_objects = []
        
        for obj in objects:
            enhanced = obj.copy()
            
            # Check if object is in center of frame (where ultrasonic points)
            if self._is_in_sensor_view(obj['bbox']) and ultrasonic_distance is not None:
                enhanced['precise_distance'] = ultrasonic_distance / 100.0  # cm to meters
                enhanced['distance_source'] = 'ultrasonic'
            else:
                # Object not in ultrasonic FOV or ultrasonic unavailable/mock, use camera estimation
                enhanced['precise_distance'] = self._estimate_from_bbox(obj['bbox'])
                enhanced['distance_source'] = 'camera_estimate'
            
            enhanced_objects.append(enhanced)
        
        return enhanced_objects
    
    def _is_in_sensor_view(self, bbox: tuple) -> bool:
        """
        Check if object bbox is within ultrasonic sensor's field of view
        
        Sensor typically points straight ahead with narrow FOV (~15 degrees)
        """
        x1, _, x2, _ = bbox
        
        # Get object horizontal center
        obj_center_x = (x1 + x2) / 2
        
        # Get frame horizontal center
        frame_center_x = self.frame_width / 2
        
        # Calculate angle offset from center
        pixel_offset_x = abs(obj_center_x - frame_center_x)
        
        # Convert pixel offset to angle
        # Approximate: angle ≈ (pixel_offset / frame_width) * camera_fov
        angle_offset = (pixel_offset_x / self.frame_width) * self.camera_fov
        
        # Object is in sensor view if within sensor FOV
        return angle_offset < (self.sensor_fov / 2)
    
    def _estimate_from_bbox(self, bbox: tuple) -> float:
        """
        Estimate distance from bbox size (fallback when ultrasonic unavailable)
        
        Returns distance in meters
        """
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = self.frame_width * self.frame_height
        area_ratio = bbox_area / frame_area
        
        # Rough estimation: distance inversely proportional to sqrt(area)
        if area_ratio > 0.35:
            return 0.5  # Very close
        elif area_ratio > 0.15:
            return 1.0  # Close
        elif area_ratio > 0.05:
            return 2.0  # Medium
        else:
            return 4.0  # Far
    
    def get_critical_obstacles(self, objects: List[Dict], distance_threshold: float = 1.5,
                               ultrasonic_distance: Optional[float] = _SENTINEL) -> List[Dict]:
        """
        Get obstacles that are critically close
        
        Args:
            objects: List of detected objects
            distance_threshold: Distance in meters to consider critical
            ultrasonic_distance: Optional distance in cm, or None if unavailable
            
        Returns:
            List of critical obstacles sorted by distance
        """
        enhanced = self.get_object_with_distance(objects, ultrasonic_distance=ultrasonic_distance)
        
        # Filter obstacles (not books, bottles, etc.)
        obstacle_classes = ['person', 'chair', 'couch', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']
        
        critical = []
        for obj in enhanced:
            if obj['class_name'] in obstacle_classes:
                if obj['precise_distance'] < distance_threshold:
                    critical.append(obj)
        
        # Sort by distance (closest first)
        critical.sort(key=lambda x: x['precise_distance'])
        
        return critical
    
    def generate_obstacle_warnings(self, objects: List[Dict],
                                   ultrasonic_distance: Optional[float] = _SENTINEL) -> List[Dict]:
        """
        Generate precise obstacle warnings with distances
        
        Args:
            objects: List of detected objects
            ultrasonic_distance: Optional distance in cm, or None if unavailable
            
        Returns:
            List of warning messages with priority
        """
        critical = self.get_critical_obstacles(objects, ultrasonic_distance=ultrasonic_distance)
        
        warnings = []
        
        for obj in critical[:3]:  # Top 3 closest obstacles
            distance = obj['precise_distance']
            class_name = obj['class_name']
            source = obj['distance_source']
            
            # Determine location in frame
            x1, y1, x2, y2 = obj['bbox']
            center_x = (x1 + x2) / 2
            
            if center_x < self.frame_width / 3:
                position = "left"
            elif center_x > 2 * self.frame_width / 3:
                position = "right"
            else:
                position = "ahead"
            
            # Determine priority and message
            if distance < 0.5:
                priority = 1  # EMERGENCY
                message = f"STOP! {class_name.capitalize()} {distance:.1f}m {position}"
            elif distance < 1.0:
                priority = 2  # HIGH WARNING
                message = f"Caution: {class_name.capitalize()} {distance:.1f}m {position}"
            else:
                priority = 3  # NOTICE
                message = f"{class_name.capitalize()} {distance:.1f}m {position}"
            
            # Add accuracy indicator
            if source == 'ultrasonic':
                accuracy = "measured"
            else:
                accuracy = "estimated"
            
            warnings.append({
                'message': message,
                'priority': priority,
                'distance': distance,
                'object': class_name,
                'position': position,
                'accuracy': accuracy
            })
        
        return warnings
    
    def get_ultrasonic_status(self) -> Dict:
        """Get current ultrasonic sensor status"""
        return self.ultrasonic_sensor.get_obstacle_status()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Testing Obstacle Fusion ===")
    
    # Mock ultrasonic sensor
    from ultrasonic_sensor import UltrasonicSensor
    sensor = UltrasonicSensor(mock_mode=True)
    sensor.start()
    
    fusion = ObstacleFusion(sensor)
    
    # Mock detected objects
    mock_objects = [
        {
            'class_name': 'person',
            'confidence': 0.95,
            'bbox': (280, 100, 360, 400)  # Center of frame
        },
        {
            'class_name': 'chair',
            'confidence': 0.87,
            'bbox': (50, 200, 150, 380)  # Left side
        },
        {
            'class_name': 'bottle',
            'confidence': 0.76,
            'bbox': (500, 300, 530, 400)  # Right side, small
        }
    ]
    
    print("\nEnhancing objects with distance:")
    enhanced = fusion.get_object_with_distance(mock_objects)
    for obj in enhanced:
        print(f"  {obj['class_name']}: {obj['precise_distance']:.2f}m ({obj['distance_source']})")
    
    print("\nGenerating warnings:")
    warnings = fusion.generate_obstacle_warnings(mock_objects)
    for warn in warnings:
        print(f"  P{warn['priority']}: {warn['message']} [{warn['accuracy']}]")
    
    sensor.stop()
    print("\n[OK] Obstacle fusion test complete!")
