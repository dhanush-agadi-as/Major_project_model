"""
Spatial Navigator for VisionMate
Provides real-time obstacle detection and directional guidance for blind users
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialNavigator:
    """Converts visual detections into spatial guidance for navigation"""
    
    # Zone definitions (3x3 grid)
    ZONES = {
        'top-left': (0, 0), 'top': (0, 1), 'top-right': (0, 2),
        'left': (1, 0), 'center': (1, 1), 'right': (1, 2),
        'bottom-left': (2, 0), 'bottom': (2, 1), 'bottom-right': (2, 2)
    }
    
    # Priority: bottom zones are most critical (immediate obstacles)
    ZONE_PRIORITY = {
        'bottom-left': 1, 'bottom': 1, 'bottom-right': 1,
        'left': 2, 'center': 2, 'right': 2,
        'top-left': 3, 'top': 3, 'top-right': 3
    }
    
    # Obstacle classes that require immediate guidance
    OBSTACLE_CLASSES = ['person', 'chair', 'couch', 'bicycle', 'car', 'motorcycle', 
                        'bus', 'truck', 'bench', 'potted plant', 'dog', 'cat']
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480, 
                 distance_threshold: float = 0.3):
        """
        Initialize spatial navigator
        
        Args:
            frame_width: Width of camera frame in pixels
            frame_height: Height of camera frame in pixels
            distance_threshold: Bbox area ratio for "close" detection (0-1)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.distance_threshold = distance_threshold
        
        # Calculate zone boundaries
        self.zone_width = frame_width / 3
        self.zone_height = frame_height / 3
        
        logger.info(f"Spatial Navigator initialized ({frame_width}x{frame_height})")
        logger.info(f"  Zone size: {self.zone_width:.0f}x{self.zone_height:.0f}")
        logger.info(f"  Distance threshold: {distance_threshold}")
    
    def get_object_zone(self, bbox: Tuple[int, int, int, int]) -> str:
        """
        Determine which zone a bounding box occupies
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box coordinates
            
        Returns:
            Zone name (e.g., 'bottom-left', 'center')
        """
        x1, y1, x2, y2 = bbox
        
        # Get center point of bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Determine column (left, center, right)
        if center_x < self.zone_width:
            col = 0  # left
        elif center_x < 2 * self.zone_width:
            col = 1  # center
        else:
            col = 2  # right
        
        # Determine row (top, middle, bottom)
        if center_y < self.zone_height:
            row = 0  # top
        elif center_y < 2 * self.zone_height:
            row = 1  # middle
        else:
            row = 2  # bottom
        
        # Map to zone name
        zone_map = {
            (0, 0): 'top-left', (0, 1): 'top', (0, 2): 'top-right',
            (1, 0): 'left', (1, 1): 'center', (1, 2): 'right',
            (2, 0): 'bottom-left', (2, 1): 'bottom', (2, 2): 'bottom-right'
        }
        
        return zone_map.get((row, col), 'center')
    
    def estimate_distance(self, bbox: Tuple[int, int, int, int]) -> str:
        """
        Estimate relative distance based on bounding box size
        Larger bbox = closer object
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box coordinates
            
        Returns:
            'very-close', 'close', 'medium', or 'far'
        """
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = self.frame_width * self.frame_height
        area_ratio = bbox_area / frame_area
        
        if area_ratio > 0.35:
            return 'very-close'  # Immediate danger
        elif area_ratio > 0.15:
            return 'close'
        elif area_ratio > 0.05:
            return 'medium'
        else:
            return 'far'
    
    def is_obstacle(self, class_name: str) -> bool:
        """Check if detected object is an obstacle that needs guidance"""
        return class_name.lower() in [c.lower() for c in self.OBSTACLE_CLASSES]
    
    def get_clear_direction(self, occupied_zones: List[str]) -> Optional[str]:
        """
        Determine the clearest direction to move based on occupied zones
        
        Args:
            occupied_zones: List of zone names that have obstacles
            
        Returns:
            Direction guidance: 'left', 'right', 'straight', or None if blocked
        """
        # Priority: Check bottom zones first (immediate path)
        bottom_zones = ['bottom-left', 'bottom', 'bottom-right']
        
        occupied_bottom = [z for z in occupied_zones if z in bottom_zones]
        
        if not occupied_bottom:
            return 'straight'  # Clear ahead
        
        # Check which side is clearer
        left_blocked = 'bottom-left' in occupied_bottom or 'left' in occupied_zones
        right_blocked = 'bottom-right' in occupied_bottom or 'right' in occupied_zones
        center_blocked = 'bottom' in occupied_bottom
        
        # If only center is blocked, can still go straight (object is directly ahead but can move around)
        if center_blocked and not left_blocked and not right_blocked:
            # Choose the clearer side
            if 'left' not in occupied_zones:
                return 'left'
            elif 'right' not in occupied_zones:
                return 'right'
            else:
                return 'straight'  # Can still proceed forward
        
        # Otherwise, guide to clearest path
        if not left_blocked and not center_blocked:
            return 'left'
        elif not right_blocked and not center_blocked:
            return 'right'
        elif not left_blocked:
            return 'left'
        elif not right_blocked:
            return 'right'
        else:
            return None  # All paths blocked, stop
    
    def generate_guidance(self, detections: List[Dict]) -> List[Dict]:
        """
        Generate spatial guidance for all detected obstacles
        
        Args:
            detections: List of detected objects with bbox, class_name, confidence
            Each detection: {'bbox': (x1,y1,x2,y2), 'class_name': str, 'confidence': float}
            
        Returns:
            List of guidance messages with priority:
            [{'message': str, 'priority': int, 'zone': str, 'distance': str}]
        """
        guidance_list = []
        occupied_zones = []
        
        for det in detections:
            class_name = det['class_name']
            bbox = det['bbox']
            
            # Only process obstacles
            if not self.is_obstacle(class_name):
                continue
            
            # Get spatial info
            zone = self.get_object_zone(bbox)
            distance = self.estimate_distance(bbox)
            priority = self.ZONE_PRIORITY.get(zone, 3)
            
            occupied_zones.append(zone)
            
            # Generate guidance message based on zone and distance
            message = self._create_guidance_message(class_name, zone, distance)
            
            # Increase priority for very close objects
            if distance == 'very-close':
                priority = 1  # Emergency
            elif distance == 'close' and priority == 2:
                priority = 1  # Upgrade to emergency
            
            guidance_list.append({
                'message': message,
                'priority': priority,
                'zone': zone,
                'distance': distance,
                'class_name': class_name,
                'bbox': bbox
            })
        
        # Add overall directional guidance if obstacles present
        if occupied_zones:
            clear_direction = self.get_clear_direction(occupied_zones)
            if clear_direction:
                direction_message = f"Move {clear_direction}"
                guidance_list.append({
                    'message': direction_message,
                    'priority': 1,
                    'zone': 'navigation',
                    'distance': 'immediate',
                    'class_name': 'navigation',
                    'bbox': None
                })
            else:
                guidance_list.append({
                    'message': "Path blocked, stop",
                    'priority': 1,
                    'zone': 'navigation',
                    'distance': 'immediate',
                    'class_name': 'emergency',
                    'bbox': None
                })
        
        # Sort by priority (1 = highest)
        guidance_list.sort(key=lambda x: x['priority'])
        
        return guidance_list
    
    def _create_guidance_message(self, class_name: str, zone: str, 
                                 distance: str) -> str:
        """
        Create a natural language guidance message
        
        Args:
            class_name: Type of obstacle
            zone: Location zone
            distance: Estimated distance
            
        Returns:
            Guidance message string
        """
        # Simplify zone names for speech
        zone_speech = {
            'top-left': 'above left',
            'top': 'above',
            'top-right': 'above right',
            'left': 'left',
            'center': 'ahead',
            'right': 'right',
            'bottom-left': 'bottom left',
            'bottom': 'bottom',
            'bottom-right': 'bottom right'
        }
        
        distance_speech = {
            'very-close': '',  # Don't mention distance, it's urgent
            'close': 'close',
            'medium': '',
            'far': ''
        }
        
        zone_desc = zone_speech.get(zone, zone)
        dist_desc = distance_speech.get(distance, '')
        
        # Build message
        if distance == 'very-close':
            # Emergency: short and direct
            return f"{class_name.capitalize()} {zone_desc}"
        else:
            # Normal: include distance if relevant
            if dist_desc:
                return f"{class_name.capitalize()} {dist_desc} {zone_desc}"
            else:
                return f"{class_name.capitalize()} {zone_desc}"
    
    def get_navigation_priority(self, distance: str) -> int:
        """
        Map distance to audio priority level (for integration with AudioFeedback)
        
        Args:
            distance: 'very-close', 'close', 'medium', 'far'
            
        Returns:
            Priority: 1 (EMERGENCY), 2 (SOCIAL), 3 (NAVIGATIONAL), 4 (STATUS)
        """
        priority_map = {
            'very-close': 1,  # EMERGENCY
            'close': 1,       # EMERGENCY
            'medium': 3,      # NAVIGATIONAL
            'far': 3          # NAVIGATIONAL
        }
        return priority_map.get(distance, 3)
