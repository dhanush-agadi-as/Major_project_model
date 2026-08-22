"""
Item Tracking Module
Uses IoU (Intersection over Union) to track items inside containers
Example: Track if 'keys' are on 'table'
"""
import numpy as np
from typing import List, Dict, Optional
import time
from config import ITEM_TRACKING_TIMEOUT, IOU_THRESHOLD
from database import VisionMateDB


class ItemTracker:
    def __init__(self, db: VisionMateDB = None):
        """
        Initialize item tracking system
        
        Args:
            db: Database instance for storing item locations
        """
        self.db = db if db else VisionMateDB()
        self.timeout = ITEM_TRACKING_TIMEOUT
        self.iou_threshold = IOU_THRESHOLD
        
        # Track item-container pairs over time
        self.tracking_state = {}  # {(item, container): {'start_time': float, 'confirmed': bool}}
        
        print(f"[OK] Item Tracker initialized")
        print(f"  IoU threshold: {self.iou_threshold}")
        print(f"  Tracking timeout: {self.timeout}s")
    
    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            box1, box2: [x1, y1, x2, y2] format
            
        Returns:
            IoU value (0 to 1)
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def is_inside(self, item_box: List[int], container_box: List[int]) -> bool:
        """
        Check if item box is inside container box using IoU
        
        Args:
            item_box: Bounding box of the item [x1, y1, x2, y2]
            container_box: Bounding box of the container [x1, y1, x2, y2]
            
        Returns:
            True if item is inside container
        """
        iou = self.calculate_iou(item_box, container_box)
        
        # Also check if item center is inside container
        item_cx = (item_box[0] + item_box[2]) / 2
        item_cy = (item_box[1] + item_box[3]) / 2
        
        inside_bounds = (
            container_box[0] <= item_cx <= container_box[2] and
            container_box[1] <= item_cy <= container_box[3]
        )
        
        return iou > self.iou_threshold or inside_bounds
    
    def update_tracking(self, detections: List[Dict]) -> List[Dict]:
        """
        Update item tracking based on current detections
        
        Args:
            detections: List of object detections from ObjectDetector
            
        Returns:
            List of confirmed item-location pairs that should be saved
        """
        current_time = time.time()
        confirmed_locations = []
        
        # Define which objects can be containers
        container_classes = ['chair', 'couch', 'bed', 'dining table', 'tv', 'sink', 'bench', 'sofa']  # Trackable containers
        item_classes = ['keys', 'bottle', 'cup', 'cell phone', 'laptop', 'backpack', 'handbag', 'umbrella', 'book', 'remote', 'keyboard', 'mouse']  # Trackable items
        
        # Extract items and containers from detections
        items = [d for d in detections if d['class_name'] in item_classes]
        containers = [d for d in detections if d['class_name'] in container_classes]
        
        # Check each item against each container
        active_pairs = set()
        
        for item in items:
            for container in containers:
                if self.is_inside(item['bbox'], container['bbox']):
                    pair_key = (item['class_name'], container['class_name'])
                    active_pairs.add(pair_key)
                    
                    # Check if we're already tracking this pair
                    if pair_key in self.tracking_state:
                        state = self.tracking_state[pair_key]
                        elapsed = current_time - state['start_time']
                        
                        # If tracked long enough and not yet confirmed
                        if elapsed >= self.timeout and not state['confirmed']:
                            state['confirmed'] = True
                            confirmed_locations.append({
                                'item': item['class_name'],
                                'location': container['class_name'],
                                'timestamp': current_time
                            })
                            # Save to database
                            self.db.update_item_location(
                                item['class_name'],
                                container['class_name']
                            )
                            print(f"[OK] Confirmed: {item['class_name']} on {container['class_name']}")
                    else:
                        # Start tracking new pair
                        self.tracking_state[pair_key] = {
                            'start_time': current_time,
                            'confirmed': False
                        }
                        print(f"⊙ Tracking: {item['class_name']} on {container['class_name']}")
        
        # Clear tracking for pairs no longer detected
        inactive_pairs = set(self.tracking_state.keys()) - active_pairs
        for pair in inactive_pairs:
            del self.tracking_state[pair]
        
        return confirmed_locations
    
    def query_item_location(self, item: str) -> Optional[str]:
        """
        Query last known location of an item
        
        Args:
            item: Item name (e.g., 'keys')
            
        Returns:
            Location string or None if not found
        """
        return self.db.get_item_location(item)
    
    def get_all_tracked_items(self) -> List[Dict]:
        """Get all currently tracked items with their locations"""
        locations = self.db.get_all_item_locations()
        return [
            {'item': item, 'location': loc, 'timestamp': ts}
            for item, loc, ts in locations
        ]


if __name__ == "__main__":
    # Quick test
    tracker = ItemTracker()
    
    # Test IoU calculation
    box1 = [100, 100, 200, 200]
    box2 = [150, 150, 250, 250]
    iou = tracker.calculate_iou(box1, box2)
    print(f"\nIoU test: {iou:.3f}")
    print("\n[OK] Item Tracker ready")
