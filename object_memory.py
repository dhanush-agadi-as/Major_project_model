"""
Object Memory Manager for VisionMate
High-level interface for enrolling and recognizing custom objects
"""
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
import logging
from datetime import datetime

from database import VisionMateDB
from visual_matcher import VisualMatcher
from config import (
    OBJECT_MATCH_HIGH_CONFIDENCE, OBJECT_MATCH_AMBIGUOUS_LOW, 
    OBJECT_ENROLLMENT_FRAMES, USE_DEEP_FEATURES,
    OBJECT_MATCH_COOLDOWN, OBJECT_CANDIDATES_CACHE_TTL
)

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


class ObjectMemory:
    """Manages enrollment and recognition of custom remembered objects"""
    
    def __init__(self, db: VisionMateDB = None, matcher: VisualMatcher = None):
        """
        Initialize object memory manager
        
        Args:
            db: Database instance (creates new if None)
            matcher: Visual matcher instance (creates new if None)
        """
        self.db = db if db is not None else VisionMateDB()
        self.matcher = matcher if matcher is not None else VisualMatcher(
            feature_type='ORB',
            match_threshold=OBJECT_MATCH_AMBIGUOUS_LOW,
            use_deep_features=USE_DEEP_FEATURES
        )
        
        self.pending_confirmation = None # For ambiguous matches
        
        # Enrollment buffer for multi-frame capture
        self.enrollment_buffer = {
            'images': [],
            'bboxes': [],
            'yolo_class': None,
            'custom_name': None
        }
        
        self._last_sighting_update = {}   # maps custom_name -> unix timestamp of last DB write
        self._candidates_cache = {}       # maps yolo_class -> (timestamp, list_of_candidates) for cached DB reads
        
        logger.info("Object memory initialized")
    
    def start_enrollment(self, custom_name: str, yolo_class: str) -> None:
        """
        Start enrolling a new object
        
        Args:
            custom_name: User-friendly name (e.g., "house keys")
            yolo_class: YOLO detection class (e.g., "keys")
        """
        # Check if object already exists
        existing = self.db.get_remembered_object(custom_name)
        if existing:
            logger.warning(f"Object '{custom_name}' already enrolled. Delete first to re-enroll.")
            raise ValueError(f"Object '{custom_name}' already exists")
        
        self.enrollment_buffer = {
            'images': [],
            'bboxes': [],
            'yolo_class': yolo_class,
            'custom_name': custom_name
        }
        
        logger.info(f"Started enrollment for '{custom_name}' ({yolo_class})")
    
    def add_enrollment_frame(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> int:
        """
        Add a frame to the enrollment buffer
        
        Args:
            image: Frame containing the object
            bbox: Bounding box (x1, y1, x2, y2) of the object
            
        Returns:
            Number of frames captured so far
        """
        if self.enrollment_buffer['custom_name'] is None:
            raise ValueError("Must call start_enrollment first")
        
        self.enrollment_buffer['images'].append(image.copy())
        self.enrollment_buffer['bboxes'].append(bbox)
        
        num_frames = len(self.enrollment_buffer['images'])
        logger.debug(f"Enrollment frame {num_frames}/{OBJECT_ENROLLMENT_FRAMES} captured")
        
        return num_frames
    
    def finish_enrollment(self) -> bool:
        """
        Complete the enrollment process and save to database
        
        Returns:
            True if successful, False otherwise
        """
        if self.enrollment_buffer['custom_name'] is None:
            logger.error("No enrollment in progress")
            return False
        
        if len(self.enrollment_buffer['images']) < 1:
            logger.error("Need at least 1 frame to enroll")
            return False
        
        custom_name = self.enrollment_buffer['custom_name']
        yolo_class = self.enrollment_buffer['yolo_class']
        
        # Extract and aggregate features from all frames
        logger.info(f"Extracting features from {len(self.enrollment_buffer['images'])} frames...")
        features = self.matcher.extract_multi_frame_features(
            self.enrollment_buffer['images'],
            self.enrollment_buffer['bboxes']
        )
        
        # Validate features (check for either deep embeddings or traditional descriptors)
        if features.get('feature_type') == 'deep':
            if not features.get('deep_embedding'):
                logger.error("Failed to extract valid deep features")
                return False
        else:
            if not features.get('descriptors'):
                logger.error("Failed to extract valid features")
                return False
        
        # Save to database
        obj_id = self.db.add_remembered_object(
            custom_name=custom_name,
            yolo_class=yolo_class,
            visual_features=features,
            enrollment_frames=len(self.enrollment_buffer['images'])
        )
        
        if obj_id > 0:
            logger.info(f"[OK] Object '{custom_name}' enrolled successfully (ID: {obj_id})")
            if features.get('feature_type') == 'deep':
                logger.info(f"  Deep embedding: {features.get('embedding_size')}D vector")
                logger.info(f"  Frames aggregated: {features.get('num_frames')}")
            else:
                logger.info(f"  Keypoints: {len(features.get('keypoints', []))}")
                logger.info(f"  Descriptors: {len(features.get('descriptors', []))}")
            
            # Clear buffer
            self.enrollment_buffer = {
                'images': [],
                'bboxes': [],
                'yolo_class': None,
                'custom_name': None
            }
            self._candidates_cache = {}
            return True
        else:
            logger.error("Failed to save object to database")
            return False
    
    def cancel_enrollment(self) -> None:
        """Cancel current enrollment"""
        self.enrollment_buffer = {
            'images': [],
            'bboxes': [],
            'yolo_class': None,
            'custom_name': None
        }
        logger.info("Enrollment cancelled")
    
    def recognize_object(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                        yolo_class: str) -> Optional[Tuple[str, float]]:
        """
        Recognize a detected object by matching against remembered objects
        
        Args:
            image: Frame containing the object
            bbox: Bounding box (x1, y1, x2, y2)
            yolo_class: YOLO class of the detection
            
        Returns:
            (custom_name, confidence) if matched, None otherwise
        """
        if yolo_class in self._candidates_cache:
            timestamp, candidates = self._candidates_cache[yolo_class]
            if (time.time() - timestamp) < OBJECT_CANDIDATES_CACHE_TTL:
                pass
            else:
                candidates = self.db.get_remembered_objects_by_class(yolo_class)
                self._candidates_cache[yolo_class] = (time.time(), candidates)
        else:
            candidates = self.db.get_remembered_objects_by_class(yolo_class)
            self._candidates_cache[yolo_class] = (time.time(), candidates)
        
        if not candidates:
            # No remembered objects of this class
            return None
        
        # Extract features from detected object
        query_features = self.matcher.extract_features(image, bbox)
        
        # Validate features (check for either deep embeddings or traditional descriptors)
        if query_features.get('feature_type') == 'deep':
            if not query_features.get('deep_embedding'):
                logger.debug("No deep features extracted from detection")
                return None
        else:
            if not query_features.get('descriptors'):
                logger.debug("No features extracted from detection")
                return None
        
        # Find best match
        match = self.matcher.find_best_match(query_features, candidates)
        
        if match:
            obj_id, custom_name, confidence = match
            
            if confidence >= OBJECT_MATCH_HIGH_CONFIDENCE:
                logger.info(f"Recognized '{custom_name}' with high confidence ({confidence:.2%})")
                return (custom_name, confidence, "high")
            elif confidence >= OBJECT_MATCH_AMBIGUOUS_LOW:
                logger.info(f"Ambiguous match for '{custom_name}' ({confidence:.2%})")
                self.pending_confirmation = {
                    'custom_name': custom_name,
                    'confidence': confidence,
                    'timestamp': time.time()
                }
                return (custom_name, confidence, "ambiguous")
            else:
                logger.debug(f"Discarding low confidence match for '{custom_name}' ({confidence:.2%})")
                
        return None
    
    def update_object_sighting(self, custom_name: str, image: np.ndarray = None,
                               location: str = None, gps_lat: float = None,
                               gps_lon: float = None, nearby_objects: List[str] = None,
                               confidence: float = 0.0) -> bool:
        """
        Record a sighting of a remembered object
        
        Args:
            custom_name: Name of the remembered object
            image: Optional frame showing the object (for context screenshot)
            location: Text description of location
            gps_lat: GPS latitude
            gps_lon: GPS longitude
            nearby_objects: List of nearby detected objects
            confidence: Recognition confidence
            
        Returns:
            True if successful
        """
        now = time.time()
        last_update = self._last_sighting_update.get(custom_name, 0)
        if (now - last_update) < OBJECT_MATCH_COOLDOWN:
            return True  # Skip redundant DB write, sighting was updated too recently
        self._last_sighting_update[custom_name] = now

        # Get object from database
        obj = self.db.get_remembered_object(custom_name)
        if not obj:
            logger.error(f"Object '{custom_name}' not found in database")
            return False
        
        obj_id = obj[0]
        
        # Build context dictionary
        context = {}
        if nearby_objects:
            context['nearby_objects'] = nearby_objects
        if image is not None:
            # Could save screenshot here in the future
            context['screenshot_saved'] = False
        
        # Record sighting
        self.db.add_object_sighting(
            object_id=obj_id,
            location=location,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            context=context if context else None,
            confidence=confidence
        )
        
        logger.info(f"Updated sighting for '{custom_name}' at {location or '(unknown location)'}")
        return True
    
    def get_object_location(self, custom_name: str) -> Optional[Dict]:
        """
        Get the last known location of a remembered object
        
        Args:
            custom_name: Name of the object
            
        Returns:
            Dictionary with location info or None
        """
        sighting = self.db.get_sighting_by_name(custom_name)
        
        if sighting:
            logger.info(f"'{custom_name}' last seen: {sighting['location']} at {sighting['timestamp']}")
            return sighting
        
        logger.info(f"No sighting recorded for '{custom_name}'")
        return None

    PENDING_CONFIRM_TIMEOUT_SECONDS = 15.0

    def get_pending_match(self):
        """Return the pending ambiguous match, or None. A match older than
        PENDING_CONFIRM_TIMEOUT_SECONDS is discarded so C/X can't act on a stale sighting."""
        p = self.pending_confirmation
        if p and (time.time() - p['timestamp']) > self.PENDING_CONFIRM_TIMEOUT_SECONDS:
            logger.info(f"Pending match for '{p['custom_name']}' expired")
            self.pending_confirmation = None
            return None
        return p

    def confirm_pending_match(self, location: str = None, gps_lat: float = None, gps_lon: float = None) -> bool:
        """Confirm the currently pending ambiguous match and update its location."""
        if not self.get_pending_match():
            return False
            
        custom_name = self.pending_confirmation['custom_name']
        confidence = self.pending_confirmation['confidence']
        self.pending_confirmation = None
        
        return self.update_object_sighting(
            custom_name=custom_name,
            location=location,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            confidence=confidence
        )
        
    def clear_pending_match(self):
        """Clear the pending confirmation state."""
        self.pending_confirmation = None
    
    def list_remembered_objects(self) -> List[Tuple[str, str]]:
        """
        Get list of all remembered objects
        
        Returns:
            List of (custom_name, yolo_class) tuples
        """
        objects = self.db.get_all_remembered_objects()
        return [(name, yolo_class) for _, name, yolo_class in objects]
    
    def delete_object(self, custom_name: str) -> bool:
        """
        Delete a remembered object
        
        Args:
            custom_name: Name of the object to delete
            
        Returns:
            True if successful
        """
        success = self.db.delete_remembered_object(custom_name)
        if success:
            logger.info(f"Deleted '{custom_name}'")
            self._candidates_cache = {}
        else:
            logger.warning(f"Failed to delete '{custom_name}' (not found?)")
        return success
    
    def get_enrollment_progress(self) -> Dict:
        """
        Get current enrollment progress
        
        Returns:
            Dictionary with enrollment status
        """
        return {
            'active': self.enrollment_buffer['custom_name'] is not None,
            'custom_name': self.enrollment_buffer['custom_name'],
            'yolo_class': self.enrollment_buffer['yolo_class'],
            'frames_captured': len(self.enrollment_buffer['images']),
            'frames_needed': OBJECT_ENROLLMENT_FRAMES
        }


if __name__ == "__main__":
    # Quick test
    print("Object Memory Test")
    print("=" * 60)
    
    memory = ObjectMemory()
    print("[OK] Object memory initialized")
    
    # Test enrollment workflow
    print("\nTest 1: Start enrollment")
    memory.start_enrollment("test keys", "keys")
    progress = memory.get_enrollment_progress()
    print(f"  Active: {progress['active']}")
    print(f"  Object: {progress['custom_name']}")
    print(f"  Frames: {progress['frames_captured']}/{progress['frames_needed']}")
    
    # Add mock frames
    print("\nTest 2: Add enrollment frames")
    for i in range(3):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 200)
        num = memory.add_enrollment_frame(img, bbox)
        print(f"  Frame {num} added")
    
    # Finish enrollment
    print("\nTest 3: Finish enrollment")
    success = memory.finish_enrollment()
    print(f"  Success: {success}")
    
    # List objects
    print("\nTest 4: List remembered objects")
    objects = memory.list_remembered_objects()
    print(f"  Found {len(objects)} objects:")
    for name, yolo_class in objects:
        print(f"    - {name} ({yolo_class})")
    
    # Update sighting
    print("\nTest 5: Update sighting")
    memory.update_object_sighting(
        "test keys",
        location="desk",
        gps_lat=37.7749,
        gps_lon=-122.4194,
        nearby_objects=["laptop", "phone"],
        confidence=0.85
    )
    
    # Query location
    print("\nTest 6: Query location")
    location = memory.get_object_location("test keys")
    if location:
        print(f"  Location: {location['location']}")
        print(f"  GPS: ({location['gps_lat']}, {location['gps_lon']})")
        print(f"  Context: {location['context']}")
    
    # Cleanup
    print("\nTest 7: Delete object")
    memory.delete_object("test keys")
    objects = memory.list_remembered_objects()
    print(f"  Remaining: {len(objects)} objects")
    
    print("\n" + "=" * 60)
    print("Object memory is ready!")
