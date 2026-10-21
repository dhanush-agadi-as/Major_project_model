"""
Object Memory Manager for VisionMate
High-level interface for enrolling and recognizing custom objects
"""
import cv2
import numpy as np
import time
import threading
import re
import collections
from typing import List, Tuple, Optional, Dict
import logging
from datetime import datetime

from database import VisionMateDB, is_legacy_62720d
from visual_matcher import VisualMatcher
from config import (
    OBJECT_MATCH_HIGH_CONFIDENCE, SINGLE_OBJECT_MATCH_MIN, OBJECT_MATCH_AMBIGUOUS_LOW, 
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
        self._lock = threading.RLock()
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
        self._last_sighting_call = {}     # maps custom_name -> unix timestamp of last update_object_sighting call
        self._candidates_cache = {}       # maps yolo_class -> (timestamp, list_of_candidates) for cached DB reads
        self._last_match_state = {}       # maps custom_name -> last_tier ('high', 'ambiguous')
        self._last_match_log_time = {}    # maps (custom_name, tier) -> timestamp of last INFO log
        
        # Timing statistics for stages log & true rolling averages (rolling down to 0.0 when unused)
        self.last_embed_ms = 0.0
        self.last_compare_ms = 0.0
        self.last_db_ms = 0.0
        self._rolling_embed_ms = collections.deque(maxlen=30)
        self._rolling_compare_ms = collections.deque(maxlen=30)
        self._rolling_db_ms = collections.deque(maxlen=30)

        # In-RAM embedding cache structures (zero DB reading inside per-frame path)
        self._cached_objects = {}         # custom_name -> dict(id, custom_name, yolo_class, visual_features, aliases)
        self._class_to_names = collections.defaultdict(list)  # yolo_class -> list of custom_name
        self._cached_class_matrices = {}  # yolo_class -> {'matrix': 2D np.ndarray, 'row_to_name': list[str], 'candidates': list[tuple]}
        self._all_embeddings_matrix = np.empty((0, 0), dtype=np.float32)
        self._all_embeddings_labels = []
        self._logged_not_found = set()    # Suppress duplicate 'not found in database' logs to at most once

        try:
            self.refresh_cache()
        except Exception as e:
            logger.debug(f"Error initializing in-RAM embedding cache: {e}")
        
        logger.info("Object memory initialized")

    @property
    def embed_ms(self) -> float:
        if not self._rolling_embed_ms:
            return 0.0
        return float(np.mean(self._rolling_embed_ms))

    @property
    def compare_ms(self) -> float:
        if not self._rolling_compare_ms:
            return 0.0
        return float(np.mean(self._rolling_compare_ms))

    @property
    def db_ms(self) -> float:
        if not self._rolling_db_ms:
            return 0.0
        return float(np.mean(self._rolling_db_ms))
    
    def refresh_cache(self) -> None:
        """
        Load all remembered objects' embeddings from the database into RAM caches.
        Refreshed only after startup, enrollment, deletion, or renaming.
        Never accessed per-frame from SQLite.
        Detects legacy 62720D rows, ignores them, and prompts the user to re-enroll.
        """
        with self._lock:
            self._logged_not_found.clear()
            self._cached_objects = {}
            self._class_to_names = collections.defaultdict(list)
            self._cached_class_matrices = {}
            all_embeddings = []
            all_labels = []

            try:
                raw_objects = self.db.get_all_remembered_objects()
            except Exception as e:
                logger.debug(f"Error reading objects from DB in refresh_cache: {e}")
                raw_objects = []

            for row in raw_objects:
                obj_id, custom_name, yolo_class = row[0], row[1], row[2]
                try:
                    full_obj = self.db.get_remembered_object(custom_name)
                except Exception:
                    full_obj = None
                if not full_obj:
                    continue
                _, _, _, features, frames = full_obj
                if is_legacy_62720d(features):
                    logger.warning(
                        f"Old 62720D row detected in database for '{custom_name}'. "
                        f"Invalid embedding dimension (62720D); ignoring object. Please re-enroll '{custom_name}'."
                    )
                    continue
                aliases = features.get('aliases', []) if isinstance(features, dict) else []
                self._cached_objects[custom_name] = {
                    'id': obj_id,
                    'custom_name': custom_name,
                    'yolo_class': yolo_class,
                    'visual_features': features,
                    'enrollment_frames': frames,
                    'aliases': aliases
                }
                self._class_to_names[yolo_class].append(custom_name)

            for yolo_class, names in self._class_to_names.items():
                class_embs = []
                class_row_labels = []
                class_candidates = []
                for name in names:
                    obj = self._cached_objects[name]
                    class_candidates.append((obj['id'], name, obj['yolo_class'], obj['visual_features'], obj.get('aliases', [])))
                    vf = obj['visual_features']
                    if isinstance(vf, dict):
                        stored = vf.get('deep_embeddings')
                        if stored and isinstance(stored, list) and len(stored) > 0:
                            for e in stored:
                                arr = np.array(e, dtype=np.float32).flatten()
                                n = np.linalg.norm(arr)
                                if n > 0:
                                    arr = arr / n
                                class_embs.append(arr)
                                class_row_labels.append(name)
                                all_embeddings.append(arr)
                                all_labels.append(name)
                        elif vf.get('deep_embedding') is not None:
                            arr = np.array(vf['deep_embedding'], dtype=np.float32).flatten()
                            n = np.linalg.norm(arr)
                            if n > 0:
                                arr = arr / n
                            class_embs.append(arr)
                            class_row_labels.append(name)
                            all_embeddings.append(arr)
                            all_labels.append(name)

                if class_embs:
                    mat = np.stack(class_embs, axis=0)
                else:
                    mat = np.empty((0, 0), dtype=np.float32)

                self._cached_class_matrices[yolo_class] = {
                    'matrix': mat,
                    'row_to_name': class_row_labels,
                    'candidates': class_candidates
                }

            if all_embeddings:
                self._all_embeddings_matrix = np.stack(all_embeddings, axis=0)
                self._all_embeddings_labels = all_labels
            else:
                self._all_embeddings_matrix = np.empty((0, 0), dtype=np.float32)
                self._all_embeddings_labels = []

            self._candidates_cache = {}
            logger.debug(f"Refreshed embedding cache: {len(self._cached_objects)} objects, {len(all_embeddings)} embeddings in RAM")

    def get_cached_candidates(self, yolo_class: str) -> List[Tuple]:
        """Return remembered object candidates from RAM cache without DB access."""
        with self._lock:
            if yolo_class in self._cached_class_matrices:
                cands = self._cached_class_matrices[yolo_class]['candidates']
                if cands:
                    return cands
            # Fallback only if cache is completely unpopulated (e.g. mock DB in tests)
            if not self._cached_objects:
                try:
                    return self.db.get_remembered_objects_by_class(yolo_class)
                except Exception:
                    return []
            return []

    def get_cached_object(self, custom_name: str) -> Optional[dict]:
        """Get remembered object details from RAM cache with case/alias tolerance."""
        if not custom_name:
            return None
        with self._lock:
            if custom_name in self._cached_objects:
                return self._cached_objects[custom_name]
            clean_q = custom_name.lower().strip()
            # Direct case-insensitive match
            for name, obj in self._cached_objects.items():
                if name.lower().strip() == clean_q:
                    return obj
            # Check aliases
            for name, obj in self._cached_objects.items():
                for alias in obj.get('aliases', []):
                    if str(alias).lower().strip() == clean_q:
                        return obj
            # Check clean object name match (e.g. 'my phone' -> 'phone')
            clean_noprefix = re.sub(r'^(?:(?:my|the|a|an)\s+)+', '', clean_q).strip()
            for name, obj in self._cached_objects.items():
                name_noprefix = re.sub(r'^(?:(?:my|the|a|an)\s+)+', '', name.lower().strip()).strip()
                if name_noprefix == clean_noprefix:
                    return obj
        return None

    def refresh_embedding_cache(self) -> None:
        """Alias for refresh_cache to maintain explicit API naming."""
        self.refresh_cache()
    
    def start_enrollment(self, custom_name: str, yolo_class: str, aliases: Optional[List[str]] = None) -> None:
        """
        Start enrolling a new object
        
        Args:
            custom_name: User-friendly name (e.g., "house keys")
            yolo_class: YOLO detection class (e.g., "keys")
            aliases: Optional list of alternate names / aliases for this object
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
            'custom_name': custom_name,
            'aliases': [str(a).strip().lower() for a in aliases if str(a).strip()] if aliases else []
        }
        
        logger.info(f"Started enrollment for '{custom_name}' ({yolo_class}) aliases={self.enrollment_buffer['aliases']}")
    
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
        
        # Attach aliases if defined during enrollment
        if self.enrollment_buffer.get('aliases'):
            features['aliases'] = self.enrollment_buffer['aliases']

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
                logger.info("Deep embedding: 1280D vector")
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
                'custom_name': None,
                'aliases': []
            }
            self._candidates_cache = {}
            try:
                self.refresh_cache()
            except Exception as e:
                logger.debug(f"Error refreshing cache after enrollment: {e}")
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
            'custom_name': None,
            'aliases': []
        }
        logger.info("Enrollment cancelled")

    def get_object_aliases(self, custom_name: str) -> List[str]:
        """Get list of aliases for a remembered object."""
        with self._lock:
            cached = self._cached_objects.get(custom_name)
            if cached:
                return cached.get('aliases', [])
        obj = self.db.get_remembered_object(custom_name)
        if obj:
            features = obj[3]
            if isinstance(features, dict):
                return features.get('aliases', [])
        return []
    
    def recognize_object(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                        yolo_class: str) -> Optional[Tuple[str, float]]:
        """
        Recognize a detected object by matching against remembered objects.
        Uses cached numpy embedding matrix in RAM (never hits DB on per-frame path).
        
        Args:
            image: Frame containing the object
            bbox: Bounding box (x1, y1, x2, y2)
            yolo_class: YOLO class of the detection
            
        Returns:
            (custom_name, confidence, tier) if matched, None otherwise
        """
        self.last_embed_ms = 0.0
        self.last_compare_ms = 0.0

        # 1. Fetch candidates from in-RAM cache
        class_info = None
        with self._lock:
            if yolo_class in self._cached_class_matrices:
                class_info = self._cached_class_matrices[yolo_class]
                candidates = class_info['candidates']
            else:
                candidates = []

        # If cache is empty or class missing, fallback to DB only if cache has not been populated
        if not candidates:
            with self._lock:
                if not self._cached_objects:
                    candidates = self.db.get_remembered_objects_by_class(yolo_class)
                else:
                    self._rolling_embed_ms.append(0.0)
                    self._rolling_compare_ms.append(0.0)
                    return None
        
        if not candidates:
            # No remembered objects of this class
            self._rolling_embed_ms.append(0.0)
            self._rolling_compare_ms.append(0.0)
            return None
        
        # Extract features from detected object
        t_embed_start = time.perf_counter()
        query_features = self.matcher.extract_features(image, bbox)
        self.last_embed_ms = (time.perf_counter() - t_embed_start) * 1000.0
        self._rolling_embed_ms.append(self.last_embed_ms)
        
        # Validate features (check for either deep embeddings or traditional descriptors)
        if query_features.get('feature_type') == 'deep':
            if query_features.get('deep_embedding') is None:
                logger.debug("No deep features extracted from detection")
                self._rolling_compare_ms.append(0.0)
                return None
        else:
            if query_features.get('descriptors') is None:
                logger.debug("No features extracted from detection")
                self._rolling_compare_ms.append(0.0)
                return None
        
        # Fast path: Vectorized matrix matching in RAM
        t_comp_start = time.perf_counter()
        match = None
        if query_features.get('feature_type') == 'deep' and class_info and class_info['matrix'].size > 0:
            best_name, confidence = self.matcher.match_vectorized(
                query_features['deep_embedding'],
                class_info['matrix'],
                class_info['row_to_name']
            )
            if best_name is not None and confidence >= getattr(self.matcher, 'match_threshold', OBJECT_MATCH_AMBIGUOUS_LOW):
                obj_id = self._cached_objects.get(best_name, {}).get('id', 0)
                match = (obj_id, best_name, confidence)
        
        if match is None:
            # Fallback for traditional features (ORB) or test mocks
            match = self.matcher.find_best_match(query_features, candidates)
        self.last_compare_ms = (time.perf_counter() - t_comp_start) * 1000.0
        self._rolling_compare_ms.append(self.last_compare_ms)
        
        if match:
            obj_id, custom_name, confidence = match
            
            # Tier rule:
            # If exactly one remembered object exists for this class, similarity >= SINGLE_OBJECT_MATCH_MIN (0.60)
            # is treated as high-confidence (no question).
            # Only ask when two or more remembered objects of the same class compete, or similarity is below 0.60.
            if len(candidates) == 1:
                # Class gating: exactly one object enrolled of this class -> similarity >= 0.35 is high tier
                if confidence >= SINGLE_OBJECT_MATCH_MIN:
                    tier = "high"
                else:
                    logger.debug(f"Discarding low confidence match for '{custom_name}' ({confidence:.2%})")
                    return None
            else:
                if confidence >= OBJECT_MATCH_HIGH_CONFIDENCE:
                    tier = "high"
                elif confidence >= OBJECT_MATCH_AMBIGUOUS_LOW:
                    tier = "ambiguous"
                    self.pending_confirmation = {
                        'custom_name': custom_name,
                        'confidence': confidence,
                        'timestamp': time.time()
                    }
                else:
                    logger.debug(f"Discarding low confidence match for '{custom_name}' ({confidence:.2%})")
                    return None

            now = time.time()
            prev_tier = self._last_match_state.get(custom_name)
            last_info_time = self._last_match_log_time.get((custom_name, tier), 0.0)

            is_new_state = (prev_tier is None or prev_tier != tier)
            can_log_info = is_new_state and (now - last_info_time >= 10.0)

            msg = (f"Recognized '{custom_name}' with high confidence ({confidence:.2%})"
                   if tier == "high"
                   else f"Ambiguous match for '{custom_name}' ({confidence:.2%})")

            if can_log_info:
                logger.info(msg)
                self._last_match_log_time[(custom_name, tier)] = now
            else:
                logger.debug(msg)

            self._last_match_state[custom_name] = tier
            return (custom_name, confidence, tier)
                
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
        if custom_name in self._logged_not_found:
            return False

        now = time.time()
        last_update = self._last_sighting_update.get(custom_name, 0)
        if (now - last_update) < OBJECT_MATCH_COOLDOWN:
            return True  # Skip redundant DB write, sighting was updated too recently

        # Get object id from cache if available, else database
        obj_id = None
        with self._lock:
            if custom_name in self._cached_objects:
                obj_id = self._cached_objects[custom_name].get('id')
        if obj_id is None:
            obj = self.db.get_remembered_object(custom_name)
            if not obj:
                if custom_name not in self._logged_not_found:
                    self._logged_not_found.add(custom_name)
                    logger.error(f"Object '{custom_name}' not found in database")
                return False
            obj_id = obj[0]

        self._last_sighting_update[custom_name] = now
        
        # Build context dictionary
        context = {}
        if nearby_objects:
            context['nearby_objects'] = nearby_objects
        if image is not None:
            # Could save screenshot here in the future
            context['screenshot_saved'] = False
        
        # Record sighting
        t_db_start = time.perf_counter()
        self.db.add_object_sighting(
            object_id=obj_id,
            location=location,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            context=context if context else None,
            confidence=confidence
        )
        self.last_db_ms = (time.perf_counter() - t_db_start) * 1000.0
        self._rolling_db_ms.append(self.last_db_ms)
        
        logger.debug(f"Updated sighting for '{custom_name}' at {location or '(unknown location)'}")
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
        with self._lock:
            if self._cached_objects:
                return [(name, o['yolo_class']) for name, o in self._cached_objects.items()]
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
            try:
                self.refresh_cache()
            except Exception as e:
                logger.debug(f"Error refreshing cache after delete: {e}")
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
