"""
Spatial Navigator for VisionMate
Provides real-time obstacle detection and directional guidance for blind users
"""
import time
import math
from typing import List, Dict, Tuple, Optional
import logging

try:
    import config
except ImportError:
    config = None

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
    
    # Guidance priority: routine guidance is navigational (Priority 3)
    ZONE_PRIORITY = {
        'bottom-left': 3, 'bottom': 3, 'bottom-right': 3,
        'left': 3, 'center': 3, 'right': 3,
        'top-left': 3, 'top': 3, 'top-right': 3
    }
    
    # Obstacle classes that require immediate guidance
    OBSTACLE_CLASSES = ['person', 'chair', 'couch', 'bicycle', 'car', 'motorcycle', 
                        'bus', 'truck', 'bench', 'potted plant', 'dog', 'cat']

    DISTANCE_BUCKET_RANKS = {
        'far': 0,
        'medium': 1,
        'close': 2,
        'very-close': 3,
        'immediate': 4
    }
    
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
        self.frame_area = frame_width * frame_height
        
        # Calculate zone boundaries
        self.zone_width = frame_width / 3
        self.zone_height = frame_height / 3

        # Tracking for speech guidance repeat suppression
        self._last_spoken_time = {}          # (class_name, zone) -> timestamp
        self._last_spoken_distance = {}      # (class_name, zone) -> distance bucket
        self._last_seen_time = {}            # (class_name, zone) -> timestamp
        self._was_absent_over_3s = {}        # (class_name, zone) -> bool
        self._last_critical_spoken_time = {} # (class_name, zone) -> timestamp

        # Event-based obstacle tracking across frames
        self._next_track_id = 1
        self._active_tracks = {}             # track_id -> dict
        self._last_person_count = 0
        self._last_person_seen_time = None
        self._person_announced = False
        self._person_was_absent_over_5s = True
        self.is_find_active = False

        # Group person announcements (count, closest zone, 6s cooldown)
        self._last_person_announce_time = -999.0
        self._last_announced_person_count = 0
        self._last_announced_person_zone = None
        self._person_announced_under_1m = False

        # Spatial speech repetition suppression
        self._last_spatial_announcement_time = -999.0
        self._last_spoken_info = {}  # class_name -> {'time': float, 'dist_m': float, 'direction': str, 'bucket': str}
        
        logger.info(f"Spatial Navigator initialized ({frame_width}x{frame_height})")
        logger.info(f"  Zone size: {self.zone_width:.0f}x{self.zone_height:.0f}")
        logger.info(f"  Distance threshold: {distance_threshold}")

    def set_frame_size(self, width: int, height: int):
        """Update frame dimensions dynamically from real camera stream."""
        self.frame_width = width
        self.frame_height = height
        self.frame_area = width * height
        self.zone_width = width / 3.0
        self.zone_height = height / 3.0
    
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
    
    def update_presence(self, detections: List[Dict], current_time: float = None):
        """Update presence and centroid tracking for obstacle objects across frames."""
        if current_time is None:
            current_time = time.time()

        # Update fallback zone-based last_seen_time
        for det in detections:
            class_name = det.get('class_name') or det.get('class') or det.get('label') or ''
            if not self.is_obstacle(class_name):
                continue
            bbox = det.get('bbox')
            if not bbox:
                continue
            zone = self.get_object_zone(bbox)
            key = (class_name, zone)

            last_seen = self._last_seen_time.get(key)
            if last_seen is not None and (current_time - last_seen) > 3.0:
                self._was_absent_over_3s[key] = True
            self._last_seen_time[key] = current_time

        # Centroid matching for obstacle objects across frames
        current_person_count = sum(1 for det in detections if (det.get('class_name') or det.get('class') or det.get('label')) == 'person')
        prev_person_count = getattr(self, '_last_person_count', 0)
        person_count_increased = (current_person_count > prev_person_count and prev_person_count > 0)
        self._last_person_count = current_person_count

        if current_person_count > 0:
            if self._last_person_seen_time is None:
                person_was_absent_over_5s = True
            else:
                person_was_absent_over_5s = ((current_time - self._last_person_seen_time) >= 5.0)
            self._last_person_seen_time = current_time
            if person_was_absent_over_5s:
                self._person_announced = False
                self._person_was_absent_over_5s = True
                self._person_announced_under_1m = False
                self._last_announced_person_count = 0
                self._last_announced_person_zone = None
            else:
                self._person_was_absent_over_5s = False
        else:
            if self._last_person_seen_time is not None:
                if (current_time - self._last_person_seen_time) >= 5.0:
                    self._person_announced = False
                    self._person_was_absent_over_5s = True
                    self._person_announced_under_1m = False
                    self._last_announced_person_count = 0
                    self._last_announced_person_zone = None
            person_was_absent_over_5s = False

        candidates = []
        for det in detections:
            cname = det.get('class_name') or det.get('class') or det.get('label') or ''
            if not self.is_obstacle(cname):
                continue
            bbox = det.get('bbox')
            if not bbox:
                continue
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            dist_bucket = self.estimate_distance(bbox)
            zone = self.get_object_zone(bbox)
            candidates.append({
                'det': det,
                'class_name': cname,
                'bbox': bbox,
                'centroid': (cx, cy),
                'distance': dist_bucket,
                'zone': zone
            })

        matched_tracks = set()
        matched_candidates = set()

        # Match candidates to active tracks of the same class_name by centroid distance
        pairs = []
        for c_idx, c in enumerate(candidates):
            for tr_id, tr in self._active_tracks.items():
                if tr['class_name'] == c['class_name']:
                    d = math.hypot(c['centroid'][0] - tr['centroid'][0], c['centroid'][1] - tr['centroid'][1])
                    pairs.append((d, tr_id, c_idx))

        pairs.sort(key=lambda x: x[0])
        max_match_dist = self.frame_width * 0.6  # Matching distance threshold

        for d, tr_id, c_idx in pairs:
            if tr_id in matched_tracks or c_idx in matched_candidates:
                continue
            if d > max_match_dist:
                continue
            matched_tracks.add(tr_id)
            matched_candidates.add(c_idx)

            track = self._active_tracks[tr_id]
            c = candidates[c_idx]

            # Check if this object was absent
            time_since_seen = current_time - track.get('last_seen_time', current_time)
            threshold_absent = 5.0 if track['class_name'] == 'person' else 3.0
            if time_since_seen > threshold_absent or (track.get('absent_since') is not None and (current_time - track['absent_since']) > threshold_absent):
                track['was_absent_over_3s'] = True
                track['announced_buckets'] = set()
                track['initial_announced'] = False
            track['absent_since'] = None

            track['centroid'] = c['centroid']
            track['bbox'] = c['bbox']
            track['zone'] = c['zone']
            track['distance'] = c['distance']
            track['last_seen_time'] = current_time

            c['det']['track_id'] = tr_id

        # Handle unmatched tracks (absent in this frame)
        for tr_id, tr in list(self._active_tracks.items()):
            if tr_id not in matched_tracks:
                if tr.get('absent_since') is None:
                    tr['absent_since'] = current_time
                elif (current_time - tr['absent_since']) > 60.0:
                    del self._active_tracks[tr_id]

        # Handle unmatched candidates (either re-entering after absence or brand new)
        for c_idx, c in enumerate(candidates):
            if c_idx not in matched_candidates:
                # Check if an existing track of same class was absent
                reassoc_id = None
                for tr_id, tr in self._active_tracks.items():
                    if tr_id not in matched_tracks and tr['class_name'] == c['class_name']:
                        if tr['class_name'] == 'person' or (tr.get('absent_since') is not None and (current_time - tr['absent_since']) > 3.0):
                            reassoc_id = tr_id
                            break

                if reassoc_id is not None:
                    matched_tracks.add(reassoc_id)
                    track = self._active_tracks[reassoc_id]
                    threshold_absent = 5.0 if track['class_name'] == 'person' else 3.0
                    was_absent = (current_time - (track.get('absent_since') or current_time)) >= threshold_absent
                    if was_absent:
                        track['was_absent_over_3s'] = True
                        track['announced_buckets'] = set()
                        track['initial_announced'] = False
                    track['absent_since'] = None
                    track['centroid'] = c['centroid']
                    track['bbox'] = c['bbox']
                    track['zone'] = c['zone']
                    track['distance'] = c['distance']
                    track['last_seen_time'] = current_time
                    c['det']['track_id'] = reassoc_id
                else:
                    tr_id = self._next_track_id
                    self._next_track_id += 1
                    self._active_tracks[tr_id] = {
                        'track_id': tr_id,
                        'class_name': c['class_name'],
                        'centroid': c['centroid'],
                        'bbox': c['bbox'],
                        'distance': c['distance'],
                        'zone': c['zone'],
                        'announced_buckets': set(),
                        'initial_announced': False,
                        'person_count_increased': person_count_increased if c['class_name'] == 'person' else False,
                        'last_seen_time': current_time,
                        'absent_since': None,
                        'was_absent_over_3s': False,
                    }
                    c['det']['track_id'] = tr_id

    def generate_guidance(self, detections: List[Dict], current_time: float = None,
                          ultrasonic_distance: Optional[float] = None,
                          is_find_active: bool = False) -> List[Dict]:
        """
        Generate spatial guidance for all detected obstacles.
        
        Args:
            detections: List of detected objects with bbox, class_name, confidence
            current_time: Optional timestamp
            ultrasonic_distance: Optional distance reading from ultrasonic sensor (cm)
            is_find_active: True if find session is active
            
        Returns:
            List of guidance messages with priority:
            [{'message': str, 'priority': int, 'zone': str, 'distance': str}]
        """
        if is_find_active:
            self.is_find_active = is_find_active
        if current_time is None:
            current_time = time.time()
        self.update_presence(detections, current_time=current_time)

        zone_speech_map = {
            'top-left': 'left', 'top': 'ahead', 'top-right': 'right',
            'left': 'left', 'center': 'ahead', 'middle': 'ahead', 'right': 'right',
            'bottom-left': 'left', 'bottom': 'ahead', 'bottom-right': 'right', 'ahead': 'ahead'
        }

        guidance_list = []
        occupied_zones = []

        person_dets = []
        non_person_dets = []

        for det in detections:
            class_name = det.get('class_name') or det.get('class') or det.get('label') or ''
            if not self.is_obstacle(class_name):
                continue
            bbox = det.get('bbox')
            if not bbox:
                continue
            zone = self.get_object_zone(bbox)
            occupied_zones.append(zone)
            if class_name == 'person':
                person_dets.append(det)
            else:
                non_person_dets.append(det)

        # 1. Non-person obstacles
        for det in non_person_dets:
            class_name = det.get('class_name') or det.get('class') or det.get('label') or ''
            bbox = det['bbox']
            zone = self.get_object_zone(bbox)
            distance = self.estimate_distance(bbox)
            message = self._create_guidance_message(class_name, zone, distance)

            x1, y1, x2, y2 = bbox
            area_ratio = ((x2 - x1) * (y2 - y1)) / self.frame_area
            if area_ratio > 0.35:
                est_m = 0.45
            elif area_ratio > 0.15:
                est_m = 1.0
            elif area_ratio > 0.05:
                est_m = 2.0
            else:
                est_m = 4.0

            priority = 1 if distance == 'very-close' else 3

            guidance_list.append({
                'message': message,
                'priority': priority,
                'zone': zone,
                'distance': distance,
                'class_name': class_name,
                'type': 'obstacle',
                'bbox': bbox,
                'estimated_distance_m': est_m,
                'track_id': det.get('track_id')
            })

        # 2. Group person obstacles into ONE combined announcement
        if person_dets:
            def _det_area(d):
                b = d.get('bbox', [0, 0, 0, 0])
                return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

            closest_det = max(person_dets, key=_det_area)
            closest_bbox = closest_det['bbox']
            closest_zone = self.get_object_zone(closest_bbox)
            closest_dist = self.estimate_distance(closest_bbox)
            closest_zone_speech = zone_speech_map.get(closest_zone, 'ahead')

            person_count = len(person_dets)

            # Check for recognized known faces
            known_names = []
            for d in person_dets:
                fn = d.get('face_name')
                if fn and fn != 'Unknown' and fn not in known_names:
                    known_names.append(fn)

            unknown_count = person_count - len(known_names)

            if known_names:
                known_str = " and ".join(known_names)
                if unknown_count == 0:
                    message = f"{known_str} {closest_zone_speech}"
                elif unknown_count == 1:
                    message = f"{known_str} {closest_zone_speech}, and one more person"
                else:
                    message = f"{known_str} {closest_zone_speech}, and {unknown_count} more people"
            else:
                if person_count == 1:
                    message = self._create_guidance_message('person', closest_zone, closest_dist)
                else:
                    message = f"{person_count} people {closest_zone_speech}"

            x1, y1, x2, y2 = closest_bbox
            area_ratio = ((x2 - x1) * (y2 - y1)) / self.frame_area
            if area_ratio > 0.35:
                est_m = 0.45
            elif area_ratio > 0.15:
                est_m = 1.0
            elif area_ratio > 0.05:
                est_m = 2.0
            else:
                est_m = 4.0

            u_cm = None
            if ultrasonic_distance is not None:
                u_cm = ultrasonic_distance * 100.0 if ultrasonic_distance <= 5.0 else ultrasonic_distance
            is_real_under_50 = (
                u_cm is not None
                and u_cm < 50.0
            )
            priority = 1 if (is_real_under_50 or closest_dist == 'very-close') else 3

            guidance_list.append({
                'message': message,
                'priority': priority,
                'zone': closest_zone,
                'zone_speech': closest_zone_speech,
                'distance': closest_dist,
                'class_name': 'person',
                'type': 'person',
                'count': person_count,
                'bbox': closest_bbox,
                'estimated_distance_m': est_m,
                'track_id': closest_det.get('track_id')
            })
        
        # Add overall directional guidance if obstacles present
        if occupied_zones:
            clear_direction = self.get_clear_direction(occupied_zones)
            if clear_direction:
                direction_message = f"Move {clear_direction}"
                dir_priority = 3  # Navigational
                guidance_list.append({
                    'message': direction_message,
                    'priority': dir_priority,
                    'zone': 'navigation',
                    'distance': 'immediate',
                    'class_name': 'navigation',
                    'type': 'navigation',
                    'bbox': None
                })
            else:
                guidance_list.append({
                    'message': "Path blocked, stop",
                    'priority': 1,
                    'zone': 'navigation',
                    'distance': 'immediate',
                    'class_name': 'emergency',
                    'type': 'emergency',
                    'bbox': None,
                    'estimated_distance_m': 0.30
                })
        
        # Sort guidance_list:
        # Obstacle guidance (e.g. Person ahead, Chair left) comes first,
        # with active events prioritized, followed by higher priority and closer distance.
        def _sort_key(item):
            is_obs = self.is_obstacle(item.get('class_name', ''))
            should_spk = self.should_speak_guidance(item, current_time=current_time)
            dist_rank = self.DISTANCE_BUCKET_RANKS.get(item.get('distance', ''), 0)
            return (
                0 if is_obs else 1,
                0 if should_spk else 1,
                item.get('priority', 3),
                -dist_rank
            )

        guidance_list.sort(key=_sort_key)
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
        # Simplify zone names for speech: use only left, ahead, right (no vertical words)
        zone_speech = {
            'top-left': 'left',
            'top': 'ahead',
            'top-right': 'right',
            'left': 'left',
            'center': 'ahead',
            'middle': 'ahead',
            'right': 'right',
            'bottom-left': 'left',
            'bottom': 'ahead',
            'bottom-right': 'right',
            'ahead': 'ahead',
        }
        
        distance_speech = {
            'very-close': '',  # Don't mention distance, it's urgent
            'close': 'close',
            'medium': '',
            'far': ''
        }
        
        zone_desc = zone_speech.get(zone)
        if not zone_desc:
            if 'left' in str(zone):
                zone_desc = 'left'
            elif 'right' in str(zone):
                zone_desc = 'right'
            else:
                zone_desc = 'ahead'
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
        Map distance to audio priority level. Only very-close (<50 cm) is emergency.
        """
        if distance == 'very-close':
            return 1  # EMERGENCY
        return 3      # NAVIGATIONAL

    def is_critical(self, guidance_item: Dict, ultrasonic_distance: float = 100.0) -> bool:
        """
        Check if guidance warning is critical (<50 cm).
        Only a warning with a real distance under 50 cm (ultrasonic reading) may use emergency priority.
        Never treat priority 1 itself as critical.
        """
        class_name = guidance_item.get('class_name', '')
        u_cm = None
        if ultrasonic_distance is not None:
            u_cm = ultrasonic_distance * 100.0 if ultrasonic_distance <= 5.0 else ultrasonic_distance
        is_real_under_50 = (
            u_cm is not None
            and u_cm < 50.0
        )
        if class_name == 'person':
            # For person, ONLY a real distance under 50 cm may be emergency priority / critical
            return is_real_under_50

        if is_real_under_50:
            return True
        if guidance_item.get('distance') == 'very-close':
            return True
        if guidance_item.get('estimated_distance_m', 999.0) < 0.50:
            return True
        if class_name == 'emergency':
            return True
        if guidance_item.get('distance') == 'immediate' and 'stop' in guidance_item.get('message', '').lower():
            return True
        return False

    def should_speak_guidance(self, guidance_item: Dict, ultrasonic_distance: float = 100.0,
                              current_time: float = None, is_find_active: bool = False) -> bool:
        """
        Event-based guidance repeat suppression:
        - Routine announcements speak ONLY on events (no time-based repeats):
            (1) A new person/object enters view
            (2) The same object left view for more than 3 seconds (or 5s for person) and came back
            (3) The distance bucket gets closer (far to medium to close to very-close)
                (each bucket announced once per object)
        - "Person ahead": only announce when a person was absent for >= 5 s (not a single missed frame),
          and never during an active find session unless the distance is under 50 cm.
        - While the same person stays in view and nothing changes, say nothing.
        - Real critical warnings under 50 cm bypass limits and may repeat every 3 seconds (non-person).
        """
        if current_time is None:
            current_time = time.time()

        class_name = guidance_item.get('class_name') or guidance_item.get('type') or ''
        zone = guidance_item.get('zone', '')
        key = (class_name, zone)
        find_active = is_find_active or getattr(self, 'is_find_active', False)

        # Check if object/obstacle is under 50 cm
        u_cm = None
        u_m = None
        if ultrasonic_distance is not None:
            u_cm = ultrasonic_distance * 100.0 if ultrasonic_distance <= 5.0 else ultrasonic_distance
            u_m = ultrasonic_distance if ultrasonic_distance <= 5.0 else (ultrasonic_distance / 100.0)

        # Real critical warning (<50 cm)
        is_crit = self.is_critical(guidance_item, ultrasonic_distance)
        is_under_50cm = (
            is_crit
            or (u_cm is not None and u_cm < 50.0)
            or (guidance_item.get('estimated_distance_m', 999.0) < 0.50)
            or (guidance_item.get('distance') == 'very-close')
        )

        # During an active find session: NO spatial speech at all ("Person ahead/left/right",
        # "Chair left", "2 people ahead", "Move left") unless an object is closer than 50 cm.
        if find_active:
            if not is_under_50cm:
                return False
            return True

        if is_crit:
            if class_name == 'person':
                # Critical emergency warning for person under 50 cm is always permitted
                return True
            else:
                last_crit = self._last_critical_spoken_time.get(key, 0.0)
                if (current_time - last_crit) >= 3.0:
                    return True
                return False

        # At most one spatial announcement per 4 s.
        if (current_time - self._last_spatial_announcement_time) < 4.0:
            return False

        # Don't repeat the same spatial announcement for the same object and distance bucket
        # within 10 s unless the distance changes by 30% or more or the direction changes.
        zone_speech_map = {
            'top-left': 'left', 'top': 'ahead', 'top-right': 'right',
            'left': 'left', 'center': 'ahead', 'middle': 'ahead', 'right': 'right',
            'bottom-left': 'left', 'bottom': 'ahead', 'bottom-right': 'right', 'ahead': 'ahead'
        }
        curr_dir = guidance_item.get('zone_speech') or zone_speech_map.get(zone, 'ahead')
        curr_bucket = guidance_item.get('distance', 'far')
        curr_dist_m = guidance_item.get('estimated_distance_m')
        if curr_dist_m is None and u_m is not None:
            curr_dist_m = u_m

        prev_info = self._last_spoken_info.get(class_name)
        if prev_info is not None:
            time_since = current_time - prev_info['time']
            if time_since < 10.0:
                prev_bucket = prev_info.get('bucket')
                prev_dir = prev_info.get('direction')
                prev_dist_m = prev_info.get('dist_m')

                direction_changed = (curr_dir != prev_dir)
                dist_changed_30pct = False
                if prev_dist_m is not None and curr_dist_m is not None and prev_dist_m > 0:
                    dist_changed_30pct = (abs(curr_dist_m - prev_dist_m) / prev_dist_m) >= 0.30

                if direction_changed or dist_changed_30pct:
                    return True

                if curr_bucket == prev_bucket:
                    return False

        # Group person announcements rules:
        # 1. "Person ahead": for a person who stays in view, don't repeat it within 30 s.
        #    Announce again only when the count changes, a new person arrives, or a person gets closer than 1 m.
        if class_name == 'person':

            p_count = guidance_item.get('count', 1)
            est_m = guidance_item.get('estimated_distance_m', 999.0)
            is_under_1m = (
                (u_m is not None and u_m < 1.0)
                or (est_m < 1.0)
                or (guidance_item.get('distance') in ('very-close', 'close') and est_m <= 1.0)
            )

            # Check if this person was not announced yet or is returning after >= 5s absence (new person arrives)
            new_person = (not getattr(self, '_person_announced', False) or getattr(self, '_person_was_absent_over_5s', False))

            # Count changed (e.g. 1 -> 2, or 2 -> 1)
            count_changed = (self._last_announced_person_count != 0 and p_count != self._last_announced_person_count)

            # Person gets closer than 1m (and wasn't announced under 1m yet)
            closer_than_1m = is_under_1m and not getattr(self, '_person_announced_under_1m', False)

            if not is_under_1m and est_m >= 1.2:
                self._person_announced_under_1m = False

            if new_person:
                return True

            if closer_than_1m:
                return True

            if count_changed and (current_time - self._last_person_announce_time) >= 1.0:
                return True

            # For a person who stays in view: don't repeat within 30 s.
            # Announce again only when count changes, a new person arrives, or a person gets closer than 1 m.
            return False

        # Directional navigation advice ('Move straight', 'Move left', etc.)
        if class_name == 'navigation':
            # Never speak separate "Person ahead", "Person left" and "Move left" within the same 6 s.
            if (current_time - self._last_person_announce_time) < 6.0:
                return False
            # 'Move straight' is normal walking state, not an obstacle event to announce
            if 'straight' in guidance_item.get('message', '').lower():
                return False
            # Steering advice ('Move left' / 'Move right') only if directional change
            msg = guidance_item.get('message', '')
            if msg != self._last_spoken_time.get('nav_direction'):
                return True
            return False

        # Event-based tracking check
        tr_id = guidance_item.get('track_id')
        if tr_id is not None and tr_id in self._active_tracks:
            track = self._active_tracks[tr_id]

            # 1. New object entering view or person count increased
            if not track.get('initial_announced', False):
                return True
            if class_name == 'person' and track.get('person_count_increased', False):
                return True

            # 2. Re-entered after absence (> 5s for person, > 3s for others)
            if track.get('was_absent_over_3s', False):
                return True

            # 3. Distance bucket gets closer (each bucket announced once per object)
            curr_b = guidance_item.get('distance', track.get('distance', 'far'))
            curr_rank = self.DISTANCE_BUCKET_RANKS.get(curr_b, 0)
            announced = track.get('announced_buckets', set())
            max_rank = max([self.DISTANCE_BUCKET_RANKS.get(b, -1) for b in announced] or [-1])
            if curr_rank > max_rank and curr_b not in announced:
                return True

            # While same person stays in view and nothing changes: say nothing!
            return False

        # Fallback (when no track_id exists)
        if class_name == 'person':
            if getattr(self, '_person_announced', False) and not getattr(self, '_person_was_absent_over_5s', False):
                return False

        last_spoken = self._last_spoken_time.get(key)
        if last_spoken is None:
            return True

        if self._was_absent_over_3s.get(key, False):
            return True

        last_distance = self._last_spoken_distance.get(key, 'far')
        curr_distance = guidance_item.get('distance', 'far')
        last_rank = self.DISTANCE_BUCKET_RANKS.get(last_distance, 0)
        curr_rank = self.DISTANCE_BUCKET_RANKS.get(curr_distance, 0)
        if curr_rank > last_rank:
            return True

        # NO time-based repeats!
        return False

    def mark_guidance_spoken(self, guidance_item: Dict, current_time: float = None):
        """Record that a guidance message was spoken."""
        if current_time is None:
            current_time = time.time()

        class_name = guidance_item.get('class_name', '')
        zone = guidance_item.get('zone', '')
        key = (class_name, zone)

        self._last_spoken_time[key] = current_time
        self._last_spoken_distance[key] = guidance_item.get('distance', 'far')
        self._was_absent_over_3s[key] = False

        self._last_spatial_announcement_time = current_time
        zone_speech_map = {
            'top-left': 'left', 'top': 'ahead', 'top-right': 'right',
            'left': 'left', 'center': 'ahead', 'middle': 'ahead', 'right': 'right',
            'bottom-left': 'left', 'bottom': 'ahead', 'bottom-right': 'right', 'ahead': 'ahead'
        }
        curr_dir = guidance_item.get('zone_speech') or zone_speech_map.get(zone, 'ahead')
        curr_dist_m = guidance_item.get('estimated_distance_m')
        if class_name:
            self._last_spoken_info[class_name] = {
                'time': current_time,
                'dist_m': curr_dist_m,
                'direction': curr_dir,
                'bucket': guidance_item.get('distance', 'far'),
            }

        if class_name == 'person':
            self._person_announced = True
            self._person_was_absent_over_5s = False
            self._last_person_announce_time = current_time
            self._last_announced_person_count = guidance_item.get('count', 1)
            zone_speech_map = {
                'top-left': 'left', 'top': 'ahead', 'top-right': 'right',
                'left': 'left', 'center': 'ahead', 'middle': 'ahead', 'right': 'right',
                'bottom-left': 'left', 'bottom': 'ahead', 'bottom-right': 'right', 'ahead': 'ahead'
            }
            self._last_announced_person_zone = guidance_item.get('zone_speech') or zone_speech_map.get(zone, 'ahead')
            self._last_person_seen_time = current_time
            est_m = guidance_item.get('estimated_distance_m', 999.0)
            if est_m < 1.0 or guidance_item.get('distance') in ('very-close',):
                self._person_announced_under_1m = True
            else:
                self._person_announced_under_1m = False

        if class_name == 'navigation':
            self._last_spoken_time['nav_direction'] = guidance_item.get('message', '')

        if self.is_critical(guidance_item):
            self._last_critical_spoken_time[key] = current_time

        tr_id = guidance_item.get('track_id')
        if tr_id is not None and tr_id in self._active_tracks:
            track = self._active_tracks[tr_id]
            track['initial_announced'] = True
            track['was_absent_over_3s'] = False
            track['person_count_increased'] = False
            dist = guidance_item.get('distance', track.get('distance', 'far'))
            track.setdefault('announced_buckets', set()).add(dist)
            track['last_spoken_time'] = current_time
