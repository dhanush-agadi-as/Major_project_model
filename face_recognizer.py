"""
Face Recognition Module using InsightFace
Handles face detection, embedding extraction, and recognition
Falls back to OpenCV if InsightFace is not available
"""
import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Optional, Tuple
from config import FACE_RECOGNITION_THRESHOLD, FACE_MODEL, FACE_DET_SIZE
from database import VisionMateDB

logger = logging.getLogger(__name__)

# Try to import InsightFace, fall back to OpenCV if not available
INSIGHTFACE_AVAILABLE = False
try:
    import onnxruntime as ort
    from insightface.model_zoo.model_zoo import PickableInferenceSession
    _orig_pickable_session_init = PickableInferenceSession.__init__

    def _patched_pickable_session_init(self, model_path, **kwargs):
        sess_options = kwargs.get('sess_options')
        if sess_options is None:
            sess_options = ort.SessionOptions()
        try:
            from config import FACE_THREADS
            sess_options.intra_op_num_threads = FACE_THREADS
        except Exception:
            sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        kwargs['sess_options'] = sess_options
        _orig_pickable_session_init(self, model_path, **kwargs)

    PickableInferenceSession.__init__ = _patched_pickable_session_init

    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    print("[WARN] InsightFace not available - using OpenCV fallback (detection only, no recognition)")
    INSIGHTFACE_AVAILABLE = False
except Exception as e:
    logger.debug(f"InsightFace setup notice: {e}")
    try:
        from insightface.app import FaceAnalysis
        INSIGHTFACE_AVAILABLE = True
    except Exception:
        INSIGHTFACE_AVAILABLE = False



class FaceRecognizer:
    def __init__(self, db: VisionMateDB = None):
        """
        Initialize face recognition system
        Uses InsightFace if available, otherwise falls back to OpenCV
        
        Args:
            db: Database instance for storing/retrieving faces
        """
        self.insightface_available = INSIGHTFACE_AVAILABLE
        self.app = None
        self.face_cascade = None
        
        if INSIGHTFACE_AVAILABLE:
            print("Loading InsightFace model...")
            self.app = FaceAnalysis(name=FACE_MODEL, providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
            self.app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)
            print(f"[OK] Face Recognizer initialized (InsightFace)")
        else:
            print("Loading OpenCV face detector (fallback mode)...")
            # Load OpenCV's Haar Cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print(f"[OK] Face Detector initialized (OpenCV - detection only, no recognition)")
        self.threshold = FACE_RECOGNITION_THRESHOLD
        self.db = db if db else VisionMateDB()
        
        self.recent_identities = {}
        # IoU face tracking & throttling
        self.face_tracks = {}  # track_id -> dict
        self._next_face_track_id = 1
        self.iou_threshold = 0.3
        self.track_max_age = 2.0
        self.last_frame_time_ms = 0.0
        self.last_detection_time = -999.0
        self.last_detected_faces = []
        self.last_recognition_results = []

        self.enrollment_buffer = {
            'images': [],
            'embeddings': [],
            'name': None
        }
        
        # Load known faces from database
        self.known_faces = []
        if INSIGHTFACE_AVAILABLE:
            self._load_known_faces()
        else:
            print("[WARN] Face recognition disabled - InsightFace required for enrollment/recognition")
    
    def _load_known_faces(self):
        """Load all known faces from database"""
        self.known_faces = self.db.get_all_faces()
    
    def detect_faces(self, frame: np.ndarray, yolo_has_person: bool = True, current_time: Optional[float] = None) -> List[Dict]:
        """
        Detect faces in frame and extract embeddings.
        Runs at most 5 times per second and only when YOLO sees a person.
        Reuses the last result between runs.
        
        Args:
            frame: Input image (BGR format)
            yolo_has_person: True if YOLO sees a person in the frame
            current_time: Optional explicit timestamp (seconds)
            
        Returns:
            List of face dictionaries containing:
                - bbox: [x1, y1, x2, y2]
                - embedding: 512-d face embedding (or None if using OpenCV)
                - confidence: Detection confidence
        """
        now = current_time if current_time is not None else time.time()
        if not yolo_has_person:
            self.last_detected_faces = []
            return []

        # Run detection at most 5 times per second (0.2s interval)
        if (now - self.last_detection_time) < 0.20 and self.last_detected_faces:
            return self.last_detected_faces

        self.last_detection_time = now

        if self.insightface_available:
            # Use InsightFace
            faces = self.app.get(frame)
            
            results = []
            for face in faces:
                # Minimum face size/quality check
                bbox = face.bbox.astype(int)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Reject tiny faces as they're prone to false matches
                if width < 30 or height < 30:
                    continue
                    
                embedding = face.normed_embedding
                
                face_dict = {
                    'bbox': bbox.tolist(),  # [x1, y1, x2, y2]
                    'embedding': embedding,
                    'confidence': float(face.det_score)
                }
                results.append(face_dict)
            
            self.last_detected_faces = results
            return results
        else:
            # Use OpenCV fallback (detection only, no embeddings)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            results = []
            for (x, y, w, h) in faces:
                # Reject tiny faces
                if w < 30 or h < 30:
                    continue
                    
                face_dict = {
                    'bbox': [x, y, x + w, y + h],
                    'embedding': None,  # No embeddings in fallback mode
                    'confidence': 0.9  # OpenCV doesn't provide confidence
                }
                results.append(face_dict)
            
            self.last_detected_faces = results
            return results
    
    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        box format: [x1, y1, x2, y2]
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
        area2 = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0
        return float(intersection / union)

    def _prune_tracks(self, now: float, max_age: float = 2.0):
        """Remove tracks that haven't been seen for more than max_age seconds"""
        to_del = [tid for tid, trk in self.face_tracks.items() if (now - trk['last_seen']) > max_age]
        for tid in to_del:
            del self.face_tracks[tid]

    def recognize_faces(self, frame: np.ndarray, current_time: Optional[float] = None,
                        frame_time_ms: Optional[float] = None, yolo_has_person: bool = True) -> List[Dict]:
        """
        Detect and recognize faces in frame.
        - Skips entirely when YOLO sees no person or when frame time is above 150 ms
        - Runs detection at most 5 times per second and reuses last result between runs
        - Only processes the 2 largest faces per frame
        - Uses an IoU tracker so each tracked face keeps its identity
        - Recognises a given track at most once per second (and once at first sight)

        Args:
            frame: Input image (BGR format)
            current_time: Optional explicit timestamp (seconds)
            frame_time_ms: Optional frame processing time in ms
            yolo_has_person: True if YOLO sees a person in the frame

        Returns:
            List of recognized faces with names or 'Unknown'
        """
        now = current_time if current_time is not None else time.time()

        if not yolo_has_person:
            self.last_recognition_results = []
            self._prune_tracks(now, self.track_max_age)
            return []

        if frame_time_ms is not None:
            self.last_frame_time_ms = frame_time_ms

        if self.last_frame_time_ms > 150.0:
            return self.last_recognition_results

        # Run detection at most 5 times per second (interval >= 0.2s); reuse last result between runs
        if (now - self.last_detection_time) < 0.20 and self.last_recognition_results:
            return self.last_recognition_results

        detected_faces = self.detect_faces(frame, yolo_has_person=yolo_has_person, current_time=now)
        if not detected_faces:
            self._prune_tracks(now, self.track_max_age)
            self.last_recognition_results = []
            return []

        # Only the 2 largest faces per frame
        if len(detected_faces) > 2:
            detected_faces.sort(
                key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]),
                reverse=True
            )
            detected_faces = detected_faces[:2]

        results = []
        unmatched_detections = list(range(len(detected_faces)))
        unmatched_tracks = set(self.face_tracks.keys())

        # Match detected faces to existing tracks using IoU >= threshold
        matches = []
        for det_idx in unmatched_detections:
            det_box = detected_faces[det_idx]['bbox']
            for trk_id in unmatched_tracks:
                trk_box = self.face_tracks[trk_id]['bbox']
                iou = self._compute_iou(det_box, trk_box)
                if iou >= self.iou_threshold:
                    matches.append((iou, det_idx, trk_id))

        # Sort matches by IoU descending (greedy matching)
        matches.sort(key=lambda x: x[0], reverse=True)
        matched_dets = set()
        matched_trks = set()

        for iou, det_idx, trk_id in matches:
            if det_idx in matched_dets or trk_id in matched_trks:
                continue
            matched_dets.add(det_idx)
            matched_trks.add(trk_id)

            face = detected_faces[det_idx]
            track = self.face_tracks[trk_id]
            track['bbox'] = face['bbox']
            track['confidence'] = face.get('confidence', 0.9)
            track['last_seen'] = now

            # Recognise a given track at most once per second (and once at first sight)
            if (now - track['last_rec_time']) >= 1.0:
                face_embedding = face.get('embedding')
                if face_embedding is not None:
                    name, distance, margin = self._match_face(face_embedding)
                    track['name'] = name
                    track['distance'] = distance
                    track['margin'] = margin
                track['last_rec_time'] = now
                track['rec_count'] += 1

            results.append({
                'bbox': face['bbox'],
                'name': track['name'],
                'distance': track['distance'],
                'margin': track['margin'],
                'confidence': track['confidence'],
                'track_id': trk_id
            })

        # New detections: first sight! (recognize once at first sight)
        for det_idx in unmatched_detections:
            if det_idx in matched_dets:
                continue
            face = detected_faces[det_idx]
            face_embedding = face.get('embedding')
            if face_embedding is not None:
                name, distance, margin = self._match_face(face_embedding)
            else:
                name, distance, margin = 'Unknown', 1.0, 0.0

            trk_id = self._next_face_track_id
            self._next_face_track_id += 1

            self.face_tracks[trk_id] = {
                'track_id': trk_id,
                'bbox': face['bbox'],
                'name': name,
                'distance': distance,
                'margin': margin,
                'confidence': face.get('confidence', 0.9),
                'first_seen': now,
                'last_seen': now,
                'last_rec_time': now,
                'rec_count': 1
            }

            results.append({
                'bbox': face['bbox'],
                'name': name,
                'distance': distance,
                'margin': margin,
                'confidence': face.get('confidence', 0.9),
                'track_id': trk_id
            })

        self._prune_tracks(now, self.track_max_age)

        # Update recent_identities for short-term identity persistence
        for r in results:
            if r['name'] != 'Unknown':
                center_x = (r['bbox'][0] + r['bbox'][2]) / 2
                center_y = (r['bbox'][1] + r['bbox'][3]) / 2
                self.recent_identities[r['name']] = {
                    'center': (center_x, center_y),
                    'timestamp': now,
                    'last_distance': r['distance']
                }

        self.last_recognition_results = results
        return results
    
    def _match_face(self, embedding: np.ndarray) -> Tuple[str, float, float]:
        """
        Match face embedding against known faces
        
        Args:
            embedding: Face embedding to match
            
        Returns:
            (name, distance, confidence_margin) tuple - name is 'Unknown' if no match
        """
        if not self.insightface_available:
            return 'Unknown', 1.0, 0.0

        if len(self.known_faces) == 0:
            return 'Unknown', 1.0, 0.0
        
        # Calculate distances to all embeddings
        distances = []
        for face_id, name, known_embedding in self.known_faces:
            if embedding.shape != known_embedding.shape:
                continue
            
            # Euclidean distance
            distance = np.linalg.norm(embedding - known_embedding)
            # Cosine similarity for logging (optional, nice to have)
            cosine = np.dot(embedding, known_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(known_embedding))
            distances.append((distance, cosine, name))
        
        if not distances:
            return 'Unknown', 1.0, 0.0
            
        distances.sort(key=lambda x: x[0])
        
        best_distance = distances[0][0]
        best_cosine = distances[0][1]
        best_name = distances[0][2]
        
        # Find distance to the best match of a DIFFERENT person
        next_best_diff_name_dist = 2.0
        for dist, cos, name in distances:
            if name != best_name:
                next_best_diff_name_dist = dist
                break
                
        confidence_margin = next_best_diff_name_dist - best_distance
        
        # Log distance and cosine
        if best_distance < self.threshold + 0.1: # Log close matches
            logger.debug(f"Face match check: best='{best_name}' dist={best_distance:.3f} cos={best_cosine:.3f} margin={confidence_margin:.3f}")
        
        # Check if distance is below threshold
        if best_distance > self.threshold:
            return 'Unknown', best_distance, confidence_margin
            
        # Check confidence margin
        if confidence_margin < 0.08:
            logger.warning(f"Low confidence margin ({confidence_margin:.3f}) for {best_name}, might be ambiguous")
        
        return best_name, best_distance, confidence_margin

    def start_enrollment(self, name: str) -> None:
        """Start enrolling a new face"""
        self.enrollment_buffer = {
            'images': [],
            'embeddings': [],
            'name': name
        }
        print(f"Started face enrollment for '{name}'")
        
    def add_enrollment_frame(self, frame: np.ndarray) -> int:
        """
        Add a frame to the face enrollment buffer
        
        Returns:
            Number of frames captured so far
        """
        if self.enrollment_buffer['name'] is None:
            raise ValueError("Must call start_enrollment first")
            
        faces = self.detect_faces(frame)
        if len(faces) == 0:
            return len(self.enrollment_buffer['embeddings'])
            
        # Use largest face
        faces.sort(key=lambda f: (f['bbox'][2]-f['bbox'][0]) * (f['bbox'][3]-f['bbox'][1]), reverse=True)
        face = faces[0]
        
        if face['embedding'] is not None:
            self.enrollment_buffer['embeddings'].append(face['embedding'])
            self.enrollment_buffer['images'].append(frame.copy())
            
        num_frames = len(self.enrollment_buffer['embeddings'])
        return num_frames
        
    def finish_enrollment(self) -> bool:
        """
        Complete face enrollment and save to database
        """
        if self.enrollment_buffer['name'] is None:
            return False
            
        if len(self.enrollment_buffer['embeddings']) == 0:
            print("[FAIL] No faces captured during enrollment")
            return False
            
        name = self.enrollment_buffer['name']
        
        # Save all embeddings to database
        for embedding in self.enrollment_buffer['embeddings']:
            self.db.add_face(name, embedding)
            
        print(f"[OK] Enrolled {name} with {len(self.enrollment_buffer['embeddings'])} embeddings")
        
        # Reload known faces
        self._load_known_faces()
        self.cancel_enrollment()
        
        return True
        
    def cancel_enrollment(self) -> None:
        """Cancel current face enrollment"""
        self.enrollment_buffer = {
            'images': [],
            'embeddings': [],
            'name': None
        }
    
    def enroll_face(self, frame: np.ndarray, name: str) -> bool:
        """
        Legacy single-frame enrollment (kept for backward compatibility)
        """
        faces = self.detect_faces(frame)
        
        if len(faces) == 0:
            print("[FAIL] No face detected in frame")
            return False
        
        if len(faces) > 1:
            print("[WARN] Multiple faces detected, using the largest one")
        
        # Use largest face
        faces.sort(key=lambda f: (f['bbox'][2]-f['bbox'][0]) * (f['bbox'][3]-f['bbox'][1]), reverse=True)
        face = faces[0]
        embedding = face['embedding']
        
        if embedding is None:
            return False
            
        # Save to database
        face_id = self.db.add_face(name, embedding)
        print(f"[OK] Enrolled {name} with ID {face_id}")
        
        # Reload known faces
        self._load_known_faces()
        
        return True

    def reload_known_faces(self):
        """Reload known faces from database"""
        self._load_known_faces()

    def delete_face(self, name: str) -> bool:
        """Delete face by name and reload known faces"""
        deleted = False
        if hasattr(self.db, 'delete_face_by_name'):
            deleted = self.db.delete_face_by_name(name)
        if deleted:
            self._load_known_faces()
            # Clean up recent identities if deleted
            self.recent_identities.pop(name, None)
            for tid, trk in list(self.face_tracks.items()):
                if trk.get('name') == name:
                    trk['name'] = 'Unknown'
        return deleted

    def rename_face(self, old_name: str, new_name: str) -> bool:
        """Rename face from old_name to new_name and reload known faces"""
        renamed = False
        if hasattr(self.db, 'rename_face'):
            renamed = self.db.rename_face(old_name, new_name)
        if renamed:
            self._load_known_faces()
            if old_name in self.recent_identities:
                self.recent_identities[new_name] = self.recent_identities.pop(old_name)
            for tid, trk in list(self.face_tracks.items()):
                if trk.get('name') == old_name:
                    trk['name'] = new_name
        return renamed
    
    def draw_faces(self, frame: np.ndarray, recognized_faces: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and names on detected faces
        
        Args:
            frame: Input image
            recognized_faces: List of recognized face dictionaries
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for face in recognized_faces:
            x1, y1, x2, y2 = face['bbox']
            name = face['name']
            distance = face['distance']
            margin = face.get('margin', 0.0)
            
            # Color: Green for recognized, Red for unknown
            color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{name} ({distance:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated


if __name__ == "__main__":
    # Quick test
    recognizer = FaceRecognizer()
    print("\n[OK] Face Recognizer ready for inference")
