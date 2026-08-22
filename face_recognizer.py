"""
Face Recognition Module using InsightFace
Handles face detection, embedding extraction, and recognition
Falls back to OpenCV if InsightFace is not available
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from config import FACE_RECOGNITION_THRESHOLD, FACE_MODEL, FACE_DET_SIZE
from database import VisionMateDB

# Try to import InsightFace, fall back to OpenCV if not available
INSIGHTFACE_AVAILABLE = False
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    print("[WARN] InsightFace not available - using OpenCV fallback (detection only, no recognition)")
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
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in frame and extract embeddings
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of face dictionaries containing:
                - bbox: [x1, y1, x2, y2]
                - embedding: 512-d face embedding (or None if using OpenCV)
                - confidence: Detection confidence
        """
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
            
            return results
    
    def recognize_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and recognize faces in frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of recognized faces with names or 'Unknown'
        """
        detected_faces = self.detect_faces(frame)
        
        results = []
        for face in detected_faces:
            face_embedding = face['embedding']
            
            if face_embedding is None:
                # OpenCV fallback
                name = 'Unknown'
                distance = 1.0
                margin = 0.0
            else:
                # Match against known faces
                name, distance, margin = self._match_face(face_embedding)
            
            result = {
                'bbox': face['bbox'],
                'name': name,
                'distance': distance,
                'margin': margin,
                'confidence': face['confidence']
            }
            results.append(result)
            
            if name != 'Unknown':
                # Short-term identity persistence
                # Store the most recently seen face centers to bias future recognitions
                center_x = (face['bbox'][0] + face['bbox'][2]) / 2
                center_y = (face['bbox'][1] + face['bbox'][3]) / 2
                self.recent_identities[name] = {
                    'center': (center_x, center_y),
                    'timestamp': cv2.getTickCount() / cv2.getTickFrequency(),
                    'last_distance': distance
                }
        
        # Identity persistence logic: if a face drops to unknown but was recently recognized nearby
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        # Clean up old identities (e.g., older than 2 seconds)
        self.recent_identities = {
            n: data for n, data in self.recent_identities.items()
            if current_time - data['timestamp'] < 2.0
        }
        
        for result in results:
            if result['name'] == 'Unknown' and result.get('embedding') is not None:
                # Check if it matches a recent identity by location
                center_x = (result['bbox'][0] + result['bbox'][2]) / 2
                center_y = (result['bbox'][1] + result['bbox'][3]) / 2
                
                best_recent_name = None
                best_recent_dist = 50.0  # pixel distance
                
                for name, data in self.recent_identities.items():
                    px_dist = np.sqrt((center_x - data['center'][0])**2 + (center_y - data['center'][1])**2)
                    if px_dist < best_recent_dist:
                        best_recent_dist = px_dist
                        best_recent_name = name
                
                # If we're very close to where a known person just was, and distance isn't terrible
                if best_recent_name and result['distance'] < self.threshold + 0.15:
                    result['name'] = best_recent_name
                    # Note: distance remains the slightly higher value so caller can see it was forced
        
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
            print(f"Face match check: best='{best_name}' dist={best_distance:.3f} cos={best_cosine:.3f} margin={confidence_margin:.3f}")
        
        # Check if distance is below threshold
        if best_distance > self.threshold:
            return 'Unknown', best_distance, confidence_margin
            
        # Check confidence margin
        if confidence_margin < 0.08:
            print(f"[WARN] Low confidence margin ({confidence_margin:.3f}) for {best_name}, might be ambiguous")
        
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
