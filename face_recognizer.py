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
    print("⚠ InsightFace not available - using OpenCV fallback (detection only, no recognition)")
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
            print(f"✓ Face Recognizer initialized (InsightFace)")
        else:
            print("Loading OpenCV face detector (fallback mode)...")
            # Load OpenCV's Haar Cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print(f"✓ Face Detector initialized (OpenCV - detection only, no recognition)")
        
        self.threshold = FACE_RECOGNITION_THRESHOLD
        self.db = db if db else VisionMateDB()
        
        # Load known faces from database
        self.known_faces = []
        if INSIGHTFACE_AVAILABLE:
            self._load_known_faces()
        else:
            print("⚠ Face recognition disabled - InsightFace required for enrollment/recognition")
    
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
                bbox = face.bbox.astype(int)
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
            
            # Match against known faces
            name, distance = self._match_face(face_embedding)
            
            result = {
                'bbox': face['bbox'],
                'name': name,
                'distance': distance,
                'confidence': face['confidence']
            }
            results.append(result)
        
        return results
    
    def _match_face(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Match face embedding against known faces
        
        Args:
            embedding: Face embedding to match
            
        Returns:
            (name, distance) tuple - name is 'Unknown' if no match
        """
        if len(self.known_faces) == 0:
            return 'Unknown', 1.0
        
        min_distance = float('inf')
        matched_name = 'Unknown'
        
        for face_id, name, known_embedding in self.known_faces:
            if embedding.shape != known_embedding.shape:
                print(f"⚠ Skipping incompatible embedding for '{name}': expected {embedding.shape[0]}-d, got {known_embedding.shape[0]}-d")
                continue
                
            # Calculate Euclidean distance
            distance = np.linalg.norm(embedding - known_embedding)
            
            if distance < min_distance:
                min_distance = distance
                matched_name = name
        
        if not self.insightface_available:
            print("✗ Cannot enroll - InsightFace required for face recognition")
            return False
        
        # Check if distance is below threshold
        if min_distance > self.threshold:
            return 'Unknown', min_distance
        
        return matched_name, min_distance
    
    def enroll_face(self, frame: np.ndarray, name: str) -> bool:
        """
        Enroll a new face into the database (Remember feature)
        
        Args:
            frame: Frame containing the face
            name: Name of the person
            
        Returns:
            True if enrollment successful, False otherwise
        """
        faces = self.detect_faces(frame)
        
        if len(faces) == 0:
            print("✗ No face detected in frame")
            return False
        
        if len(faces) > 1:
            print("⚠ Multiple faces detected, using the first one")
        
        # Use the first detected face
        face = faces[0]
        embedding = face['embedding']
        
        # Save to database
        face_id = self.db.add_face(name, embedding)
        print(f"✓ Enrolled {name} with ID {face_id}")
        
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
    print("\n✓ Face Recognizer ready for inference")
