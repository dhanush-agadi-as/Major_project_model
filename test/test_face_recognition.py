"""
Test Face Recognition Module
Tests face detection, enrollment, and recognition
"""
import cv2
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_recognizer import FaceRecognizer
from database import VisionMateDB


def test_face_enrollment():
    """Test face enrollment (Remember feature)"""
    print("=" * 60)
    print("TESTING FACE ENROLLMENT")
    print("=" * 60)
    
    db = VisionMateDB()
    recognizer = FaceRecognizer(db)
    
    print("\nOpening webcam for enrollment...")
    print("Position your face in frame and press SPACE to capture")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Could not open webcam")
        return
    
    enrolled = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = recognizer.detect_faces(frame)
        
        # Draw rectangles
        display = frame.copy()
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"Confidence: {face['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Instructions
        cv2.putText(
            display,
            "Press SPACE to enroll | Q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.imshow('Face Enrollment', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and len(faces) > 0:
            name = input("\nEnter name for this face: ").strip()
            if name:
                success = recognizer.enroll_face(frame, name)
                if success:
                    print(f"[OK] Successfully enrolled {name}")
                    enrolled = True
                else:
                    print("[FAIL] Enrollment failed")
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if enrolled:
        print("\n[OK] Enrollment test complete!")
    
    db.close()


def test_face_recognition():
    """Test face recognition with webcam"""
    print("=" * 60)
    print("TESTING FACE RECOGNITION")
    print("=" * 60)
    
    db = VisionMateDB()
    recognizer = FaceRecognizer(db)
    
    # Check if we have enrolled faces
    known_faces = db.get_all_faces()
    print(f"\nKnown faces in database: {len(known_faces)}")
    if len(known_faces) == 0:
        print("[WARN] No faces enrolled! Run enrollment first.")
        print("Continuing with recognition test (will show 'Unknown')...")
    else:
        for face_id, name, _ in known_faces:
            print(f"  - {name} (ID: {face_id})")
    
    print("\nOpening webcam for recognition...")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Could not open webcam")
        db.close()
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Recognize faces (every 5th frame for performance)
        if frame_count % 5 == 0:
            recognized = recognizer.recognize_faces(frame)
            
            # Draw results
            display = recognizer.draw_faces(frame, recognized)
            
            # Print recognition results
            if recognized:
                for face in recognized:
                    print(f"\rRecognized: {face['name']} (distance: {face['distance']:.3f})", end='')
        else:
            display = frame
        
        cv2.putText(
            display,
            "Press Q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.imshow('Face Recognition Test', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n\n[OK] Recognition test complete!")
    db.close()


def test_database_operations():
    """Test database face storage and retrieval"""
    print("=" * 60)
    print("TESTING DATABASE OPERATIONS")
    print("=" * 60)
    
    db = VisionMateDB()
    
    # Get all faces
    faces = db.get_all_faces()
    print(f"\nTotal faces in database: {len(faces)}")
    
    for face_id, name, embedding in faces:
        print(f"  ID {face_id}: {name} (embedding shape: {embedding.shape})")
    
    print("\n[OK] Database test complete!")
    db.close()


def test_automated_scenarios():
    """Automated tests for multi-frame enrollment and partial occlusion"""
    print("=" * 60)
    print("TESTING AUTOMATED SCENARIOS")
    print("=" * 60)
    
    import numpy as np
    
    # Use an in-memory or temporary database for testing if possible, 
    # but since VisionMateDB uses sqlite3, we will use the default and just clean up.
    db = VisionMateDB()
    recognizer = FaceRecognizer(db)
    
    # 1. Multi-sample enrollment
    print("\n--- Test 1: Multi-sample enrollment ---")
    test_name = "test_person_xyz"
    recognizer.start_enrollment(test_name)
    
    # Create dummy embeddings to mock detection
    # InsightFace embeddings are 512-d
    base_embedding = np.random.randn(512).astype(np.float32)
    base_embedding /= np.linalg.norm(base_embedding)
    
    for i in range(5):
        # Slightly perturb the embedding to simulate different angles
        perturbed = base_embedding + np.random.randn(512).astype(np.float32) * 0.05
        perturbed /= np.linalg.norm(perturbed)
        
        # Add directly to buffer since we can't easily fake the detect_faces call without a real image
        recognizer.enrollment_buffer['embeddings'].append(perturbed)
        
    success = recognizer.finish_enrollment()
    assert success, "Multi-sample enrollment failed"
    print("[OK] Multi-sample enrollment succeeded")
    
    # 2. Matching against multiple stored embeddings
    print("\n--- Test 2: Matching against multiple stored embeddings ---")
    # A new query embedding close to the third enrolled embedding
    query_embedding = recognizer.known_faces[-3][2] + np.random.randn(512).astype(np.float32) * 0.01
    query_embedding /= np.linalg.norm(query_embedding)
    
    name, distance, margin = recognizer._match_face(query_embedding)
    print(f"Match result: {name}, distance: {distance:.3f}, margin: {margin:.3f}")
    assert name == test_name, f"Expected {test_name}, got {name}"
    print("[OK] Successfully matched against multiple stored embeddings")
    
    # 3. Partial-occlusion scenario (mock low-quality bbox)
    print("\n--- Test 3: Partial-occlusion / Low-quality rejection ---")
    # We will pass a dummy frame to detect_faces. Since we can't fake insightface output easily,
    # we'll test the logic in recognize_faces that rejects small faces.
    # We'll mock detect_faces to return a small face and a normal face.
    original_detect = recognizer.detect_faces
    
    def mock_detect_faces(frame):
        return [
            {'bbox': [0, 0, 20, 20], 'embedding': query_embedding, 'confidence': 0.9},  # Too small (20x20)
            {'bbox': [100, 100, 200, 200], 'embedding': query_embedding, 'confidence': 0.9} # Normal (100x100)
        ]
    
    
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Actually wait, our recognize_faces calls detect_faces. BUT in our FaceRecognizer class,
    # the small face rejection is inside detect_faces itself! 
    # Let's mock the InsightFace app output instead.
    
    if recognizer.insightface_available:
        class MockFace:
            def __init__(self, bbox):
                self.bbox = np.array(bbox)
                self.normed_embedding = query_embedding
                self.det_score = 0.9
                
        class MockApp:
            def get(self, frame):
                return [
                    MockFace([0, 0, 20, 20]),
                    MockFace([100, 100, 200, 200])
                ]
                
        original_app = recognizer.app
        recognizer.app = MockApp()
        
        faces = recognizer.detect_faces(dummy_frame)
        assert len(faces) == 1, f"Expected 1 face to pass size check, got {len(faces)}"
        assert faces[0]['bbox'] == [100, 100, 200, 200], "Wrong face passed size check"
        print("[OK] Small/occluded face correctly rejected")
        
        recognizer.app = original_app
    
    recognizer.detect_faces = original_detect
    
    # Clean up database
    # Find IDs for the test person
    cursor = db.conn.cursor()
    cursor.execute("DELETE FROM faces WHERE name = ?", (test_name,))
    db.conn.commit()
    print("[OK] Cleanup completed")
    
    print("\nAutomated scenarios passed!")
    db.close()


if __name__ == "__main__":
    print("\nVISIONMATE FACE RECOGNITION TEST SUITE")
    print("=" * 60)
    
    print("\nSelect test mode:")
    print("1. Test enrollment (Remember feature)")
    print("2. Test recognition")
    print("3. View database")
    print("4. Full test (enrollment + recognition)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        test_face_enrollment()
    elif choice == '2':
        test_face_recognition()
    elif choice == '3':
        test_database_operations()
    elif choice == '4':
        test_database_operations()
        print("\n")
        test_face_enrollment()
        print("\n")
        test_face_recognition()
    elif choice == '5':
        test_automated_scenarios()
    else:
        print("Running database test...")
        test_database_operations()
