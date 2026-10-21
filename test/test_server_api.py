"""
Test Flask API Server
Tests all API endpoints
"""
import requests
import base64
import cv2
import numpy as np
import sys
import os
import time
import pytest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_source import open_camera

BASE_URL = "http://localhost:5000"


def encode_image(frame):
    """Encode frame to base64"""
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


def test_health_endpoint():
    """Test /health endpoint"""
    print("=" * 60)
    print("TESTING /health ENDPOINT")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print("\n[OK] Server is healthy!")
            print(f"  Status: {data['status']}")
            print(f"  FPS: {data['fps']}")
            print(f"  Frames processed: {data['frames_processed']}")
            print(f"  Uptime: {data['uptime_seconds']}s")
            return True
        else:
            print(f"\n[FAIL] Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n[FAIL] Could not connect to server!")
        print("Make sure the server is running: python server.py")
        return False


def test_process_frame_endpoint():
    """Test /process_frame endpoint"""
    print("=" * 60)
    print("TESTING /process_frame ENDPOINT")
    print("=" * 60)
    
    # Create test frame
    cap = open_camera()
    if not cap.isOpened():
        print("\n[WARN] No webcam, using blank frame")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        ret, frame = cap.read()
        cap.release()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Encode frame
    encoded = encode_image(frame)
    
    print("\nSending frame to server...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/process_frame",
            json={
                'image': encoded,
                'return_annotated': False
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print("\n[OK] Frame processed successfully!")
            print(f"  Objects detected: {len(data['objects'])}")
            print(f"  Faces detected: {len(data['faces'])}")
            print(f"  Traffic signal: {data['traffic_signal']}")
            print(f"  Item locations: {len(data['item_locations'])}")
            print(f"  Audio messages: {len(data['audio_messages'])}")
            
            # Show details
            if data['objects']:
                print("\n  Objects:")
                for obj in data['objects']:
                    print(f"    - {obj['class_name']} ({obj['confidence']:.2f})")
            
            if data['faces']:
                print("\n  Faces:")
                for face in data['faces']:
                    print(f"    - {face['name']} (distance: {face['distance']:.3f})")
            
            return True
        else:
            print(f"\n[FAIL] Request failed: {response.status_code}")
            print(response.json())
            return False
            
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


@pytest.mark.skip(reason="manual/interactive test")
def test_enroll_face_endpoint():
    """Test /enroll_face endpoint"""
    print("=" * 60)
    print("TESTING /enroll_face ENDPOINT")
    print("=" * 60)
    
    print("\nOpening webcam to capture face...")
    cap = open_camera()
    
    if not cap.isOpened():
        print("[FAIL] Could not open webcam")
        return False
    
    print("Position yourself in frame and press SPACE to capture")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Enroll Face - Press SPACE', frame)
        
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    name = input("\nEnter name for enrollment: ").strip()
    if not name:
        print("[FAIL] No name provided")
        return False
    
    # Encode and send
    encoded = encode_image(frame)
    
    try:
        response = requests.post(
            f"{BASE_URL}/enroll_face",
            json={
                'image': encoded,
                'name': name
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n[OK] {data['message']}")
            return True
        else:
            print(f"\n[FAIL] Enrollment failed: {response.status_code}")
            print(response.json())
            return False
            
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


@pytest.mark.skip(reason="manual/interactive test")
def test_query_item_endpoint():
    """Test /query_item endpoint"""
    print("=" * 60)
    print("TESTING /query_item ENDPOINT")
    print("=" * 60)
    
    item = input("\nEnter item to query (default: keys): ").strip() or 'keys'
    
    try:
        response = requests.get(f"{BASE_URL}/query_item?item={item}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n[OK] Query result:")
            print(f"  Item: {data['item']}")
            print(f"  Found: {data['found']}")
            print(f"  Location: {data['location']}")
            print(f"  Message: {data['message']}")
            return True
        else:
            print(f"\n[FAIL] Query failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


def test_get_all_faces_endpoint():
    """Test /get_all_faces endpoint"""
    print("=" * 60)
    print("TESTING /get_all_faces ENDPOINT")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/get_all_faces")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n[OK] Found {data['count']} enrolled faces:")
            for face in data['faces']:
                print(f"  - {face['name']} (ID: {face['id']})")
            return True
        else:
            print(f"\n[FAIL] Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


def test_performance():
    """Test server performance with multiple requests"""
    print("=" * 60)
    print("TESTING SERVER PERFORMANCE")
    print("=" * 60)
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    encoded = encode_image(frame)
    
    num_requests = 10
    print(f"\nSending {num_requests} requests...")
    
    start_time = time.time()
    success_count = 0
    
    for i in range(num_requests):
        try:
            response = requests.post(
                f"{BASE_URL}/process_frame",
                json={'image': encoded, 'return_annotated': False},
                timeout=5
            )
            if response.status_code == 200:
                success_count += 1
                print(f"  Request {i+1}/{num_requests}: [OK]", end='\r')
        except:
            print(f"  Request {i+1}/{num_requests}: [FAIL]")
    
    elapsed = time.time() - start_time
    
    print(f"\n\n[OK] Performance test complete!")
    print(f"  Successful requests: {success_count}/{num_requests}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Average latency: {elapsed/num_requests:.3f}s per request")
    print(f"  Throughput: {num_requests/elapsed:.2f} requests/second")


if __name__ == "__main__":
    print("\nVISIONMATE API SERVER TEST SUITE")
    print("=" * 60)
    print("\n[WARN] IMPORTANT: Server must be running!")
    print("Start server in another terminal: python server.py\n")
    
    # Check if server is running
    if not test_health_endpoint():
        print("\n[FAIL] Server is not running. Please start the server first.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("\nSelect test mode:")
    print("1. Health check")
    print("2. Process frame")
    print("3. Enroll face")
    print("4. Query item location")
    print("5. Get all faces")
    print("6. Performance test")
    print("7. All tests")
    
    choice = input("\nEnter choice (1-7): ").strip()
    
    if choice == '1':
        test_health_endpoint()
    elif choice == '2':
        test_process_frame_endpoint()
    elif choice == '3':
        test_enroll_face_endpoint()
    elif choice == '4':
        test_query_item_endpoint()
    elif choice == '5':
        test_get_all_faces_endpoint()
    elif choice == '6':
        test_performance()
    elif choice == '7':
        print("\nRunning all tests...\n")
        test_health_endpoint()
        print("\n")
        test_get_all_faces_endpoint()
        print("\n")
        test_process_frame_endpoint()
        print("\n")
        test_query_item_endpoint()
        print("\n")
        test_performance()
    else:
        print("Running health check...")
        test_health_endpoint()
