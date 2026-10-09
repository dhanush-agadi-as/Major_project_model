"""
VisionMate Flask API Server
Main inference server that integrates all AI modules
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import time
from typing import Dict, List
import threading

from object_detector import ObjectDetector
from face_recognizer import FaceRecognizer
# from traffic_signal_detector import TrafficSignalDetector  # DISABLED: Too many false positives
from item_tracker import ItemTracker
from audio_feedback import get_audio_feedback
from database import VisionMateDB
from object_memory import ObjectMemory
from gps_tracker import GPSTracker
from ip_geolocation import reverse_geocode
import config
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG

app = Flask(__name__)
CORS(app)

# Initialize all AI modules
print("=" * 60)
print("INITIALIZING VISIONMATE AI SERVER")
print("=" * 60)

db = VisionMateDB()
object_detector = ObjectDetector()
face_recognizer = FaceRecognizer(db)
# traffic_detector = TrafficSignalDetector()  # DISABLED: Too many false positives
item_tracker = ItemTracker(db)
object_memory = ObjectMemory(db)
audio = get_audio_feedback(use_gtts=False)
gps_tracker = GPSTracker(mock_mode=getattr(config, 'GPS_MOCK_MODE', True))
gps_tracker.start()

print("=" * 60)
print("[OK] ALL MODULES LOADED SUCCESSFULLY")
print("=" * 60)

# Performance tracking
frame_count = 0
start_time = time.time()


def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    img_bytes = base64.b64decode(image_data.split(',')[-1])
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return frame


def encode_image(frame: np.ndarray) -> str:
    """Encode numpy array to base64 image"""
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global frame_count, start_time
    
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    
    return jsonify({
        'status': 'healthy',
        'fps': round(fps, 2),
        'frames_processed': frame_count,
        'uptime_seconds': round(elapsed, 2)
    })


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    Main endpoint for processing video frames
    Performs all AI inference and returns results
    """
    global frame_count
    frame_count += 1
    
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode frame
        frame = decode_image(image_data)
        
        # Run all detections
        results = {
            'objects': [],
            'faces': [],
            'traffic_signal': None,
            'item_locations': [],
            'remembered_objects': [],
            'audio_messages': []
        }
        
        # 1. Object Detection
        objects = object_detector.detect(frame)
        results['objects'] = objects
        
        # 1b. Recognize remembered objects
        remembered_detections = []
        for obj in objects:
            bbox = (obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3])
            match = object_memory.recognize_object(frame, bbox, obj['class_name'])
            if match:
                custom_name, confidence, match_tier = match
                remembered_detections.append({
                    'custom_name': custom_name,
                    'yolo_class': obj['class_name'],
                    'confidence': confidence,
                    'bbox': obj['bbox'],
                    'tier': match_tier
                })
                
                if match_tier == 'high':
                    # Update sighting automatically
                    nearby_objects = [o['class_name'] for o in objects if o != obj]
                    pos = gps_tracker.get_position() if gps_tracker else None
                    lat, lon = (pos[0], pos[1]) if pos else (None, None)
                    if config.INDOOR_MODE:
                        # Skip reverse-geocoding indoors; build context from visible objects
                        nearby_classes = [o['class_name'] for o in objects if o != obj]
                        loc_str = ("near " + ", ".join(nearby_classes[:3])) if nearby_classes else None
                        # GPS coords are not meaningful indoors
                        lat, lon = None, None
                    else:
                        loc_str = reverse_geocode(lat, lon) if (lat is not None and lon is not None) else None

                    object_memory.update_object_sighting(
                        custom_name=custom_name,
                        image=frame,
                        location=loc_str,
                        nearby_objects=nearby_objects[:5],  # Limit context
                        gps_lat=lat,
                        gps_lon=lon,
                        confidence=confidence
                    )
                elif match_tier == 'ambiguous':
                    # Trigger voice prompt for confirmation
                    audio.speak_social(f"Is this your {custom_name}?")
                    
        results['remembered_objects'] = remembered_detections
        
        # 2. Face Recognition
        faces = face_recognizer.recognize_faces(frame)
        results['faces'] = faces
        
        # 3. Traffic Signal Detection - DISABLED (too many false positives)
        # traffic_signal = traffic_detector.detect_signal(frame)
        # results['traffic_signal'] = traffic_signal
        traffic_signal = None  # Disabled for now
        results['traffic_signal'] = None
        
        # 4. Item Tracking
        item_updates = item_tracker.update_tracking(objects)
        results['item_locations'] = item_updates
        
        # Generate audio feedback based on priority
        audio_messages = generate_audio_feedback(
            objects, faces, traffic_signal, item_updates, remembered_detections
        )
        results['audio_messages'] = audio_messages
        
        # Optionally return annotated frame
        if data.get('return_annotated', False):
            annotated = frame.copy()
            annotated = object_detector.draw_detections(annotated, objects)
            annotated = face_recognizer.draw_faces(annotated, faces)
            # Traffic signal detection disabled
            # if traffic_signal:
            #     annotated = traffic_detector.draw_signal(annotated, traffic_signal)
            results['annotated_image'] = encode_image(annotated)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/enroll_face', methods=['POST'])
def enroll_face():
    """
    Endpoint for enrolling a new face (Remember feature)
    """
    try:
        data = request.json
        image_data = data.get('image')
        name = data.get('name')
        
        if not image_data or not name:
            return jsonify({'error': 'Image and name are required'}), 400
        
        # Decode frame
        frame = decode_image(image_data)
        
        # Enroll face
        success = face_recognizer.enroll_face(frame, name)
        
        if success:
            audio.speak_social(f"Enrolled {name} successfully")
            return jsonify({
                'success': True,
                'message': f'Face enrolled for {name}'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No face detected or enrollment failed'
            }), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/query_item', methods=['GET'])
def query_item():
    """
    Endpoint for querying item location (Find My Keys feature)
    """
    try:
        item = request.args.get('item', 'keys')
        location = item_tracker.query_item_location(item)
        
        if location:
            message = f"Your {item} are on the {location}"
            audio.speak_navigational(message)
            return jsonify({
                'item': item,
                'location': location,
                'found': True,
                'message': message
            })
        else:
            message = f"I haven't seen your {item} recently"
            audio.speak_navigational(message)
            return jsonify({
                'item': item,
                'location': None,
                'found': False,
                'message': message
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_all_faces', methods=['GET'])
def get_all_faces():
    """Get all enrolled faces"""
    try:
        faces = db.get_all_faces()
        return jsonify({
            'count': len(faces),
            'faces': [{'id': f[0], 'name': f[1]} for f in faces]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/enroll_object', methods=['POST'])
def enroll_object():
    """
    Endpoint for enrolling remembered objects
    
    Actions:
    - start: Begin enrollment (requires: custom_name, yolo_class)
    - add_frame: Add frame to enrollment (requires: image, bbox)
    - finish: Complete enrollment
    - cancel: Cancel enrollment
    """
    try:
        data = request.json
        action = data.get('action')
        
        if action == 'start':
            custom_name = data.get('custom_name')
            yolo_class = data.get('yolo_class')
            
            if not custom_name or not yolo_class:
                return jsonify({'error': 'custom_name and yolo_class required'}), 400
            
            try:
                object_memory.start_enrollment(custom_name, yolo_class)
                return jsonify({
                    'success': True,
                    'message': f'Started enrollment for {custom_name}',
                    'status': object_memory.get_enrollment_progress()
                })
            except ValueError as e:
                return jsonify({'error': str(e)}), 400
        
        elif action == 'add_frame':
            image_data = data.get('image')
            bbox = data.get('bbox')  # [x1, y1, x2, y2]
            
            if not image_data or not bbox:
                return jsonify({'error': 'image and bbox required'}), 400
            
            frame = decode_image(image_data)
            num_frames = object_memory.add_enrollment_frame(frame, tuple(bbox))
            
            return jsonify({
                'success': True,
                'frames_captured': num_frames,
                'status': object_memory.get_enrollment_progress()
            })
        
        elif action == 'finish':
            success = object_memory.finish_enrollment()
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Object enrolled successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Enrollment failed - not enough features'
                }), 400
        
        elif action == 'cancel':
            object_memory.cancel_enrollment()
            return jsonify({
                'success': True,
                'message': 'Enrollment cancelled'
            })
        
        else:
            return jsonify({'error': 'Invalid action'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/enrollment_status', methods=['GET'])
def enrollment_status():
    """Get current enrollment progress"""
    try:
        status = object_memory.get_enrollment_progress()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/remembered_objects', methods=['GET'])
def get_remembered_objects():
    """List all remembered objects"""
    try:
        objects = object_memory.list_remembered_objects()
        return jsonify({
            'count': len(objects),
            'objects': [{'name': name, 'yolo_class': yolo_class} for name, yolo_class in objects]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/find_object/<custom_name>', methods=['GET'])
def find_object(custom_name):
    """Query last known location of a remembered object"""
    try:
        location = object_memory.get_object_location(custom_name)
        
        if location:
            message = f"{custom_name} last seen at {location['location'] or 'unknown location'}"
            audio.speak_navigational(message)
            
            return jsonify({
                'found': True,
                'object': custom_name,
                'location': location['location'],
                'gps_lat': location['gps_lat'],
                'gps_lon': location['gps_lon'],
                'context': location['context'],
                'confidence': location['confidence'],
                'timestamp': location['timestamp'],
                'message': message
            })
        else:
            message = f"I haven't seen your {custom_name} recently"
            audio.speak_navigational(message)
            
            return jsonify({
                'found': False,
                'object': custom_name,
                'message': message
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/remembered_object/<custom_name>', methods=['DELETE'])
def delete_remembered_object(custom_name):
    """Delete a remembered object"""
    try:
        success = object_memory.delete_object(custom_name)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Deleted {custom_name}'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Object {custom_name} not found'
            }), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/confirm_match', methods=['POST'])
def confirm_match():
    """Confirm a pending ambiguous object match"""
    try:
        if not object_memory.pending_confirmation:
            return jsonify({'success': False, 'message': 'No pending match to confirm'}), 400
        
        name = object_memory.pending_confirmation.get('custom_name')
        data = request.get_json(silent=True) or {}
        pos = gps_tracker.get_position() if gps_tracker else None
        lat = data.get('gps_lat', pos[0] if pos else None)
        lon = data.get('gps_lon', pos[1] if pos else None)

        location = data.get('location')
        if config.INDOOR_MODE:
            # Indoors: skip reverse-geocoding; GPS coords not meaningful
            if not location:
                location = None  # caller may have supplied a context string via POST body
            lat, lon = None, None
        elif not location and lat is not None and lon is not None:
            location = reverse_geocode(lat, lon)

        success = object_memory.confirm_pending_match(
            location=location,
            gps_lat=lat,
            gps_lon=lon
        )
        if success:
            audio.speak_social("Confirmed")
            return jsonify({
                'success': True,
                'message': f'Confirmed match for {name}',
                'location': location
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to confirm match'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reject_match', methods=['POST'])
def reject_match():
    """Reject a pending ambiguous object match"""
    try:
        if not object_memory.pending_confirmation:
            return jsonify({'success': False, 'message': 'No pending match to reject'}), 400
        
        object_memory.clear_pending_match()
        audio.speak_social("Okay, never mind.")
        return jsonify({
            'success': True,
            'message': 'Pending match rejected'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500



def generate_audio_feedback(objects, faces, traffic_signal, item_updates, remembered_objects=[]):
    """
    Generate priority-based audio feedback
    
    Priority hierarchy:
    1. EMERGENCY: Traffic signals (DISABLED)
    2. SOCIAL: Recognized faces, Remembered objects
    3. NAVIGATIONAL: Important obstacles
    4. STATUS: Item locations
    """
    messages = []
    
    # 1. EMERGENCY: Traffic Signal - DISABLED (too many false positives)
    # if traffic_signal:
    #     if traffic_signal['signal_type'] == 'RED':
    #         message = "STOP - Red Light detected"
    #         audio.speak_emergency(message)
    #         messages.append({'priority': 1, 'type': 'emergency', 'message': message})
    #     elif traffic_signal['signal_type'] == 'GREEN':
    #         message = "Green Light - Safe to cross"
    #         audio.speak_emergency(message)
    #         messages.append({'priority': 1, 'type': 'emergency', 'message': message})
    
    # 2. SOCIAL: Recognized faces
    for face in faces:
        if face['name'] != 'Unknown':
            message = f"Hello {face['name']}"
            audio.speak_social(message)
            messages.append({'priority': 2, 'type': 'social', 'message': message})
    
    # 2b. SOCIAL: Remembered objects (custom items)
    for obj in remembered_objects:
        message = f"Found your {obj['custom_name']}"
        audio.speak_social(message)
        messages.append({'priority': 2, 'type': 'social', 'message': message})
    
    # 3. NAVIGATIONAL: Important objects (obstacles)
    obstacle_classes = ['chair', 'sofa', 'person']
    for obj in objects:
        if obj['class_name'] in obstacle_classes:
            message = f"{obj['class_name'].capitalize()} ahead"
            audio.speak_navigational(message)
            messages.append({'priority': 3, 'type': 'navigational', 'message': message})
            break  # Only announce one obstacle to avoid spam
    
    # 4. STATUS: Item location updates
    for item_update in item_updates:
        message = f"{item_update['item']} on {item_update['location']}"
        audio.speak_status(message)
        messages.append({'priority': 4, 'type': 'status', 'message': message})
    
    return messages


if __name__ == '__main__':
    print(f"\n[INFO] Starting VisionMate Server on {FLASK_HOST}:{FLASK_PORT}")
    print(f"[INFO] Endpoints available:")
    print(f"   POST /process_frame - Main AI inference")
    print(f"   POST /enroll_face - Enroll new face")
    print(f"   POST /api/enroll_object - Remember custom objects")
    print(f"   GET /api/remembered_objects - List remembered objects")
    print(f"   GET /api/find_object/<name> - Find object location")
    print(f"   DELETE /api/remembered_object/<name> - Delete object")
    print(f"   POST /api/confirm_match - Confirm pending ambiguous match")
    print(f"   POST /api/reject_match - Reject pending ambiguous match")
    print(f"   GET /query_item?item=keys - Find item location")
    print(f"   GET /get_all_faces - List enrolled faces")
    print(f"   GET /health - Health check")
    print(f"\n[OK] Server ready!\n")
    
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG, threaded=True)
