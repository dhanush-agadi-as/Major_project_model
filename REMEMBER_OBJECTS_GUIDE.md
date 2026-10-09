# Remember Objects Feature - Quick Guide

## Overview
The "Remember Objects" feature allows VisionMate to distinguish your specific objects (like "house keys") from generic objects (like "keys"). It uses visual feature matching with ORB descriptors to recognize custom-enrolled objects.

## How It Works

### 1. Enrollment Process
- User shows object to camera  
- System captures 5 frames from different angles
- Extracts visual features (keypoints + descriptors + color histogram)
- Stores in database with custom name

### 2. Recognition Process
- YOLO detects object class (e.g., "keys")
- Extracts features from detected object
- Matches against all enrolled objects of same class
- Returns custom name if confidence ≥ 70%

### 3. Sighting Tracking
- Records when/where object was last seen
- Stores nearby objects as context
- GPS coordinates (when available)
- Queryable via "Find" command

## Files Created

### Core Modules
1. **visual_matcher.py** - ORB feature extraction and matching
2. **object_memory.py** - High-level enrollment/recognition manager
3. **database.py** - Updated with `remembered_objects` and `object_sightings` tables

### API Endpoints (server.py)
```
POST   /api/enroll_object        - Enroll custom objects
GET    /api/enrollment_status     - Check enrollment progress
GET    /api/remembered_objects    - List all enrolled objects
GET    /api/find_object/<name>    - Query last seen location
DELETE /api/remembered_object/<name> - Delete object
```

### Test Scripts
- `test/test_database_objects.py` - Database schema tests
- `test/test_object_memory.py` - Enrollment/recognition tests
- `test/test_object_memory_api.py` - API workflow tests
- `test/test_server_compatibility.py` - Server integration tests

## Using the Integration Test

### Run Test
```bash
conda activate D:\conda_envs\visionmate
python test\test_integration.py
```

### Keyboard Controls
| Key | Action |
|-----|--------|
| **R** | Remember object - Start enrollment |
| **L** | List all remembered objects |
| **F** | Find object - Query last seen location |
| **E** | Enroll face |
| **A** | Toggle audio feedback |
| **Q** | Quit |

### Enrollment Workflow (Press 'R')
1. Shows list of detected objects
2. Select object number
3. Enter custom name (e.g., "house keys")
4. System captures 5 frames automatically
5. Keep object steady in view
6. Purple box shows recording progress
7. Audio confirms when complete

### Visual Feedback
- **Purple box** during enrollment
- **Purple label** with custom name when recognized
- **Confidence percentage** displayed
- **Audio announcement**: "Found your [custom name]"

## API Usage Example

### Enroll Object (Multi-Step)
```python
import requests
import base64

# Step 1: Start enrollment
response = requests.post('http://localhost:5000/api/enroll_object', json={
    'action': 'start',
    'custom_name': 'house keys',
    'yolo_class': 'keys'
})

# Step 2: Add frames (repeat 3-5 times)
with open('frame.jpg', 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode()

response = requests.post('http://localhost:5000/api/enroll_object', json={
    'action': 'add_frame',
    'image': f'data:image/jpeg;base64,{img_base64}',
    'bbox': [100, 100, 200, 200]  # [x1, y1, x2, y2]
})

# Step 3: Finish enrollment
response = requests.post('http://localhost:5000/api/enroll_object', json={
    'action': 'finish'
})
```

### Query Object Location
```python
response = requests.get('http://localhost:5000/api/find_object/house keys')
data = response.json()

if data['found']:
    print(f"Location: {data['location']}")
    print(f"GPS: ({data['gps_lat']}, {data['gps_lon']})")
    print(f"Context: {data['context']}")
    print(f"Timestamp: {data['timestamp']}")
```

### List Remembered Objects
```python
response = requests.get('http://localhost:5000/api/remembered_objects')
data = response.json()

print(f"Found {data['count']} objects:")
for obj in data['objects']:
    print(f"  - {obj['name']} ({obj['yolo_class']})")
```

### Delete Object
```python
response = requests.delete('http://localhost:5000/api/remembered_object/house keys')
```

## Database Schema

### remembered_objects Table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| custom_name | TEXT | User-friendly name (unique) |
| yolo_class | TEXT | Base YOLO class |
| visual_features | TEXT | JSON with keypoints/descriptors/histogram |
| enrollment_frames | INTEGER | Number of frames used |
| timestamp | DATETIME | Enrollment time |

### object_sightings Table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| object_id | INTEGER | FK to remembered_objects |
| location | TEXT | Description of location |
| gps_lat | REAL | GPS latitude |
| gps_lon | REAL | GPS longitude |
| context | TEXT | JSON with nearby objects |
| confidence | REAL | Recognition confidence |
| timestamp | DATETIME | Sighting time |

## Configuration (config.py)
```python
OBJECT_MEMORY_MATCH_THRESHOLD = 0.7    # Min confidence for recognition
OBJECT_ENROLLMENT_FRAMES = 5           # Frames to capture during enrollment
OBJECT_MATCH_COOLDOWN = 2.0            # Seconds between announcements
```

## Tips for Best Results

### Enrollment
✅ **Good:**
- Capture from multiple angles
- Good lighting
- Steady camera
- Clear view of object
- Object fills bounding box

❌ **Avoid:**
- Blurry images
- Object too small
- Extreme angles
- Heavy shadows
- Motion blur

### Recognition
- Works best with objects that have distinctive features
- Keys, wallets, badges work well
- Plain/uniform objects harder to distinguish
- Confidence ≥ 70% = reliable match
- Confidence 50-70% = possible match
- Confidence < 50% = no match

## Testing Checklist

- [x] Database schema with 2 new tables
- [x] Visual feature extraction with ORB
- [x] Multi-frame enrollment (aggregation)
- [x] Feature matching with Lowe's ratio test
- [x] Object memory manager API
- [x] Server endpoints (5 new routes)
- [x] Integration test with 'R' key
- [x] Audio feedback for recognized objects
- [x] Sighting tracking with context
- [x] Location query by name

## Next Steps (Hardware Integration)

When deploying to Raspberry Pi with 5MP camera:
1. No code changes needed - OpenCV VideoCapture works with both
2. Ensure adequate lighting for feature extraction
3. Consider GPU acceleration for YOLO inference
4. Add GPS module integration for real coordinates
5. Optimize for real-time performance (30+ FPS)

## Support

For issues or questions:
1. Check test scripts in `test/` folder
2. Review logs (INFO/DEBUG level)
3. Verify database tables exist
4. Test with well-lit, textured objects first
