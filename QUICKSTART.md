# VisionMate Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (2-3 minutes)

```powershell
cd d:\Model_for_VisionMate
pip install -r requirements.txt
```

Wait for installation to complete. First run will download AI models (~200MB).

---

### Step 2: Run Your First Test (1 minute)

```powershell
# Test the database
python database.py
```

Expected output: `✓ Database initialized successfully`

---

### Step 3: Test with Webcam (2 minutes)

```powershell
# Run full integration test
python test\test_integration.py
```

**Choose option 1** when prompted.

This will:
- ✅ Detect objects in real-time
- ✅ Recognize faces
- ✅ Detect traffic signals
- ✅ Track items
- ✅ Provide audio feedback

**Controls in the test**:
- Press **Q** to quit
- Press **E** to enroll a face
- Press **F** to find an item
- Press **A** to toggle audio

---

## 🎯 Quick Test Sequence

### 1. Object Detection (30 seconds)
```powershell
python test\test_object_detection.py
# Choose option 2 (static image test)
```

### 2. Face Recognition (2 minutes)
```powershell
python test\test_face_recognition.py
# Choose option 4 (full test)
# Enroll your face when prompted
```

### 3. Traffic Signals (1 minute)
```powershell
python test\test_traffic_signals.py
# Choose option 1 (synthetic test)
```

### 4. Item Tracking (1 minute)
```powershell
python test\test_item_tracking.py
# Choose option 1 (IoU test)
```

---

## 🌐 Start the API Server

### Terminal 1: Start Server
```powershell
python server.py
```

Wait for: `✓ Server ready!`

### Terminal 2: Test API
```powershell
python test\test_server_api.py
# Choose option 7 (all tests)
```

---

## 🎮 Demo Workflow for Hackathon

### Preparation (10 minutes before demo)

1. **Enroll faces of judges/team members**
   ```powershell
   python test\test_face_recognition.py
   # Option 1: Enroll 2-3 people
   ```

2. **Test all features once**
   ```powershell
   python test\test_integration.py
   # Option 1: Run full test
   ```

3. **Start the server**
   ```powershell
   python server.py
   ```

### During Demo (5 minutes)

**Show these features in order**:

1. **Object Detection** (30 sec)
   - Point camera at chair, bottle, person
   - Show real-time bounding boxes

2. **Face Recognition** (1 min)
   - Show enrolled face → "Hello John"
   - Show unknown face → "Unknown"

3. **Traffic Signals** (30 sec)
   - Show red object → "STOP - Red Light" (emergency audio)
   - Show green object → "Green Light - Safe to cross"

4. **Item Tracking** (1 min)
   - Place keys on chair for 3+ seconds
   - Query: "Where are my keys?"
   - Response: "Your keys are on the chair"

5. **Audio Priority** (30 sec)
   - Trigger multiple events
   - Show emergency interrupts others

6. **API Server** (1 min)
   - Show `/health` endpoint
   - Demonstrate `/process_frame` response
   - Show database with enrolled faces

---

## 📊 Check Your System Performance

```powershell
# Run this to measure FPS
python test\test_integration.py
# Let it run for 30 seconds, then press Q
# Note the final "Average FPS" value
```

**Target**: 
- With GPU: 30+ FPS ✅
- With CPU: 8-15 FPS ⚠️ (still usable)

---

## 🐛 Common Issues & Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| "No module named X" | `pip install X` |
| Webcam not opening | Change camera index to 1 in test script |
| Low FPS | Process every 5th frame, reduce resolution |
| Face always "Unknown" | Re-enroll with better lighting |
| No audio | Check speakers, or use print-only mode |
| Indoor GPS inaccurate | GPS and reverse-geocoded addresses are for OUTDOOR navigation only. Indoors, set `INDOOR_MODE = True` in `config.py` (already the default) so object-finding guidance relies on the camera and nearby-object context instead of GPS, which has no signal indoors. |

---

## 📱 Contact & Support

For the hackathon team:
- **Test first**: Run individual module tests before integration
- **Check README.md**: Full documentation available
- **Debug with tests**: Each test script is standalone

---

## ✅ Pre-Demo Checklist

Before your presentation:

- [ ] All tests pass without errors
- [ ] Webcam works and displays correctly
- [ ] At least 2 faces enrolled in database
- [ ] Server starts without errors
- [ ] Audio feedback is audible
- [ ] FPS is acceptable (>8 FPS)
- [ ] Know which test to run for each feature

---

## 🎉 You're Ready!

Everything is set up. Just run the tests and start building your demo!

**Main command for demo**:
```powershell
python test\test_integration.py
```

Good luck! 🚀
