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

---

### Step 3: Test with Webcam & Voice-Only Interface (2 minutes)

```powershell
# Run laptop wireless server / pipeline
python wireless\laptop_server.py
```

VisionMate is now completely **VOICE-ONLY**:
- 🎙️ **Wake Word**: Continuous listening — say `"Hey Vision"` (or `"Vision"`) then your command.
- 🔔 **Wake Word Alone**: Say `"Hey Vision"`, hear a short ready beep, then speak your command within 6 seconds.
- 🗣️ **One-shot Command**: Say `"Hey Vision find my mobile"` or `"Hey Vision stop finding"` in one sentence.
- 💬 **Confirmations**: When VisionMate asks `"Is this your <object>?"` or `"Say yes to confirm"`, reply `"yes"`, `"correct"`, `"yeah"`, or `"no"`, `"wrong"`, `"nope"` within 6 seconds without repeating the wake word.
- 📝 **Auto-Read Voice Notes**: Say `"Hey Vision take note <text>"` to save a note. VisionMate immediately reads it back: `"Saved. You said: <text>"`.
- 🔇 **Echo Suppression**: TTS audio is automatically suppressed while VisionMate is speaking and for 0.25s after speech finishes.
- ⚙️ **Keyboard Fallback**: Fully disabled by default (`KEYBOARD_FALLBACK = False` in `config.py`). Set to `True` only for debug hotkeys (`Q` to quit).

**Voice Commands Reference**:
- `"remember this <object>"` - Enroll custom object currently in view
- `"find my <object>"` - Search and guide to an enrolled object
- `"stop finding"` / `"stop search"` - Cancel active object search
- `"enroll face <name>"` - Enroll person's face in camera view
- `"take note <text>"` - Save a voice note (auto-read back)
- `"read notes"` - Read all saved voice notes
- `"delete note <number>"` - Delete a specific voice note
- `"clear notes"` - Delete all voice notes
- `"navigate"` / `"stop navigation"` - Toggle obstacle navigation
- `"navigate to <destination>"` / `"stop GPS"` - GPS walking directions
- `"advance GPS"` - Simulate GPS progress in mock mode
- `"toggle audio"` - Turn audio announcements on/off
- `"list all items"` - Spoken list of remembered objects
- `"show database"` - Spoken summary of database contents
- `"shut down"` / `"quit"` - Request shutdown (confirms on `"yes"`)
- `"help"` - Spoken command list

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
