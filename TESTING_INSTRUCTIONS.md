# 🎤 VOICE COMMAND TESTING GUIDE

## 🚀 Quick Start

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
D:\conda_envs\visionmate\python.exe demo_voice_remember_find.py
```

---

## ✅ COMPLETE TEST CHECKLIST

### **PART 1: Basic Setup (2 min)**

**Steps:**
1. Run the command above
2. Wait for "✓ All systems ready!"
3. Webcam window should open
4. Check that objects are being detected (yellow boxes)

**Expected:**
- ✅ Webcam shows video feed
- ✅ Yellow boxes around detected objects
- ✅ Text showing "Objects: X" in top-left
- ✅ "Detected:" list at bottom showing object names

**If fails:**
- ❌ No webcam → Check if another app is using it
- ❌ No boxes → Move objects into view (phone, bottle, etc.)

---

### **PART 2: Voice Recognition Test (3 min)**

**Test 1: Basic Voice Recognition**
1. Press **ENTER**
2. Say: **"Hello"** (just to test mic)
3. Check console output

**Expected:**
- ✅ System says "Listening"
- ✅ Console shows: "Could not parse command: 'hello'"
- ✅ This is CORRECT - means mic is working!

**Test 2: Valid Command**
1. Press **ENTER**
2. Say: **"List all items"**

**Expected:**
- ✅ System recognizes "List"
- ✅ Says: "No items remembered yet" (if first time)
- ✅ Console shows: "✗ No items remembered"

---

### **PART 3: Remember Objects (5 min)**

**Test 3: Remember Phone**
1. Hold your **phone** in front of webcam
2. Wait until you see yellow box labeled "cell phone"
3. Press **ENTER**
4. Say clearly: **"Remember this phone"**
5. Keep phone steady
6. Watch for purple box and "Recording 1/5 ... 2/5 ... 3/5 ... 4/5 ... 5/5"
7. Wait for confirmation

**Expected:**
- ✅ Console: "🎤 Listening for command..."
- ✅ Console: "✓ Enrolling 'myphone'..."
- ✅ Screen: Purple box around phone
- ✅ Screen: "Recording 1/5", "Recording 2/5", etc.
- ✅ Audio: "Remembered your myphone"
- ✅ Console: "✓ Enrolled 'myphone'!"

**If fails:**
- ❌ "No phone detected" → Make sure phone is visible and recognized as "cell phone"
- ❌ "Could not understand audio" → Speak louder and clearer
- ❌ No purple box → Check console for error messages

---

**Test 4: Remember Bottle**
1. Hold a **bottle** in front of webcam
2. Wait for yellow box labeled "bottle"
3. Press **ENTER**
4. Say: **"Remember this bottle"**
5. Keep bottle steady for 5 frames

**Expected:**
- ✅ Same process as phone
- ✅ Enrolls as "mybottle"
- ✅ Console: "✓ Enrolled 'mybottle'!"

---

**Test 5: Try to Remember Without Object**
1. Make sure NO phone is visible
2. Press **ENTER**
3. Say: **"Remember this phone"**

**Expected:**
- ✅ Audio: "No phone detected in view"
- ✅ Console shows objects currently detected

---

### **PART 4: Find Objects (3 min)**

**Test 6: Find Enrolled Phone**
1. Move phone OUT of camera view
2. Press **ENTER**
3. Say: **"Find my phone"**

**Expected:**
- ✅ Audio: "Your phone last seen near [objects]"
- ✅ Console: "✓ Your phone last seen near..."

---

**Test 7: Find Bottle**
1. Press **ENTER**
2. Say: **"Find my bottle"**

**Expected:**
- ✅ System tells you where bottle was last seen
- ✅ Lists nearby objects (chair, person, etc.)

---

**Test 8: Find Non-existent Object**
1. Press **ENTER**
2. Say: **"Find my keys"** (if you haven't enrolled keys)

**Expected:**
- ✅ Audio: "Haven't seen keys"
- ✅ Console: "✗ No record of keys"

---

### **PART 5: List Items (1 min)**

**Test 9: List Everything**
1. Press **ENTER**
2. Say: **"List all items"**

**Expected:**
- ✅ Audio: "I remember: myphone, mybottle"
- ✅ Console: "✓ Remembered: myphone, mybottle"

---

### **PART 6: Alternative Commands (2 min)**

**Test 10: Different "Find" Phrases**
Try these variations:
- Press ENTER → "**Where is my phone**" → Should work
- Press ENTER → "**Locate my bottle**" → Should work

**Test 11: Different "Remember" Phrases**
- Press ENTER → "**Save this phone**" → Should work
- Press ENTER → "**Enroll this bottle**" → Should work

---

### **PART 7: Edge Cases (3 min)**

**Test 12: Rapid Commands**
1. Press ENTER → Say command
2. Immediately press ENTER again
3. Try to say another command

**Expected:**
- ✅ Console: "⏳ Please wait a moment..."
- ✅ System has 1-second cooldown (prevents spam)

---

**Test 13: Gibberish**
1. Press ENTER
2. Say random nonsense

**Expected:**
- ✅ System handles gracefully
- ✅ Shows "Unknown command" or "Could not parse"
- ✅ No crashes

---

**Test 14: Silent Input**
1. Press ENTER
2. Stay completely silent for 5 seconds

**Expected:**
- ✅ Console: "Could not understand audio"
- ✅ Console: "No valid speech recognized"
- ✅ System ready for next command

---

**Test 15: Multiple Objects Visible**
1. Put phone, bottle, and chair in view
2. Press ENTER
3. Say "Remember this bottle"

**Expected:**
- ✅ System correctly identifies bottle (not phone or chair)
- ✅ Enrolls only the bottle

---

## 📊 TEST RESULTS FORM

After testing, fill this out:

```
========================================
VOICE COMMAND TEST RESULTS
Date: _____________
Tester: ___________
========================================

BASIC SETUP:
[ ] Webcam opens
[ ] Objects detected with yellow boxes
[ ] Status text visible

VOICE RECOGNITION:
[ ] Microphone captures voice
[ ] Speech-to-text working
[ ] "List all items" recognized

REMEMBER FEATURE:
[ ] "Remember this phone" works
[ ] Purple box shows during recording
[ ] Captures 5 frames successfully
[ ] Audio confirmation plays
[ ] "Remember this bottle" works
[ ] Handles "No object detected" correctly

FIND FEATURE:
[ ] "Find my phone" works
[ ] "Find my bottle" works
[ ] Shows nearby context
[ ] Handles non-existent objects

LIST FEATURE:
[ ] "List all items" works
[ ] Speaks all enrolled objects
[ ] Shows "No items" when empty

ALTERNATIVE PHRASES:
[ ] "Where is my phone" works
[ ] "Locate my bottle" works
[ ] "Save this phone" works

EDGE CASES:
[ ] Cooldown prevents spam
[ ] Handles gibberish gracefully
[ ] Handles silence timeout
[ ] Handles multiple objects correctly

OVERALL STATUS:
[ ] All tests passed
[ ] Minor issues (describe below)
[ ] Major issues (describe below)

NOTES:
_________________________________
_________________________________
_________________________________

BUGS FOUND:
_________________________________
_________________________________
_________________________________
========================================
```

---

## 🐛 Common Issues & Solutions

### Issue 1: "Could not understand audio"
**Solution:**
- Speak louder and clearer
- Reduce background noise
- Move closer to microphone
- Check microphone volume in Windows settings

### Issue 2: "No [object] detected in view"
**Solution:**
- Make sure object is visible in webcam
- Check bottom of screen for "Detected:" list
- Object must match YOLO classes: phone, bottle, chair, person, couch, tv, keys, clock

### Issue 3: Enrollment stuck at "Recording 2/5"
**Solution:**
- Keep object completely still
- Make sure object stays in frame
- Good lighting helps detection
- Wait for all 5 frames

### Issue 4: Webcam doesn't open
**Solution:**
- Close other apps using webcam (Teams, Zoom, etc.)
- Check webcam is connected
- Try index 1: `cv2.VideoCapture(1)` if needed

---

## 🎯 SUCCESS CRITERIA

You should be able to:
1. ✅ Enroll 2-3 different objects by voice
2. ✅ Find those objects by voice after moving them
3. ✅ List all enrolled objects
4. ✅ Handle errors gracefully (wrong commands, missing objects)
5. ✅ Use alternative phrases ("where is" vs "find")

**If all 5 work → SYSTEM IS READY! 🚀**

---

## 💡 Pro Tips

1. **Best objects to test:** Phone, bottle, TV remote, keys (if detected)
2. **Speak naturally:** Don't need to shout or speak robotically
3. **Wait for "Listening":** Make sure audio feedback plays before speaking
4. **Keep objects visible:** System needs to see object during enrollment
5. **Test finding immediately:** Move object away right after enrollment to test find feature

---

## 🔄 Quick Test Sequence (2 minutes)

If you're in a hurry, do this:

```
1. Run program
2. Hold phone in view
3. ENTER → "Remember this phone"
4. Wait for 5 frames
5. Move phone away
6. ENTER → "Find my phone"
7. ENTER → "List all items"
8. Press Q to quit
```

If all 3 work → Everything works! ✅

---

## 📝 What to Report Back

After testing, tell me:

1. **What worked:** Which features passed all tests
2. **What failed:** Which tests failed and what error messages
3. **Performance:** Was speech recognition fast/accurate?
4. **User experience:** Was it easy to use? Confusing?
5. **Bugs:** Any crashes or unexpected behavior?

---

## 🎬 Ready to Test!

Run this command and follow the checklist above:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
D:\conda_envs\visionmate\python.exe demo_voice_remember_find.py
```

**Good luck! Test everything and let me know how it goes!** 🚀
