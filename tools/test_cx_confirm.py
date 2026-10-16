"""
Forces the 50-75% 'ambiguous' band so C / X can be tested for real.

Part A (automatic, no camera): runs ObjectMemory.recognize_object with a stubbed matcher that
returns confidence 0.62, then exercises confirm_pending_match / clear_pending_match exactly as
the C and X key handlers in laptop_server.py do.

Part B (live, optional):  python tools/test_cx_confirm.py --live
  Temporarily raises OBJECT_MATCH_HIGH_CONFIDENCE to 0.99 so ANY real match of a remembered
  object lands in 'ambiguous'. Then start laptop_server normally with
  OBJECT_MATCH_HIGH_CONFIDENCE=0.99 (printed below) and press C / X while the log shows
  'Ambiguous match'. Revert afterwards.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, config

if "--live" in sys.argv:
    print("Set in config.py for the live test:\n  OBJECT_MATCH_HIGH_CONFIDENCE = 0.99\n"
          "Then remember an object, show it to the camera, wait for 'Ambiguous match' in the log,\n"
          "and press C (expect 'Confirmed match') then repeat and press X (expect 'Rejected pending match').\n"
          "Set it back to 0.75 afterwards.")
    sys.exit(0)

from database import VisionMateDB
import object_memory as om

class StubMatcher:
    def __init__(self, conf): self.conf = conf
    def extract_features(self, img, bbox): return {"feature_type": "deep", "deep_embedding": [0.1]}
    def find_best_match(self, q, cands): return (1, "my mug", self.conf)

def run(conf):
    db = VisionMateDB(os.path.join(tempfile.mkdtemp(), "t.db")) if "db_path" in VisionMateDB.__init__.__code__.co_varnames else VisionMateDB()
    db.add_remembered_object("my mug", "cup", {"feature_type": "deep", "deep_embedding": [0.1]}, 1)
    mem = om.ObjectMemory(db, StubMatcher(conf))
    mem._candidates_cache["cup"] = (__import__("time").time(), [object()])
    return mem, mem.recognize_object(np.zeros((100, 100, 3), np.uint8), (0, 0, 50, 50), "cup")

fails = 0
def check(name, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + name); fails += (not cond)

mem, r = run(0.62)
check("0.62 -> 'ambiguous' label", r and r[2] == "ambiguous")
check("pending_confirmation populated", mem.pending_confirmation and mem.pending_confirmation["custom_name"] == "my mug")
ok = mem.confirm_pending_match(location="near laptop", gps_lat=None, gps_lon=None)
check("C: confirm_pending_match returns True", ok is True)
check("C: pending cleared afterwards", mem.pending_confirmation is None)

mem, r = run(0.62)
mem.clear_pending_match()
check("X: pending cleared", mem.pending_confirmation is None)
mem, r = run(0.90); check("0.90 -> 'high', nothing pending", r[2] == "high" and mem.pending_confirmation is None)
mem, r = run(0.30); check("0.30 -> discarded", r is None and mem.pending_confirmation is None)
mem, r = run(0.62)
mem.pending_confirmation['timestamp'] -= 60   # make it 60s old
check("expired pending: get_pending_match() is None", mem.get_pending_match() is None)
mem, r = run(0.62)
mem.pending_confirmation['timestamp'] -= 60
check("expired pending: confirm_pending_match() returns False", mem.confirm_pending_match() is False)
sys.exit(1 if fails else 0)
