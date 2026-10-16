"""
Phone GPS -> mobile_backend -> laptop_server poller -> gps_tracker, without a phone.
Run from the project root:  python tools/test_phone_gps_pipeline.py

Uses Flask's test client for mobile_backend (so no ports/SSL), then feeds the exact JSON that
GET /api/location returns through the same acceptance rule the laptop poller uses.
LIVE steps (real phone) are printed at the end.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mobile_backend as mb
from gps_tracker import GPSTracker

c = mb.app.test_client()
fails = 0
def check(n, cond):
    global fails; print(("PASS " if cond else "FAIL ") + n); fails += (not cond)

before = c.get("/api/location").get_json()
check(f"before any phone report source != 'phone' (got {before['source']})", before["source"] != "phone")

r = c.post("/api/location", json={"latitude": 13.0827, "longitude": 80.2707, "accuracy": 12})
check("POST /api/location accepted", r.status_code == 200)
after = c.get("/api/location").get_json()
check("GET now reports source 'phone' with posted coords",
      after["source"] == "phone" and abs(after["latitude"] - 13.0827) < 1e-6)
check("bad POST rejected (400)", c.post("/api/location", json={"latitude": "x"}).status_code == 400)

# laptop side: same rule as _poll_mobile_backend_location
laptop = GPSTracker(mock_mode=True) if "mock_mode" in GPSTracker.__init__.__code__.co_varnames else GPSTracker()
def poll_accept(d):
    if d.get("source") == "phone" and d.get("latitude") is not None:
        laptop.set_phone_position(d["latitude"], d["longitude"])
poll_accept(before)
check("laptop ignores non-phone fallback", laptop._get_fresh_phone_position() is None)
poll_accept(after)
check("laptop adopts real phone fix", laptop._get_fresh_phone_position() == (13.0827, 80.2707))
check("laptop.get_position() returns phone fix", laptop.get_position() == (13.0827, 80.2707))
print("\nLIVE (needs you): 1) python mobile_backend.py  2) phone on same Wi-Fi -> https://<laptop-ip>:5001/locate, accept the cert warning + allow location"
      "\n3) python wireless/laptop_server.py  4) log must show 'Phone GPS report received: (lat, lon)' every ~2s"
      "\n5) say 'find my <object>' and confirm guidance still runs while that log line keeps appearing.")
sys.exit(1 if fails else 0)
