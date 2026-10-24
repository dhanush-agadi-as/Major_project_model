#!/usr/bin/env python3
"""
tools/send_test_alert.py
========================
Creates a test emergency alert in Firebase Firestore and sends FCM data pushes,
updates its location a few times to simulate live movement, and then cancels it.
Used to test the family mobile app end-to-end without needing the smart glasses hardware.
"""

import os
import sys
import time
import argparse
import logging

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("send_test_alert")

try:
    import config
    DEFAULT_SA = getattr(config, 'FIREBASE_SERVICE_ACCOUNT_PATH', 'firebase_service_account.json')
    DEFAULT_USER = getattr(config, 'EMERGENCY_USER_NAME', 'Bharath (Test)')
except Exception:
    DEFAULT_SA = "firebase_service_account.json"
    DEFAULT_USER = "Bharath (Test)"

from alert_sender import AlertSender


class MockGPSTracker:
    """Mock tracker for simulating location updates in test alert."""
    def __init__(self, start_lat=12.9265, start_lon=77.5264):
        self.lat = start_lat
        self.lon = start_lon

    def _get_fresh_phone_position_info(self):
        return ((self.lat, self.lon), 1.0, 10.0)

    def _get_ip_tracker(self):
        return None

    def get_city(self):
        return "Bangalore"


def main():
    parser = argparse.ArgumentParser(description="Send test emergency alert to Firebase family app.")
    parser.add_argument("--sa", default=DEFAULT_SA, help="Path to firebase_service_account.json")
    parser.add_argument("--user", default=DEFAULT_USER, help="User name in alert")
    parser.add_argument("--severity", default="severe", choices=["normal", "severe"], help="Severity level")
    parser.add_argument("--reason", default="Test fall detection alert", help="Alert reason")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode without contacting Firebase")
    parser.add_argument("--updates", type=int, default=3, help="Number of location updates to perform")
    parser.add_argument("--interval", type=float, default=3.0, help="Interval between location updates in seconds")
    args = parser.parse_args()

    # If service account file doesn't exist and not explicitly passed, inform user
    dry_run = args.dry_run
    if not os.path.exists(args.sa) and not dry_run:
        logger.warning(f"Service account file '{args.sa}' not found. Falling back to --dry-run mode.")
        dry_run = True

    mock_gps = MockGPSTracker()

    def speak_print(text):
        print(f"[TTS Announcement]: {text}")

    print("=" * 60)
    print("  VisionMate Emergency Alert Test Tool")
    print(f"  User: {args.user}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'REAL FIREBASE'}")
    print("=" * 60)

    sender = AlertSender(
        service_account_path=args.sa,
        user_name=args.user,
        dry_run=dry_run,
        speak_fn=speak_print,
        gps_tracker=mock_gps
    )

    print("\n1. Creating test emergency alert...")
    alert_id = sender.send_alert(severity=args.severity, reason=args.reason)
    if not alert_id:
        print("[FAIL] Failed to create alert.")
        sys.exit(1)

    print(f"[OK] Alert created with ID: {alert_id}")

    # Location updates
    for i in range(1, args.updates + 1):
        print(f"\nWaiting {args.interval}s before location update {i}/{args.updates}...")
        time.sleep(args.interval)

        # Shift GPS coordinates slightly
        mock_gps.lat += 0.00015
        mock_gps.lon += 0.00015

        loc = sender.resolve_location()
        print(f"-> Updating location {i}: lat={loc.get('lat'):.6f}, lon={loc.get('lon'):.6f}, place={loc.get('placeName')}")

        if sender.db and not sender.dry_run:
            try:
                sender.db.collection("alerts").document(alert_id).update({"location": loc})
                print(f"[OK] Updated location in Firestore alerts/{alert_id}")
            except Exception as e:
                print(f"[WARN] Error updating location in Firestore: {e}")
        else:
            print(f"[DRY RUN] Would update Firestore alerts/{alert_id} with new location")

    print(f"\nWaiting {args.interval}s before cancelling alert...")
    time.sleep(args.interval)

    print(f"2. Cancelling alert {alert_id}...")
    sender.cancel_alert(alert_id)
    print(f"[OK] Alert {alert_id} cancelled.")
    print("Test alert workflow completed successfully.")


if __name__ == "__main__":
    main()
