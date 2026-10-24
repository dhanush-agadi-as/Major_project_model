#!/usr/bin/env python3
"""
tools/record_motion.py
======================
Polls http://<PI_HOST>:5000/motion for 30 seconds and saves readings
to a CSV file for analyzing IMU telemetry and tuning fall detection thresholds.
"""

import os
import sys
import time
import csv
import argparse
import requests

# Ensure repository root is on path for config import
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import config
    DEFAULT_HOST = getattr(config, "PI_HOST", "127.0.0.1")
except Exception:
    DEFAULT_HOST = "127.0.0.1"


def record_motion(host: str = DEFAULT_HOST,
                  port: int = 5000,
                  duration_s: float = 30.0,
                  poll_interval_s: float = 0.05,
                  output_file: str = "motion_data.csv") -> str:
    url = f"http://{host}:{port}/motion"
    print(f"Connecting to {url}...")
    print(f"Recording motion data for {duration_s} seconds (interval: {poll_interval_s*1000:.0f} ms)...")

    rows = []
    seen_event_ids = set()
    start_time = time.time()
    next_sample_time = start_time

    while (time.time() - start_time) < duration_s:
        loop_now = time.time()
        elapsed = loop_now - start_time

        try:
            resp = requests.get(url, timeout=0.5)
            if resp.status_code == 200:
                data = resp.json()
                ax = data.get("ax", 0.0)
                ay = data.get("ay", 0.0)
                az = data.get("az", 0.0)
                mag = data.get("magnitude", 0.0)
                gyro_mag = data.get("gyro_mag", 0.0)
                ts = data.get("ts", loop_now)
                events = data.get("events", [])

                new_event_types = []
                for ev in events:
                    ev_id = ev.get("id")
                    if ev_id and ev_id not in seen_event_ids:
                        seen_event_ids.add(ev_id)
                        new_event_types.append(f"{ev.get('type')}:{ev.get('peak_g')}g")

                event_str = ";".join(new_event_types) if new_event_types else ""

                rows.append({
                    "timestamp": f"{ts:.4f}",
                    "elapsed_s": f"{elapsed:.2f}",
                    "ax": f"{ax:.3f}",
                    "ay": f"{ay:.3f}",
                    "az": f"{az:.3f}",
                    "magnitude": f"{mag:.3f}",
                    "gyro_mag": f"{gyro_mag:.3f}",
                    "events": event_str
                })
            else:
                print(f"[{elapsed:.1f}s] Non-200 response: {resp.status_code}")
        except Exception as e:
            # Server might be slow or temporarily disconnected
            pass

        next_sample_time += poll_interval_s
        sleep_dur = max(0.0, next_sample_time - time.time())
        time.sleep(sleep_dur)

    out_path = os.path.abspath(output_file)
    fieldnames = ["timestamp", "elapsed_s", "ax", "ay", "az", "magnitude", "gyro_mag", "events"]
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Recorded {len(rows)} samples in {duration_s:.1f} s.")
    print(f"Data saved to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Record MPU6050 motion telemetry from Pi Hub.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Pi IP address or hostname")
    parser.add_argument("--port", type=int, default=5000, help="Pi Hub port (default 5000)")
    parser.add_argument("--duration", type=float, default=30.0, help="Recording duration in seconds (default 30)")
    parser.add_argument("--interval", type=float, default=0.05, help="Polling interval in seconds (default 0.05)")
    parser.add_argument("--output", default="motion_data.csv", help="Output CSV path")
    args = parser.parse_args()

    record_motion(
        host=args.host,
        port=args.port,
        duration_s=args.duration,
        poll_interval_s=args.interval,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
