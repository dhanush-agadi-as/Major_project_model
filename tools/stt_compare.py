"""
VisionMate STT Comparison Tool
==============================
Records N voice commands from the microphone, runs both offline Cactus Whistle
and online Google Speech Recognition, and prints text, latency, and match comparison.

Usage:
    python tools/stt_compare.py [N]
    python tools/stt_compare.py -n 5
    python tools/stt_compare.py --mock (run synthetic comparison without microphone)
"""

import os
import sys
import time
import argparse
import difflib
import logging

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import speech_recognition as sr
import config
import stt_engine


def run_comparison(num_commands: int = 5, timeout: float = 10.0, phrase_limit: float = 8.0, mock: bool = False):
    print("=" * 70)
    print("                  VisionMate STT Comparison Tool")
    print("      Offline Cactus Whistle  vs  Online Google Speech Recognition")
    print("=" * 70)

    # 1. Check Whistle availability
    if not stt_engine.is_available():
        print("[!] ERROR: cactus-needle (Whistle) is not installed or failed to import.")
        print("    Please install it using: pip install cactus-needle")
        return False

    # 2. Check Internet connectivity
    online = stt_engine.is_internet_reachable()
    print(f"[*] Internet connection: {'ONLINE' if online else 'OFFLINE (Google STT may fail)'}")

    # 3. Warm up Whistle model
    print("[*] Warming up Cactus Whistle model...")
    t0 = time.perf_counter()
    stt_engine.warm_up()
    warmup_ms = int((time.perf_counter() - t0) * 1000)
    print(f"[*] Model ready (warmup took {warmup_ms} ms)")

    # 4. Load keywords
    keywords = stt_engine.get_keywords()
    print(f"[*] Loaded {len(keywords)} biasing keywords (commands + DB names):")
    print(f"    {', '.join(keywords[:10])}...")

    recognizer = sr.Recognizer()
    results = []

    if mock:
        print("\n[*] Running in MOCK mode (synthetic audio samples)...")
        import numpy as np
        synthetic_commands = [
            ("hey vision remember", np.sin(np.linspace(0, 100, 32000)).astype(np.float32) * 0.4),
            ("find my phone", np.sin(np.linspace(0, 80, 24000)).astype(np.float32) * 0.4),
            ("navigate", np.sin(np.linspace(0, 120, 16000)).astype(np.float32) * 0.4),
        ]
        num_commands = min(num_commands, len(synthetic_commands))
        for i in range(num_commands):
            expected_text, audio_samples = synthetic_commands[i]
            print(f"\n--- [Mock Command {i+1}/{num_commands}] ---")
            
            # Whistle
            t_w = time.perf_counter()
            w_text = stt_engine.transcribe(audio_samples, keywords=keywords)
            w_ms = int((time.perf_counter() - t_w) * 1000)

            # Simulated Google
            g_text = expected_text
            g_ms = 450

            clean_w = w_text.lower().strip()
            clean_g = g_text.lower().strip()
            exact = (clean_w == clean_g)
            ratio = difflib.SequenceMatcher(None, clean_w, clean_g).ratio()

            print(f"  Whistle: '{w_text}' ({w_ms} ms)")
            print(f"  Google:  '{g_text}' ({g_ms} ms)")
            print(f"  Match:   {'EXACT MATCH' if exact else f'{ratio*100:.1f}% similarity'}")
            results.append({
                "whistle_text": w_text, "whistle_ms": w_ms,
                "google_text": g_text, "google_ms": g_ms,
                "exact": exact, "ratio": ratio
            })

    else:
        # Live microphone recording
        try:
            mic = sr.Microphone()
        except Exception as e:
            print(f"[!] ERROR: Could not open microphone: {e}")
            print("    Run with --mock to test synthetically.")
            return False

        with mic as source:
            print("\n[*] Calibrating for ambient noise (1 second)...")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print("[*] Calibration complete. Ready to record.")

            for i in range(1, num_commands + 1):
                print(f"\n" + "-" * 50)
                print(f"--- [Command {i}/{num_commands}] ---")
                print(">>> Speak now (e.g., 'hey vision find my phone', 'remember this', 'navigate')...")

                try:
                    audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
                except sr.WaitTimeoutError:
                    print("  [!] Listening timeout - no speech detected. Skipping to next.")
                    continue
                except Exception as e:
                    print(f"  [!] Audio capture error: {e}")
                    continue

                print("  Processing with both STT engines...")

                # 1. Whistle transcription
                t_w = time.perf_counter()
                try:
                    w_text = stt_engine.transcribe(audio, keywords=keywords)
                except Exception as e:
                    w_text = f"<Whistle Error: {e}>"
                w_ms = int((time.perf_counter() - t_w) * 1000)

                # 2. Google transcription
                t_g = time.perf_counter()
                try:
                    g_text = recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    g_text = "<Unrecognized Speech>"
                except sr.RequestError as e:
                    g_text = f"<Google Request Error: {e}>"
                except Exception as e:
                    g_text = f"<Google Error: {e}>"
                g_ms = int((time.perf_counter() - t_g) * 1000)

                # Comparison
                clean_w = w_text.lower().strip()
                clean_g = g_text.lower().strip()
                exact = (clean_w == clean_g)
                ratio = difflib.SequenceMatcher(None, clean_w, clean_g).ratio()

                match_desc = "EXACT MATCH (100%)" if exact else f"DIFF ({ratio*100:.1f}% similarity)"

                print(f"  Whistle: '{w_text}' ({w_ms} ms)")
                print(f"  Google:  '{g_text}' ({g_ms} ms)")
                print(f"  Match:   {match_desc}")

                results.append({
                    "whistle_text": w_text, "whistle_ms": w_ms,
                    "google_text": g_text, "google_ms": g_ms,
                    "exact": exact, "ratio": ratio
                })

    # Summary
    print("\n" + "=" * 70)
    print("                           SUMMARY")
    print("=" * 70)
    if not results:
        print("No commands were successfully recorded.")
        return True

    total = len(results)
    exacts = sum(1 for r in results if r["exact"])
    avg_w_ms = sum(r["whistle_ms"] for r in results) / total
    avg_g_ms = sum(r["google_ms"] for r in results) / total
    avg_ratio = sum(r["ratio"] for r in results) / total

    print(f"  Total Commands Tested:      {total}")
    print(f"  Exact Matches:              {exacts}/{total} ({exacts*100/total:.1f}%)")
    print(f"  Average Similarity:         {avg_ratio*100:.1f}%")
    print(f"  Avg Whistle Latency:        {avg_w_ms:.1f} ms")
    print(f"  Avg Google Latency:         {avg_g_ms:.1f} ms")
    if avg_w_ms < avg_g_ms:
        speedup = avg_g_ms / max(1.0, avg_w_ms)
        print(f"  Speedup:                    Whistle is {speedup:.1f}x faster than Google")
    print("=" * 70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Cactus Whistle vs Google STT")
    parser.add_argument("commands_pos", nargs="?", type=int, default=None, help="Number of commands to record (positional)")
    parser.add_argument("-n", "--num-commands", type=int, default=5, help="Number of commands to record (default: 5)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Microphone listen timeout (default: 10s)")
    parser.add_argument("--phrase-limit", type=float, default=8.0, help="Phrase time limit (default: 8s)")
    parser.add_argument("--mock", action="store_true", help="Run with synthetic test audio instead of live microphone")
    args = parser.parse_args()

    n = args.commands_pos if args.commands_pos is not None else args.num_commands
    run_comparison(num_commands=n, timeout=args.timeout, phrase_limit=args.phrase_limit, mock=args.mock)
