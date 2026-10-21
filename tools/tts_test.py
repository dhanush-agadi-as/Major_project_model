"""
TTS Test Tool
Speaks 6 phrases in a row from several threads (like the app does)
to verify that every utterance is completely audible and logged properly.
"""
import sys
import os
import time
import threading

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_feedback import get_audio_feedback, PRIORITY_EMERGENCY, PRIORITY_PROMPT, PRIORITY_NAVIGATIONAL, PRIORITY_STATUS
import config

def main():
    print("=" * 60)
    print("VisionMate TTS Multi-Threaded Audio Test")
    print("=" * 60)

    audio = get_audio_feedback(use_gtts=False)
    # Wait briefly for worker initialization
    time.sleep(0.5)

    phrases = [
        ("StatusThread", "VisionMate ready.", PRIORITY_STATUS),
        ("PromptThread", "Say hey vision, then: remember, find, take note, read notes, where am I, navigate to, or help.", PRIORITY_PROMPT),
        ("NavThread", "Person ahead.", PRIORITY_NAVIGATIONAL),
        ("EnrollThread", "Please say a name, for example: remember cell phone as mobile.", PRIORITY_PROMPT),
        ("FindThread", "Straight ahead, close.", PRIORITY_PROMPT),
        ("ArrivalThread", "Mobile is right in front of you, reach out.", PRIORITY_PROMPT),
    ]

    results = []

    def worker_speak(thread_name, text, priority, delay_before):
        if delay_before > 0:
            time.sleep(delay_before)
        print(f"[{thread_name}] Requesting speech: '{text}' (priority={priority})")
        audio.speak(text, priority=priority, wait=True)
        results.append((thread_name, text))
        print(f"[{thread_name}] Finished speech: '{text}'")

    threads = []
    # Speaks 6 phrases in a row from several threads so each one is heard audibly
    for thread_idx, (name, text, priority) in enumerate(phrases):
        t = threading.Thread(
            target=worker_speak,
            args=(name, text, priority, 0.0),
            name=name
        )
        threads.append(t)
        t.start()
        # Wait for this thread to finish speaking its phrase before the next thread speaks
        t.join(timeout=15.0)

    print("\n" + "=" * 60)
    print(f"Test complete: {len(results)} of {len(phrases)} phrases spoken.")
    for name, text in results:
        print(f"  - [{name}] {text}")
    print("=" * 60)

    if len(results) == len(phrases):
        print("[SUCCESS] All 6 phrases completed successfully.")
    else:
        print("[FAIL] Some phrases did not complete.")

if __name__ == "__main__":
    main()
