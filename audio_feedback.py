"""
Audio Feedback System
Priority-based Text-to-Speech for VisionMate
"""
import threading
import queue
from typing import Optional
from gtts import gTTS
import os
import tempfile
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except:
    PYTTSX3_AVAILABLE = False

from config import (
    PRIORITY_EMERGENCY, PRIORITY_SOCIAL,
    PRIORITY_NAVIGATIONAL, PRIORITY_STATUS
)


class AudioFeedback:
    def __init__(self, use_gtts: bool = False):
        """
        Initialize audio feedback system
        
        Args:
            use_gtts: If True, use gTTS (requires internet), else use pyttsx3 (offline)
        """
        self.use_gtts = use_gtts
        self.audio_queue = queue.PriorityQueue()
        self.is_playing = False
        self.current_priority = None
        
        # Initialize pyttsx3 if available and not using gTTS
        self.engine = None
        if not use_gtts and PYTTSX3_AVAILABLE:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speed
            self.engine.setProperty('volume', 1.0)  # Volume
        
        # Start audio playback thread
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        
        print(f"[OK] Audio Feedback initialized")
        print(f"  Engine: {'gTTS' if use_gtts else 'pyttsx3'}")
        print(f"  Priority levels: EMERGENCY(1) > SOCIAL(2) > NAVIGATIONAL(3) > STATUS(4)")
    
    def speak(self, text: str, priority: int = PRIORITY_NAVIGATIONAL, interrupt: bool = False):
        """
        Add text to speech queue
        
        Args:
            text: Text to speak
            priority: Priority level (1=highest, 4=lowest)
            interrupt: If True, interrupt current speech for high priority messages
        """
        # Emergency messages always interrupt
        if priority == PRIORITY_EMERGENCY:
            interrupt = True
        
        if interrupt and self.is_playing:
            # Clear queue and stop current playback
            self._clear_queue()
            self._stop_current()
        
        # Add to priority queue
        self.audio_queue.put((priority, text))
    
    def speak_emergency(self, text: str):
        """Speak emergency message (highest priority, interrupts everything)"""
        self.speak(text, PRIORITY_EMERGENCY, interrupt=True)
    
    def speak_social(self, text: str):
        """Speak social message (recognized person)"""
        self.speak(text, PRIORITY_SOCIAL)
    
    def speak_navigational(self, text: str):
        """Speak navigational message (obstacles)"""
        self.speak(text, PRIORITY_NAVIGATIONAL)
    
    def speak_status(self, text: str):
        """Speak status message (lowest priority)"""
        self.speak(text, PRIORITY_STATUS)
    
    def _playback_worker(self):
        """Background thread for playing audio"""
        while True:
            try:
                priority, text = self.audio_queue.get(timeout=1)
                self.current_priority = priority
                self.is_playing = True
                
                print(f"🔊 Speaking (Priority {priority}): {text}")
                self._play_text(text)
                
                self.is_playing = False
                self.current_priority = None
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[FAIL] Audio playback error: {e}")
                self.is_playing = False
    
    def _play_text(self, text: str):
        """Play text using selected TTS engine"""
        if self.use_gtts:
            self._play_gtts(text)
        elif self.engine:
            self._play_pyttsx3(text)
        else:
            # Fallback: just print
            print(f"[AUDIO]: {text}")
    
    def _play_gtts(self, text: str):
        """Play text using gTTS"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_file = fp.name
            
            # Generate speech
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_file)
            
            # Play audio (platform-specific)
            if os.name == 'nt':  # Windows
                os.system(f'start /min "" "{temp_file}"')
            else:  # Linux/Mac
                os.system(f'mpg123 -q "{temp_file}"')
            
            # Clean up
            os.remove(temp_file)
            
        except Exception as e:
            print(f"[FAIL] gTTS error: {e}")
            print(f"[AUDIO]: {text}")
    
    def _play_pyttsx3(self, text: str):
        """Play text using pyttsx3"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"[FAIL] pyttsx3 error: {e}")
            print(f"[AUDIO]: {text}")
    
    def _clear_queue(self):
        """Clear all pending messages"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def _stop_current(self):
        """Stop currently playing audio"""
        if self.engine and PYTTSX3_AVAILABLE:
            try:
                self.engine.stop()
            except:
                pass
    
    def get_queue_size(self) -> int:
        """Get number of pending messages"""
        return self.audio_queue.qsize()
    
    def is_busy(self) -> bool:
        """Check if currently speaking"""
        return self.is_playing


# Singleton instance
_audio_instance = None

def get_audio_feedback(use_gtts: bool = False) -> AudioFeedback:
    """Get singleton AudioFeedback instance"""
    global _audio_instance
    if _audio_instance is None:
        _audio_instance = AudioFeedback(use_gtts=use_gtts)
    return _audio_instance


if __name__ == "__main__":
    # Quick test
    audio = AudioFeedback(use_gtts=False)
    
    audio.speak_status("System initialized")
    audio.speak_navigational("Chair ahead")
    audio.speak_social("Hello John")
    audio.speak_emergency("STOP - Red Light")
    
    import time
    time.sleep(5)
    
    print("\n[OK] Audio Feedback test complete")
