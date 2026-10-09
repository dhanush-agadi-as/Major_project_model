"""
Voice Command System for VisionMate
Handles button-triggered voice input, speech-to-text, and command parsing
"""

import speech_recognition as sr
import threading
import queue
import time
import logging
from typing import Optional, Callable, Dict
import re

logger = logging.getLogger(__name__)


class ButtonHandler:
    """
    Monitors button press events
    On Windows: Uses keyboard spacebar for testing
    On Raspberry Pi: Can be adapted to use GPIO
    """
    def __init__(self, callback: Callable):
        self.callback = callback
        self.running = False
        self.thread = None
        
    def start(self):
        """Start monitoring for button presses"""
        self.running = True
        # For Windows testing, we'll integrate with keyboard in the main loop
        logger.info("ButtonHandler started (use SPACEBAR for voice input)")
        
    def stop(self):
        """Stop monitoring"""
        self.running = False
        logger.info("ButtonHandler stopped")
        
    def trigger(self):
        """Manually trigger the button press callback"""
        if self.running:
            self.callback()


class VoiceCapture:
    """
    Captures audio and converts to text using speech recognition
    """
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.is_listening = False
        
        # Adjust recognition settings for better accuracy
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Seconds of silence to consider end of phrase
        
    def initialize_microphone(self) -> bool:
        """Initialize the microphone device"""
        try:
            self.microphone = sr.Microphone()
            # Adjust for ambient noise
            with self.microphone as source:
                logger.info("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Microphone initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize microphone: {e}")
            return False
            
    def listen_and_recognize(self, timeout: int = 5) -> Optional[str]:
        """
        Listen for voice input and convert to text
        
        Args:
            timeout: Maximum seconds to wait for speech
            
        Returns:
            Recognized text or None if failed
        """
        if not self.microphone:
            logger.warning("Microphone not initialized")
            return None
            
        try:
            self.is_listening = True
            logger.info("Listening for voice input...")
            
            with self.microphone as source:
                # Listen with timeout
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                
            logger.info("Processing speech...")
            
            # Try Google Speech Recognition (online)
            try:
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Recognized: '{text}'")
                return text.lower()
            except sr.UnknownValueError:
                logger.warning("Could not understand audio")
                return None
            except sr.RequestError as e:
                logger.error(f"Speech recognition service error: {e}")
                return None
                
        except sr.WaitTimeoutError:
            logger.warning("Listening timeout - no speech detected")
            return None
        except Exception as e:
            logger.error(f"Error during voice capture: {e}")
            return None
        finally:
            self.is_listening = False


class VoiceCommandParser:
    """
    Parses voice commands and extracts intent and parameters
    """
    
    # Command patterns
    PATTERNS = {
        'remember': [
            r'remember\s+(?:this\s+)?(\w+)',
            r'save\s+(?:this\s+)?(\w+)',
            r'enroll\s+(?:this\s+)?(\w+)'
        ],
        'stop_find': [
            r'stop\s+find(?:ing)?',
            r'stop\s+search(?:ing)?',
            r'cancel\s+search',
            r'i\s+found\s+it',
        ],
        'find': [
            r'find\s+(?:my\s+)?(\w+)',
            r'where\s+is\s+(?:my\s+)?(\w+)',
            r'locate\s+(?:my\s+)?(\w+)'
        ],
        'navigate_to': [
            r'navigate\s+to\s+([\w\s]+)',
            r'take\s+me\s+to\s+([\w\s]+)',
            r'go\s+to\s+([\w\s]+)',
            r'directions\s+to\s+([\w\s]+)'
        ],
        'navigate': [
            r'navigate',
            r'start\s+navigation',
            r'guide\s+me',
            r'help\s+me\s+walk'
        ],
        'stop_gps': [
            r'stop\s+gps',
            r'stop\s+gps\s+navigation',
            r'end\s+navigation'
        ],
        'stop_navigate': [
            r'stop\s+navigation',
            r'stop\s+navigating',
            r'turn\s+off\s+navigation'
        ],
        'list': [
            r'list\s+(?:all\s+)?(?:objects|items|things)',
            r'what\s+do\s+you\s+remember',
            r'show\s+me\s+everything'
        ],
        'help': [
            r'help',
            r'what\s+can\s+you\s+do',
            r'commands'
        ]
    }
    
    def parse(self, text: str) -> Dict[str, any]:
        """
        Parse voice command text and extract intent
        
        Args:
            text: Recognized speech text
            
        Returns:
            Dictionary with 'intent' and 'parameters'
        """
        if not text:
            return {'intent': 'unknown', 'parameters': {}}
            
        text = text.lower().strip()
        
        # Try to match each command pattern
        for intent, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    params = {}
                    
                    # Extract parameters based on intent
                    if intent in ['remember', 'find']:
                        if match.groups():
                            params['object_name'] = match.group(1)
                    elif intent == 'navigate_to':
                        if match.groups():
                            params['destination'] = match.group(1).strip()
                    
                    logger.info(f"Parsed command - Intent: {intent}, Params: {params}")
                    return {'intent': intent, 'parameters': params}
        
        # No match found
        logger.warning(f"Could not parse command: '{text}'")
        return {'intent': 'unknown', 'parameters': {'raw_text': text}}


class VoiceCommandSystem:
    """
    Main voice command system that ties everything together
    """
    def __init__(self):
        self.voice_capture = VoiceCapture()
        self.parser = VoiceCommandParser()
        self.button_handler = None
        self.command_queue = queue.Queue()
        self.enabled = False
        self.command_callbacks = {}
        
    def initialize(self) -> bool:
        """Initialize the voice command system"""
        logger.info("Initializing Voice Command System...")
        
        # Initialize microphone
        if not self.voice_capture.initialize_microphone():
            logger.error("Failed to initialize microphone")
            return False
            
        # Setup button handler
        self.button_handler = ButtonHandler(self._on_button_press)
        
        logger.info("Voice Command System initialized successfully")
        return True
        
    def start(self):
        """Start the voice command system"""
        self.enabled = True
        if self.button_handler:
            self.button_handler.start()
        logger.info("Voice Command System started")
        
    def stop(self):
        """Stop the voice command system"""
        self.enabled = False
        if self.button_handler:
            self.button_handler.stop()
        logger.info("Voice Command System stopped")
        
    def register_command_handler(self, intent: str, callback: Callable):
        """
        Register a callback function for a specific command intent
        
        Args:
            intent: Command intent (e.g., 'remember', 'find')
            callback: Function to call when intent is recognized
        """
        self.command_callbacks[intent] = callback
        logger.info(f"Registered handler for intent: {intent}")
        
    def _on_button_press(self):
        """Called when button is pressed"""
        if not self.enabled:
            return
            
        logger.info("Button pressed - starting voice capture")
        
        # Capture and recognize speech
        text = self.voice_capture.listen_and_recognize(timeout=5)
        
        if text:
            # Parse the command
            command = self.parser.parse(text)
            
            # Execute the command
            self._execute_command(command)
        else:
            logger.warning("No valid speech recognized")
            
    def trigger_voice_input(self):
        """Manually trigger voice input (for testing with keyboard)"""
        if self.button_handler:
            self.button_handler.trigger()
            
    def _execute_command(self, command: Dict):
        """Execute a parsed command by calling registered handlers"""
        intent = command.get('intent')
        parameters = command.get('parameters', {})
        
        if intent == 'unknown':
            logger.warning(f"Unknown command: {parameters.get('raw_text', '')}")
            return
            
        # Call registered handler
        if intent in self.command_callbacks:
            try:
                self.command_callbacks[intent](parameters)
            except Exception as e:
                logger.error(f"Error executing command '{intent}': {e}")
        else:
            logger.warning(f"No handler registered for intent: {intent}")
            
    def get_help_text(self) -> str:
        """Get help text describing available commands"""
        return """
Voice Commands:
- "Remember this [object]" - Enroll a new object
- "Find my [object]" - Locate an enrolled object
- "Navigate" - Start obstacle navigation
- "Stop navigation" - Turn off navigation
- "List all items" - Show remembered objects
- "Help" - Show this message
"""


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create voice command system
    vcs = VoiceCommandSystem()
    
    # Register test handlers
    def handle_remember(params):
        print(f"[OK] REMEMBER command: {params}")
        
    def handle_find(params):
        print(f"[OK] FIND command: {params}")
        
    def handle_navigate(params):
        print(f"[OK] NAVIGATE command activated")
        
    vcs.register_command_handler('remember', handle_remember)
    vcs.register_command_handler('find', handle_find)
    vcs.register_command_handler('navigate', handle_navigate)
    
    # Initialize and start
    if vcs.initialize():
        vcs.start()
        print("\nVoice Command System Ready!")
        print("Press SPACEBAR to trigger voice input, or 'q' to quit")
        print(vcs.get_help_text())
        
        # Simple test loop
        try:
            while True:
                cmd = input("\nPress ENTER to simulate button press (q to quit): ")
                if cmd.lower() == 'q':
                    break
                vcs.trigger_voice_input()
        except KeyboardInterrupt:
            pass
        finally:
            vcs.stop()
            print("\nVoice Command System stopped")
    else:
        print("Failed to initialize voice command system")
