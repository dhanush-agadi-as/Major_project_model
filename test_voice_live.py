"""
Live Voice Command Test
Tests voice commands with actual microphone input
"""

import os
import sys

# Suppress OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_command_system import VoiceCommandSystem

def test_live_voice():
    """Test voice commands with real microphone"""
    print("\n" + "=" * 70)
    print("LIVE VOICE COMMAND TEST")
    print("=" * 70)
    
    # Initialize voice system
    voice_system = VoiceCommandSystem()
    
    print("\nInitializing microphone...")
    if not voice_system.initialize():
        print("[FAIL] Failed to initialize microphone")
        print("\nPossible issues:")
        print("  - No microphone detected")
        print("  - Microphone permissions denied")
        print("  - Audio drivers not working")
        return False
    
    voice_system.start()
    
    print("\n[OK] Voice system ready!")
    print("\nTest Plan:")
    print("  1. When prompted, say a command")
    print("  2. System will recognize and parse it")
    print("  3. You can test multiple commands")
    
    # Register test handlers
    def handle_remember(params):
        print(f"\n[OK] REMEMBER command recognized!")
        print(f"   Object: {params.get('object_name', 'unknown')}")
    
    def handle_find(params):
        print(f"\n[OK] FIND command recognized!")
        print(f"   Object: {params.get('object_name', 'unknown')}")
    
    def handle_navigate(params):
        print(f"\n[OK] NAVIGATE command recognized!")
    
    def handle_stop_navigate(params):
        print(f"\n[OK] STOP NAVIGATION command recognized!")
    
    def handle_list(params):
        print(f"\n[OK] LIST command recognized!")
    
    voice_system.register_command_handler('remember', handle_remember)
    voice_system.register_command_handler('find', handle_find)
    voice_system.register_command_handler('navigate', handle_navigate)
    voice_system.register_command_handler('stop_navigate', handle_stop_navigate)
    voice_system.register_command_handler('list', handle_list)
    
    print("\n" + "-" * 70)
    print("Ready to test! Try these commands:")
    print("  - 'Remember this phone'")
    print("  - 'Find my keys'")
    print("  - 'Navigate'")
    print("  - 'Stop navigation'")
    print("  - 'List all items'")
    print("-" * 70)
    
    # Test loop
    try:
        while True:
            input("\nPress ENTER to speak a command (q to quit): ")
            choice = input("Type 'q' to quit or press ENTER to continue: ").strip()
            if choice.lower() == 'q':
                break
            
            print("\n[MIC] Listening... (speak now)")
            voice_system.trigger_voice_input()
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    finally:
        voice_system.stop()
        print("\n[OK] Voice system stopped")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_live_voice()
    sys.exit(0 if success else 1)
