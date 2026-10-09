"""
Quick test for voice command integration without microphone
Tests command parsing and handler registration
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_command_system import VoiceCommandParser

def test_command_parser():
    """Test voice command parsing"""
    parser = VoiceCommandParser()
    
    test_cases = [
        ("remember this phone", "remember", {"object_name": "phone"}),
        ("find my keys", "find", {"object_name": "keys"}),
        ("navigate", "navigate", {}),
        ("stop navigation", "stop_navigate", {}),
        ("list all items", "list", {}),
        ("help", "help", {}),
        ("random text", "unknown", {}),
    ]
    
    print("Testing Voice Command Parser")
    print("=" * 60)
    
    passed = 0
    for text, expected_intent, expected_params in test_cases:
        result = parser.parse(text)
        intent = result['intent']
        params = result['parameters']
        
        # Check intent
        if intent == expected_intent:
            # Check parameters
            if expected_params:
                params_match = all(params.get(k) == v for k, v in expected_params.items())
            else:
                params_match = True
            
            if params_match:
                print(f"[OK] '{text}'")
                print(f"  -> Intent: {intent}, Params: {params}")
                passed += 1
            else:
                print(f"[FAIL] '{text}'")
                print(f"  -> Expected params: {expected_params}, Got: {params}")
        else:
            print(f"[FAIL] '{text}'")
            print(f"  -> Expected: {expected_intent}, Got: {intent}")
    
    print("\n" + "=" * 60)
    print(f"Parser Test: {passed}/{len(test_cases)} passed")
    print("=" * 60)
    
    return passed == len(test_cases)


if __name__ == "__main__":
    success = test_command_parser()
    sys.exit(0 if success else 1)
