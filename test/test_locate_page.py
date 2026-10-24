"""
Test for mobile_backend /locate page geolocation functionality.
Verifies that the phone location page satisfies all requirements:
1. Uses navigator.geolocation.watchPosition
2. Configures low accuracy with timeout 30000 and maximumAge 60000
3. Configures high accuracy watch in parallel with timeout 60000
4. Handles code 3 (TIMEOUT) with 'Waiting for location... try near a window' and auto-retry
5. Handles code 1 (PERMISSION_DENIED) and code 2 (POSITION_UNAVAILABLE)
6. Displays latitude, longitude, accuracy in metres, time, 'Sending...', 'Sent OK', and positions sent
7. Checks window.isSecureContext and displays "Open the https link and accept the certificate"
8. POSTs to /api/location
"""
import os
import sys
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mobile_backend as mb


@pytest.fixture
def client():
    return mb.app.test_client()


def test_locate_page_status_and_title(client):
    """Test that /locate route returns HTTP 200 and has the expected title."""
    response = client.get('/locate')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "<title>VisionMate Location Sharing</title>" in html
    assert "VisionMate Location Sharing" in html


def test_locate_page_contains_watch_position_and_timeouts(client):
    """Test that page uses watchPosition and specifies 30000 and 60000 timeouts."""
    response = client.get('/locate')
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    # 1. Use navigator.geolocation.watchPosition (not getCurrentPosition)
    assert "watchPosition" in html
    assert "getCurrentPosition" not in html

    # 2. Low-accuracy watch with timeout 30000 and maximumAge 60000
    assert "30000" in html
    assert "enableHighAccuracy: false" in html
    assert "timeout: 30000" in html
    assert "maximumAge: 60000" in html

    # High-accuracy watch with timeout 60000
    assert "60000" in html
    assert "enableHighAccuracy: true" in html
    assert "timeout: 60000" in html


def test_locate_page_error_handling_messages(client):
    """Test error handling messages for TIMEOUT, PERMISSION_DENIED, and POSITION_UNAVAILABLE."""
    response = client.get('/locate')
    html = response.get_data(as_text=True)

    # 3. TIMEOUT (code 3): show "Waiting for location... try near a window" without red error stop
    assert "Waiting for location... try near a window" in html
    assert "err.code === 3" in html or "err.TIMEOUT" in html

    # Code 1: permission denied
    assert "Location permission denied" in html
    assert "err.code === 1" in html or "err.PERMISSION_DENIED" in html

    # Code 2: position unavailable
    assert "Location unavailable, turn on phone location" in html
    assert "err.code === 2" in html or "err.POSITION_UNAVAILABLE" in html


def test_locate_page_details_and_sending_status(client):
    """Test that status, coordinates, accuracy in metres, time, and positions sent are on page."""
    response = client.get('/locate')
    html = response.get_data(as_text=True)

    # 4 & 5. Sending... / Sent OK status, coordinates, accuracy, time, positions sent
    assert "Sending..." in html
    assert "Sent OK" in html
    assert "Latitude" in html
    assert "Longitude" in html
    assert "Accuracy" in html
    assert "Positions Sent" in html
    assert "/api/location" in html


def test_locate_page_secure_context_check(client):
    """Test that window.isSecureContext is checked with exact required message."""
    response = client.get('/locate')
    html = response.get_data(as_text=True)

    # 6. Secure context check
    assert "window.isSecureContext" in html
    assert "Open the https link and accept the certificate" in html


def test_api_location_post_integration(client):
    """Test that POST /api/location continues to receive coords from the page."""
    payload = {"latitude": 12.9716, "longitude": 77.5946, "accuracy": 15.0}
    response = client.post('/api/location', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert abs(data["latitude"] - 12.9716) < 1e-6
    assert abs(data["longitude"] - 77.5946) < 1e-6
    assert data["accuracy"] == 15.0


if __name__ == '__main__':
    fails = 0
    c = mb.app.test_client()

    def run_check(name, fn):
        global fails
        try:
            fn(c)
            print(f"PASS {name}")
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            fails += 1
        except Exception as e:
            print(f"ERROR {name}: {e}")
            fails += 1

    run_check("test_locate_page_status_and_title", test_locate_page_status_and_title)
    run_check("test_locate_page_contains_watch_position_and_timeouts", test_locate_page_contains_watch_position_and_timeouts)
    run_check("test_locate_page_error_handling_messages", test_locate_page_error_handling_messages)
    run_check("test_locate_page_details_and_sending_status", test_locate_page_details_and_sending_status)
    run_check("test_locate_page_secure_context_check", test_locate_page_secure_context_check)
    run_check("test_api_location_post_integration", test_api_location_post_integration)

    sys.exit(1 if fails else 0)
