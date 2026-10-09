"""
GPS Position Tracking
Supports both real GPS modules and mock GPS for testing.

Position-source priority order (highest to lowest):
  1. Manual override  -- fixed lat/lon from config; instant, no network.
                         Use for indoor/bench testing at a known location.
  2. Phone geolocation -- lat/lon reported by a phone/browser via
                         navigator.geolocation, POSTed to /api/location.
                         Accuracy ~5-50 m. Free, no hardware needed.
                         Only used while the last report is fresh
                         (see PHONE_GPS_MAX_AGE_SECONDS below).
  3. IP geolocation   -- ip-api.com city-level lookup. Accuracy ~1-10 km.
                         No hardware required; works anywhere with internet.
  4. Mock mode        -- simulated path in RR Nagar, Bangalore. Keyboard-
                         driven (M key). Zero dependencies. Safe fallback.
  5. Real serial GPS  -- NMEA data from a physical GPS module on GPS_PORT.
                         Falls back to mock if the serial port won't open.

Each layer is independently enable/disable via config.py flags.
Setting GPS_USE_WIFI_GEOLOCATION=False and GPS_USE_IP_GEOLOCATION=False
restores byte-identical behaviour to the original mock/serial-only code.
"""

import logging
import time
import threading
from typing import Optional, Tuple
from config import (
    GPS_MOCK_MODE, GPS_PORT, GPS_BAUDRATE,
    GPS_USE_IP_GEOLOCATION, IP_GEOLOCATION_CACHE_SECONDS,
    GPS_USE_WIFI_GEOLOCATION,
    GPS_MANUAL_OVERRIDE_ENABLED, GPS_MANUAL_OVERRIDE_LAT, GPS_MANUAL_OVERRIDE_LON,
)
from ip_geolocation import IPGeolocationTracker

logger = logging.getLogger(__name__)

# How long a phone-reported position stays "fresh" before we stop trusting
# it and fall back to the next layer (IP geolocation / mock).
PHONE_GPS_MAX_AGE_SECONDS = 30.0


class GPSTracker:
    """
    GPS position tracker with mock mode for testing
    """
    
    def __init__(
        self,
        mock_mode: bool = GPS_MOCK_MODE,
        use_ip_geolocation: bool = GPS_USE_IP_GEOLOCATION,
        manual_override: bool = GPS_MANUAL_OVERRIDE_ENABLED,
    ):
        self.mock_mode = mock_mode
        self.use_ip_geolocation = use_ip_geolocation
        self.manual_override = manual_override
        # BUGFIX: this was previously always left as None, so IP geolocation
        # (Layer 3) silently never ran even when GPS_USE_IP_GEOLOCATION=True.
        self.ip_tracker = (
            IPGeolocationTracker(cache_seconds=IP_GEOLOCATION_CACHE_SECONDS)
            if use_ip_geolocation else None
        )
        self.current_position = None
        self.last_update = None
        self.speed = 0.0  # m/s
        self.heading = 0.0  # degrees
        self.is_running = False
        # True only when a real serial GPS port was successfully opened.
        # Kept separate from mock_mode so that Wi-Fi/IP layers enabling
        # mock_mode=False never accidentally triggers the serial read thread.
        self.use_serial = False

        # Layer 2: phone/browser geolocation -- populated later via
        # set_phone_position() whenever the mobile backend receives a
        # POST /api/location report from a phone. Starts empty.
        self._phone_position: Optional[Tuple[float, float]] = None
        self._phone_position_time: float = 0.0
        self._phone_position_lock = threading.Lock()

        # Layer 1: manual override -- no network, instant
        if manual_override:
            logger.info(
                f"GPS Tracker: manual override ACTIVE "
                f"({GPS_MANUAL_OVERRIDE_LAT}, {GPS_MANUAL_OVERRIDE_LON}). "
                "All network/serial/mock sources bypassed."
            )

        # Layers 4 & 5: mock path or real serial GPS (existing logic, unchanged)
        if not use_ip_geolocation and not manual_override:
            if mock_mode:
                logger.info("GPS Tracker initialized in MOCK mode")
                self.mock_path_index = 0
                self.mock_path = self._generate_mock_path()
            else:
                logger.info(f"GPS Tracker initialized on {GPS_PORT}")
                try:
                    import serial
                    self.serial_port = serial.Serial(GPS_PORT, GPS_BAUDRATE, timeout=1)
                    self.use_serial = True   # only set here — serial port is actually open
                    logger.info("GPS serial connection established")
                except Exception as e:
                    logger.error(f"Failed to open GPS serial port: {e}")
                    logger.warning("Falling back to MOCK mode")
                    self.mock_mode = True
                    self.mock_path = self._generate_mock_path()
        elif mock_mode and not manual_override:
            # At least one network layer is active, but keep mock data ready
            # so that advance_mock_position() still works during testing
            self.mock_path_index = 0
            self.mock_path = self._generate_mock_path()
    
    def _generate_mock_path(self) -> list:
        """Generate a mock GPS path for testing (RR Nagar, Bangalore)"""
        return [
            (12.926516, 77.526422),  # Start: RR Nagar
            (12.926600, 77.526500),
            (12.926700, 77.526600),
            (12.926800, 77.526700),
            (12.926900, 77.526800),
            (12.927000, 77.526900),
            (12.927100, 77.527000),
            (12.927200, 77.527100),
            (12.927300, 77.527200),
            (12.927400, 77.527300),
        ]
    
    def start(self):
        """Start GPS tracking"""
        self.is_running = True
        if self.use_serial:
            self.tracking_thread = threading.Thread(target=self._read_gps_loop, daemon=True)
            self.tracking_thread.start()
        logger.info("GPS tracking started")

    def stop(self):
        """Stop GPS tracking"""
        self.is_running = False
        if self.use_serial and hasattr(self, 'serial_port'):
            self.serial_port.close()
        logger.info("GPS tracking stopped")
    
    def set_phone_position(self, lat: float, lon: float, accuracy: Optional[float] = None):
        """
        Record a position reported by a phone/browser via navigator.geolocation
        (called from mobile_backend.py's POST /api/location handler).

        This is treated as Layer 2 in get_position()'s priority order, and
        only trusted while fresh (see PHONE_GPS_MAX_AGE_SECONDS) -- an old
        report from a phone that's since gone offline should not keep
        overriding IP/mock fallbacks forever.
        """
        with self._phone_position_lock:
            self._phone_position = (lat, lon)
            self._phone_position_time = time.time()
        acc_str = f", accuracy={accuracy:.0f}m" if accuracy is not None else ""
        logger.info(f"Phone GPS report received: ({lat:.6f}, {lon:.6f}){acc_str}")

    def _get_fresh_phone_position(self) -> Optional[Tuple[float, float]]:
        with self._phone_position_lock:
            if self._phone_position is None:
                return None
            age = time.time() - self._phone_position_time
            if age > PHONE_GPS_MAX_AGE_SECONDS:
                return None
            return self._phone_position

    def get_position(self) -> Optional[Tuple[float, float]]:
        """
        Get current GPS position using the highest-priority available source.

        Priority: manual override > phone geolocation > IP geolocation > mock > serial GPS.
        Returns (latitude, longitude) or None if unavailable.

        NOTE: The resolved position is always stored in self.current_position so
        callers that read the attribute directly (e.g. server.py) see a real value
        rather than None when only network-based layers are active.
        """
        # Layer 1: manual override -- instant, no network
        if self.manual_override:
            pos = (GPS_MANUAL_OVERRIDE_LAT, GPS_MANUAL_OVERRIDE_LON)
            self.current_position = pos
            return pos

        # Layer 2: phone/browser geolocation -- most accurate free source
        # when no dedicated GPS hardware is attached, as long as it's fresh
        phone_pos = self._get_fresh_phone_position()
        if phone_pos is not None:
            self.current_position = phone_pos
            self.last_update = time.time()
            return phone_pos

        # Layer 3: IP geolocation (ip-api.com)
        if self.ip_tracker is not None:
            pos = self.ip_tracker.get_position()
            if pos is not None:
                self.current_position = pos
                self.last_update = time.time()
                return pos

        # Layers 4 & 5: mock path or real serial GPS (original logic, unchanged)
        if self.mock_mode:
            return self._get_mock_position()
        else:
            return self.current_position
    
    def _get_mock_position(self) -> Tuple[float, float]:
        """Get simulated GPS position"""
        if self.mock_path_index >= len(self.mock_path):
            self.mock_path_index = len(self.mock_path) - 1
        
        position = self.mock_path[self.mock_path_index]
        self.current_position = position
        self.last_update = time.time()
        return position
    
    def get_city(self) -> Optional[str]:
        """
        Get current city string if available from IP geolocation.
        Otherwise returns None.
        """
        if self.ip_tracker is not None:
            return self.ip_tracker.get_city()
        return None
    
    def advance_mock_position(self):
        """Move to next position in mock path (for testing)"""
        if self.mock_mode:
            self.mock_path_index += 1
            logger.info(f"Mock GPS advanced to position {self.mock_path_index}")
    
    def set_mock_path(self, path: list):
        """Set custom mock GPS path"""
        self.mock_path = path
        self.mock_path_index = 0
        logger.info(f"Set custom mock path with {len(path)} points")
    
    def _read_gps_loop(self):
        """Read GPS data from serial port (for real GPS module)"""
        while self.is_running:
            try:
                line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                
                if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                    position = self._parse_nmea(line)
                    if position:
                        self.current_position = position
                        self.last_update = time.time()
                        
            except Exception as e:
                logger.error(f"Error reading GPS data: {e}")
                time.sleep(1)
    
    def _parse_nmea(self, nmea_sentence: str) -> Optional[Tuple[float, float]]:
        """
        Parse NMEA sentence to extract GPS coordinates
        
        Supports GPGGA and GPRMC formats
        """
        try:
            parts = nmea_sentence.split(',')
            
            if nmea_sentence.startswith('$GPGGA'):
                # $GPGGA format: ..., lat, N/S, lon, E/W, ...
                if len(parts) < 6:
                    return None
                
                lat_str = parts[2]
                lat_dir = parts[3]
                lon_str = parts[4]
                lon_dir = parts[5]
                
                if not lat_str or not lon_str:
                    return None
                
                # Convert DDMM.MMMM to DD.DDDDDD
                lat = self._convert_to_degrees(lat_str, lat_dir)
                lon = self._convert_to_degrees(lon_str, lon_dir)
                
                return (lat, lon)
                
            elif nmea_sentence.startswith('$GPRMC'):
                # $GPRMC format: ..., lat, N/S, lon, E/W, speed, ...
                if len(parts) < 8:
                    return None
                
                lat_str = parts[3]
                lat_dir = parts[4]
                lon_str = parts[5]
                lon_dir = parts[6]
                
                if not lat_str or not lon_str:
                    return None
                
                lat = self._convert_to_degrees(lat_str, lat_dir)
                lon = self._convert_to_degrees(lon_str, lon_dir)
                
                # Parse speed (knots)
                if len(parts) > 7 and parts[7]:
                    self.speed = float(parts[7]) * 0.514444  # knots to m/s
                
                return (lat, lon)
                
        except Exception as e:
            logger.error(f"Error parsing NMEA: {e}")
            return None
    
    def _convert_to_degrees(self, coord_str: str, direction: str) -> float:
        """Convert NMEA coordinate format to decimal degrees"""
        # NMEA format: DDMM.MMMM or DDDMM.MMMM
        if len(coord_str) < 4:
            return 0.0
        
        # Find decimal point
        dot_index = coord_str.find('.')
        if dot_index < 0:
            return 0.0
        
        # Degrees are before the last 2 digits before decimal
        if dot_index >= 4:  # Longitude (DDDMM.MMMM)
            degrees = float(coord_str[:dot_index-2])
            minutes = float(coord_str[dot_index-2:])
        else:  # Latitude (DDMM.MMMM)
            degrees = float(coord_str[:dot_index-2])
            minutes = float(coord_str[dot_index-2:])
        
        decimal_degrees = degrees + (minutes / 60.0)
        
        # Apply direction
        if direction in ['S', 'W']:
            decimal_degrees = -decimal_degrees
        
        return decimal_degrees
    
    def is_available(self) -> bool:
        """Check if GPS is available and working (mirrors get_position() priority order)"""
        if self.manual_override:
            return GPS_MANUAL_OVERRIDE_LAT is not None and GPS_MANUAL_OVERRIDE_LON is not None
        if self._get_fresh_phone_position() is not None:
            return True
        if self.ip_tracker is not None:
            return self.ip_tracker.get_position() is not None
        if self.mock_mode:
            return True
        return self.current_position is not None
    
    def get_speed(self) -> float:
        """Get current speed in m/s"""
        return self.speed
    
    def get_heading(self) -> float:
        """Get current heading in degrees"""
        return self.heading


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Testing GPS Tracker ===")
    gps = GPSTracker(mock_mode=True)
    gps.start()
    
    print("\nSimulating movement along path:")
    for i in range(10):
        pos = gps.get_position()
        if pos:
            print(f"  Position {i+1}: {pos[0]:.6f}, {pos[1]:.6f}")
            gps.advance_mock_position()
            time.sleep(0.5)
        else:
            print(f"  Position {i+1}: GPS not available")
    
    gps.stop()
    print("\n[OK] GPS Tracker test complete!")
