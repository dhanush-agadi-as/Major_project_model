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
try:
    from config import PHONE_GPS_URL, PHONE_GPS_MAX_AGE_S
except ImportError:
    PHONE_GPS_URL = "https://127.0.0.1:5001/api/location"
    PHONE_GPS_MAX_AGE_S = 30

from ip_geolocation import IPGeolocationTracker

logger = logging.getLogger(__name__)

# How long a phone-reported position stays "fresh" before we stop trusting
# it and fall back to the next layer (IP geolocation / mock).
PHONE_GPS_MAX_AGE_SECONDS = float(getattr(config, 'PHONE_GPS_MAX_AGE_S', 30.0)) if 'config' in locals() else 30.0


class GPSTracker:
    """
    GPS position tracker with mock mode for testing
    """
    
    def __init__(
        self,
        mock_mode: bool = GPS_MOCK_MODE,
        use_ip_geolocation: bool = GPS_USE_IP_GEOLOCATION,
        manual_override: bool = GPS_MANUAL_OVERRIDE_ENABLED,
        start_phone_polling: bool = True,
    ):
        import config
        mock_effective = mock_mode or getattr(config, 'GPS_MOCK_MODE', False)
        indoor_effective = getattr(config, 'INDOOR_MODE', False)
        self.mock_mode = mock_mode
        self.manual_override = manual_override
        if mock_effective or indoor_effective:
            self.use_ip_geolocation = False
            self.ip_tracker = None
        else:
            self.use_ip_geolocation = use_ip_geolocation
            self.ip_tracker = (
                IPGeolocationTracker(cache_seconds=IP_GEOLOCATION_CACHE_SECONDS)
                if use_ip_geolocation else None
            )
        self.current_position = None
        self.last_update = None
        self.speed = 0.0  # m/s
        self.heading = 0.0  # degrees
        self.is_running = False
        self.geolocation_thread = None
        self.use_serial = False
        self.last_source = None

        # Layer 2: phone/browser geolocation
        self._phone_position: Optional[Tuple[float, float]] = None
        self._phone_position_time: float = 0.0
        self._phone_position_accuracy: Optional[float] = None
        self._phone_position_lock = threading.Lock()

        # Phone GPS background polling thread
        self._phone_poll_thread = None
        self._phone_poll_running = False
        self._phone_poll_stop_event = threading.Event()
        if start_phone_polling:
            self.start_phone_polling()

        # Layer 1: manual override -- no network, instant
        if manual_override:
            logger.info(
                f"GPS Tracker: manual override ACTIVE "
                f"({GPS_MANUAL_OVERRIDE_LAT}, {GPS_MANUAL_OVERRIDE_LON}). "
                "All network/serial/mock sources bypassed."
            )

        self.mock_path_index = 0
        self.mock_path = self._generate_mock_path()

        # Layers 4 & 5: mock path or real serial GPS (existing logic, unchanged)
        import sys
        skip_serial = False
        import config
        gps_src = getattr(config, 'GPS_SOURCE', 'phone').lower() if config else 'phone'
        # Don't try /dev/ttyUSB0 on laptop when Pi or phone is the source (or on Windows)
        if gps_src in ('phone', 'pi') or getattr(config, 'USE_PI_CAMERA', False) or getattr(config, 'PHONE_GPS_URL', None) or sys.platform.startswith('win'):
            skip_serial = True

        if not self.use_ip_geolocation and not manual_override:
            if mock_mode:
                logger.info("GPS Tracker initialized in MOCK mode")
                self.mock_path_index = 0
            elif skip_serial:
                logger.info("GPS Tracker: phone/Pi is position source; skipping serial port %s", GPS_PORT)
                self.use_serial = False
            else:
                logger.info(f"GPS Tracker initialized on {GPS_PORT}")
                try:
                    import serial
                    self.serial_port = serial.Serial(GPS_PORT, GPS_BAUDRATE, timeout=1)
                    self.use_serial = True   # only set here — serial port is actually open
                    logger.info("GPS serial connection established")
                except Exception as e:
                    logger.error(f"Failed to open GPS serial port: {e}")
                    if getattr(config, 'GPS_MOCK_MODE', False):
                        logger.warning("Falling back to MOCK mode")
                        self.mock_mode = True
                    else:
                        logger.warning("Serial GPS unavailable and GPS_MOCK_MODE is False; mock disabled")
                        self.mock_mode = False
                    self.use_serial = False
        elif mock_mode and not manual_override:
            self.mock_path_index = 0

    def start_phone_polling(self):
        """Start background polling thread for mobile_backend phone GPS."""
        if self._phone_poll_thread is not None and self._phone_poll_thread.is_alive():
            return
        self._phone_poll_running = True
        self._phone_poll_stop_event.clear()
        self._phone_poll_thread = threading.Thread(
            target=self._poll_phone_gps_loop,
            name="PhoneGPSPoller",
            daemon=True
        )
        self._phone_poll_thread.start()

    def stop_phone_polling(self):
        """Stop background polling thread."""
        self._phone_poll_running = False
        self._phone_poll_stop_event.set()
        if self._phone_poll_thread and self._phone_poll_thread.is_alive():
            self._phone_poll_thread.join(timeout=0.2)
        self._phone_poll_thread = None

    def _poll_phone_gps_loop(self):
        """Poll PHONE_GPS_URL ~once per second in background without blocking."""
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass

        while self._phone_poll_running and not self._phone_poll_stop_event.is_set():
            self._poll_phone_gps_once()
            self._phone_poll_stop_event.wait(1.0)

    def _poll_phone_gps_once(self):
        """Perform a single non-blocking fetch from mobile_backend."""
        import requests
        import config
        url = getattr(config, 'PHONE_GPS_URL', "https://127.0.0.1:5001/api/location")
        try:
            resp = requests.get(url, timeout=1.0, verify=False)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and data.get("source") == "phone":
                    lat = data.get("latitude")
                    lon = data.get("longitude")
                    if lat is not None and lon is not None:
                        acc = data.get("accuracy")
                        ts = data.get("timestamp")
                        self.set_phone_position(
                            float(lat), float(lon),
                            accuracy=float(acc) if acc is not None else None,
                            timestamp=float(ts) if ts is not None else None
                        )
        except Exception as e:
            logger.debug(f"Phone GPS polling: {e}")

    def _get_ip_tracker(self):
        """Get or lazily instantiate IPGeolocationTracker if enabled."""
        if self.ip_tracker is not None:
            return self.ip_tracker
        import config
        if getattr(config, 'GPS_USE_IP_GEOLOCATION', True):
            try:
                self.ip_tracker = IPGeolocationTracker(cache_seconds=IP_GEOLOCATION_CACHE_SECONDS)
                return self.ip_tracker
            except Exception:
                return None
        return None
    
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
        self.start_phone_polling()
        if self.use_serial:
            self.tracking_thread = threading.Thread(target=self._read_gps_loop, daemon=True)
            self.tracking_thread.start()

        self.geolocation_thread = None
        import config
        mock_effective = self.mock_mode or getattr(config, 'GPS_MOCK_MODE', False)
        indoor_effective = getattr(config, 'INDOOR_MODE', False)
        if self.ip_tracker and not mock_effective and not indoor_effective:
            if hasattr(self.ip_tracker, 'start'):
                self.ip_tracker.start()
                self.geolocation_thread = getattr(self.ip_tracker, 'geolocation_thread', None)
            else:
                self.geolocation_thread = threading.Thread(target=self.ip_tracker.get_position, daemon=True)
                self.geolocation_thread.start()

        logger.info("GPS tracking started")

    def stop(self):
        """Stop GPS tracking"""
        self.is_running = False
        self.stop_phone_polling()
        if self.ip_tracker and hasattr(self.ip_tracker, 'stop'):
            self.ip_tracker.stop()
        self.geolocation_thread = None
        if self.use_serial and hasattr(self, 'serial_port'):
            self.serial_port.close()
        logger.info("GPS tracking stopped")
    
    def set_phone_position(
        self,
        lat: float,
        lon: float,
        accuracy: Optional[float] = None,
        timestamp: Optional[float] = None
    ):
        """
        Record a position reported by a phone/browser via navigator.geolocation.
        """
        with self._phone_position_lock:
            self._phone_position = (lat, lon)
            self._phone_position_time = float(timestamp) if timestamp is not None else time.time()
            self._phone_position_accuracy = float(accuracy) if accuracy is not None else None
        acc_str = f", accuracy={accuracy:.0f}m" if accuracy is not None else ""
        logger.debug(f"Phone GPS coordinates: ({lat:.6f}, {lon:.6f}){acc_str}")
        try:
            from ip_geolocation import get_place_name
            place_str = get_place_name(lat, lon, city_fallback=self.get_city())
        except Exception:
            place_str = self.get_city() or "your current location"
        logger.info(f"Phone GPS report received: near {place_str}{acc_str}")

    def _get_fresh_phone_position_info(self) -> Optional[Tuple[Tuple[float, float], float, Optional[float]]]:
        """Returns ((lat, lon), age, accuracy) if phone position is fresh, else None."""
        with self._phone_position_lock:
            if self._phone_position is None:
                return None
            import config
            max_age = float(getattr(config, 'PHONE_GPS_MAX_AGE_S', PHONE_GPS_MAX_AGE_SECONDS))
            age = time.time() - self._phone_position_time
            if age > max_age:
                return None
            return self._phone_position, age, self._phone_position_accuracy

    def _get_fresh_phone_position(self) -> Optional[Tuple[float, float]]:
        info = self._get_fresh_phone_position_info()
        return info[0] if info is not None else None

    def _log_source_once(self, source_name: str):
        # Extract base source name (e.g. 'phone', 'ip', 'mock', 'manual', 'serial')
        base_src = source_name.split()[0].lower()
        if getattr(self, "_last_logged_source", None) != base_src:
            logger.info(f"GPS position source used: {source_name}")
            self._last_logged_source = base_src

    def get_position(self) -> Optional[Tuple[float, float]]:
        """
        Get current GPS position using the highest-priority available source:
          1. Manual override
          2. Phone GPS (if fresh <= 30 s)
          3. IP geolocation
          4. Mock mode (ONLY if GPS_MOCK_MODE is True)
          5. Serial GPS
        Never silently uses mock when GPS_MOCK_MODE is False.
        Returns (latitude, longitude) or None if unavailable.
        """
        import config
        # Layer 1: manual override
        if self.manual_override:
            pos = (GPS_MANUAL_OVERRIDE_LAT, GPS_MANUAL_OVERRIDE_LON)
            self.current_position = pos
            self.last_source = "manual"
            self._log_source_once("manual override")
            return pos

        # Layer 2: phone GPS if fresher than 30 s
        phone_info = self._get_fresh_phone_position_info()
        if phone_info is not None:
            pos, age, acc = phone_info
            self.current_position = pos
            self.last_update = time.time()
            self.last_source = "phone"
            acc_str = f"{acc:.0f}m" if acc is not None else "unknown"
            self._log_source_once(f"phone (age={int(age)}s, accuracy={acc_str})")
            return pos

        # Layer 3: IP geolocation
        ip_tracker = self._get_ip_tracker()
        if ip_tracker is not None:
            pos = ip_tracker.get_position()
            if pos is not None:
                self.current_position = pos
                self.last_update = time.time()
                self.last_source = "ip"
                self._log_source_once("ip")
                return pos

        # Layers 4 & 5: mock path or real serial GPS
        mock_allowed = bool(getattr(config, 'GPS_MOCK_MODE', False))
        if mock_allowed:
            self.last_source = "mock"
            self._log_source_once("mock")
            return self._get_mock_position()

        if self.current_position is not None and self.use_serial:
            self.last_source = "serial"
            self._log_source_once("serial GPS")
            return self.current_position

        # When GPS_MOCK_MODE is False and neither phone nor IP is available, never use mock!
        self.last_source = None
        return None

    def get_where_am_i_speech(self) -> str:
        """
        Build the spoken 'where am I' response:
        Priority:
          (a) Phone GPS if fresher than 30 s:
              Speak street-level name from Nominatim reverse geocode.
              If accuracy is worse than 100 m, say 'You are approximately near ...'
              Else say 'You are near ...'
          (b) IP geolocation:
              Spoken note 'approximate location', say 'You are approximately near ...'
          (c) Mock GPS ONLY if GPS_MOCK_MODE is True:
              Say 'You are near ...'
          (d) Nothing available:
              Say 'I can't get your location. Open the location page on your phone.'
        """
        import config
        # (a) Check fresh phone GPS
        phone_info = self._get_fresh_phone_position_info()
        if phone_info is not None:
            pos, age, acc = phone_info
            acc_val = acc if acc is not None else 10.0
            acc_str = f"{acc_val:.0f}m"
            logger.info(f"GPS position source used: phone (age={int(age)}s, accuracy={acc_str})")
            place = None
            try:
                from ip_geolocation import reverse_geocode
                place = reverse_geocode(pos[0], pos[1])
            except Exception:
                pass
            if not place:
                try:
                    from ip_geolocation import get_place_name
                    place = get_place_name(pos[0], pos[1], city_fallback=self.get_city())
                except Exception:
                    place = self.get_city() or "your current location"

            if acc is not None and acc > 100.0:
                return f"You are approximately near {place}"
            return f"You are near {place}"

        # (b) IP geolocation
        ip_tracker = self._get_ip_tracker()
        ip_pos = ip_tracker.get_position() if ip_tracker else None
        if ip_pos is not None:
            logger.info("GPS position source used: ip")
            place = None
            try:
                from ip_geolocation import reverse_geocode
                place = reverse_geocode(ip_pos[0], ip_pos[1])
            except Exception:
                pass
            if not place:
                place = self.get_city() or "your current location"
            return f"You are approximately near {place}. Approximate location."

        # (c) Mock only if GPS_MOCK_MODE is True
        mock_allowed = bool(getattr(config, 'GPS_MOCK_MODE', False))
        if mock_allowed:
            logger.info("GPS position source used: mock")
            mock_pos = self._get_mock_position()
            place = self.get_current_place_name()
            return f"You are near {place}"

        # (d) Nothing is available and GPS_MOCK_MODE is False
        logger.warning("GPS position unavailable: no fresh phone GPS or IP geolocation")
        return "I can't get your location. Open the location page on your phone."
    
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

    def get_current_place_name(self) -> str:
        """
        Get current human-friendly place name (road + suburb/neighbourhood + city)
        using reverse geocoding, falling back to IP city, or 'your current location'.
        Never returns raw coordinates.
        """
        pos = self.get_position()
        city = self.get_city()
        if pos:
            try:
                from ip_geolocation import get_place_name
                return get_place_name(pos[0], pos[1], city_fallback=city)
            except Exception:
                pass
        if city:
            return city
        return "your current location"
    
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
