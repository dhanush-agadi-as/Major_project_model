"""
Ultrasonic Sensor Interface
Reads distance data from ESP32 ultrasonic sensor
Supports mock mode for testing without hardware
"""

import logging
import time
import random
import threading
from typing import Optional
from config import (ULTRASONIC_MOCK_MODE, ULTRASONIC_PORT, ULTRASONIC_BAUDRATE,
                   ULTRASONIC_MAX_DISTANCE, ULTRASONIC_OBSTACLE_THRESHOLD)

logger = logging.getLogger(__name__)


class UltrasonicSensor:
    """
    Interface to ESP32 ultrasonic sensor
    Reads distance measurements via serial connection
    """
    
    def __init__(self, mock_mode: Optional[bool] = None):
        try:
            import config
            use_pi = getattr(config, 'USE_PI_CAMERA', False)
        except Exception:
            use_pi = False

        if mock_mode is None:
            self.mock_mode = False if use_pi else ULTRASONIC_MOCK_MODE
        else:
            self.mock_mode = mock_mode

        self.current_distance = None
        self.last_update = None
        self.is_running = False
        self.serial_port = None
        self._warned_no_distance = False
        
        if self.mock_mode:
            logger.info("Ultrasonic Sensor initialized in MOCK mode")
        elif use_pi:
            logger.info("Ultrasonic Sensor: Pi is source (USE_PI_CAMERA=True). Local serial port skipped.")
        else:
            logger.info(f"Ultrasonic Sensor initializing on {ULTRASONIC_PORT}")
            try:
                import serial
                self.serial_port = serial.Serial(ULTRASONIC_PORT, ULTRASONIC_BAUDRATE, timeout=1)
                logger.info("ESP32 serial connection established")
                time.sleep(2)  # Wait for ESP32 to initialize
            except Exception as e:
                logger.error(f"Failed to open ultrasonic serial port: {e}")
                logger.warning("Falling back to MOCK mode")
                self.mock_mode = True
    
    def start(self):
        """Start reading sensor data"""
        self.is_running = True
        if not self.mock_mode and self.serial_port is not None:
            self.reading_thread = threading.Thread(target=self._read_sensor_loop, daemon=True)
            self.reading_thread.start()
        logger.info("Ultrasonic sensor reading started")
    
    def stop(self):
        """Stop reading sensor data"""
        self.is_running = False
        if self.serial_port:
            self.serial_port.close()
        logger.info("Ultrasonic sensor stopped")
    
    def get_distance(self) -> Optional[float]:
        """
        Get current distance measurement
        
        Returns:
            Distance in centimeters, or None if unavailable
        """
        if self.mock_mode:
            return self._get_mock_distance()
        if self.current_distance is None:
            try:
                import config
                use_pi = getattr(config, 'USE_PI_CAMERA', False)
            except Exception:
                use_pi = False
            if use_pi and not getattr(self, '_warned_no_distance', False):
                logger.warning("No ultrasonic distance received from Pi")
                self._warned_no_distance = True
            return None
        return self.current_distance

    
    def _get_mock_distance(self) -> float:
        """
        Generate simulated distance readings
        Mock mode must never generate obstacle warnings
        """
        # Vary distance safely above threshold (no fake obstacle warnings)
        distance = random.uniform(ULTRASONIC_OBSTACLE_THRESHOLD + 20, 300)
        
        # Add some noise
        distance += random.uniform(-5, 5)
        
        # Clamp to valid range above obstacle threshold
        distance = max(ULTRASONIC_OBSTACLE_THRESHOLD + 5, min(distance, ULTRASONIC_MAX_DISTANCE))
        
        self.current_distance = distance
        self.last_update = time.time()
        return distance
    
    def _read_sensor_loop(self):
        """Read sensor data from ESP32 via serial"""
        while self.is_running:
            try:
                if self.serial_port and self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Expected format from ESP32: "DIST:123" (distance in cm)
                    if line.startswith("DIST:"):
                        try:
                            distance_str = line.split(":")[1]
                            distance = float(distance_str)
                            
                            # Validate reading
                            if 2 <= distance <= ULTRASONIC_MAX_DISTANCE:
                                self.current_distance = distance
                                self.last_update = time.time()
                            else:
                                logger.warning(f"Invalid distance reading: {distance} cm")
                                
                        except (ValueError, IndexError) as e:
                            logger.error(f"Error parsing distance: {e}")
                    
                time.sleep(0.05)  # 20 Hz reading rate
                
            except Exception as e:
                logger.error(f"Error reading ultrasonic sensor: {e}")
                time.sleep(1)
    
    def is_obstacle_close(self) -> bool:
        """Check if obstacle is closer than threshold"""
        if self.mock_mode:
            return False
        distance = self.get_distance()
        if distance is None:
            return False
        return distance < ULTRASONIC_OBSTACLE_THRESHOLD
    
    def get_obstacle_status(self) -> dict:
        """
        Get detailed obstacle status
        
        Returns:
            Dictionary with distance and warning level
        """
        if self.mock_mode:
            return {
                'available': False,
                'distance': None,
                'warning_level': 'clear',
                'below_threshold': False
            }
        distance = self.get_distance()
        
        if distance is None:
            return {
                'available': False,
                'distance': None,
                'warning_level': 'unknown'
            }
        
        # Determine warning level
        if distance < 30:
            level = 'emergency'  # Very close - immediate danger
        elif distance < 60:
            level = 'warning'  # Close - caution needed
        elif distance < ULTRASONIC_OBSTACLE_THRESHOLD:
            level = 'notice'  # Approaching - be aware
        else:
            level = 'clear'  # Safe distance
        
        return {
            'available': True,
            'distance': distance,
            'distance_meters': distance / 100.0,
            'warning_level': level,
            'below_threshold': distance < ULTRASONIC_OBSTACLE_THRESHOLD
        }
    
    def is_available(self) -> bool:
        """Check if sensor is working"""
        if self.mock_mode:
            return True
        
        if self.current_distance is None:
            return False
        
        # Check if data is recent (within last 2 seconds)
        if self.last_update:
            age = time.time() - self.last_update
            return age < 2.0
        
        return False


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Testing Ultrasonic Sensor ===")
    sensor = UltrasonicSensor(mock_mode=True)
    sensor.start()
    
    print("\nReading distance measurements:")
    for i in range(15):
        status = sensor.get_obstacle_status()
        
        if status['available']:
            distance = status['distance']
            level = status['warning_level']
            print(f"  Reading {i+1}: {distance:.1f} cm - {level.upper()}")
            
            if level == 'emergency':
                print(f"    [WARN] EMERGENCY: Obstacle < 30 cm!")
            elif level == 'warning':
                print(f"    [WARN] WARNING: Obstacle < 60 cm")
        else:
            print(f"  Reading {i+1}: Sensor unavailable")
        
        time.sleep(0.5)
    
    sensor.stop()
    print("\n[OK] Ultrasonic sensor test complete!")
