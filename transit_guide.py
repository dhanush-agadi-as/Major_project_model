"""
Transit Guide
Handles public transportation specific navigation
Bus stop detection, transfer instructions, and stop counting
"""

import logging
from typing import Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


class TransitGuide:
    """
    Specialized guide for public transportation navigation
    Handles bus/train stops, transfers, and transit-specific announcements
    """
    
    def __init__(self, gps_tracker, stop_arrival_distance: float = 50.0):
        """
        Args:
            gps_tracker: GPSTracker instance
            stop_arrival_distance: Distance (m) to detect stop arrival (default 50m)
        """
        self.gps = gps_tracker
        self.stop_arrival_distance = stop_arrival_distance
        
        self.active_transit = None
        self.current_stop_index = 0
        self.stops_passed = 0
        self.on_vehicle = False
        
        logger.info("Transit Guide initialized")
        logger.info(f"  Stop arrival distance: {stop_arrival_distance}m")
    
    def start_transit_step(self, transit_step: Dict) -> bool:
        """
        Start following a transit step
        
        Args:
            transit_step: Step dict from Google Maps with transit_details
            
        Returns:
            True if transit step loaded successfully
        """
        if 'transit_details' not in transit_step:
            logger.error("Not a transit step")
            return False
        
        transit = transit_step['transit_details']
        
        self.active_transit = {
            'line': transit['line'],
            'departure_stop': transit['departure_stop'],
            'arrival_stop': transit['arrival_stop'],
            'num_stops': transit['num_stops'],
            'headsign': transit.get('headsign', ''),
            'departure_time': transit.get('departure_time'),
            'arrival_time': transit.get('arrival_time')
        }
        
        self.current_stop_index = 0
        self.stops_passed = 0
        self.on_vehicle = False
        
        logger.info(f"Transit loaded: {transit['line']} ({transit['num_stops']} stops)")
        
        return True
    
    def get_boarding_instruction(self) -> str:
        """
        Get instruction for boarding the vehicle
        
        Returns formatted boarding message
        """
        if not self.active_transit:
            return "No active transit"
        
        line = self.active_transit['line']
        departure_stop = self.active_transit['departure_stop']
        headsign = self.active_transit['headsign']
        
        msg = f"Board {line} at {departure_stop}"
        if headsign:
            msg += f" towards {headsign}"
        
        return msg
    
    def check_at_stop(self) -> Optional[Dict]:
        """
        Check if user is at a stop location
        
        Returns:
            Dict with stop info if at stop, None otherwise
        """
        if not self.active_transit:
            return None
        
        current_pos = self.gps.get_position()
        if not current_pos:
            return None
        
        # Check departure stop
        if not self.on_vehicle:
            departure = self.active_transit['departure_stop']
            if 'location' in departure:
                distance = self._calculate_distance(
                    current_pos,
                    (departure['location']['lat'], departure['location']['lng'])
                )
                
                if distance < self.stop_arrival_distance:
                    return {
                        'type': 'departure',
                        'name': departure['name'],
                        'distance': distance,
                        'action': 'board'
                    }
        
        # Check arrival stop
        else:
            arrival = self.active_transit['arrival_stop']
            if 'location' in arrival:
                distance = self._calculate_distance(
                    current_pos,
                    (arrival['location']['lat'], arrival['location']['lng'])
                )
                
                if distance < self.stop_arrival_distance:
                    return {
                        'type': 'arrival',
                        'name': arrival['name'],
                        'distance': distance,
                        'action': 'exit'
                    }
        
        return None
    
    def mark_boarded(self):
        """Mark that user has boarded the vehicle"""
        self.on_vehicle = True
        self.stops_passed = 0
        logger.info("User boarded vehicle")
    
    def mark_stop_passed(self):
        """Increment stop counter"""
        if self.on_vehicle:
            self.stops_passed += 1
            logger.info(f"Stop passed: {self.stops_passed}/{self.active_transit['num_stops']}")
    
    def get_exit_warning(self) -> Optional[str]:
        """
        Get warning when approaching exit stop
        
        Returns:
            Warning message if approaching exit, None otherwise
        """
        if not self.active_transit or not self.on_vehicle:
            return None
        
        stops_remaining = self.active_transit['num_stops'] - self.stops_passed
        
        if stops_remaining == 3:
            return f"Get off in 3 stops at {self.active_transit['arrival_stop']['name']}"
        elif stops_remaining == 2:
            return f"Get off in 2 stops at {self.active_transit['arrival_stop']['name']}"
        elif stops_remaining == 1:
            return f"Next stop: {self.active_transit['arrival_stop']['name']}. Prepare to exit."
        
        return None
    
    def get_current_status(self) -> Dict:
        """
        Get current transit status
        
        Returns:
            Dict with transit progress and next action
        """
        if not self.active_transit:
            return {'status': 'inactive'}
        
        if not self.on_vehicle:
            return {
                'status': 'waiting',
                'message': self.get_boarding_instruction(),
                'departure_stop': self.active_transit['departure_stop']['name'],
                'line': self.active_transit['line']
            }
        else:
            stops_remaining = self.active_transit['num_stops'] - self.stops_passed
            
            status = {
                'status': 'on_vehicle',
                'line': self.active_transit['line'],
                'stops_passed': self.stops_passed,
                'stops_remaining': stops_remaining,
                'arrival_stop': self.active_transit['arrival_stop']['name']
            }
            
            # Add exit warning if applicable
            warning = self.get_exit_warning()
            if warning:
                status['warning'] = warning
            
            return status
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate distance between two GPS coordinates
        
        Returns distance in meters
        """
        R = 6371000  # Earth radius in meters
        
        lat1, lon1 = pos1
        lat2, lon2 = pos2
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return distance
    
    def is_complete(self) -> bool:
        """Check if transit step is complete"""
        if not self.active_transit or not self.on_vehicle:
            return False
        
        return self.stops_passed >= self.active_transit['num_stops']
    
    def reset(self):
        """Reset transit guide for next step"""
        self.active_transit = None
        self.current_stop_index = 0
        self.stops_passed = 0
        self.on_vehicle = False
        logger.info("Transit guide reset")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Testing Transit Guide ===")
    
    from gps_tracker import GPSTracker
    
    # Initialize GPS
    gps = GPSTracker(mock_mode=True)
    gps.start()
    
    guide = TransitGuide(gps)
    
    # Mock transit step (like from Google Maps API)
    mock_transit_step = {
        'html_instructions': 'Bus towards Majestic',
        'transit_details': {
            'line': 'Bus 201',
            'departure_stop': {
                'name': 'Indiranagar Bus Stop',
                'location': {'lat': 12.9716, 'lng': 77.6412}
            },
            'arrival_stop': {
                'name': 'Majestic Bus Station',
                'location': {'lat': 12.9767, 'lng': 77.5715}
            },
            'num_stops': 8,
            'headsign': 'Majestic',
            'departure_time': '10:30 AM',
            'arrival_time': '11:15 AM'
        }
    }
    
    print("\nStarting transit step...")
    guide.start_transit_step(mock_transit_step)
    
    print("\nBoarding instruction:")
    print(f"  {guide.get_boarding_instruction()}")
    
    print("\nSimulating waiting at stop...")
    stop_check = guide.check_at_stop()
    if stop_check:
        print(f"  At {stop_check['name']} - Action: {stop_check['action']}")
    
    print("\nUser boards vehicle...")
    guide.mark_boarded()
    
    print("\nSimulating journey (passing stops)...")
    for i in range(8):
        status = guide.get_current_status()
        print(f"\n  Stop {status['stops_passed'] + 1}/{status['stops_passed'] + status['stops_remaining']}")
        print(f"  Status: {status['status']}")
        
        if 'warning' in status:
            print(f"  🔊 WARNING: {status['warning']}")
        
        guide.mark_stop_passed()
    
    print("\nChecking if complete...")
    if guide.is_complete():
        print("  [OK] Transit step complete!")
    
    gps.stop()
    print("\n[OK] Transit guide test complete!")
