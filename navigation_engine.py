"""
Navigation Engine
Core turn-by-turn navigation logic with GPS tracking
Calculates distances, triggers announcements, handles rerouting
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
import time

logger = logging.getLogger(__name__)


class NavigationEngine:
    """
    Main navigation engine for turn-by-turn directions
    Combines GPS tracking with Google Maps directions
    """
    
    def __init__(self, gps_tracker, maps_navigator, 
                 announcement_distance: float = 50.0,
                 reroute_distance: float = 30.0,
                 arrival_distance: float = 20.0):
        """
        Args:
            gps_tracker: GPSTracker instance
            maps_navigator: GoogleMapsNavigator instance
            announcement_distance: Distance (m) before turn to announce (default 50m)
            reroute_distance: Distance (m) off route to trigger reroute (default 30m)
            arrival_distance: Distance (m) from destination to mark arrived (default 20m)
        """
        self.gps = gps_tracker
        self.maps = maps_navigator
        
        self.announcement_distance = announcement_distance
        self.reroute_distance = reroute_distance
        self.arrival_distance = arrival_distance
        
        self.active_route = None
        self.current_step = 0
        self.last_announcement = None
        self.navigation_active = False
        
        logger.info("Navigation Engine initialized")
        logger.info(f"  Announcement distance: {announcement_distance}m")
        logger.info(f"  Reroute distance: {reroute_distance}m")
        logger.info(f"  Arrival distance: {arrival_distance}m")
    
    def start_navigation(self, origin: str, destination: str, mode: str = 'transit') -> bool:
        """
        Start navigation from origin to destination
        
        Args:
            origin: Starting address or coordinates
            destination: Ending address
            mode: 'transit', 'walking', or 'driving'
            
        Returns:
            True if route found and navigation started
        """
        logger.info(f"Starting navigation: {origin} -> {destination} ({mode})")
        
        # Get current GPS position
        current_pos = self.gps.get_position()
        if not current_pos:
            logger.error("Cannot get GPS position")
            return False
        
        # If origin is string, geocode it; otherwise use current position
        if isinstance(origin, str):
            origin_coords = self.maps.geocode_address(origin)
            if not origin_coords:
                logger.warning(f"Could not geocode origin: {origin}, using current GPS position")
                origin = current_pos
            else:
                origin = origin_coords
        
        # Get directions
        self.active_route = self.maps.get_directions(origin, destination, mode)
        
        if not self.active_route:
            logger.error("Failed to get directions")
            return False
        
        self.current_step = 0
        self.last_announcement = None
        self.navigation_active = True
        
        logger.info(f"Route loaded: {self.active_route['total_distance']}, {self.active_route['total_duration']}")
        logger.info(f"Steps: {len(self.active_route['steps'])}")
        
        return True
    
    def update(self) -> Dict:
        """
        Update navigation state based on current GPS position
        
        Returns:
            Dict with navigation status and actions to take
        """
        if not self.navigation_active or not self.active_route:
            return {'status': 'inactive'}
        
        current_pos = self.gps.get_position()
        if not current_pos:
            return {'status': 'no_gps', 'message': 'Waiting for GPS signal'}
        
        # Check if arrived
        if self._is_arrived(current_pos):
            self.navigation_active = False
            return {
                'status': 'arrived',
                'message': 'You have arrived at your destination',
                'distance_to_destination': self._calculate_distance_to_destination(current_pos)
            }
        
        # Check if off route
        if self._is_off_route(current_pos):
            return {
                'status': 'off_route',
                'message': 'You are off route. Recalculating...',
                'deviation': self._calculate_deviation(current_pos)
            }
        
        # Get current instruction
        instruction = self.maps.get_current_instruction()
        if not instruction:
            return {'status': 'no_instruction'}
        
        # Calculate distance to next waypoint
        distance_to_waypoint = self._calculate_distance_to_waypoint(current_pos)
        
        # Check if should announce
        should_announce = self._should_announce(distance_to_waypoint)
        
        result = {
            'status': 'navigating',
            'instruction': instruction,
            'distance_to_waypoint': distance_to_waypoint,
            'current_step': self.current_step + 1,
            'total_steps': len(self.active_route['steps']),
            'announce': should_announce
        }
        
        # If very close to waypoint, advance to next step
        if distance_to_waypoint < 10:  # Within 10 meters
            if self.maps.advance_step():
                self.current_step += 1
                self.last_announcement = None  # Reset announcement flag
                logger.info(f"Advanced to step {self.current_step + 1}")
        
        return result
    
    def _calculate_distance_to_waypoint(self, current_pos: Tuple[float, float]) -> float:
        """
        Calculate distance from current position to next waypoint
        
        Returns distance in meters
        """
        step = self.active_route['steps'][self.current_step]
        
        # Use end location of current step
        end_lat = step['end_location']['lat']
        end_lon = step['end_location']['lng']
        
        return self._haversine_distance(
            current_pos[0], current_pos[1],
            end_lat, end_lon
        )
    
    def _calculate_distance_to_destination(self, current_pos: Tuple[float, float]) -> float:
        """Calculate distance to final destination"""
        final_step = self.active_route['steps'][-1]
        dest_lat = final_step['end_location']['lat']
        dest_lon = final_step['end_location']['lng']
        
        return self._haversine_distance(
            current_pos[0], current_pos[1],
            dest_lat, dest_lon
        )
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two GPS coordinates using Haversine formula
        
        Returns distance in meters
        """
        R = 6371000  # Earth radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return distance
    
    def _is_arrived(self, current_pos: Tuple[float, float]) -> bool:
        """Check if arrived at destination"""
        distance = self._calculate_distance_to_destination(current_pos)
        return distance < self.arrival_distance
    
    def _is_off_route(self, current_pos: Tuple[float, float]) -> bool:
        """Check if user has deviated from route"""
        deviation = self._calculate_deviation(current_pos)
        return deviation > self.reroute_distance
    
    def _calculate_deviation(self, current_pos: Tuple[float, float]) -> float:
        """
        Calculate perpendicular distance from current position to route path
        
        Simplified: distance to current waypoint
        """
        return self._calculate_distance_to_waypoint(current_pos)
    
    def _should_announce(self, distance_to_waypoint: float) -> bool:
        """
        Determine if should announce upcoming turn
        
        Only announce once when entering announcement zone
        """
        if distance_to_waypoint <= self.announcement_distance:
            if self.last_announcement != self.current_step:
                self.last_announcement = self.current_step
                return True
        return False
    
    def get_progress(self) -> Dict:
        """Get current navigation progress"""
        if not self.navigation_active:
            return {'status': 'inactive'}
        
        return {
            'status': 'active',
            'current_step': self.current_step + 1,
            'total_steps': len(self.active_route['steps']),
            'total_distance': self.active_route['total_distance'],
            'total_duration': self.active_route['total_duration']
        }
    
    def stop_navigation(self):
        """Stop active navigation"""
        self.navigation_active = False
        self.active_route = None
        self.current_step = 0
        self.last_announcement = None
        logger.info("Navigation stopped")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Testing Navigation Engine ===")
    
    from gps_tracker import GPSTracker
    from google_maps_navigator import GoogleMapsNavigator
    import config
    
    # Initialize components
    gps = GPSTracker(mock_mode=True)
    gps.start()
    
    maps = GoogleMapsNavigator(config.GOOGLE_MAPS_API_KEY)
    
    engine = NavigationEngine(gps, maps)
    
    print("\nStarting navigation from Indiranagar to Majestic...")
    success = engine.start_navigation(
        "Indiranagar, Bangalore",
        "Majestic, Bangalore",
        mode='transit'
    )
    
    if success:
        print("[OK] Route loaded!")
        
        progress = engine.get_progress()
        print(f"\nRoute: {progress['total_distance']}, {progress['total_duration']}")
        print(f"Steps: {progress['total_steps']}")
        
        print("\nSimulating navigation updates...")
        for i in range(10):
            # Update navigation
            status = engine.update()
            
            if status['status'] == 'navigating':
                print(f"\nStep {status['current_step']}/{status['total_steps']}")
                print(f"  Instruction: {status['instruction']}")
                print(f"  Distance to waypoint: {status['distance_to_waypoint']:.1f}m")
                
                if status['announce']:
                    print("  [AUDIO] ANNOUNCE: " + status['instruction'])
            
            elif status['status'] == 'arrived':
                print(f"\n[OK] {status['message']}")
                break
            
            # Advance mock GPS position
            gps.advance_mock_position()
            time.sleep(0.5)
    else:
        print("[FAIL] Failed to start navigation")
    
    gps.stop()
    print("\n[OK] Navigation engine test complete!")
