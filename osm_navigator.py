"""
OpenStreetMap Navigator
Free and open-source navigation using OSRM and Nominatim
No API keys required, unlimited usage
"""

import requests
import logging
import time
import math
from typing import Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class OSMNavigator:
    """
    Free navigation using OpenStreetMap services
    - Nominatim for geocoding
    - OSRM for routing
    """
    
    def __init__(self):
        self.nominatim_url = "https://nominatim.openstreetmap.org"
        self.osrm_url = "http://router.project-osrm.org"
        self.user_agent = "VisionMate/1.0"
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Nominatim requires 1 req/sec max
        
        logger.info("OSM Navigator initialized (FREE, no API key needed)")
    
    def _rate_limit(self):
        """Respect Nominatim's rate limit (1 request/second)"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Convert address to GPS coordinates using Nominatim
        
        Args:
            address: Place name or address (e.g., "Majestic, Bangalore")
            
        Returns:
            (latitude, longitude) or None if not found
        """
        try:
            self._rate_limit()
            
            url = f"{self.nominatim_url}/search"
            params = {
                'q': address,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            headers = {'User-Agent': self.user_agent}
            
            logger.info(f"Geocoding: {address}")
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                display_name = data[0]['display_name']
                
                logger.info(f"[OK] Found: {display_name}")
                logger.info(f"  Coordinates: {lat:.6f}, {lon:.6f}")
                
                return (lat, lon)
            else:
                logger.warning(f"Location not found: {address}")
                return None
                
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            return None
    
    def get_directions(self, origin: Tuple[float, float], destination: Tuple[float, float], 
                       mode: str = 'foot') -> Optional[Dict]:
        """
        Get directions from origin to destination using OSRM
        
        Args:
            origin: (lat, lon) tuple
            destination: (lat, lon) tuple
            mode: 'foot' (walking), 'car', or 'bike'
            
        Returns:
            Route dictionary with steps and navigation info
        """
        try:
            if mode == 'transit':
                logger.warning("Transit mode not supported in free version, using walking")
                mode = 'foot'
            
            # OSRM uses lon,lat order (not lat,lon!)
            origin_str = f"{origin[1]},{origin[0]}"
            dest_str = f"{destination[1]},{destination[0]}"
            
            url = f"{self.osrm_url}/route/v1/{mode}/{origin_str};{dest_str}"
            params = {
                'overview': 'full',
                'steps': 'true',
                'geometries': 'geojson'
            }
            
            logger.info(f"Getting {mode} directions...")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data['code'] != 'Ok':
                logger.error(f"OSRM error: {data['code']}")
                return None
            
            route = data['routes'][0]
            
            # Parse into our format
            parsed_route = self._parse_route(route)
            
            logger.info(f"[OK] Route found: {parsed_route['total_distance']}, {parsed_route['total_duration']}")
            
            return parsed_route
            
        except Exception as e:
            logger.error(f"Routing error: {e}")
            return None
    
    def _parse_route(self, route: Dict) -> Dict:
        """Parse OSRM route into simplified format"""
        
        total_distance = route['distance']  # meters
        total_duration = route['duration']  # seconds
        
        steps = []
        
        for leg in route['legs']:
            for step in leg['steps']:
                instruction = step['maneuver']['type']
                
                # Convert instruction to readable format
                readable = self._format_instruction(step)
                
                steps.append({
                    'instruction': readable,
                    'distance': step['distance'],  # meters
                    'duration': step['duration'],  # seconds
                    'start_location': {
                        'lat': step['maneuver']['location'][1],
                        'lng': step['maneuver']['location'][0]
                    },
                    'end_location': {
                        'lat': step['maneuver']['location'][1],  # Simplified
                        'lng': step['maneuver']['location'][0]
                    }
                })
        
        return {
            'steps': steps,
            'total_distance': self._format_distance(total_distance),
            'total_duration': self._format_duration(total_duration),
            'distance_meters': total_distance,
            'duration_seconds': total_duration
        }
    
    def _format_instruction(self, step: Dict) -> str:
        """Convert OSRM maneuver to readable instruction"""
        maneuver = step['maneuver']
        maneuver_type = maneuver['type']
        
        name = step.get('name', 'unnamed road')
        distance = step['distance']
        
        if maneuver_type == 'depart':
            return f"Head {self._get_direction(maneuver.get('bearing_after', 0))} on {name}"
        elif maneuver_type == 'arrive':
            return "You have arrived at your destination"
        elif maneuver_type == 'turn':
            modifier = maneuver.get('modifier', 'straight')
            return f"Turn {modifier} onto {name}"
        elif maneuver_type == 'new name':
            return f"Continue on {name}"
        elif maneuver_type == 'continue':
            return f"Continue straight on {name}"
        elif maneuver_type == 'roundabout':
            exit_num = maneuver.get('exit', 1)
            return f"At roundabout, take exit {exit_num} onto {name}"
        else:
            return f"Continue on {name}"
    
    def calculate_straight_line(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> Dict:
        """
        Calculate straight-line distance and bearing between two GPS points using the Haversine formula.
        Does NOT call any external API - pure math, for quick 'how far and which way' queries
        that don't need full road routing (e.g. finding a remembered object).
        Returns dict with: distance_meters (float), direction (str, using existing _get_direction), 
        formatted_distance (str, using existing _format_distance)
        """
        lat1, lon1 = origin
        lat2, lon2 = destination
        
        R = 6371e3  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2.0) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * \
            math.sin(delta_lambda / 2.0) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c

        y = math.sin(delta_lambda) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - \
            math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
        bearing = (math.degrees(math.atan2(y, x)) + 360) % 360

        return {
            'distance_meters': distance,
            'direction': self._get_direction(bearing),
            'formatted_distance': self._format_distance(distance)
        }

    def _get_direction(self, bearing: float) -> str:
        """Convert bearing to cardinal direction"""
        directions = ['north', 'northeast', 'east', 'southeast', 
                     'south', 'southwest', 'west', 'northwest']
        index = round(bearing / 45) % 8
        return directions[index]
    
    def _format_distance(self, meters: float) -> str:
        """Format distance for display"""
        if meters < 1000:
            return f"{int(meters)} meters"
        else:
            km = meters / 1000
            return f"{km:.1f} km"
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration for display"""
        minutes = int(seconds / 60)
        if minutes < 60:
            return f"{minutes} min"
        else:
            hours = minutes // 60
            mins = minutes % 60
            return f"{hours} hr {mins} min"
    
    def get_current_instruction(self, step_index: int = 0) -> Optional[str]:
        """Get current navigation instruction"""
        if not hasattr(self, 'current_route') or not self.current_route:
            return None
        
        if step_index >= len(self.current_route['steps']):
            return "You have arrived"
        
        return self.current_route['steps'][step_index]['instruction']
    
    def advance_step(self, current_step: int) -> bool:
        """Check if can advance to next step"""
        if not hasattr(self, 'current_route') or not self.current_route:
            return False
        
        return current_step + 1 < len(self.current_route['steps'])
    
    def get_progress(self, current_step: int) -> Dict:
        """Get navigation progress"""
        if not hasattr(self, 'current_route') or not self.current_route:
            return {'status': 'inactive'}
        
        total_steps = len(self.current_route['steps'])
        
        return {
            'status': 'active',
            'current_step': current_step + 1,
            'total_steps': total_steps,
            'total_distance': self.current_route['total_distance'],
            'total_duration': self.current_route['total_duration']
        }
    
    def get_all_instructions(self) -> List[str]:
        """Get all turn-by-turn instructions"""
        if not hasattr(self, 'current_route') or not self.current_route:
            return []
        
        return [step['instruction'] for step in self.current_route['steps']]


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("OpenStreetMap Navigator Test (100% FREE)")
    print("="*60)
    
    nav = OSMNavigator()
    
    # Test 1: Geocoding
    print("\n1. Testing Geocoding...")
    majestic = nav.geocode_address("Majestic, Bangalore, India")
    if majestic:
        print(f"   [OK] Majestic coordinates: {majestic[0]:.6f}, {majestic[1]:.6f}")
    
    indiranagar = nav.geocode_address("Indiranagar, Bangalore, India")
    if indiranagar:
        print(f"   [OK] Indiranagar coordinates: {indiranagar[0]:.6f}, {indiranagar[1]:.6f}")
    
    # Test 2: Routing
    if majestic and indiranagar:
        print("\n2. Testing Routing (Walking)...")
        route = nav.get_directions(indiranagar, majestic, mode='foot')
        
        if route:
            nav.current_route = route
            print(f"   [OK] Route found!")
            print(f"   Distance: {route['total_distance']}")
            print(f"   Duration: {route['total_duration']}")
            print(f"   Steps: {len(route['steps'])}")
            
            print("\n3. Turn-by-turn instructions:")
            for i, step in enumerate(route['steps'][:5], 1):
                print(f"   {i}. {step['instruction']} ({nav._format_distance(step['distance'])})")
            
            if len(route['steps']) > 5:
                print(f"   ... and {len(route['steps']) - 5} more steps")
    
    print("\n" + "="*60)
    print("[OK] Test complete! No API keys needed, unlimited usage!")
    print("="*60)
