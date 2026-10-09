"""
Google Maps Navigation Integration
Handles route planning, directions, and transit information
"""

import requests
import logging
from typing import Dict, List, Optional, Tuple
from config import GOOGLE_MAPS_API_KEY

logger = logging.getLogger(__name__)


class GoogleMapsNavigator:
    """
    Integrates with Google Maps APIs for navigation
    """
    
    def __init__(self, api_key: str = GOOGLE_MAPS_API_KEY):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api"
        self.current_route = None
        self.current_step_index = 0
        
        logger.info(f"Google Maps Navigator initialized")
        if api_key == "YOUR_API_KEY_HERE":
            logger.warning("Using placeholder API key - get real key from Google Cloud Console")
    
    def geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Convert address to GPS coordinates
        
        Args:
            address: Human-readable address (e.g., "Majestic, Bangalore")
            
        Returns:
            Tuple of (latitude, longitude) or None if failed
        """
        try:
            url = f"{self.base_url}/geocode/json"
            params = {
                'address': address,
                'key': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data['status'] == 'OK':
                location = data['results'][0]['geometry']['location']
                lat, lon = location['lat'], location['lng']
                logger.info(f"Geocoded '{address}' to ({lat}, {lon})")
                return (lat, lon)
            else:
                logger.error(f"Geocoding failed: {data['status']}")
                return None
                
        except Exception as e:
            logger.error(f"Error geocoding address: {e}")
            return None
    
    def get_directions(self, origin: Tuple[float, float], destination: str,
                      mode: str = "transit") -> Optional[Dict]:
        """
        Get directions from origin to destination
        
        Args:
            origin: Tuple of (lat, lon) for starting point
            destination: Address or place name
            mode: "transit", "walking", "driving"
            
        Returns:
            Dictionary with route information or None if failed
        """
        try:
            url = f"{self.base_url}/directions/json"
            params = {
                'origin': f"{origin[0]},{origin[1]}",
                'destination': destination,
                'mode': mode,
                'key': self.api_key
            }
            
            # For transit, add departure time
            if mode == "transit":
                import time
                params['departure_time'] = int(time.time())
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data['status'] == 'OK':
                route = data['routes'][0]
                self.current_route = self._parse_route(route, mode)
                self.current_step_index = 0
                logger.info(f"Got directions to '{destination}' via {mode}")
                return self.current_route
            else:
                logger.error(f"Directions failed: {data['status']}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting directions: {e}")
            return None
    
    def _parse_route(self, route: Dict, mode: str) -> Dict:
        """Parse Google Maps route into simplified format"""
        leg = route['legs'][0]
        
        parsed = {
            'distance': leg['distance']['text'],
            'duration': leg['duration']['text'],
            'start_address': leg['start_address'],
            'end_address': leg['end_address'],
            'steps': [],
            'mode': mode
        }
        
        for step in leg['steps']:
            step_info = {
                'instruction': self._clean_html(step['html_instructions']),
                'distance': step['distance']['text'],
                'duration': step['duration']['text'],
                'start_location': (step['start_location']['lat'], step['start_location']['lng']),
                'end_location': (step['end_location']['lat'], step['end_location']['lng']),
                'travel_mode': step['travel_mode']
            }
            
            # Add transit details if available
            if 'transit_details' in step:
                transit = step['transit_details']
                step_info['transit'] = {
                    'line': transit['line']['short_name'],
                    'vehicle': transit['line']['vehicle']['name'],
                    'departure_stop': transit['departure_stop']['name'],
                    'arrival_stop': transit['arrival_stop']['name'],
                    'num_stops': transit['num_stops']
                }
            
            parsed['steps'].append(step_info)
        
        return parsed
    
    def _clean_html(self, html_text: str) -> str:
        """Remove HTML tags from instruction text"""
        import re
        clean = re.sub('<.*?>', '', html_text)
        clean = clean.replace('&nbsp;', ' ')
        return clean.strip()
    
    def get_current_instruction(self) -> Optional[str]:
        """Get current navigation instruction"""
        if not self.current_route or self.current_step_index >= len(self.current_route['steps']):
            return None
        
        step = self.current_route['steps'][self.current_step_index]
        instruction = step['instruction']
        
        # Add transit info if available
        if 'transit' in step:
            transit = step['transit']
            instruction = f"Take {transit['vehicle']} {transit['line']} from {transit['departure_stop']}"
        
        return instruction
    
    def advance_step(self):
        """Move to next navigation step"""
        if self.current_route:
            self.current_step_index += 1
            logger.info(f"Advanced to step {self.current_step_index}")
    
    def get_progress(self) -> Dict:
        """Get navigation progress information"""
        if not self.current_route:
            return {'active': False}
        
        total_steps = len(self.current_route['steps'])
        remaining_steps = total_steps - self.current_step_index
        
        return {
            'active': True,
            'current_step': self.current_step_index + 1,
            'total_steps': total_steps,
            'remaining_steps': remaining_steps,
            'distance': self.current_route['distance'],
            'duration': self.current_route['duration']
        }
    
    def get_all_instructions(self) -> List[str]:
        """Get all navigation instructions as a list"""
        if not self.current_route:
            return []
        
        instructions = []
        for i, step in enumerate(self.current_route['steps']):
            instruction = f"{i+1}. {step['instruction']} ({step['distance']})"
            if 'transit' in step:
                transit = step['transit']
                instruction += f"\n   {transit['vehicle']} {transit['line']}: {transit['num_stops']} stops"
            instructions.append(instruction)
        
        return instructions
    
    def is_navigation_complete(self) -> bool:
        """Check if navigation is complete"""
        if not self.current_route:
            return True
        return self.current_step_index >= len(self.current_route['steps'])


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    nav = GoogleMapsNavigator()
    
    # Test geocoding
    print("\n=== Testing Geocoding ===")
    coords = nav.geocode_address("Majestic, Bangalore")
    if coords:
        print(f"Majestic coordinates: {coords}")
    
    # Test directions (mock origin)
    if coords:
        print("\n=== Testing Directions ===")
        origin = (12.9716, 77.5946)  # Example: Indiranagar
        route = nav.get_directions(origin, "Majestic, Bangalore", mode="transit")
        
        if route:
            print(f"\nRoute Overview:")
            print(f"  Distance: {route['distance']}")
            print(f"  Duration: {route['duration']}")
            print(f"  From: {route['start_address']}")
            print(f"  To: {route['end_address']}")
            
            print(f"\nStep-by-step instructions:")
            for instruction in nav.get_all_instructions():
                print(f"  {instruction}")
