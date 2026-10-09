"""
Traffic Signal Detection Module
Detects and classifies traffic lights using HSV color masking
"""
import cv2
import numpy as np
from typing import Optional, List, Dict
from config import (
    TRAFFIC_HSV_RED_LOWER, TRAFFIC_HSV_RED_UPPER,
    TRAFFIC_HSV_GREEN_LOWER, TRAFFIC_HSV_GREEN_UPPER,
    TRAFFIC_MIN_AREA
)


class TrafficSignalDetector:
    def __init__(self):
        """Initialize traffic signal detector"""
        self.red_lower = np.array(TRAFFIC_HSV_RED_LOWER)
        self.red_upper = np.array(TRAFFIC_HSV_RED_UPPER)
        self.green_lower = np.array(TRAFFIC_HSV_GREEN_LOWER)
        self.green_upper = np.array(TRAFFIC_HSV_GREEN_UPPER)
        self.min_area = TRAFFIC_MIN_AREA
        
        print(f"[OK] Traffic Signal Detector initialized")
        print(f"  Minimum detection area: {self.min_area} pixels")
    
    def detect_signal(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect traffic signal and classify color
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            Dictionary with:
                - signal_type: 'RED' or 'GREEN'
                - bbox: [x, y, w, h]
                - confidence: Detection strength (area-based)
            Returns None if no signal detected
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect red signal
        red_mask = cv2.inRange(hsv, self.red_lower, self.red_upper)
        red_contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Detect green signal
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        green_contours, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Find largest red contour
        red_detection = self._get_largest_contour(red_contours, 'RED')
        
        # Find largest green contour
        green_detection = self._get_largest_contour(green_contours, 'GREEN')
        
        # Return the detection with higher confidence
        if red_detection and green_detection:
            if red_detection['confidence'] > green_detection['confidence']:
                return red_detection
            else:
                return green_detection
        elif red_detection:
            return red_detection
        elif green_detection:
            return green_detection
        else:
            return None
    
    def _get_largest_contour(self, contours: List, signal_type: str) -> Optional[Dict]:
        """
        Find largest contour and return detection info
        
        Args:
            contours: List of contours
            signal_type: 'RED' or 'GREEN'
            
        Returns:
            Detection dictionary or None
        """
        if len(contours) == 0:
            return None
        
        # Find largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < self.min_area:
            return None
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return {
            'signal_type': signal_type,
            'bbox': [x, y, w, h],
            'confidence': float(area),
            'area': float(area)
        }
    
    def draw_signal(self, frame: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Draw traffic signal detection on frame
        
        Args:
            frame: Input image
            detection: Detection dictionary
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        x, y, w, h = detection['bbox']
        signal_type = detection['signal_type']
        
        # Color based on signal type
        color = (0, 0, 255) if signal_type == 'RED' else (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        
        # Draw label
        label = f"{signal_type} LIGHT"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(
            annotated,
            (x, y - label_size[1] - 15),
            (x + label_size[0], y),
            color,
            -1
        )
        cv2.putText(
            annotated,
            label,
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Add warning text for red light
        if signal_type == 'RED':
            warning = "STOP - RED LIGHT"
            cv2.putText(
                annotated,
                warning,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )
        
        return annotated
    
    def get_color_masks(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get HSV color masks for debugging
        
        Args:
            frame: Input image
            
        Returns:
            Dictionary with 'red' and 'green' masks
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        red_mask = cv2.inRange(hsv, self.red_lower, self.red_upper)
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        return {
            'red': red_mask,
            'green': green_mask
        }


if __name__ == "__main__":
    # Quick test
    detector = TrafficSignalDetector()
    print("\n[OK] Traffic Signal Detector ready")
