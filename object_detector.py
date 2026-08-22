"""
YOLOv8 Object Detection Module
Detects objects including Person, Chair, Bottle, Sofa, TV, and custom Keys class
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple
import os
from config import (
    YOLO_MODEL, YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD,
    TARGET_CLASSES, MODELS_DIR, YOLO_IMGSZ,
    USE_OPENVINO, YOLO_OPENVINO_MODEL, OPENVINO_DEVICE
)


class ObjectDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize YOLOv8 object detector
        
        Args:
            model_path: Path to YOLO model file (default: yolov8n.pt)
        """
        self.imgsz = YOLO_IMGSZ
        self.using_openvino = False
        
        if USE_OPENVINO and os.path.exists(YOLO_OPENVINO_MODEL):
            print(f"Loading OpenVINO-accelerated YOLO model: {YOLO_OPENVINO_MODEL}")
            # Ultralytics auto-detects OpenVINO format from the folder path.
            # Device is selected via the `device` kwarg in predict(), not at load time.
            self.model = YOLO(YOLO_OPENVINO_MODEL, task='detect')
            self.using_openvino = True
            
            # Determine which OpenVINO device is available to use at inference time.
            # IMPORTANT: Ultralytics v8.4.x OpenVINO backend (backends/openvino.py line 37)
            # only activates device routing when self.device starts with "intel:".
            # A bare "GPU" or "CPU" string is silently ignored and falls through to AUTO/CPU.
            # Correct format: "intel:gpu", "intel:npu", "intel:cpu", "intel:auto".
            try:
                import openvino as ov
                core = ov.Core()
                available = core.available_devices
                # Pick the requested device if present; fall back to AUTO (not CPU) so
                # OpenVINO's own scheduler can still pick GPU/NPU even if the exact
                # device string wasn't in the list.
                selected = OPENVINO_DEVICE if OPENVINO_DEVICE in available else "AUTO"
                self.device = f"intel:{selected.lower()}"
                print(f"  Available OpenVINO devices: {available}")
                print(f"  Device: OpenVINO ({self.device}) | Inference size: {self.imgsz}px")
            except Exception as e:
                print(f"  OpenVINO device query failed ({e}), defaulting to intel:cpu")
                self.device = "intel:cpu"
                print(f"  Device: OpenVINO (intel:cpu) | Inference size: {self.imgsz}px")
        else:
            if model_path is None:
                model_path = YOLO_MODEL
            print(f"Loading YOLOv8 model: {model_path}")
            self.model = YOLO(model_path)
            self.device = 0 if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            print(f"  Device: {'GPU (cuda:0)' if self.device == 0 else 'CPU'} | Inference size: {self.imgsz}px")
            
        self.confidence = YOLO_CONFIDENCE
        self.iou_threshold = YOLO_IOU_THRESHOLD
        self.target_classes = TARGET_CLASSES
        
        # COCO class name mapping
        self.coco_mapping = {
            'sofa': 'couch',  # Map COCO 'sofa' to our 'couch' (if returned by some models)
            'couch': 'couch', # Ensure it remains couch
        }
        
        print(f"[OK] Object Detector initialized")
        print(f"  Target classes: {self.target_classes}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Perform object detection on a frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of detections, each containing:
                - class_name: Object class
                - confidence: Detection confidence
                - bbox: [x1, y1, x2, y2] bounding box
                - center: (cx, cy) center point
        """
        # device= is honoured by both the plain PyTorch backend and the OpenVINO backend
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract box data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id].lower()
                
                # Map COCO names to our target names
                class_name = self.coco_mapping.get(class_name, class_name)
                
                # Filter to only target classes
                if class_name in self.target_classes:
                    # Calculate center point
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    detection = {
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'center': (cx, cy)
                    }
                    detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input image
            detections: List of detection dictionaries
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Draw bounding box
            color = self._get_color(class_name)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return annotated
    
    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get consistent color for each class"""
        colors = {
            'person': (0, 255, 0),      # Green
            'chair': (255, 0, 0),       # Blue
            'bottle': (0, 165, 255),    # Orange
            'sofa': (203, 192, 255),    # Pink
            'tv': (255, 255, 0),        # Cyan
            'keys': (0, 0, 255),        # Red
        }
        return colors.get(class_name, (128, 128, 128))
    
    def get_detection_by_class(self, detections: List[Dict], class_name: str) -> List[Dict]:
        """Filter detections by class name"""
        return [d for d in detections if d['class_name'] == class_name]


if __name__ == "__main__":
    # Quick test
    detector = ObjectDetector()
    print("\n[OK] Object Detector ready for inference")
