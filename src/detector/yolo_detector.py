"""
YOLO-based person detector - OPTIMAL VERSION
"""
from ultralytics import YOLO
import numpy as np
import cv2
from typing import List, Tuple
from ..core.config import Config


class YOLODetector:
    """YOLO-based person detection - optimized for real-time performance"""
    
    def __init__(self, model_path: str = None, confidence: float = None):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model
            confidence: Confidence threshold
        """
        self.model_path = model_path or Config.YOLO_MODEL
        self.confidence = confidence or Config.CONFIDENCE_THRESHOLD
        
        print(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        print("Warming up model...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)
        
        print("✓ YOLO model ready!")
        print(f"  Model: {self.model_path}")
        print(f"  Confidence: {self.confidence}")
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect persons in frame - FAST & ACCURATE
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
        """
        results = self.model.predict(
            frame,
            verbose=False,
            conf=self.confidence,
            iou=0.45,  
            classes=[Config.PERSON_CLASS_ID],
            max_det=30,  
            imgsz=Config.PROCESS_WIDTH,  
            device='cpu',
            half=False,
            augment=False,  
            agnostic_nms=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # Confidence
                conf = float(box.conf[0])
                
                # Bbox coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Size validation
                width = x2 - x1
                height = y2 - y1
                
                # Minimum size filter
                if width < 20 or height < 40:
                    continue
                
                aspect_ratio = height / max(width, 1)
                if aspect_ratio < 1.0 or aspect_ratio > 5.0:
                    continue
                
                detections.append((x1, y1, x2, y2, conf))
        
        return detections