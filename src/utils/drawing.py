"""
Drawing utilities for visualization
"""
import cv2
import numpy as np
from typing import Tuple


def draw_bbox(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
              track_id: int = None, color: Tuple[int, int, int] = (0, 255, 0)):
    """
    Draw bounding box with track ID
    
    Args:
        frame: Video frame
        bbox: (x1, y1, x2, y2)
        track_id: Optional track ID
        color: BGR color
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    if track_id is not None:
        label = f"ID: {track_id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def draw_alarm(frame: np.ndarray, message: str = "ALARM!", 
               color: Tuple[int, int, int] = (0, 0, 255)):
    """
    Draw alarm message on frame
    
    Args:
        frame: Video frame
        message: Alarm message
        color: BGR color
    """
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, 2.5, 4)[0]
    text_x = (w - text_size[0]) // 2
    text_y = 70
    
    cv2.putText(frame, message, (text_x, text_y), font, 2.5, (255, 255, 255), 4)