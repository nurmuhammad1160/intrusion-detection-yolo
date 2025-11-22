"""
Geometric utilities for zone detection
"""
import cv2
import numpy as np
from typing import List, Tuple


def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm
    
    Args:
        point: (x, y) coordinates
        polygon: List of (x, y) vertices
    
    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def get_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Get center point of bounding box
    
    Args:
        bbox: (x1, y1, x2, y2)
    
    Returns:
        (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


def get_bbox_bottom_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Get bottom center point of bounding box (more accurate for person detection)
    
    Args:
        bbox: (x1, y1, x2, y2)
    
    Returns:
        (center_x, bottom_y)
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    return center_x, y2


def draw_polygon(frame: np.ndarray, polygon: List[Tuple[int, int]], 
                 color: Tuple[int, int, int], thickness: int = 2):
    """
    Draw polygon on frame
    
    Args:
        frame: Video frame
        polygon: List of vertices
        color: BGR color
        thickness: Line thickness
    """
    if len(polygon) < 3:
        return
    
    pts = np.array(polygon, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)