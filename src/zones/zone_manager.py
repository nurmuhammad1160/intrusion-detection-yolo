"""
Zone manager for drawing and managing restricted zones
"""
import json
import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from ..core.config import Config
from ..utils.geometry import point_in_polygon, draw_polygon


class ZoneManager:
    """Manage restricted zones with interactive drawing"""
    
    def __init__(self, zones_file: str = None):
        """
        Args:
            zones_file: Path to zones JSON file
        """
        self.zones_file = zones_file or str(Config.ZONES_FILE)
        self.zones: List[List[Tuple[int, int]]] = []
        self.current_zone: List[Tuple[int, int]] = []
        self.drawing_mode = False
        
        # Load existing zones
        self.load_zones()
    
    def load_zones(self):
        """Load zones from JSON file"""
        if not Path(self.zones_file).exists():
            print(f"No zones file found at {self.zones_file}")
            return
        
        try:
            with open(self.zones_file, 'r') as f:
                data = json.load(f)
                self.zones = [zone['points'] for zone in data.get('zones', [])]
            print(f"Loaded {len(self.zones)} zone(s) from {self.zones_file}")
        except Exception as e:
            print(f"Error loading zones: {e}")
    
    def save_zones(self):
        """Save zones to JSON file"""
        try:
            data = {
                'zones': [
                    {'points': zone, 'id': idx} 
                    for idx, zone in enumerate(self.zones)
                ]
            }
            
            with open(self.zones_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Saved {len(self.zones)} zone(s) to {self.zones_file}")
        except Exception as e:
            print(f"Error saving zones: {e}")
    
    def start_drawing(self, frame: np.ndarray) -> bool:
        """
        Start interactive zone drawing
        
        Args:
            frame: First frame for drawing
        
        Returns:
            True if drawing was completed
        """
        self.drawing_mode = True
        self.current_zone = []
        
        print("\n" + "="*60)
        print("ZONE DRAWING MODE")
        print("="*60)
        print("Instructions:")
        print("  - LEFT CLICK: Add point to zone")
        print("  - RIGHT CLICK: Remove last point")
        print("  - PRESS 'c': Complete zone")
        print("  - PRESS 'r': Reset current zone")
        print("  - PRESS 'q': Quit without saving")
        print("="*60 + "\n")
        
        window_name = "Draw Restricted Zone"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        temp_frame = frame.copy()
        
        while True:
            display_frame = temp_frame.copy()
            
            for zone in self.zones:
                draw_polygon(display_frame, zone, (0, 255, 0), 2)
            
            if len(self.current_zone) > 0:
                for point in self.current_zone:
                    cv2.circle(display_frame, point, 5, (0, 0, 255), -1)
                
                if len(self.current_zone) > 1:
                    pts = np.array(self.current_zone, np.int32)
                    cv2.polylines(display_frame, [pts], False, (0, 0, 255), 2)
            
            cv2.putText(display_frame, f"Points: {len(self.current_zone)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'c' to complete, 'r' to reset, 'q' to quit", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'): 
                if len(self.current_zone) >= 3:
                    self.zones.append(self.current_zone.copy())
                    self.save_zones()
                    print(f"✓ Zone completed with {len(self.current_zone)} points")
                    self.current_zone = []
                else:
                    print("⚠ Need at least 3 points to create a zone")
            
            elif key == ord('r'):  
                self.current_zone = []
                print("↻ Zone reset")
            
            elif key == ord('q'):  
                break
        
        cv2.destroyWindow(window_name)
        self.drawing_mode = False
        
        return len(self.zones) > 0
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for zone drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_zone.append((x, y))
            print(f"+ Point added: ({x}, {y})")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_zone) > 0:
                removed = self.current_zone.pop()
                print(f"- Point removed: {removed}")
    
    def is_point_in_any_zone(self, point: Tuple[int, int]) -> bool:
        """
        Check if point is in any restricted zone
        
        Args:
            point: (x, y) coordinates
        
        Returns:
            True if point is in any zone
        """
        for zone in self.zones:
            if point_in_polygon(point, zone):
                return True
        return False
    
    def draw_zones(self, frame: np.ndarray):
        """Draw all zones on frame"""
        for zone in self.zones:
            draw_polygon(frame, zone, Config.ZONE_COLOR, Config.ZONE_THICKNESS)