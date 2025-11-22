"""
Configuration management for the intrusion detection system
"""
import os
from pathlib import Path


class Config:
    """Central configuration class"""
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    ZONES_FILE = PROJECT_ROOT / "restricted_zones.json"
    
    YOLO_MODEL = "yolov8n.pt" # --> fast
    # YOLO_MODEL = "yolov8s.pt" # --> medium
    # YOLO_MODEL = "yolov8m.pt" # --> slow

    CONFIDENCE_THRESHOLD = 0.35 
    PERSON_CLASS_ID = 0
    
    PROCESS_WIDTH = 640  
    
    MAX_AGE = 40 
    MIN_HITS = 2 
    IOU_THRESHOLD = 0.25  
    
    ALARM_COOLDOWN_SECONDS = 3
    
    VIDEO_PATH = str(PROJECT_ROOT / "test.mp4")
    WINDOW_NAME = "Intrusion Detection System"
    PROCESS_EVERY_N_FRAMES = 2  # Har 2-frameni process qilish
    
    ZONE_COLOR = (0, 0, 255)
    ZONE_THICKNESS = 3
    BBOX_COLOR = (0, 255, 0)
    ALARM_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        cls.DATA_DIR.mkdir(exist_ok=True)