"""
Main processing pipeline for intrusion detection - OPTIMIZED
"""
import cv2
import numpy as np
import time
from typing import Optional
from .detector import YOLODetector
from .tracker import DeepSORTTracker
from .zones import ZoneManager
from .core.config import Config
from .utils import get_bbox_bottom_center, draw_bbox, draw_alarm, Timer


class IntrusionDetectionPipeline:
    """Main pipeline for intrusion detection system - OPTIMIZED"""
    
    def __init__(self):
        """Initialize pipeline components"""
        print("\n" + "="*60)
        print("Initializing Intrusion Detection Pipeline...")
        print("="*60)
        
        # Components
        self.detector = YOLODetector()
        self.tracker = DeepSORTTracker(
            max_age=Config.MAX_AGE,
            min_hits=Config.MIN_HITS,
            iou_threshold=Config.IOU_THRESHOLD
        )
        self.zone_manager = ZoneManager()
        
        # Alarm state
        self.alarm_active = False
        self.alarm_timer = Timer(Config.ALARM_COOLDOWN_SECONDS)
        self.intruders_in_zone = set()
        
        # Frame tracking
        self.frame_count = 0
        self.last_tracks = []  
        
        # Performance tracking
        self.fps_history = []
        self.last_time = time.time()
        
        print("✓ Pipeline initialized successfully!")
        print("="*60 + "\n")
    
    def setup_zones(self, frame: np.ndarray) -> bool:
        """Setup zones interactively"""
        if len(self.zone_manager.zones) == 0:
            print("\n⚠ No zones found. Starting zone drawing mode...")
            return self.zone_manager.start_drawing(frame)
        else:
            print(f"✓ Found {len(self.zone_manager.zones)} existing zone(s)")
            
            print("\nOptions:")
            print("  1. Use existing zones")
            print("  2. Redraw zones")
            
            choice = input("Enter choice (1/2): ").strip()
            
            if choice == '2':
                self.zone_manager.zones = []
                return self.zone_manager.start_drawing(frame)
            
            return True
    
    def process_frame(self, frame: np.ndarray, force_detect: bool = False) -> np.ndarray:
        """
        Process single frame - OPTIMIZED
        
        Args:
            frame: Input frame
            force_detect: Force detection even if skipping
        
        Returns:
            Processed frame with visualizations
        """
        self.frame_count += 1
        
        # FPS calculation
        current_time = time.time()
        fps = 1.0 / max(current_time - self.last_time, 0.001)
        self.last_time = current_time
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # Selective processing - har N-frameda detect qilish
        should_detect = (self.frame_count % Config.PROCESS_EVERY_N_FRAMES == 0) or force_detect
        
        if should_detect:
            # 1. Detect persons
            detections = self.detector.detect_persons(frame)
            
            # 2. Track persons
            tracks = self.tracker.update(detections)
            
            # Cache for next frames
            self.last_tracks = tracks
        else:
            # Use cached tracks
            tracks = self.last_tracks
        
        # 3. Check intrusions
        current_intruders = set()
        
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            
            # Get bottom center point (more stable for people)
            point = get_bbox_bottom_center((x1, y1, x2, y2))
            
            # Check if in restricted zone
            in_zone = self.zone_manager.is_point_in_any_zone(point)
            
            # Draw bbox (red if in zone, green otherwise)
            color = (0, 0, 255) if in_zone else (0, 255, 0)
            draw_bbox(frame, (x1, y1, x2, y2), track_id, color)
            
            # Draw tracking point
            cv2.circle(frame, point, 6, color, -1)
            cv2.circle(frame, point, 8, (255, 255, 255), 2)
            
            # Track intruders
            if in_zone:
                current_intruders.add(track_id)
        
        # 4. Update alarm state
        self._update_alarm_state(current_intruders)
        
        # 5. Draw zones
        self.zone_manager.draw_zones(frame)
        
        # 6. Draw alarm if active
        if self.alarm_active:
            draw_alarm(frame, "ПРОНИКНОВЕНИЕ!", (0, 0, 255))
            
            # Show intruder IDs
            intruder_text = f"IDs: {', '.join(map(str, self.intruders_in_zone))}"
            cv2.putText(frame, intruder_text, (frame.shape[1] // 2 - 100, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 7. Draw stats
        self._draw_stats(frame, len(tracks), len(current_intruders), avg_fps)
        
        return frame
    
    def _update_alarm_state(self, current_intruders: set):
        """Update alarm state based on current intruders"""
        if len(current_intruders) > 0:
            if not self.alarm_active:
                self.alarm_active = True
                print(f"\n{'='*60}")
                print(f"🚨 ALARM ACTIVATED!")
                print(f"   Intruders: {current_intruders}")
                print(f"   Frame: {self.frame_count}")
                print(f"{'='*60}")
            
            # Reset timer while intruder is present
            self.alarm_timer.reset()
            self.intruders_in_zone = current_intruders.copy()
        
        else:
            if self.alarm_active:
                # Start cooldown timer
                if not self.alarm_timer.is_running:
                    self.alarm_timer.start()
                    print(f"\n⏱ Cooldown started (3s)...")
                
                # Check if cooldown expired
                if self.alarm_timer.is_expired():
                    self.alarm_active = False
                    self.alarm_timer.stop()
                    print(f"✓ Alarm deactivated\n")
                    self.intruders_in_zone.clear()
    
    def _draw_stats(self, frame: np.ndarray, total_persons: int, 
                    intruders: int, fps: float):
        """Draw statistics on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 280, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw stats
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 270, y_offset),
                   font, 0.7, (0, 255, 0), 2)
        y_offset += 35
        
        # Persons
        cv2.putText(frame, f"Persons: {total_persons}", (w - 270, y_offset),
                   font, 0.7, (255, 255, 255), 2)
        y_offset += 35
        
        # Intruders
        color = (0, 0, 255) if intruders > 0 else (0, 255, 0)
        cv2.putText(frame, f"Intruders: {intruders}", (w - 270, y_offset),
                   font, 0.7, color, 2)
        y_offset += 35
        
        # Zones
        cv2.putText(frame, f"Zones: {len(self.zone_manager.zones)}", (w - 270, y_offset),
                   font, 0.7, (255, 255, 255), 2)
    
    def run(self, video_path: str = None):
        """Run the pipeline on video"""
        video_path = video_path or Config.VIDEO_PATH
        
        print(f"\nOpening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Total frames: {total_frames}")
        
        # Get first frame for zone setup
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
        
        # Setup zones
        if not self.setup_zones(first_frame):
            print("\n⚠ No zones configured. Exiting...")
            return
        
        # Reset video to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_count = 0
        
        print("\n" + "="*60)
        print("INTRUSION DETECTION ACTIVE")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit")
        print("  'p' - Pause/Resume")
        print("  'r' - Restart video")
        print("="*60 + "\n")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("\n↻ End of video - restarting...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_count = 0
                    continue
                
                # Process frame
                processed_frame = self.process_frame(frame)
            
            else:
                # Paused - just show frame
                processed_frame = frame.copy()
                cv2.putText(processed_frame, "PAUSED - Press 'p' to resume", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 255), 2)
            
            # Display
            cv2.imshow(Config.WINDOW_NAME, processed_frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nShutting down...")
                break
            
            elif key == ord('p'):
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"\n{status}")
            
            elif key == ord('r'):
                print("\n↻ Restarting video...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_count = 0
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("Pipeline finished")
        print(f"Total frames processed: {self.frame_count}")
        print("="*60)