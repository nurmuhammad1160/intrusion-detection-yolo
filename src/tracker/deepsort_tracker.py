"""
DeepSORT tracker for person tracking with ID persistence
"""
import numpy as np
from typing import List, Tuple
from collections import deque


class Track:
    """Single track object"""
    
    _id_counter = 0
    
    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float):
        """
        Args:
            bbox: (x1, y1, x2, y2)
            confidence: Detection confidence
        """
        Track._id_counter += 1
        self.id = Track._id_counter
        self.bbox = bbox
        self.confidence = confidence
        self.age = 0
        self.hits = 1
        self.history = deque(maxlen=30)
        self.history.append(bbox)
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float):
        """Update track with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.age = 0
        self.history.append(bbox)
    
    def predict(self):
        """Predict next position (simple: use last position)"""
        self.age += 1
        return self.bbox


class DeepSORTTracker:
    """Simplified DeepSORT-like tracker"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits to consider track confirmed
            iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
    
    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Tuple[int, int, int, int, int]]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of [(x1, y1, x2, y2, conf), ...]
        
        Returns:
            List of active tracks: [(x1, y1, x2, y2, track_id), ...]
        """
        if len(detections) == 0:
            for track in self.tracks:
                track.predict()
            
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            
            active_tracks = []
            for track in self.tracks:
                if track.hits >= self.min_hits:
                    x1, y1, x2, y2 = track.bbox
                    active_tracks.append((x1, y1, x2, y2, track.id))
            
            return active_tracks
        
        matched_indices = self._match_detections_to_tracks(detections)
        
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))
        
        for det_idx, track_idx in matched_indices:
            bbox = detections[det_idx][:4]
            conf = detections[det_idx][4]
            self.tracks[track_idx].update(bbox, conf)
            unmatched_detections.discard(det_idx)
            unmatched_tracks.discard(track_idx)
        
        for det_idx in unmatched_detections:
            bbox = detections[det_idx][:4]
            conf = detections[det_idx][4]
            self.tracks.append(Track(bbox, conf))
        
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].predict()
        
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        
        active_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits:
                x1, y1, x2, y2 = track.bbox
                active_tracks.append((x1, y1, x2, y2, track.id))
        
        return active_tracks
    
    def _match_detections_to_tracks(self, detections: List[Tuple]) -> List[Tuple[int, int]]:
        """
        Match detections to existing tracks using IOU
        
        Returns:
            List of (detection_index, track_index) pairs
        """
        if len(self.tracks) == 0:
            return []
        
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        
        for d, det in enumerate(detections):
            for t, track in enumerate(self.tracks):
                iou_matrix[d, t] = self._iou(det[:4], track.bbox)
        
        matched_indices = []
        
        while True:
            max_iou = iou_matrix.max()
            
            if max_iou < self.iou_threshold:
                break
            
            det_idx, track_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            
            matched_indices.append((int(det_idx), int(track_idx)))
            
            iou_matrix[det_idx, :] = -1
            iou_matrix[:, track_idx] = -1
        
        return matched_indices
    
    @staticmethod
    def _iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IOU between two bboxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union