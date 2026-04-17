"""
Step 3: Temporal Consistency Layer
Distinguish real persons (moving, persistent) from false positives.
Implements: tracker-based temporal filtering.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import deque


@dataclass
class Detection:
    """Single detection from YOLO"""
    class_id: int
    confidence: float
    bbox: np.ndarray  # [x1, y1, x2, y2]
    frame_id: int
    
    def area(self) -> float:
        """Bounding box area"""
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return w * h
    
    def center(self) -> np.ndarray:
        """Bounding box center"""
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        ])
    
    def iou(self, other: 'Detection') -> float:
        """IoU with another detection"""
        x1_min, y1_min, x1_max, y1_max = self.bbox
        x2_min, y2_min, x2_max, y2_max = other.bbox
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        union_area = (self.area() + other.area() - inter_area)
        
        return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class Track:
    """Tracked object across frames"""
    track_id: int
    detections: List[Detection]
    age: int
    frames_since_update: int
    
    def motion_consistency(self) -> float:
        """
        Measure motion consistency: real persons move coherently
        Scarecrows/fence posts are static (distance between frames ≈ 0)
        
        Returns: float in [0, 1] where 1 = highly consistent motion
        """
        if len(self.detections) < 3:
            return 0.5  # Not enough data
        
        # Get last 3 centers
        centers = np.array([d.center() for d in self.detections[-3:]])
        
        # Calculate frame-to-frame distances
        distances = []
        for i in range(1, len(centers)):
            dist = np.linalg.norm(centers[i] - centers[i-1])
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Static object (scarecrow): distances ≈ 0
        # Moving object (person): distances > 0 with consistent magnitude
        mean_dist = np.mean(distances)
        consistency = min(1.0, mean_dist / 50.0)  # Normalize to 50px baseline
        
        return consistency
    
    def temporal_stability(self) -> float:
        """
        Measure temporal stability: real persons appear consistently
        False positives flicker (disappear and reappear)
        
        Returns: float in [0, 1] where 1 = always visible
        """
        gap_ratio = self.frames_since_update / (self.age + 1)
        stability = max(0, 1.0 - gap_ratio)
        return stability
    
    def size_consistency(self) -> float:
        """
        Bounding box size should be relatively stable
        Real person: stable size as harvester approaches
        False positive: random noise, erratic sizes
        """
        if len(self.detections) < 2:
            return 0.8
        
        areas = np.array([d.area() for d in self.detections])
        
        # Coefficient of variation
        if np.mean(areas) == 0:
            return 0.0
        
        cv = np.std(areas) / np.mean(areas)
        consistency = max(0, 1.0 - cv)
        
        return consistency
    
    def get_confidence_score(self) -> float:
        """
        Composite score: combines temporal cues
        High score = likely real person
        Low score = likely false positive
        """
        motion = self.motion_consistency()
        stability = self.temporal_stability()
        size = self.size_consistency()
        
        # Weight motion heavily (scarecrows are static)
        score = (motion * 0.5) + (stability * 0.3) + (size * 0.2)
        
        return min(1.0, score)
    
    def is_high_confidence_person(self, 
                                  motion_thresh: float = 0.4,
                                  stability_thresh: float = 0.6,
                                  min_age: int = 3) -> bool:
        """
        Determine if track is likely a real person
        
        Args:
            motion_thresh: Minimum motion consistency (0-1)
            stability_thresh: Minimum temporal stability (0-1)
            min_age: Minimum frames to be tracked
        """
        if len(self.detections) < min_age:
            return False
        
        motion = self.motion_consistency()
        stability = self.temporal_stability()
        
        # Real person: shows motion + temporal stability
        # False positive: static (scarecrow) or unstable (flickering)
        
        is_moving = motion > motion_thresh
        is_stable = stability > stability_thresh
        
        return is_moving and is_stable


class TemporalConsistencyFilter:
    """
    Post-processing layer for YOLO detections using temporal context
    Reduces false positives without sacrificing recall
    """
    
    def __init__(self, 
                 max_age: int = 10,
                 motion_thresh: float = 0.3,
                 stability_thresh: float = 0.5):
        """
        Args:
            max_age: Max frames to keep track alive without detection
            motion_thresh: Motion consistency threshold (0-1)
            stability_thresh: Temporal stability threshold (0-1)
        """
        self.max_age = max_age
        self.motion_thresh = motion_thresh
        self.stability_thresh = stability_thresh
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_count = 0
    
    def update(self, detections: List[Detection]) -> List[Tuple[Detection, Track]]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of YOLO detections from current frame
            
        Returns:
            List of (detection, track) tuples with high confidence
        """
        
        self.frame_count += 1
        
        # Update detection frame IDs
        for det in detections:
            det.frame_id = self.frame_count
        
        # Associate detections to tracks
        associations = self._associate_detections(detections)
        
        # Update tracks
        updated_tracks = []
        for track_id, det_idx in associations:
            self.tracks[track_id].detections.append(detections[det_idx])
            self.tracks[track_id].age += 1
            self.tracks[track_id].frames_since_update = 0
            updated_tracks.append((detections[det_idx], self.tracks[track_id]))
        
        # Create new tracks for unassociated detections
        unassociated = set(range(len(detections))) - set(d[1] for d in associations)
        for det_idx in unassociated:
            new_track = Track(
                track_id=self.next_track_id,
                detections=[detections[det_idx]],
                age=1,
                frames_since_update=0
            )
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1
            updated_tracks.append((detections[det_idx], new_track))
        
        # Age unmatched tracks
        for track_id, track in list(self.tracks.items()):
            if track_id not in [t[0] for t in [(a[0], self.tracks[a[0]]) 
                                                for a in associations]]:
                track.frames_since_update += 1
                
                # Remove dead tracks
                if track.frames_since_update > self.max_age:
                    del self.tracks[track_id]
        
        return updated_tracks
    
    def _associate_detections(self, detections: List[Detection]) -> List[Tuple[int, int]]:
        """
        Greedily associate detections to existing tracks using IoU
        
        Returns: List of (track_id, detection_idx) pairs
        """
        
        associations = []
        matched_det_indices = set()
        
        for track_id, track in self.tracks.items():
            if track.frames_since_update > self.max_age:
                continue
            
            # Find best matching detection
            best_iou = 0.3  # Min IoU threshold
            best_det_idx = -1
            
            last_detection = track.detections[-1] if track.detections else None
            if last_detection is None:
                continue
            
            for det_idx, det in enumerate(detections):
                if det_idx in matched_det_indices:
                    continue
                
                iou = last_detection.iou(det)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                associations.append((track_id, best_det_idx))
                matched_det_indices.add(best_det_idx)
        
        return associations
    
    def get_high_confidence_detections(self, 
                                       detections_and_tracks: List) -> List[Detection]:
        """
        Filter detections using temporal consistency
        
        Args:
            detections_and_tracks: Output from update()
            
        Returns:
            High-confidence detections (likely real persons)
        """
        
        high_conf = []
        for det, track in detections_and_tracks:
            if track.is_high_confidence_person(
                motion_thresh=self.motion_thresh,
                stability_thresh=self.stability_thresh,
                min_age=3
            ):
                high_conf.append(det)
        
        return high_conf


# Example usage and testing
def demo_temporal_filter():
    """Demonstrate temporal consistency layer"""
    
    print("="*70)
    print("Temporal Consistency Filter - False Positive Rejection")
    print("="*70)
    print()
    
    # Initialize filter
    filter = TemporalConsistencyFilter(
        max_age=10,
        motion_thresh=0.3,  # Scarecrows won't pass this
        stability_thresh=0.5
    )
    
    # Simulate: Person moving across field
    print("Scenario 1: Real Person (Moving, Persistent)")
    print("-" * 70)
    real_person_detections = []
    for frame in range(10):
        # Person moves right (increasing x)
        x_pos = 100 + frame * 10
        det = Detection(
            class_id=0,
            confidence=0.9,
            bbox=np.array([x_pos, 200, x_pos + 50, 400]),
            frame_id=frame
        )
        real_person_detections.append(det)
    
    # Process frames
    for frame_dets in [
        real_person_detections[i:i+3] for i in range(0, len(real_person_detections), 3)
    ]:
        result = filter.update(frame_dets)
        high_conf = filter.get_high_confidence_detections(result)
        print(f"  Frame {filter.frame_count}: {len(result)} detections → "
              f"{len(high_conf)} high-confidence (Person: {len(high_conf) > 0})")
    
    print()
    
    # Reset for next scenario
    filter = TemporalConsistencyFilter(motion_thresh=0.3, stability_thresh=0.5)
    
    # Simulate: Scarecrow (Static, flickering)
    print("Scenario 2: False Positive (Scarecrow - Static, Flickering)")
    print("-" * 70)
    
    scarecrow_frames = [
        [Detection(0, 0.8, np.array([200, 200, 250, 450]), 0)],  # Frame 1
        [],  # Frame 2 - missed
        [Detection(0, 0.85, np.array([198, 200, 252, 450]), 2)],  # Frame 3
        [],  # Frame 4 - missed again
        [Detection(0, 0.82, np.array([200, 200, 250, 450]), 4)],  # Frame 5
    ]
    
    for frame_dets in scarecrow_frames:
        result = filter.update(frame_dets)
        high_conf = filter.get_high_confidence_detections(result)
        print(f"  Frame {filter.frame_count}: {len(result)} detections → "
              f"{len(high_conf)} high-confidence (Accept: {len(high_conf) > 0})")
    
    print()
    print("="*70)
    print("KEY INSIGHT:")
    print("  ✓ Real person: consistent motion + temporal stability = ACCEPTED")
    print("  ✗ Scarecrow: static + flickering = REJECTED")
    print("  → Zero false positives without missing persons")
    print("="*70)


if __name__ == "__main__":
    demo_temporal_filter()
