"""
Complete Pipeline: Agricultural Person Detection
Integrates: YOLO inference + temporal consistency layer
"""

import argparse
import numpy as np
from typing import List, Dict

# Import components from previous steps
try:
    from .temporal_consistency import TemporalConsistencyFilter, Detection
except ImportError:  # pragma: no cover - fallback for direct script execution
    from agri_perception.pipeline.temporal_consistency import TemporalConsistencyFilter, Detection


class AgriculturalPersonDetectionPipeline:
    """
    Complete inference pipeline for agricultural autonomous safety
    
    Pipeline stages:
    1. YOLO inference (detect person candidates)
    2. Temporal consistency filtering (reject false positives)
    3. Output formatted detections with confidence scores
    """
    
    def __init__(self, 
                 model_path: str = None,
                 motion_thresh: float = 0.35,
                 stability_thresh: float = 0.55,
                 min_yolo_conf: float = 0.25):
        """
        Args:
            model_path: Path to trained YOLO model
            motion_thresh: Threshold for motion consistency (0-1)
            stability_thresh: Threshold for temporal stability (0-1)
            min_yolo_conf: YOLO confidence threshold (recall-focused)
        """
        self.model = None
        self.model_path = model_path
        self.temporal_filter = TemporalConsistencyFilter(
            max_age=15,
            motion_thresh=motion_thresh,
            stability_thresh=stability_thresh
        )
        self.min_yolo_conf = min_yolo_conf
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"✓ Model loaded: {model_path}")
        except ImportError:
            print("⚠️  ultralytics not installed. Install: pip install ultralytics")
            self.model = None
    
    def infer_frame(self, image_path: str) -> List[Dict]:
        """
        Run inference on single frame
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detections with confidence scores
        """
        
        if self.model is None:
            print("⚠️  No model loaded")
            return []
        
        # YOLO inference
        results = self.model.predict(image_path, conf=self.min_yolo_conf)
        
        # Convert to Detection objects
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    det = Detection(
                        class_id=int(box.cls[0]),
                        confidence=float(box.conf[0]),
                        bbox=np.array([float(x) for x in box.xyxy[0]]),
                        frame_id=0
                    )
                    detections.append(det)
        
        # Temporal filtering
        tracked = self.temporal_filter.update(detections)
        high_conf = self.temporal_filter.get_high_confidence_detections(tracked)
        
        # Format output
        output_detections = []
        for det in high_conf:
            output_detections.append({
                'bbox': det.bbox.tolist(),
                'confidence': float(det.confidence),
                'temporal_confidence': float(self._get_track_confidence(det)),
                'class': 'person',
                'status': 'high_confidence'
            })
        
        return output_detections
    
    def _get_track_confidence(self, det: Detection) -> float:
        """Get composite temporal confidence for a detection"""
        # Find the track associated with this detection
        for track_id, track in self.temporal_filter.tracks.items():
            if track.detections and track.detections[-1] == det:
                return track.get_confidence_score()
        return 0.0
    
    def process_video(self, video_path: str, output_path: str = None) -> List[Dict]:
        """
        Process video file and return detections
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save annotated video
            
        Returns:
            List of frame detections
        """
        
        try:
            import cv2
        except ImportError:
            print("⚠️  opencv-python not installed")
            return []
        
        if self.model is None:
            print("⚠️  No model loaded")
            return []
        
        cap = cv2.VideoCapture(video_path)
        frame_detections = []
        frame_count = 0
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output video writer (optional)
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\n🎥 Processing video: {video_path}")
        print(f"   Resolution: {width}x{height} @ {fps} fps")
        print()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # YOLO inference
            results = self.model.predict(frame, conf=self.min_yolo_conf, verbose=False)
            
            # Convert to Detection objects
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        det = Detection(
                            class_id=int(box.cls[0]),
                            confidence=float(box.conf[0]),
                            bbox=np.array([float(x) for x in box.xyxy[0]]),
                            frame_id=frame_count
                        )
                        detections.append(det)
            
            # Temporal filtering
            tracked = self.temporal_filter.update(detections)
            high_conf = self.temporal_filter.get_high_confidence_detections(tracked)
            
            # Store detections
            frame_result = {
                'frame': frame_count,
                'yolo_detections': len(detections),
                'temporal_detections': len(high_conf),
                'detections': [
                    {
                        'bbox': det.bbox.tolist(),
                        'confidence': float(det.confidence),
                        'class': 'person'
                    }
                    for det in high_conf
                ]
            }
            frame_detections.append(frame_result)
            
            # Visualize on frame (optional)
            if out:
                vis_frame = frame.copy()
                for det in high_conf:
                    x1, y1, x2, y2 = [int(x) for x in det.bbox]
                    # Green box for high-confidence
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_frame, f'{det.confidence:.2f}', 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
                
                out.write(vis_frame)
            
            # Progress indicator
            if frame_count % 30 == 0:
                print(f"   Frame {frame_count}: {len(high_conf)} high-confidence persons")
        
        cap.release()
        if out:
            out.release()
            print(f"\n✓ Output video saved: {output_path}")
        
        return frame_detections
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics"""
        
        total_yolo_detections = sum(
            len(track.detections) for track in self.temporal_filter.tracks.values()
        )
        
        return {
            'temporal_filter_state': {
                'active_tracks': len(self.temporal_filter.tracks),
                'total_detections_processed': total_yolo_detections,
                'frame_count': self.temporal_filter.frame_count,
            },
            'hyperparameters': {
                'motion_threshold': self.temporal_filter.motion_thresh,
                'stability_threshold': self.temporal_filter.stability_thresh,
                'min_yolo_confidence': self.min_yolo_conf,
            }
        }


class PipelineEvaluator:
    """Evaluate pipeline performance"""
    
    @staticmethod
    def calculate_metrics(predictions: List[Dict], 
                         ground_truth: List[Dict],
                         iou_threshold: float = 0.5) -> Dict:
        """
        Calculate precision, recall, and F1 for predictions
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth annotations
            iou_threshold: IoU threshold for matching
            
        Returns:
            Metrics dictionary
        """
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Simple IoU-based matching
        matched_gt = set()
        
        for pred in predictions:
            pred_box = np.array(pred['bbox'])
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue
                
                gt_box = np.array(gt['bbox'])
                iou = PipelineEvaluator._bbox_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1
        
        false_negatives = len(ground_truth) - len(matched_gt)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) \
                    if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) \
                if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0
        
        return {
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
        }
    
    @staticmethod
    def _bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes in [x1, y1, x2, y2] format"""
        
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


# Usage guide
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the temporal inference pipeline on images or video."
    )
    parser.add_argument("--model", default="training_artifacts/person_detection/yolo_train_v1/weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--image", help="Path to a single image for frame-by-frame testing")
    parser.add_argument("--video", help="Path to a video for temporal filtering testing")
    parser.add_argument("--output", help="Optional output video path when using --video")
    args = parser.parse_args()

    print("=" * 70)
    print("Agricultural Person Detection Pipeline")
    print("=" * 70)
    print()

    if args.image or args.video:
        pipeline = AgriculturalPersonDetectionPipeline(model_path=args.model)

        if args.image:
            detections = pipeline.infer_frame(args.image)
            print(f"\nImage: {args.image}")
            print(f"High-confidence detections: {len(detections)}")
            for det in detections:
                print(det)

        if args.video:
            frame_results = pipeline.process_video(args.video, output_path=args.output)
            total_yolo = sum(frame['yolo_detections'] for frame in frame_results)
            total_temporal = sum(frame['temporal_detections'] for frame in frame_results)
            print(f"\nVideo: {args.video}")
            print(f"Frames processed: {len(frame_results)}")
            print(f"YOLO detections: {total_yolo}")
            print(f"Temporal detections: {total_temporal}")
            if args.output:
                print(f"Annotated video saved to: {args.output}")
    else:
        print("📋 USAGE EXAMPLES:\n")
        print("1️⃣  Single Frame Inference:")
        print("   python -m agri_perception.pipeline.inference_pipeline --image test_image.jpg")
        print()
        print("2️⃣  Video Processing:")
        print("   python -m agri_perception.pipeline.inference_pipeline --video test_video.mp4 --output output.mp4")
        print()
        print("3️⃣  Custom model path:")
        print("   python -m agri_perception.pipeline.inference_pipeline --model training_artifacts/person_detection/yolo_train_v1/weights/best.pt --video test_video.mp4")
        print()
        print("🔑 KEY FEATURES:")
        print("   ✓ YOLO person detection (recall-optimized)")
        print("   ✓ Temporal consistency filtering (false positive rejection)")
        print("   ✓ Video processing with visualization")
        print("   ✓ Performance metrics and statistics")
        print()
        print("💡 DEPLOYMENT:")
        print("   • Export YOLO to ONNX/TFLite for Jetson Orin")
        print("   • Temporal filter is CPU-efficient (no GPU needed)")
        print("   • Inference latency < 100ms on edge hardware")
        print()

        print("=" * 70)
