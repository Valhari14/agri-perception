"""Generate severity overlays from detector outputs for portfolio demonstration."""

import argparse
import cv2
import json
import os
from pathlib import Path

from ultralytics import YOLO

# Risk thresholds
THRESHOLDS = {
    "LOW": (0.0, 0.25),      # Green: Safe
    "CAUTION": (0.25, 0.50),  # Yellow: Be careful
    "WARNING": (0.50, 0.75),  # Orange: Reduce speed
    "CRITICAL": (0.75, 1.0),  # Red: Emergency stop
}

# ============================================================================
# STEP 1: Define Severity Scoring Function
# ============================================================================
def compute_severity_score(bbox_height, bbox_width, confidence_score, image_height=640):
    """
    Compute risk score based on:
    1. Target size (smaller = closer = higher risk)
    2. Detection confidence (lower confidence = higher uncertainty)
    
    Formula: risk = (1 - relative_size) × (1 - confidence)
    """
    # Normalize bbox size (0-1, where 1 = full image height)
    relative_size = bbox_height / image_height
    
    # Invert: smaller boxes = higher risk
    size_factor = 1 - relative_size
    
    # Confidence factor: lower confidence = higher risk
    uncertainty_factor = 1 - confidence_score
    
    # Combined risk score
    risk_score = size_factor * uncertainty_factor
    
    return risk_score

def get_risk_band(score):
    """Map score to risk band."""
    for band, (low, high) in THRESHOLDS.items():
        if low <= score < high:
            return band
    return "CRITICAL"

def get_band_color(band):
    """Map risk band to BGR color."""
    colors = {
        "LOW": (0, 255, 0),      # Green
        "CAUTION": (0, 255, 255),  # Yellow
        "WARNING": (0, 140, 255),  # Orange
        "CRITICAL": (0, 0, 255),   # Red
    }
    return colors.get(band, (128, 128, 128))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Severity scoring demonstration")
    parser.add_argument("--model", default="training_artifacts/person_detection/yolo_train_v1/weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--images", default="yolo_dataset/images/test", help="Input image directory")
    parser.add_argument("--output-dir", default="demo_outputs/tier2_severity", help="Output directory")
    parser.add_argument("--sample-count", type=int, default=10, help="Number of images to process")
    parser.add_argument("--conf", type=float, default=0.4, help="Inference confidence threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    test_images = sorted(Path(args.images).glob("*.jpg"))[: args.sample_count]
    if not test_images:
        raise FileNotFoundError(f"No images found in {args.images}")

    severity_results = []

    for idx, img_path in enumerate(test_images, start=1):
        print(f"Processing {idx}/{len(test_images)}: {img_path.name}")
        result = model.predict(source=str(img_path), conf=args.conf, verbose=False)[0]

        image = cv2.imread(str(img_path))
        image_height = image.shape[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        image_detections = []
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            bbox_height = y2 - y1
            bbox_width = x2 - x1
            risk_score = compute_severity_score(bbox_height, bbox_width, float(conf), image_height)
            risk_band = get_risk_band(risk_score)

            image_detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "risk_score": float(risk_score),
                    "risk_band": risk_band,
                }
            )

            color = get_band_color(risk_band)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(image, f"{risk_band} ({risk_score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        legend_y = 30
        for band in ["LOW", "CAUTION", "WARNING", "CRITICAL"]:
            color = get_band_color(band)
            cv2.rectangle(image, (10, legend_y), (30, legend_y + 15), color, -1)
            cv2.putText(image, band, (40, legend_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            legend_y += 20

        cv2.imwrite(os.path.join(output_dir, f"severity_{idx:02d}.jpg"), image)
        severity_results.append(
            {
                "image": img_path.name,
                "detections": image_detections,
                "total_detections": int(len(boxes)),
            }
        )

    with open(os.path.join(output_dir, "severity_scores.json"), "w", encoding="utf-8") as f:
        json.dump(severity_results, f, indent=2)

    all_risk_bands = []
    for result in severity_results:
        for detection in result["detections"]:
            all_risk_bands.append(detection["risk_band"])

    band_counts = {band: all_risk_bands.count(band) for band in ["LOW", "CAUTION", "WARNING", "CRITICAL"]}
    summary = {
        "total_detections": int(sum(r["total_detections"] for r in severity_results)),
        "risk_distribution": band_counts,
        "severity_formula": "risk_score = (1 - relative_bbox_height) * (1 - confidence)",
    }

    with open(os.path.join(output_dir, "severity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Completed severity demo. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
