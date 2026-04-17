# Agricultural Person Detection Module

This repository is one part of a larger autonomous safety solution built during the 24-hour HackHPI 2026 hackathon.

It contains the perception module for safety-critical person detection in autonomous agricultural settings. It is not the full autonomous system (for example: planning, controls, and hardware integration are out of scope here).

## What This Repo Covers

- Dataset analysis for small and occluded person targets
- Data preparation and COCO to YOLO conversion
- YOLO training configuration for recall-focused detection
- Inference pipeline with temporal consistency filtering

## Scope In The Full Solution

This module provides perception outputs used by the larger stack. In a full deployment pipeline, these detections are consumed by downstream safety and decision logic.

## Result Snapshot

Internal validation on the challenge setup:

- Precision: 0.966
- Recall: 0.932
- mAP50: 0.955
- mAP50-95: 0.765
- Inference latency: 7.8 ms/image (approximately 125 FPS)

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Run the main pipeline:

```bash
python -m agri_perception.pipeline.inference_pipeline --model training_artifacts/person_detection/yolo_train_v1/weights/best.pt --video path/to/video.mp4 --output output.mp4
```



## Public Repository Note

Private challenge data is intentionally excluded from this repository.


