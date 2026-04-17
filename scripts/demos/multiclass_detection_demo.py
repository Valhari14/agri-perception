"""Run a COCO-pretrained multi-class detection demonstration on test images."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-class detection demonstration")
    parser.add_argument("--model", default="yolov8m.pt", help="Path to COCO-pretrained YOLO weights")
    parser.add_argument("--images", default="yolo_dataset/images/test", help="Input image directory")
    parser.add_argument("--output-dir", default="demo_outputs/tier2_multiclass", help="Output directory")
    parser.add_argument("--sample-count", type=int, default=5, help="Number of images to visualize")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    test_images = sorted(Path(args.images).glob("*.jpg"))[: args.sample_count]
    if not test_images:
        raise FileNotFoundError(f"No images found in {args.images}")

    color_map = {
        "person": (0, 255, 0),
        "dog": (255, 0, 0),
        "car": (0, 0, 255),
        "bicycle": (255, 255, 0),
    }

    for idx, img_path in enumerate(test_images, start=1):
        result = model.predict(source=str(img_path), conf=args.conf, verbose=False)[0]
        image = cv2.imread(str(img_path))

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names.get(int(cls_id), f"class_{int(cls_id)}")
            color = color_map.get(class_name, (128, 128, 128))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{class_name} {conf:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        out_path = output_dir / f"multiclass_{idx:02d}.jpg"
        cv2.imwrite(str(out_path), image)
        print(f"Saved: {out_path}")

    print(f"Completed multi-class demo. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
