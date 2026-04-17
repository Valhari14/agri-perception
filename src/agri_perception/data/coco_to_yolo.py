"""
Convert HackHPI COCO annotations into YOLO dataset format.

Output layout:
  yolo_dataset/
    images/{train,val,test}
    labels/{train,val,test}
  yolo_data.yaml

Notes:
- Every copied image gets a corresponding .txt label file.
- Background images intentionally get empty .txt files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def coco_bbox_to_yolo(bbox: list[float], width: float, height: float) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    xc = (x + w / 2.0) / width
    yc = (y + h / 2.0) / height
    wn = w / width
    hn = h / height
    return xc, yc, wn, hn


def safe_rel_name(recording_name: str, file_name: str) -> str:
    # Keep a deterministic, unique base name across recordings.
    return f"{recording_name}__{file_name}"


def build_dataset(root: Path, train_ratio: float, val_ratio: float, seed: int) -> dict[str, int]:
    ann_root = root / "annotation"
    data_root = root / "data"
    out_root = root / "yolo_dataset"

    if not ann_root.exists():
        raise FileNotFoundError(f"Missing annotation directory: {ann_root}")
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data directory: {data_root}")

    if out_root.exists():
        shutil.rmtree(out_root)

    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    json_files = sorted(ann_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON annotation files found under {ann_root}")

    counts = {
        "json_files": len(json_files),
        "images_total": 0,
        "labels_total": 0,
        "images_missing_on_disk": 0,
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "train_empty_labels": 0,
        "val_empty_labels": 0,
        "test_empty_labels": 0,
    }

    # Cache dataset-level filename indexes for robust path resolution.
    file_index_cache: dict[str, dict[str, list[Path]]] = {}

    def get_dataset_file_index(dataset_name: str) -> dict[str, list[Path]]:
        if dataset_name in file_index_cache:
            return file_index_cache[dataset_name]
        dataset_dir = data_root / dataset_name
        index: dict[str, list[Path]] = defaultdict(list)
        if dataset_dir.exists():
            for p in dataset_dir.rglob("*"):
                if p.is_file():
                    index[p.name].append(p)
        file_index_cache[dataset_name] = index
        return index

    def resolve_recording_dir(dataset_dir: Path, recording_name: str) -> Path:
        # Prefer exact folder match, then common suffix-normalized match.
        exact = dataset_dir / recording_name
        if exact.exists() and exact.is_dir():
            return exact

        base_name = recording_name[:-3] if recording_name.endswith("_11") else recording_name
        alt = dataset_dir / base_name
        if alt.exists() and alt.is_dir():
            return alt

        # Fall back to the dataset root; file index lookup will pick precise files.
        return dataset_dir

    for jf in json_files:
        with jf.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        dataset_name = jf.parent.name
        recording_name = jf.stem
        dataset_dir = data_root / dataset_name
        recording_img_dir = resolve_recording_dir(dataset_dir, recording_name)
        dataset_file_index = get_dataset_file_index(dataset_name)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])

        ann_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in annotations:
            ann_by_image[ann["image_id"]].append(ann)

        indices = np.arange(len(images))
        rng.shuffle(indices)

        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)

        split_map: dict[int, str] = {}
        for idx in indices[:n_train]:
            split_map[int(idx)] = "train"
        for idx in indices[n_train:n_train + n_val]:
            split_map[int(idx)] = "val"
        for idx in indices[n_train + n_val:]:
            split_map[int(idx)] = "test"

        for i, img in enumerate(images):
            split = split_map[i]
            src = recording_img_dir / img["file_name"]
            if not src.exists():
                candidates = dataset_file_index.get(img["file_name"], [])
                if not candidates:
                    counts["images_missing_on_disk"] += 1
                    continue

                # Prefer file from the matching recording folder when multiple candidates exist.
                preferred = None
                for c in candidates:
                    if recording_name in c.parts or recording_name[:-3] in c.parts:
                        preferred = c
                        break
                src = preferred or candidates[0]

            unique_name = safe_rel_name(recording_name, img["file_name"])
            dst_img = out_root / "images" / split / unique_name
            dst_lbl = out_root / "labels" / split / f"{Path(unique_name).stem}.txt"

            shutil.copy2(src, dst_img)

            img_anns = ann_by_image.get(img["id"], [])
            with dst_lbl.open("w", encoding="utf-8") as lf:
                for ann in img_anns:
                    # All human annotations are mapped to one YOLO class: person (class 0).
                    xc, yc, wn, hn = coco_bbox_to_yolo(ann["bbox"], img["width"], img["height"])
                    lf.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

            counts["images_total"] += 1
            counts["labels_total"] += len(img_anns)
            counts[f"{split}_images"] += 1
            if len(img_anns) == 0:
                counts[f"{split}_empty_labels"] += 1

    yaml_path = root / "yolo_data.yaml"
    yaml_content = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["person"],
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f, sort_keys=False)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO dataset format.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counts = build_dataset(args.root, args.train_ratio, args.val_ratio, args.seed)

    print("COCO -> YOLO conversion completed")
    print(f"Annotation files: {counts['json_files']}")
    print(f"Converted images: {counts['images_total']}")
    print(f"Total boxes: {counts['labels_total']}")
    print(f"Missing source images: {counts['images_missing_on_disk']}")
    print(
        "Split images: "
        f"train={counts['train_images']} val={counts['val_images']} test={counts['test_images']}"
    )
    print(
        "Empty label files (backgrounds): "
        f"train={counts['train_empty_labels']} val={counts['val_empty_labels']} test={counts['test_empty_labels']}"
    )
    print(f"Wrote YAML: {args.root.resolve() / 'yolo_data.yaml'}")


if __name__ == "__main__":
    main()
