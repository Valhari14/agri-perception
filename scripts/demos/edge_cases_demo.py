"""Edge-case quantitative analysis for YOLO label outputs.

Supports two tagging modes:
1) JSON map file: {"image_name.jpg": "condition", ...}
2) Folder structure under edge_cases/: edge_cases/<condition>/<image files>

Outputs:
- edge_case_metrics.csv
- edge_case_metrics.json
- edge_case_comparison.html
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


KNOWN_CONDITIONS = ["tiny_target", "occlusion", "glare", "motion_blur", "normal"]


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def normalize_condition(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float):
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return (x1, y1, x2, y2)


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_yolo_boxes(label_file: Path):
    boxes = []
    if not label_file.exists() or label_file.stat().st_size == 0:
        return boxes

    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue
            boxes.append(yolo_to_xyxy(xc, yc, w, h))
    return boxes


def greedy_match(gt_boxes, pred_boxes, iou_threshold: float):
    pairs = []
    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(pred_boxes):
            iou = box_iou(g, p)
            if iou >= iou_threshold:
                pairs.append((iou, gi, pi))

    pairs.sort(reverse=True, key=lambda x: x[0])
    used_gt = set()
    used_pred = set()
    matched_ious = []

    for iou, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matched_ious.append(iou)

    tp = len(matched_ious)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    mean_iou = sum(matched_ious) / tp if tp else 0.0
    return tp, fp, fn, mean_iou, tp


def load_tags(tags_file: Path, edge_cases_dir: Path):
    if tags_file.exists():
        with open(tags_file, "r", encoding="utf-8") as f:
            tags = json.load(f)
        normalized = {k: normalize_condition(v) for k, v in tags.items()}
        return normalized, "json"

    tags = {}
    if edge_cases_dir.exists():
        for cond_dir in sorted(edge_cases_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            condition = normalize_condition(cond_dir.name)
            for file_path in cond_dir.iterdir():
                if not file_path.is_file():
                    continue
                tags[file_path.name] = condition
    return tags, "folders"


def load_baseline_map(baseline_metrics_file: Path):
    if not baseline_metrics_file.exists():
        return None, None

    with open(baseline_metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    baseline = data.get("baseline_test_metrics", {})
    map50 = baseline.get("mAP50")
    map50_95 = baseline.get("mAP50-95")
    return map50, map50_95


def build_gt_label_index(gt_labels_dir: Path):
    stems = []
    stem_to_paths = {}
    if not gt_labels_dir.exists():
        return stems, stem_to_paths

    label_files = list(gt_labels_dir.glob("*.txt"))
    label_files.extend(gt_labels_dir.glob("**/*.txt"))

    seen = set()
    deduped = []
    for p in label_files:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(p)

    for label_file in deduped:
        stem = label_file.stem
        stems.append(stem)
        stem_to_paths.setdefault(stem, []).append(label_file)
    return stems, stem_to_paths


def resolve_gt_label_path(img_name: str, gt_stems, stem_to_paths):
    stem = Path(img_name).stem

    # 1) Direct match
    if stem in stem_to_paths and len(stem_to_paths[stem]) == 1:
        return stem_to_paths[stem][0], "direct"

    # 2) Unique suffix match (common when edge case images drop date prefix)
    suffix_hits = [s for s in gt_stems if s.endswith(stem)]
    if len(suffix_hits) == 1:
        return stem_to_paths[suffix_hits[0]][0], "suffix"

    # 3) Unique substring match as fallback
    contains_hits = [s for s in gt_stems if stem in s]
    if len(contains_hits) == 1:
        return stem_to_paths[contains_hits[0]][0], "substring"

    # Ambiguous or unresolved
    return None, "unresolved"


def extract_frame_token(stem: str):
    m = re.search(r"(\d{10})", stem)
    return m.group(1) if m else None


def build_suggestions(unresolved_img_names, gt_stems):
    suggestions = {}
    for img_name in unresolved_img_names:
        stem = Path(img_name).stem
        token = extract_frame_token(stem)
        candidates = []
        if token:
            candidates = [s for s in gt_stems if f"__{token}_" in s]
        suggestions[img_name] = {
            "frame_token": token,
            "candidate_label_stems": candidates[:20],
        }
    return suggestions


def label_name_to_image_name(label_name: str) -> str:
    return f"{Path(label_name).stem}.jpg"


def main():
    parser = argparse.ArgumentParser(description="Edge-case metrics by condition")
    parser.add_argument("--tags-file", default="demo_outputs/edge_case_tags.json")
    parser.add_argument("--edge-cases-dir", default="edge_cases")
    parser.add_argument("--test-labels-dir", default="yolo_dataset/labels")
    parser.add_argument("--pred-labels-dir", default="training_artifacts/person_detection/yolo_train_v1/metrics/val_labels")
    parser.add_argument("--output-dir", default="demo_outputs")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--baseline-metrics", default="demo_outputs/tier1_metrics/metrics_summary.json")
    args = parser.parse_args()

    tags_file = Path(args.tags_file)
    edge_cases_dir = Path(args.edge_cases_dir)
    gt_labels_dir = Path(args.test_labels_dir)
    pred_labels_dir = Path(args.pred_labels_dir)
    output_dir = Path(args.output_dir)
    baseline_metrics_file = Path(args.baseline_metrics)

    output_dir.mkdir(parents=True, exist_ok=True)

    edge_case_tags, source = load_tags(tags_file, edge_cases_dir)
    if not edge_case_tags:
        print("No edge-case tags found.")
        print(f"Checked: {tags_file} and {edge_cases_dir}")
        raise SystemExit(1)

    discovered = sorted({normalize_condition(v) for v in edge_case_tags.values()})
    condition_order = [c for c in KNOWN_CONDITIONS if c in discovered] + [
        c for c in discovered if c not in KNOWN_CONDITIONS
    ]

    metrics_by_condition = {
        cond: {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "total_gt": 0,
            "images": 0,
            "iou_sum": 0.0,
            "iou_matches": 0,
        }
        for cond in condition_order
    }

    gt_stems, stem_to_paths = build_gt_label_index(gt_labels_dir)
    if not gt_stems:
        print(f"No ground-truth labels found in: {gt_labels_dir}")
        raise SystemExit(1)

    processed_images = 0
    missing_label_files = 0
    resolution_stats = {"direct": 0, "suffix": 0, "substring": 0, "unresolved": 0}
    unresolved_examples = []
    unresolved_by_condition = {}
    resolved_tags = {}
    for img_name, condition in edge_case_tags.items():
        condition = normalize_condition(condition)
        if condition not in metrics_by_condition:
            continue

        gt_file, mode = resolve_gt_label_path(img_name, gt_stems, stem_to_paths)
        resolution_stats[mode] += 1
        if gt_file is None:
            unresolved_examples.append(img_name)
            unresolved_by_condition[img_name] = condition
            missing_label_files += 1
            continue

        label_name = gt_file.name
        resolved_tags[label_name_to_image_name(label_name)] = condition
        pred_file = pred_labels_dir / label_name

        if not gt_file.exists():
            missing_label_files += 1
            continue

        gt_boxes = parse_yolo_boxes(gt_file)
        pred_boxes = parse_yolo_boxes(pred_file)

        tp, fp, fn, mean_iou, matched = greedy_match(gt_boxes, pred_boxes, args.iou_threshold)

        m = metrics_by_condition[condition]
        m["TP"] += tp
        m["FP"] += fp
        m["FN"] += fn
        m["total_gt"] += len(gt_boxes)
        m["images"] += 1
        m["iou_sum"] += mean_iou * matched
        m["iou_matches"] += matched
        processed_images += 1

    map50, map50_95 = load_baseline_map(baseline_metrics_file)

    rows = []
    for condition in condition_order:
        m = metrics_by_condition[condition]
        precision = safe_div(m["TP"], m["TP"] + m["FP"])
        recall = safe_div(m["TP"], m["TP"] + m["FN"])
        f1 = safe_div(2 * precision * recall, precision + recall)
        mean_iou = safe_div(m["iou_sum"], m["iou_matches"])

        rows.append(
            {
                "condition": condition,
                "images": m["images"],
                "TP": m["TP"],
                "FP": m["FP"],
                "FN": m["FN"],
                "total_gt": m["total_gt"],
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "mean_iou": round(mean_iou, 4),
                "map50_global": map50,
                "map50_95_global": map50_95,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        print("\nEdge-case metrics:\n")
        print(df.to_string(index=False))
    else:
        print("No rows produced. Check tags and label paths.")
        raise SystemExit(1)

    csv_path = output_dir / "edge_case_metrics.csv"
    df.to_csv(csv_path, index=False)

    total_tp = int(df["TP"].sum())
    total_fp = int(df["FP"].sum())
    total_fn = int(df["FN"].sum())
    overall_precision = safe_div(total_tp, total_tp + total_fp)
    overall_recall = safe_div(total_tp, total_tp + total_fn)
    overall_f1 = safe_div(2 * overall_precision * overall_recall, overall_precision + overall_recall)

    weighted_iou_num = 0.0
    weighted_iou_den = 0
    for condition in condition_order:
        row = df[df["condition"] == condition].iloc[0]
        matches = int(metrics_by_condition[condition]["iou_matches"])
        weighted_iou_num += float(row["mean_iou"]) * matches
        weighted_iou_den += matches
    overall_iou = safe_div(weighted_iou_num, weighted_iou_den)

    summary = {
        "tag_source": source,
        "processed_images": processed_images,
        "skipped_missing_ground_truth_labels": missing_label_files,
        "name_resolution": resolution_stats,
        "unresolved_examples": unresolved_examples[:10],
        "iou_threshold": args.iou_threshold,
        "overall": {
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "f1": round(overall_f1, 4),
            "mean_iou": round(overall_iou, 4),
            "map50_global": map50,
            "map50_95_global": map50_95,
        },
    }

    json_path = output_dir / "edge_case_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    html_rows = []
    for _, row in df.iterrows():
        html_rows.append(
            "<tr>"
            f"<td><strong>{row['condition']}</strong></td>"
            f"<td>{int(row['images'])}</td>"
            f"<td>{float(row['precision']):.2%}</td>"
            f"<td>{float(row['recall']):.2%}</td>"
            f"<td>{float(row['f1']):.2%}</td>"
            f"<td>{float(row['mean_iou']):.3f}</td>"
            f"<td>{int(row['TP'])}</td>"
            f"<td>{int(row['FP'])}</td>"
            f"<td>{int(row['FN'])}</td>"
            "</tr>"
        )

    map50_text = "N/A" if map50 is None else f"{float(map50):.2%}"
    map50_95_text = "N/A" if map50_95 is None else f"{float(map50_95):.2%}"

    html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\">
    <title>Edge-Case Performance Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f6f7f9; color: #212529; }}
        h1, h2 {{ margin-bottom: 8px; }}
        .card {{ background: #ffffff; border: 1px solid #dde2e7; border-radius: 8px; padding: 18px; margin: 16px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #e8ecf0; }}
        th {{ background: #1f2937; color: #ffffff; }}
    </style>
</head>
<body>
    <h1>Edge-Case Performance Analysis</h1>
    <p>Tag source: {source}. IoU threshold: {args.iou_threshold}.</p>

    <div class=\"card\">
        <h2>Overall Metrics</h2>
        <p>Precision: {overall_precision:.2%} | Recall: {overall_recall:.2%} | F1: {overall_f1:.2%} | Mean IoU: {overall_iou:.3f}</p>
        <p>Global mAP50: {map50_text} | Global mAP50-95: {map50_95_text}</p>
    </div>

    <div class=\"card\">
        <h2>Per-Condition Metrics</h2>
        <table>
            <tr>
                <th>Condition</th>
                <th>Images</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>Mean IoU</th>
                <th>TP</th>
                <th>FP</th>
                <th>FN</th>
            </tr>
            {''.join(html_rows)}
        </table>
    </div>

    <div class=\"card\">
        <h2>Notes</h2>
        <ul>
            <li>mAP is loaded from baseline metrics if available and is reported as global.</li>
            <li>Per-condition values in this report are precision, recall, F1, and mean IoU.</li>
            <li>Missing ground-truth labels skipped: {missing_label_files}</li>
        </ul>
    </div>
</body>
</html>
"""

    html_path = output_dir / "edge_case_comparison.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_report)

    suggestions = build_suggestions(unresolved_examples, gt_stems)
    suggestions_path = output_dir / "edge_case_tag_suggestions.json"
    with open(suggestions_path, "w", encoding="utf-8") as f:
        json.dump(suggestions, f, indent=2)

    auto_tags_path = output_dir / "edge_case_tags_autoresolved.json"
    with open(auto_tags_path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(resolved_tags.items())), f, indent=2)

    unresolved_todo = {}
    for name in unresolved_examples:
        unresolved_todo[name] = {
            "condition": unresolved_by_condition.get(name),
            "candidate_label_stems": suggestions.get(name, {}).get("candidate_label_stems", []),
        }

    unresolved_todo_path = output_dir / "edge_case_tags_todo.json"
    with open(unresolved_todo_path, "w", encoding="utf-8") as f:
        json.dump(unresolved_todo, f, indent=2)

    print(f"Name resolution: {resolution_stats}")
    if unresolved_examples:
        print("Unresolved edge-case files (up to 10):")
        for name in unresolved_examples[:10]:
            print(f"  - {name}")
        print(f"Suggestions file: {suggestions_path}")
    print(f"Auto-resolved tags: {auto_tags_path}")
    print(f"Unresolved TODO tags: {unresolved_todo_path}")

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved HTML: {html_path}")


if __name__ == "__main__":
    main()
