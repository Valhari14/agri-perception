"""Dataset analysis utility for agricultural person-detection data."""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

def analyze_coco_dataset(annotation_dir):
    """Analyze COCO format annotations for class imbalance and edge cases"""
    
    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'images_with_persons': 0,
        'images_without_persons': 0,
        'category_counts': defaultdict(int),
        'bbox_sizes': [],
        'occlusion_cases': {
            'tiny': 0,  # bbox area < 5000 pixels (partial occlusion)
            'small': 0,  # 5000-15000
            'medium': 0,  # 15000-50000
            'large': 0   # > 50000 pixels
        },
        'files_processed': [],
        'errors': []
    }
    
    # Find all JSON annotation files
    annotation_path = Path(annotation_dir)
    json_files = sorted(annotation_path.rglob('*.json'))
    
    if not json_files:
        print(f"⚠️  No JSON files found in {annotation_dir}")
        return stats
    
    print(f"Found {len(json_files)} annotation files\n")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                coco_data = json.load(f)
            
            dataset_name = json_file.parent.name
            stats['files_processed'].append(str(json_file))
            
            # Count images and annotations
            num_images = len(coco_data.get('images', []))
            num_annotations = len(coco_data.get('annotations', []))
            categories = {c['id']: c['name'] for c in coco_data.get('categories', [])}
            
            stats['total_images'] += num_images
            stats['total_annotations'] += num_annotations
            
            print(f"📊 {dataset_name}")
            print(f"   Images: {num_images} | Annotations: {num_annotations}")
            
            # Track images with/without persons
            images_with_ann = set()
            for ann in coco_data.get('annotations', []):
                img_id = ann['image_id']
                images_with_ann.add(img_id)
                
                cat_id = ann['category_id']
                cat_name = categories.get(cat_id, f'unknown_{cat_id}')
                stats['category_counts'][cat_name] += 1
                
                # Analyze bbox size for occlusion
                bbox = ann.get('bbox', [0, 0, 0, 0])
                area = ann.get('area', bbox[2] * bbox[3])
                stats['bbox_sizes'].append(area)
                
                if area < 5000:
                    stats['occlusion_cases']['tiny'] += 1
                elif area < 15000:
                    stats['occlusion_cases']['small'] += 1
                elif area < 50000:
                    stats['occlusion_cases']['medium'] += 1
                else:
                    stats['occlusion_cases']['large'] += 1
            
            images_with_persons = len(images_with_ann)
            images_without_persons = num_images - images_with_persons
            
            stats['images_with_persons'] += images_with_persons
            stats['images_without_persons'] += images_without_persons
            
            print(f"   With persons: {images_with_persons} ({100*images_with_persons/num_images:.1f}%)")
            print(f"   Without persons: {images_without_persons} ({100*images_without_persons/num_images:.1f}%)")
            print()
            
        except Exception as e:
            error_msg = f"Error processing {json_file}: {e}"
            print(f"⚠️  {error_msg}")
            stats['errors'].append(error_msg)
    
    return stats

def print_analysis_report(stats):
    """Print formatted analysis report"""
    
    print("\n" + "="*70)
    print("DATASET ANALYSIS REPORT")
    print("="*70)
    
    print(f"\n📈 OVERALL STATISTICS")
    print(f"  Total Images: {stats['total_images']}")
    print(f"  Total Annotations: {stats['total_annotations']}")
    print(f"  Total Images with Persons: {stats['images_with_persons']}")
    print(f"  Total Images without Persons: {stats['images_without_persons']}")
    
    # Class imbalance ratio
    if stats['total_images'] > 0:
        ratio = stats['images_without_persons'] / max(stats['images_with_persons'], 1)
        print(f"\n⚠️  CLASS IMBALANCE RATIO: {ratio:.1f}:1 (non-person : person)")
        print(f"    Person frames: {100*stats['images_with_persons']/stats['total_images']:.2f}%")
        if ratio > 100:
            print(f"    ⚠️  EXTREME IMBALANCE - This is your training challenge!")
    
    # Category breakdown
    print(f"\n🏷️  CATEGORY BREAKDOWN")
    for cat, count in sorted(stats['category_counts'].items(), key=lambda x: -x[1]):
        pct = 100 * count / max(stats['total_annotations'], 1)
        print(f"  {cat}: {count} annotations ({pct:.1f}%)")
    
    # Occlusion analysis
    print(f"\n👥 OBJECT SIZE ANALYSIS (Occlusion Risk)")
    total_bbox = sum(stats['occlusion_cases'].values())
    if total_bbox > 0:
        for size_cat, count in sorted(stats['occlusion_cases'].items()):
            pct = 100 * count / total_bbox
            bar = "█" * int(pct/2) + "░" * (50 - int(pct/2))
            print(f"  {size_cat:7s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Bbox statistics
    if stats['bbox_sizes']:
        arr = np.array(stats['bbox_sizes'])
        print(f"\n📏 BOUNDING BOX STATISTICS")
        print(f"  Mean area: {np.mean(arr):,.0f} px²")
        print(f"  Median area: {np.median(arr):,.0f} px²")
        print(f"  Min area: {np.min(arr):,.0f} px²")
        print(f"  Max area: {np.max(arr):,.0f} px²")
        print(f"  Std dev: {np.std(arr):,.0f} px²")
        print(f"  25th percentile: {np.percentile(arr, 25):,.0f} px²")
        print(f"  75th percentile: {np.percentile(arr, 75):,.0f} px²")
    
    # Recommendations
    print(f"\n💡 TRAINING RECOMMENDATIONS")
    if stats['images_with_persons'] / stats['total_images'] < 0.05:
        print(f"  ✓ Use HEAVY class weighting (person loss weight: 5-10x)")
        print(f"  ✓ Consider focal loss or OHEM (Online Hard Example Mining)")
        print(f"  ✓ Optimize for RECALL over precision (false negatives = critical)")
    
    if stats['occlusion_cases']['tiny'] + stats['occlusion_cases']['small'] > total_bbox * 0.3:
        print(f"  ✓ Heavy occlusion present - use augmentation:")
        print(f"    - Random crop to simulate partial visibility")
        print(f"    - Zoom (scale) augmentation")
        print(f"    - Motion blur to simulate harvester movement")
    
    print(f"  ✓ Augmentation strategy (Albumentations):")
    print(f"    - Gaussian blur + dust simulation (agricultural noise)")
    print(f"    - Random brightness/contrast (lighting variations)")
    print(f"    - Affine transforms (harvester motion)")
    
    print(f"\n📁 FILES PROCESSED: {len(stats['files_processed'])}")
    for fpath in stats['files_processed']:
        print(f"  ✓ {fpath}")
    
    if stats['errors']:
        print(f"\n⚠️  ERRORS: {len(stats['errors'])}")
        for err in stats['errors']:
            print(f"  - {err}")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze COCO annotation files and summarize dataset characteristics."
    )
    parser.add_argument(
        "--annotation-dir",
        default="annotation",
        help="Path to directory containing COCO JSON annotations.",
    )
    parser.add_argument(
        "--output",
        default="dataset_stats.json",
        help="Output JSON file for summarized statistics.",
    )
    args = parser.parse_args()

    annotation_dir = Path(args.annotation_dir)
    output_file = Path(args.output)

    print("🔍 Analyzing agricultural person-detection dataset...\n")
    stats = analyze_coco_dataset(annotation_dir)
    print_analysis_report(stats)

    with open(output_file, "w") as f:
        stats_dict = {
            "total_images": stats["total_images"],
            "total_annotations": stats["total_annotations"],
            "images_with_persons": stats["images_with_persons"],
            "images_without_persons": stats["images_without_persons"],
            "category_counts": dict(stats["category_counts"]),
            "occlusion_cases": stats["occlusion_cases"],
            "bbox_sizes": stats["bbox_sizes"],
        }
        json.dump(stats_dict, f, indent=2)

    print(f"📊 Statistics saved to: {output_file.resolve()}")
