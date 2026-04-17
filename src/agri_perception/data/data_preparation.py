"""
Step 1: Data Preparation and Augmentation Strategy
Focus: Small object detection (majority of detections are tiny bbox)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import cv2

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("⚠️  Install albumentations: pip install albumentations")
    A = None
    ToTensorV2 = None

try:
    from pycocotools.coco import COCO
except ImportError:
    print("⚠️  Install pycocotools: pip install pycocotools")


class COCODatasetLoader:
    """Load and organize COCO-format agricultural dataset"""
    
    def __init__(self, annotation_dir: str, data_dir: str):
        """
        Args:
            annotation_dir: Path to annotation folder containing JSON files
            data_dir: Path to data folder containing image directories
        """
        self.annotation_dir = Path(annotation_dir)
        self.data_dir = Path(data_dir)
        self.datasets = {}
        self.image_registry = {}  # Map {image_path -> annotation_data}
        
    def load_all_datasets(self) -> Dict:
        """Load all COCO annotation files"""
        
        json_files = sorted(self.annotation_dir.rglob('*.json'))
        
        print(f"📁 Found {len(json_files)} annotation files\n")
        
        for json_file in json_files:
            dataset_name = json_file.parent.name
            
            try:
                with open(json_file, 'r') as f:
                    coco_data = json.load(f)
                
                self.datasets[dataset_name] = coco_data
                
                # Build registry: map image paths to annotations
                self._build_image_registry(dataset_name, coco_data, json_file.parent)
                
                num_imgs = len(coco_data.get('images', []))
                num_anns = len(coco_data.get('annotations', []))
                print(f"✓ {dataset_name}: {num_imgs} images, {num_anns} annotations")
                
            except Exception as e:
                print(f"⚠️  Error loading {dataset_name}: {e}")
        
        print(f"\n📊 Total: {len(self.image_registry)} images with registry")
        return self.datasets
    
    def _build_image_registry(self, dataset_name: str, coco_data: Dict, 
                              img_base_path: Path):
        """Build lookup table for images and their annotations"""
        
        # Create mapping: image_id -> image_info
        id_to_image = {img['id']: img for img in coco_data.get('images', [])}
        
        # Create mapping: image_id -> list of annotations
        id_to_annos = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in id_to_annos:
                id_to_annos[img_id] = []
            id_to_annos[img_id].append(ann)
        
        # Register images
        for img_id, img_info in id_to_image.items():
            img_path = img_base_path / img_info['file_name']
            
            self.image_registry[str(img_path)] = {
                'path': img_path,
                'dataset': dataset_name,
                'image_info': img_info,
                'annotations': id_to_annos.get(img_id, []),
                'categories': {c['id']: c['name'] 
                              for c in coco_data.get('categories', [])}
            }


class AugmentationStrategy:
    """
    Augmentation pipeline for small object detection in agricultural domain
    Key challenges:
    - 97% of objects are tiny/small bbox (< 50k px²)
    - Heavy occlusion (people partially hidden by crops)
    - Visual noise: dust, motion blur, lighting variation
    """
    
    @staticmethod
    def get_train_transforms(img_size: int = 640):
        """Training augmentations - aggressive to handle edge cases"""
        if A is None or ToTensorV2 is None:
            raise ImportError("albumentations is required. Install with: pip install albumentations")
        return A.Compose([
            # 1. SPATIAL AUGMENTATIONS (for small object robustness)
            A.OneOf([
                A.Affine(scale=(0.9, 1.2), rotate=(-15, 15), 
                        translate_percent=(-0.1, 0.1), p=0.5),  # Harvester motion
                A.Perspective(scale=(0.05, 0.1), p=0.3),  # Viewpoint change
            ], p=0.6),
            
            # 2. CROP & ZOOM (simulate partial occlusion)
            A.OneOf([
                A.RandomCrop(height=int(img_size*0.8), width=int(img_size*0.8), p=0.4),
                A.CropNonEmptyMaskIfExists(height=int(img_size*0.75), 
                                          width=int(img_size*0.75), p=0.3),
            ], p=0.5),
            
            # 3. AGRICULTURAL NOISE SIMULATION
            A.OneOf([
                # Dust/chaff simulation
                A.GaussNoise(p=0.3),
                A.ISONoise(p=0.2),
                # Motion blur from moving harvester
                A.MotionBlur(blur_limit=7, p=0.3),
                # Dust overlay (yellow tint)
                A.RandomRain(p=0.1),
            ], p=0.7),
            
            # 4. LIGHTING VARIATIONS
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, 
                                          contrast_limit=0.3, p=0.5),
                A.RandomShadow(p=0.2),
                A.RandomFog(p=0.1),
                # Simulate sunrise/sunset
                A.RandomGamma(gamma_limit=(70, 130), p=0.2),
            ], p=0.7),
            
            # 5. COLOR/FILTER AUGMENTATION
            A.OneOf([
                A.Blur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.2),
                A.CLAHE(p=0.1),  # Contrast enhancement
            ], p=0.4),
            
            # 6. GEOMETRIC NORMALIZATION
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', 
                                    label_fields=['labels'],
                                    min_visibility=0.1))  # Keep partially visible
    
    @staticmethod
    def get_val_transforms(img_size: int = 640):
        """Validation augmentations - minimal, deterministic"""
        if A is None or ToTensorV2 is None:
            raise ImportError("albumentations is required. Install with: pip install albumentations")
        return A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', 
                                    label_fields=['labels']))
    
    @staticmethod
    def get_test_transforms(img_size: int = 640):
        """Test augmentations - identity + normalization"""
        if A is None or ToTensorV2 is None:
            raise ImportError("albumentations is required. Install with: pip install albumentations")
        return A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', 
                                    label_fields=['labels']))


class COCODataset:
    """PyTorch-compatible dataset wrapper"""
    
    def __init__(self, image_paths: List[str], transforms=None):
        """
        Args:
            image_paths: List of full image paths (keys from image_registry)
            transforms: Albumentations transform pipeline
        """
        self.image_paths = image_paths
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Return image + bboxes for training"""
        img_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get bboxes (will be filled in when registry is available)
        # For now, placeholder
        bboxes = []
        labels = []
        
        if self.transforms:
            transformed = self.transforms(image=image, 
                                         bboxes=bboxes, 
                                         labels=labels)
            image = transformed['image']
        
        return {
            'image': image,
            'bboxes': np.array(bboxes),
            'labels': np.array(labels),
            'image_path': img_path
        }


def create_train_val_split(image_registry: Dict, train_ratio: float = 0.8, 
                          seed: int = 42) -> Tuple[List, List, List]:
    """
    Split dataset into train/val/test (80/10/10)
    Strategy: Split by dataset to avoid temporal leakage
    """
    np.random.seed(seed)
    
    # Group by dataset
    datasets_dict = {}
    for img_path, data in image_registry.items():
        dataset_name = data['dataset']
        if dataset_name not in datasets_dict:
            datasets_dict[dataset_name] = []
        datasets_dict[dataset_name].append(img_path)
    
    train_paths = []
    val_paths = []
    test_paths = []
    
    for dataset_name, paths in datasets_dict.items():
        # Shuffle paths for this dataset
        np.random.shuffle(paths)
        
        n_train = int(len(paths) * 0.7)
        n_val = int(len(paths) * 0.15)
        
        train_paths.extend(paths[:n_train])
        val_paths.extend(paths[n_train:n_train+n_val])
        test_paths.extend(paths[n_train+n_val:])
    
    total = len(train_paths) + len(val_paths) + len(test_paths)
    if total == 0:
        raise ValueError(
            "No images found for splitting. Check annotation path and file structure."
        )

    print(f"\n📊 TRAIN/VAL/TEST SPLIT")
    print(f"  Train: {len(train_paths)} images ({100*len(train_paths)/total:.1f}%)")
    print(f"  Val:   {len(val_paths)} images ({100*len(val_paths)/total:.1f}%)")
    print(f"  Test:  {len(test_paths)} images ({100*len(test_paths)/total:.1f}%)")
    
    return train_paths, val_paths, test_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare COCO data splits and augmentation metadata for training."
    )
    parser.add_argument("--annotation-dir", default="annotation", help="Path to COCO annotation directory")
    parser.add_argument("--data-dir", default="data", help="Path to image data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility")
    args = parser.parse_args()

    # Step 1: Load dataset
    annotation_dir = Path(args.annotation_dir)
    data_dir = Path(args.data_dir)

    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")
    
    print("🔄 Loading COCO Datasets...\n")
    loader = COCODatasetLoader(annotation_dir, data_dir)
    datasets = loader.load_all_datasets()
    
    # Step 2: Create train/val/test splits
    train_paths, val_paths, test_paths = create_train_val_split(
        loader.image_registry, 
        train_ratio=0.7,
        seed=args.seed,
    )
    
    # Step 3: Initialize datasets with transforms
    aug_strategy = AugmentationStrategy()
    
    train_transforms = aug_strategy.get_train_transforms(img_size=640)
    val_transforms = aug_strategy.get_val_transforms(img_size=640)
    test_transforms = aug_strategy.get_test_transforms(img_size=640)
    
    print(f"\n🎨 Augmentation Strategy Initialized")
    print(f"  Train transforms: Heavy augmentation (small object + occlusion)")
    print(f"  Val transforms: Minimal augmentation")
    print(f"  Test transforms: Identity + normalization")
    
    # Save metadata for next step
    metadata_file = Path(annotation_dir).parent / 'dataset_metadata.json'
    with open(metadata_file, 'w') as f:
        # Convert Path objects to strings for JSON
        json.dump({
            'train_count': len(train_paths),
            'val_count': len(val_paths),
            'test_count': len(test_paths),
            'num_datasets': len(datasets),
        }, f, indent=2)
    
    print(f"\n✅ Data preparation complete!")
    print(f"📄 Metadata saved: {metadata_file}")
