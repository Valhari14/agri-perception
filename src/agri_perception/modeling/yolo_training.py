"""
Step 2: YOLOv8/v11 Fine-tuning Setup
Focus: Small object detection with weighted loss toward recall
"""

import json
from pathlib import Path
import yaml
import numpy as np

# Try importing ultralytics (will need: pip install ultralytics)
try:
    from ultralytics import YOLO
    print("✓ ultralytics imported")
except ImportError:
    print("⚠️  Install ultralytics: pip install ultralytics")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("⚠️  Install PyTorch: pip install torch torchvision")


def create_yolo_dataset_yaml(data_dir: str, output_path: str = 'data.yaml'):
    """
    Create YOLO-format data.yaml for training
    YOLO expects: folder structure with images/ and labels/ subdirs
    Since we have COCO format, we need a custom loader
    """
    
    yaml_content = {
        'path': str(Path(data_dir).parent),  # Root directory
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,  # Number of classes (person/manikin)
        'names': ['person']  # Class names
    }
    
    data_dir_path = Path(data_dir)
    output_file = data_dir_path.parent / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"📄 YOLO data.yaml created: {output_file}")
    return output_file


class YOLOTrainingConfig:
    """
    Training hyperparameters optimized for agricultural small object detection
    """
    
    def __init__(self):
        self.config = {
            # Model selection
            'model': 'yolov8m',  # Medium model (good balance for edge deployment)
            # 'model': 'yolov8s',  # Small (faster, for edge compute)
            # 'model': 'yolov8l',  # Large (best accuracy)
            
            # Input
            'imgsz': 640,
            'batch': 16,  # Adjust based on GPU memory
            
            # Training duration
            'epochs': 100,
            'patience': 20,  # Early stopping patience
            
            # Optimization
            'optimizer': 'SGD',  # or 'Adam'
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            
            # Loss weights - CRITICAL for recall prioritization
            # Keep to broadly supported Ultralytics args for compatibility across versions.
            'cls': 0.7,   # Slight class-loss emphasis
            'box': 7.5,
            'dfl': 1.5,
            
            # Augmentation (handled by albumentations in data prep)
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10,
            'translate': 0.1,
            'scale': 0.5,
            'flipud': 0.5,
            'fliplr': 0.5,
            'mosaic': 1.0,  # Mosaic augmentation helps small objects
            'mixup': 0.1,
            'copy_paste': 0.1,
            
            # Data loading
            'workers': 8,
            'device': 0,  # GPU device
            
            # Callbacks and logging
            'save': True,
            'save_period': 10,
            'verbose': True,
            
            # Metrics
            'plots': True,
            'conf': 0.25,  # Confidence threshold (lower for recall)
            'iou': 0.6,
            'max_det': 100,
            
            # Model architecture tweaks for small objects
            # NOTE: Avoid deprecated args (e.g., obj_pw, iou_t) on newer Ultralytics versions.
        }
    
    def get_dict(self):
        return self.config
    
    def save(self, output_path: str):
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"💾 Training config saved: {output_path}")


class WeightedLossTrainer:
    """
    Custom trainer with weighted loss to prioritize recall
    """
    
    def __init__(self, model_name: str = 'yolov8m'):
        """
        Args:
            model_name: YOLO model size (s, m, l, x)
        """
        self.model_name = model_name
        self.model = None
        self.results = None
    
    def load_pretrained(self):
        """Load COCO-pretrained weights"""
        self.model = YOLO(f'{self.model_name}.pt')
        print(f"✓ Loaded pretrained {self.model_name}.pt")
        return self.model
    
    def train(self, data_yaml: str, **kwargs):
        """
        Fine-tune on agricultural dataset
        
        Args:
            data_yaml: Path to data.yaml
            **kwargs: Override default training parameters
        """
        
        if self.model is None:
            self.load_pretrained()
        
        # Default training config optimized for agriculture
        training_args = {
            'data': data_yaml,
            'epochs': 100,
            'imgsz': 640,
            'batch': 16,
            'optimizer': 'SGD',
            'lr0': 0.001,
            'patience': 20,
            'cls': 0.7,  # Class-loss emphasis for person detection
            'box': 7.5,
            'dfl': 1.5,
            'conf': 0.25,  # Lower confidence for higher recall
            'iou': 0.6,
            'save': True,
            'plots': True,
            'device': 0,
        }
        
        # Override with provided kwargs
        training_args.update(kwargs)
        
        print(f"\n🚀 Starting Fine-tuning ({self.model_name})")
        print(f"   Data: {data_yaml}")
        print(f"   Epochs: {training_args['epochs']}")
        print(f"   Batch size: {training_args['batch']}")
        print(f"   Loss weights: cls={training_args['cls']}, box={training_args['box']}, dfl={training_args['dfl']}")
        print()
        
        # Train model
        self.results = self.model.train(**training_args)
        
        return self.results
    
    def validate(self, data_yaml: str = None):
        """Validate model"""
        if self.model is None:
            print("⚠️  No model loaded")
            return None
        
        results = self.model.val(data=data_yaml)
        return results
    
    def predict(self, source: str, conf: float = 0.5):
        """
        Run inference
        
        Args:
            source: Image path, directory, or video
            conf: Confidence threshold
        """
        if self.model is None:
            print("⚠️  No model loaded")
            return None
        
        results = self.model.predict(source=source, conf=conf, save=True)
        return results
    
    def export(self, format: str = 'onnx', dynamic: bool = True):
        """
        Export model for deployment
        
        Args:
            format: 'onnx', 'torchscript', 'tflite', 'pb', etc.
            dynamic: Dynamic shape (useful for edge deploy)
        """
        if self.model is None:
            print("⚠️  No model loaded")
            return None
        
        path = self.model.export(format=format, dynamic=dynamic)
        print(f"✓ Model exported to {format}: {path}")
        return path


# Example training script
if __name__ == "__main__":
    
    # Use current working directory by default so this runs on Linux HPC and Windows.
    workspace_dir = Path.cwd()
    
    print("="*70)
    print("YOLO Fine-tuning for Agricultural Person Detection")
    print("="*70)
    print()
    
    # Step 1: Resolve YOLO data.yaml (prefer generated yolo_data.yaml)
    print("📝 Step 1: Resolving YOLO dataset config")
    yolo_yaml = workspace_dir / 'yolo_data.yaml'
    if yolo_yaml.exists():
        data_yaml = yolo_yaml
        print(f"✓ Using existing YOLO config: {data_yaml}")
    else:
        yolo_dataset_dir = workspace_dir / 'yolo_dataset'
        if yolo_dataset_dir.exists():
            data_yaml = create_yolo_dataset_yaml(str(yolo_dataset_dir), output_path='yolo_data.yaml')
            print(f"✓ Created YOLO config from dataset folder: {data_yaml}")
        else:
            print("⚠️  Missing yolo_data.yaml and yolo_dataset directory.")
            print("   Run: python -m agri_perception.data.coco_to_yolo --root .")
            raise SystemExit(1)
    print()
    
    # Step 2: Save training config
    print("⚙️  Step 2: Preparing training configuration")
    config = YOLOTrainingConfig()
    config_file = workspace_dir / 'yolo_training_config.yaml'
    config.save(str(config_file))
    print()
    
    # Step 3: Initialize trainer
    print("🔧 Step 3: Initializing YOLO trainer")
    trainer = WeightedLossTrainer('yolov8m')
    trainer.load_pretrained()
    print()
    
    # Step 4: Ready to train (uncomment to run)
    print("📋 Step 4: Ready for training")
    print("   To start training, run:")
    print(f"   → trainer.train(data_yaml='{data_yaml}')")
    print()
    print("   Key settings for agricultural domain:")
    print("   • cls=0.7: Slightly prioritize class learning")
    print("   • box=7.5: Strong box regression for tiny objects")
    print("   • dfl=1.5: Stable localization for difficult examples")
    print("   • conf=0.25: Lower threshold = higher recall")
    print()
    print("💡 IMPORTANT NOTES:")
    print("   1. Data must be in COCO format (see data_preparation.py)")
    print("   2. For production: Use temporal consistency layer (Step 3)")
    print("   3. Optimize for Recall, not Precision (safety-critical)")
    print("   4. Export to ONNX/TFLite for Jetson edge deployment")
    
    print("\n" + "="*70)
