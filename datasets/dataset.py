import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, Dict, List, Tuple

class HER2Dataset(Dataset):
    """
    Dataset class for HER2 stained breast cancer images.
    """
    EXPECTED_CLASSES = ["class_0", "class_1+", "class_2+", "class_3+"]
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        """
        Args:
            root_dir: Root directory of the dataset.
            transform: Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = self.EXPECTED_CLASSES
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        self._validate_and_collect()

    def _validate_and_collect(self) -> None:
        if not self.root_dir.exists():
            raise ValueError(f"Dataset root directory does not exist: {self.root_dir}")
            
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise ValueError(f"Missing expected class directory: {class_dir}")
                
            label = self.class_to_idx[class_name]
            for file_path in class_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                    self.image_paths.append(file_path)
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
        
    def get_num_classes(self) -> int:
        return len(self.classes)
        
    def get_class_names(self) -> List[str]:
        return self.classes
        
    def get_class_distribution(self) -> Dict[str, int]:
        counts = {cls_name: 0 for cls_name in self.classes}
        for label in self.labels:
            counts[self.classes[label]] += 1
        return counts
