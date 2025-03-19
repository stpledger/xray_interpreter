from pathlib import Path

import torch
from PIL import Image
import pandas as pd

IMAGES_PATH = Path('/Users/stple/Documents/Projects_Local/xray_interpreter/data')


class ImageDataset:
    image_paths: list[Path]
    _image_cache: list[torch.Tensor | None]
    _cache_images: bool
    labels: dict[str, int]  # Adjust the label type as needed

    def __init__(self, split: str, cache_images: bool = True, transform=None):
        # Create a list of image paths that have a corresponding label.
        labels_csv = pd.read_csv(f"{split}.csv", index_col=0).drop(columns=['Patient ID'])
        self.label_names = labels_csv.columns.tolist()
        self.labels = labels_csv.apply(lambda row: row.tolist(), axis=1).to_dict()

        all_paths = list((IMAGES_PATH).rglob("*.png"))
        self.image_paths = [p for p in all_paths if p.name in self.labels]
        self._image_cache = [None] * len(self.image_paths)
        self._cache_images = cache_images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        import numpy as np

        if self._cache_images:
            cached_image = self._image_cache[idx]
            if cached_image is not None:
                img = cached_image
            else:
                img = Image.open(self.image_paths[idx]).copy()
                self._image_cache[idx] = img
        else:
            img = Image.open(self.image_paths[idx]).copy()
        
        # Apply transforms if provided
        if self.transform is not None:
            img = self.transform(img)

        # Retrieve the label using the image file name
        label = torch.Tensor(self.labels[self.image_paths[idx].name])
        return img, label
