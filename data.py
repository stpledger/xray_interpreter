from pathlib import Path

import torch
from PIL import Image

DATASET_PATH = Path(__file__).parent.parent / "data"


class ImageDataset:
    image_paths: list[Path]
    _image_cache: list[torch.Tensor | None]
    _cache_images: bool

    def __init__(self, split: str, cache_images: bool = True):
        self.image_paths = list((DATASET_PATH / split).rglob("*.jpg"))
        self._image_cache = [None] * len(self.image_paths)
        self._cache_images = cache_images

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import numpy as np

        cached_image = self._image_cache[idx]
        if cached_image is not None:
            return cached_image

        img = torch.tensor(np.array(Image.open(self.image_paths[idx])), dtype=torch.uint8)
        if self._cache_images:
            self._image_cache[idx] = img
        return img
