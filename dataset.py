import random
import torch
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms as T
from typing import List, Tuple
from slide_detector import extract_film_from_slide

def decode_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        w, h = im.size
        if w >= 3700 and h >= 2600: im.draft("RGB", (w // 2, h // 2))
        rgb = im.convert("RGB")
    return rgb

class SlideTransform:
    def __init__(self, size: int = 224):
        self.size = size
        # ImageNet normalization standard for DINOv2
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.geometric_transform = T.Compose([
            T.Resize((size + 32, size + 32)),
            T.RandomResizedCrop(size, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
            T.RandomRotation(degrees=(-3, 3)),
            T.RandomHorizontalFlip(p=0.05), # small probability of mirror scan
        ])

        self.color_jitter = T.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.20,
            hue=0.05
        )

        self.to_tensor = T.ToTensor()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Apply geometric variations
        augmented = self.geometric_transform(img)

        # Photometric variations
        if random.random() < 0.8:
            augmented = self.color_jitter(augmented)

        # Occasional slight blur (simulating slight focus difference between shots)
        if random.random() < 0.2:
            augmented = augmented.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

        tensor = self.to_tensor(augmented)
        return self.normalize(tensor)

class SlideDuplicateDataset(Dataset):
    def __init__(self, image_paths: List[str], target_size: int = 224, cache_film_crops: bool = True):
        self.image_paths = image_paths
        self.transform = SlideTransform(size=target_size)
        self.cache_film_crops = cache_film_crops
        self._cache = {}

    def __len__(self) -> int:
        return len(self.image_paths)

    def prewarm_cache(self, max_workers: int = 8) -> int:
        if not self.cache_film_crops: return 0

        todo = [p for p in self.image_paths if p not in self._cache]
        if not todo: return 0

        def _load(path: str):
            rgb = decode_rgb(path)
            crop = extract_film_from_slide(rgb).resize((480, 320), Image.Resampling.BILINEAR)
            return path, crop

        workers = max(1, min(max_workers, len(todo)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for path, crop in ex.map(_load, todo):
                self._cache[path] = crop
        return len(todo)

    def _get_film_image(self, path: str) -> Image.Image:
        if self.cache_film_crops and path in self._cache:
            return self._cache[path]

        rgb = decode_rgb(path)
        film_crop = extract_film_from_slide(rgb)
        # Store at a manageable resolution to save RAM while maintaining quality
        film_crop_resized = film_crop.resize((480, 320), Image.Resampling.BILINEAR)

        if self.cache_film_crops:
            self._cache[path] = film_crop_resized

        return film_crop_resized

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        film_img = self._get_film_image(path)

        # Generate two distinct augmented views of the same slide
        view1 = self.transform(film_img)
        view2 = self.transform(film_img)

        return view1, view2
