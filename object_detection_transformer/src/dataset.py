from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class YoloLabel:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class YoloDataset(Dataset):
    """Reads images and YOLO `*.txt` annotations into tensors."""

    def __init__(
        self,
        data_dir: Path,
        image_size: int,
        max_detections: int = 25,
        transforms_override: transforms.Compose | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.image_size = image_size
        self.max_detections = max_detections

        images_dir = data_dir / "images"
        labels_dir = data_dir / "labels"
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing images directory: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

        image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.image_paths = sorted(self._gather_many(images_dir, image_exts))
        if not self.image_paths:
            raise FileNotFoundError(f"No training images found in {images_dir}")
        self.label_paths = {p.stem: p for p in labels_dir.glob("*.txt")}
        self.transforms = transforms_override or transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:  # pragma: no cover - simple passthrough
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        image_path = self.image_paths[index]
        label_path = self.label_paths.get(image_path.stem)

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transforms(image)

        boxes, classes = self._load_labels(label_path)
        return image_tensor, boxes, classes

    def _gather_many(self, directory: Path, patterns: Iterable[str]):
        for pattern in patterns:
            yield from directory.glob(pattern)

    def _load_labels(self, label_path: Path | None) -> Tuple[Tensor, Tensor]:
        boxes = torch.zeros((self.max_detections, 4), dtype=torch.float32)
        classes = torch.full((self.max_detections,), -1, dtype=torch.long)

        if label_path is None or not label_path.exists():
            return boxes, classes

        with label_path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.readlines() if line.strip()]

        for i, line in enumerate(lines[: self.max_detections]):
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, parts)
            boxes[i] = torch.tensor([x_center, y_center, width, height], dtype=torch.float32)
            classes[i] = int(class_id)

        return boxes, classes


def collate_yolo(batch: List[Tuple[Tensor, Tensor, Tensor]]):
    images, boxes, classes = zip(*batch)
    images_tensor = torch.stack(images)
    boxes_tensor = torch.stack(boxes)
    classes_tensor = torch.stack(classes)
    return images_tensor, boxes_tensor, classes_tensor
