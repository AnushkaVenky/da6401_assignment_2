"""Oxford-IIIT Pet Dataset loader for multi-task learning.

Provides:
  - Class label  (0-indexed breed id, 37 classes)
  - Bounding box (x_center, y_center, width, height) in 224×224 pixel space
  - Segmentation mask (H×W long tensor, values 0/1/2)
      0 = pet foreground, 1 = background, 2 = boundary

Download URLs:
  Images      : https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
  Annotations : https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
"""

import os
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Fixed input resolution (VGG11 paper)
_IMG_SIZE = 224

# ImageNet normalisation statistics
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_URLS = {
    "images":      "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    "annotations": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
}


def _download_and_extract(root: Path) -> None:
    """Download and extract dataset archives if not already present."""
    root.mkdir(parents=True, exist_ok=True)
    for key, url in _URLS.items():
        archive = root / url.split("/")[-1]
        if not archive.exists():
            print(f"Downloading {key} from {url} …")
            urllib.request.urlretrieve(url, archive)
        # Check that key directories exist before extracting
        expected = root / ("images" if key == "images" else "annotations")
        if not expected.exists():
            print(f"Extracting {archive.name} …")
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(root)


def _parse_split_file(path: Path) -> List[Tuple[str, int]]:
    """Parse trainval.txt or test.txt → list of (image_name, class_id 0-idx)."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            img_name = parts[0]
            class_id = int(parts[1]) - 1   # 1-indexed → 0-indexed
            samples.append((img_name, class_id))
    return samples


def _load_bbox(xml_path: Path, orig_w: int, orig_h: int) -> Optional[List[float]]:
    """Load head bounding box from Pascal-VOC XML, scaled to 224×224.

    Returns [cx, cy, w, h] in pixel space or None if file is missing/invalid.
    """
    if not xml_path.exists():
        return None
    try:
        root = ET.parse(xml_path).getroot()
        obj = root.find("object")
        if obj is None:
            return None
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
    except Exception:
        return None

    # Scale to target resolution
    sx = _IMG_SIZE / orig_w
    sy = _IMG_SIZE / orig_h
    cx = ((xmin + xmax) / 2.0) * sx
    cy = ((ymin + ymax) / 2.0) * sy
    w  = (xmax - xmin) * sx
    h  = (ymax - ymin) * sy
    return [cx, cy, w, h]


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset.

    Each sample returns:
        image : Float tensor [3, 224, 224] – normalised with ImageNet stats.
        target: dict with keys
            'label' : int        – 0-indexed breed id.
            'bbox'  : FloatTensor[4] – (cx, cy, w, h) in pixel space.
                      Falls back to (112, 112, 224, 224) if XML is missing.
            'mask'  : LongTensor[224, 224] – 0=pet, 1=background, 2=boundary.
    """

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        transform: Optional[Callable] = None,
        download: bool = True,
        require_bbox: bool = False,
    ):
        """
        Args:
            root: Directory where the dataset is stored / will be downloaded.
            split: 'trainval' or 'test'.
            transform: Optional callable applied to the PIL image *before*
                       normalisation. Must return a PIL Image.
            download: If True, download archives when missing.
            require_bbox: If True, drop samples without an XML annotation.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform

        if download:
            _download_and_extract(self.root)

        split_file = self.root / "annotations" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {split_file}\n"
                "Set download=True or check the dataset path."
            )

        all_samples = _parse_split_file(split_file)

        if require_bbox:
            xml_dir = self.root / "annotations" / "xmls"
            all_samples = [
                s for s in all_samples
                if (xml_dir / f"{s[0]}.xml").exists()
            ]

        self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_name, class_id = self.samples[idx]

        # ── Load image ────────────────────────────────────────────────────────
        img_path = self.root / "images" / f"{img_name}.jpg"
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Optional user transform (e.g. albumentations on PIL)
        if self.transform is not None:
            image = self.transform(image)

        # Resize to 224×224
        image = image.resize((_IMG_SIZE, _IMG_SIZE), Image.BILINEAR)

        # ── Load bounding box ─────────────────────────────────────────────────
        xml_path = self.root / "annotations" / "xmls" / f"{img_name}.xml"
        bbox = _load_bbox(xml_path, orig_w, orig_h)
        if bbox is None:
            # Fallback: whole-image box (cx=112, cy=112, w=224, h=224)
            bbox = [_IMG_SIZE / 2.0, _IMG_SIZE / 2.0,
                    float(_IMG_SIZE),  float(_IMG_SIZE)]

        # ── Load segmentation mask ────────────────────────────────────────────
        mask_path = self.root / "annotations" / "trimaps" / f"{img_name}.png"
        if mask_path.exists():
            mask = np.array(
                Image.open(mask_path).resize(
                    (_IMG_SIZE, _IMG_SIZE), Image.NEAREST
                ),
                dtype=np.int64,
            )
            mask -= 1   # {1,2,3} → {0,1,2}
            mask = mask.clip(0, 2)
        else:
            mask = np.zeros((_IMG_SIZE, _IMG_SIZE), dtype=np.int64)

        # ── Convert image to normalised float tensor ──────────────────────────
        img_arr = np.array(image, dtype=np.float32) / 255.0
        img_arr = (img_arr - _MEAN) / _STD                  # [H, W, 3]
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)  # [3, H, W]

        return img_tensor, {
            "label": class_id,
            "bbox":  torch.tensor(bbox, dtype=torch.float32),
            "mask":  torch.tensor(mask, dtype=torch.long),
        }
