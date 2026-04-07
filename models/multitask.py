"""Unified multi-task perception model (Task 4).

Loads three independently-trained checkpoints (classifier, localizer, UNet),
extracts a shared VGG11 backbone, and branches into three task heads.
A single forward(x) call returns classification logits, bounding-box
coordinates, AND a segmentation mask simultaneously.

Usage (after training individual models):
    model = MultiTaskPerceptionModel(
        classifier_path="checkpoints/classifier.pth",
        localizer_path="checkpoints/localizer.pth",
        unet_path="checkpoints/unet.pth",
    )
    out = model(image_tensor)
    # out['classification'] : [B, 37]
    # out['localization']   : [B, 4]   pixel-space (cx,cy,w,h)
    # out['segmentation']   : [B, 3, 224, 224]
"""

import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet

_IMG_SIZE = 224


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    The shared backbone is taken from the trained classifier.
    Task-specific heads (cls head, regression head, segmentation decoder)
    are loaded from their respective checkpoints.
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
    ):
        """
        Initialise shared backbone and task heads from saved checkpoints.

        Args:
            num_breeds: Number of breed classes (37 for Oxford-IIIT Pet).
            seg_classes: Number of segmentation classes (3 for trimap).
            in_channels: Number of input image channels.
            classifier_path: Path to classifier.pth checkpoint.
            localizer_path: Path to localizer.pth checkpoint.
            unet_path: Path to unet.pth checkpoint.
        """
        super().__init__()

        # ── Download checkpoints from Google Drive ───────────────────────────
        import gdown
        gdown.download(id="1yiRNVrHHPM-3-Iwx_rAyZXOp2ZUwUVww", output=classifier_path, quiet=False)
        gdown.download(id="19G6uylzVPN4XTUe6CRV4Yl5haefzs_ZM",  output=localizer_path,  quiet=False)
        gdown.download(id="1qNRy4Y2p9lEnMTOl1w2wRXhoDkQlOhqi",       output=unet_path,       quiet=False)

        # ── Initialise sub-models and load weights ───────────────────────────
        clf = VGG11Classifier(num_breeds, in_channels)
        loc = VGG11Localizer(in_channels)
        seg = VGG11UNet(seg_classes, in_channels)

        clf.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        loc.load_state_dict(torch.load(localizer_path,  map_location="cpu"))
        seg.load_state_dict(torch.load(unet_path,       map_location="cpu"))

        # ── Extract shared backbone (from classifier) ────────────────────────
        # All three models share the same VGG11 topology; we use the
        # classifier's trained backbone as the unified representation.
        self.backbone = clf.encoder       # VGG11Encoder

        # ── Task-specific heads ──────────────────────────────────────────────
        # cls_head  : nn.Sequential starting with nn.Flatten()
        self.cls_head     = clf.classifier
        # loc_head  : nn.Sequential starting with nn.Flatten(), raw logits
        self.loc_head     = loc.regressor
        # seg_decoder: SegmentationDecoder that takes (bottleneck, feats)
        self.seg_decoder  = seg.decoder

    def forward(self, x: torch.Tensor) -> dict:
        """Single forward pass over the shared backbone.

        Args:
            x: Input tensor [B, in_channels, H, W]. Expects H=W=224.

        Returns:
            dict with keys:
                'classification' : [B, num_breeds] logits.
                'localization'   : [B, 4] bbox (cx, cy, w, h) in pixel space.
                'segmentation'   : [B, seg_classes, H, W] logits.
        """
        # Shared encoder forward – also collect skip features for segmentation
        bottleneck, feats = self.backbone(x, return_features=True)
        # bottleneck: [B, 512, 7, 7]

        # ── Classification ───────────────────────────────────────────────────
        cls_logits = self.cls_head(bottleneck)          # [B, num_breeds]

        # ── Localisation ─────────────────────────────────────────────────────
        raw_loc = self.loc_head(bottleneck)             # [B, 4] raw logits
        # Map sigmoid output to pixel space (same as VGG11Localizer.forward)
        bbox = torch.sigmoid(raw_loc) * _IMG_SIZE       # [B, 4]

        # ── Segmentation ─────────────────────────────────────────────────────
        seg_logits = self.seg_decoder(bottleneck, feats)  # [B, seg_classes, H, W]

        return {
            "classification": cls_logits,
            "localization":   bbox,
            "segmentation":   seg_logits,
        }
