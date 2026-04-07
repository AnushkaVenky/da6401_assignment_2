"""Localization module: VGG11 encoder + bounding-box regression head.

Output format: [x_center, y_center, width, height] in *pixel* coordinates
for a 224×224 input image (values in (0, 224)).

Training loss (as per README): MSE + custom IoU loss.
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

# VGG11 fixed input size (per the paper)
_IMG_SIZE = 224


class VGG11Localizer(nn.Module):
    """VGG11-based single-object localizer.

    Architecture:
        VGG11Encoder → Flatten → FC(25088→4096) → ReLU → Dropout
                                → FC(4096→4096)  → ReLU → Dropout
                                → FC(4096→4)
        Output activations: sigmoid × 224 to map to pixel space.
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialise VGG11Localizer.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability in the regression head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Regression head – outputs 4 raw logits
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4),
        )

        # Init FC layers
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, in_channels, H, W]. Expects H=W=224.

        Returns:
            Bounding box tensor [B, 4] in (x_center, y_center, width, height)
            format, with values in pixel space (0 … 224).
        """
        features = self.encoder(x)          # [B, 512, 7, 7]
        raw = self.regressor(features)      # [B, 4]  (unbounded logits)

        # Sigmoid maps raw logits → (0, 1), then scale to pixel space
        coords = torch.sigmoid(raw) * _IMG_SIZE   # [B, 4] in (0, 224)
        return coords
