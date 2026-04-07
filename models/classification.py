"""Classification head and full VGG11 classifier."""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + FC classification head.

    FC head mirrors the original VGG paper:
        Flatten → Linear(25088→4096) → ReLU → Dropout
                → Linear(4096→4096)  → ReLU → Dropout
                → Linear(4096→num_classes)
    """

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_bn: bool = True,
    ):
        """
        Initialise VGG11Classifier.

        Args:
            num_classes: Number of output classes (37 pet breeds).
            in_channels: Number of input channels.
            dropout_p: Dropout probability in the FC head.
            use_bn: Whether to use BatchNorm in the encoder.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, use_bn=use_bn)

        # VGG11 bottleneck: 512 × 7 × 7 = 25088
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_classes),
        )

        # Init FC layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, in_channels, H, W]. Expects H=W=224.

        Returns:
            Classification logits [B, num_classes].
        """
        features = self.encoder(x)          # [B, 512, 7, 7]
        return self.classifier(features)    # [B, num_classes]
