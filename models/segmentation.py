"""U-Net style segmentation model built on VGG11 encoder.

Decoder is symmetric to the VGG11 encoder:
  - 5 upsampling stages, each using ConvTranspose2d (no bilinear/unpooling).
  - At each stage, upsampled features are concatenated with the matching
    encoder skip map (feature fusion along the channel dimension).
  - Final 1×1 conv produces per-pixel class logits.

Skip-map sizes (input 224×224):
  s1: [B,  64, 224, 224]   s2: [B, 128, 112, 112]
  s3: [B, 256,  56,  56]   s4: [B, 512,  28,  28]
  s5: [B, 512,  14,  14]   bottleneck: [B, 512, 7, 7]
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class _DecoderBlock(nn.Module):
    """Single decoder stage: TransposedConv upsample → concat skip → conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # Learnable 2× upsampling (no bilinear/unpooling as per spec)
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        # After concat: (out_channels + skip_channels) → out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SegmentationDecoder(nn.Module):
    """Symmetric decoder that accepts the VGG11 skip features dict."""

    def __init__(self, num_classes: int = 3, dropout_p: float = 0.5):
        super().__init__()
        # bottleneck [B,512,7,7] → up → cat s5[B,512,14,14] → [B,512,14,14]
        self.dec5 = _DecoderBlock(512, 512, 512)
        # [B,512,14,14] → up → cat s4[B,512,28,28] → [B,256,28,28]
        self.dec4 = _DecoderBlock(512, 512, 256)
        # [B,256,28,28] → up → cat s3[B,256,56,56] → [B,128,56,56]
        self.dec3 = _DecoderBlock(256, 256, 128)
        # [B,128,56,56] → up → cat s2[B,128,112,112] → [B,64,112,112]
        self.dec2 = _DecoderBlock(128, 128, 64)
        # [B,64,112,112] → up → cat s1[B,64,224,224] → [B,32,224,224]
        self.dec1 = _DecoderBlock(64, 64, 32)

        self.dropout = CustomDropout(dropout_p)
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        features: dict,
    ) -> torch.Tensor:
        x = self.dec5(bottleneck,  features["s5"])
        x = self.dec4(x,           features["s4"])
        x = self.dec3(x,           features["s3"])
        x = self.dec2(x,           features["s2"])
        x = self.dec1(x,           features["s1"])
        x = self.dropout(x)
        return self.out_conv(x)     # [B, num_classes, 224, 224]


class VGG11UNet(nn.Module):
    """U-Net style segmentation network with VGG11 encoder.

    num_classes=3 matches the Oxford-IIIT trimap labels:
        0 = pet foreground, 1 = background, 2 = boundary.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ):
        """
        Initialise VGG11UNet.

        Args:
            num_classes: Number of segmentation classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability applied before the output conv.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.decoder = SegmentationDecoder(
            num_classes=num_classes, dropout_p=dropout_p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, in_channels, H, W]. Expects H=W=224.

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        bottleneck, feats = self.encoder(x, return_features=True)
        return self.decoder(bottleneck, feats)
