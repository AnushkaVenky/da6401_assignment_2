"""VGG11 encoder following Simonyan & Zisserman (arXiv:1409.1556), Table 1 col-A.

Architecture (11 weight layers):
  Block1: Conv(64)  + BN + ReLU, MaxPool/2  → 112×112
  Block2: Conv(128) + BN + ReLU, MaxPool/2  → 56×56
  Block3: Conv(256) × 2 + BN + ReLU, MaxPool/2 → 28×28
  Block4: Conv(512) × 2 + BN + ReLU, MaxPool/2 → 14×14
  Block5: Conv(512) × 2 + BN + ReLU, MaxPool/2 → 7×7
  Bottleneck: [B, 512, 7, 7]

BatchNorm and CustomDropout placements are design choices left to the
implementer; the conv-pool topology must match the paper exactly.
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int, use_bn: bool = True):
    """Return a list of layers: Conv2d → (BN) → ReLU."""
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return layers


class VGG11Encoder(nn.Module):
    """VGG11-style convolutional encoder with optional skip-feature returns.

    The encoder consists of 5 conv-pool blocks whose outputs are named
    block1 … block5 / pool1 … pool5 so the autograder can probe them.
    """

    def __init__(self, in_channels: int = 3, use_bn: bool = True):
        """Initialise VGG11Encoder.

        Args:
            in_channels: Number of input image channels (default 3).
            use_bn: Whether to insert BatchNorm2d after each Conv2d.
        """
        super().__init__()

        # ── Block 1: 1 conv, 64 filters ──────────────────────────────────────
        self.block1 = nn.Sequential(*_conv_bn_relu(in_channels, 64, use_bn))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Block 2: 1 conv, 128 filters ─────────────────────────────────────
        self.block2 = nn.Sequential(*_conv_bn_relu(64, 128, use_bn))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Block 3: 2 convs, 256 filters ────────────────────────────────────
        self.block3 = nn.Sequential(
            *_conv_bn_relu(128, 256, use_bn),
            *_conv_bn_relu(256, 256, use_bn),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Block 4: 2 convs, 512 filters ────────────────────────────────────
        self.block4 = nn.Sequential(
            *_conv_bn_relu(256, 512, use_bn),
            *_conv_bn_relu(512, 512, use_bn),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Block 5: 2 convs, 512 filters ────────────────────────────────────
        self.block5 = nn.Sequential(
            *_conv_bn_relu(512, 512, use_bn),
            *_conv_bn_relu(512, 512, use_bn),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Weight initialisation (Kaiming for conv, as per He et al.)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: Input image tensor [B, in_channels, H, W].
                VGG11 expects H = W = 224.
            return_features: If True, also return a dict of pre-pool skip maps
                keyed 's1' … 's5' for use by the U-Net decoder.

        Returns:
            - return_features=False: bottleneck tensor [B, 512, 7, 7].
            - return_features=True: (bottleneck, {'s1':…, 's2':…, …}).
        """
        s1 = self.block1(x)          # [B,  64, 224, 224]
        x  = self.pool1(s1)          # [B,  64, 112, 112]

        s2 = self.block2(x)          # [B, 128, 112, 112]
        x  = self.pool2(s2)          # [B, 128,  56,  56]

        s3 = self.block3(x)          # [B, 256,  56,  56]
        x  = self.pool3(s3)          # [B, 256,  28,  28]

        s4 = self.block4(x)          # [B, 512,  28,  28]
        x  = self.pool4(s4)          # [B, 512,  14,  14]

        s5 = self.block5(x)          # [B, 512,  14,  14]
        x  = self.pool5(s5)          # [B, 512,   7,   7]  ← bottleneck

        if return_features:
            return x, {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}
        return x


# Alias used by the autograder: `from models.vgg11 import VGG11`
VGG11 = VGG11Encoder
