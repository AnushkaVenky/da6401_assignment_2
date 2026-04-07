"""Reusable custom layers."""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout using a binary mask with inverted scaling.

    During training, each element is zeroed independently with probability p,
    and surviving elements are scaled by 1/(1-p) to preserve expected values.
    During evaluation (self.training=False), the layer is a no-op.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability. Must be in [0, 1).
        """
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor of any shape.

        Returns:
            Output tensor with same shape as input.
        """
        if not self.training or self.p == 0.0:
            return x
        # Binary mask: 1 with probability (1-p), 0 with probability p
        mask = torch.bernoulli(torch.full_like(x, 1.0 - self.p))
        # Inverted scaling keeps E[output] == E[input]
        return x * mask / (1.0 - self.p)

    def extra_repr(self) -> str:
        return f"p={self.p}"
