"""Custom Intersection-over-Union (IoU) loss for bounding-box regression.

Boxes are expected in (x_center, y_center, width, height) format,
with coordinates in pixel space.

Loss value is always in [0, 1]:
    IoU ∈ [0, 1]  →  loss = 1 − IoU ∈ [0, 1]

Supported reductions: 'mean' (default) | 'sum' | 'none'.
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding-box regression.

    Args:
        eps: Small constant added to denominator to avoid division by zero.
        reduction: 'mean' | 'sum' | 'none'.  Default 'mean'.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"reduction must be one of 'mean', 'sum', 'none'; got '{reduction}'"
            )
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU loss.

        Args:
            pred_boxes:   [B, 4] predicted boxes  (cx, cy, w, h).
            target_boxes: [B, 4] ground-truth boxes (cx, cy, w, h).

        Returns:
            Scalar loss if reduction ∈ {'mean','sum'}, else per-sample [B].
        """
        # ── Convert (cx, cy, w, h) → (x1, y1, x2, y2) ───────────────────────
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2.0
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2.0
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2.0
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2.0

        tgt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2.0
        tgt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2.0
        tgt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2.0
        tgt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2.0

        # ── Intersection ──────────────────────────────────────────────────────
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
        inter_area = inter_w * inter_h                       # [B]

        # ── Union ─────────────────────────────────────────────────────────────
        pred_area = pred_boxes[:, 2].clamp(min=0.0) * pred_boxes[:, 3].clamp(min=0.0)
        tgt_area  = target_boxes[:, 2].clamp(min=0.0) * target_boxes[:, 3].clamp(min=0.0)
        union_area = pred_area + tgt_area - inter_area + self.eps

        # ── IoU loss ∈ [0, 1] ─────────────────────────────────────────────────
        iou  = inter_area / union_area                       # [B], ∈ [0, 1]
        loss = 1.0 - iou                                     # [B], ∈ [0, 1]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss   # 'none'

    def extra_repr(self) -> str:
        return f"eps={self.eps}, reduction='{self.reduction}'"
