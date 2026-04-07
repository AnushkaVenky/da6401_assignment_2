"""Inference, evaluation, and W&B visualisation utilities.

Covers all W&B report sections that require model outputs:
  §2.4  – Feature-map visualisation (first & last conv layers)
  §2.5  – Detection confidence & IoU table (≥10 test images)
  §2.6  – Segmentation evaluation: Pixel Accuracy vs Dice Score
  §2.7  – Final pipeline showcase on 3 novel internet images

Usage examples
--------------
# §2.4 – feature maps
python inference.py --mode feat_maps \
    --cls_ckpt checkpoints/classifier.pth \
    --image_path path/to/dog.jpg

# §2.5 – detection table
python inference.py --mode detection \
    --cls_ckpt  checkpoints/classifier.pth \
    --det_ckpt  checkpoints/localizer.pth

# §2.6 – segmentation eval
python inference.py --mode segmentation \
    --seg_ckpt checkpoints/unet.pth

# §2.7 – pipeline showcase
python inference.py --mode pipeline \
    --cls_ckpt checkpoints/classifier.pth \
    --det_ckpt checkpoints/localizer.pth \
    --seg_ckpt checkpoints/unet.pth \
    --image_paths img1.jpg img2.jpg img3.jpg
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import wandb

from data.pets_dataset import OxfordIIITPetDataset, _MEAN, _STD
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

DATASET_ROOT = "./data/oxford_pet"
_IMG_SIZE    = 224

# 37 Oxford-IIIT Pet breed names (class index = list index)
BREED_NAMES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
    "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
    "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
    "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
]

TRIMAP_COLORS = np.array([
    [0,   200,  0  ],   # 0 = pet foreground  (green)
    [200,  0,   0  ],   # 1 = background       (red)
    [200, 200,  0  ],   # 2 = boundary         (yellow)
], dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_pil(pil_img: Image.Image) -> torch.Tensor:
    """Resize to 224×224, normalise, return [1, 3, 224, 224] tensor."""
    pil_img = pil_img.convert("RGB").resize((_IMG_SIZE, _IMG_SIZE), Image.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def denorm(tensor: torch.Tensor) -> np.ndarray:
    """Denormalise a [3, H, W] tensor back to uint8 HWC."""
    arr = tensor.permute(1, 2, 0).numpy()
    arr = arr * _STD + _MEAN
    return (arr.clip(0, 1) * 255).astype(np.uint8)


def iou_from_boxes(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    loss_fn = IoULoss(reduction="mean")
    return float(1.0 - loss_fn(pred.unsqueeze(0), tgt.unsqueeze(0)).item())


# ══════════════════════════════════════════════════════════════════════════════
# §2.4  Feature-map visualisation
# ══════════════════════════════════════════════════════════════════════════════

def visualize_feature_maps(cls_ckpt: str, image_path: str,
                           wandb_run) -> None:
    """Extract and log feature maps from first and last conv blocks.

    Logs a grid of feature maps for block1 (first conv) and block5 (last conv
    before pooling) to the W&B run.
    """
    device = load_device()
    model  = VGG11Classifier(num_classes=37).to(device)
    model.load_state_dict(torch.load(cls_ckpt, map_location="cpu"))
    model.eval()

    pil_img = Image.open(image_path).convert("RGB")
    inp     = preprocess_pil(pil_img).to(device)

    activations = {}

    def _make_hook(name):
        def _hook(module, inp_, out):
            activations[name] = out.detach().cpu()
        return _hook

    h1 = model.encoder.block1.register_forward_hook(_make_hook("block1"))
    h5 = model.encoder.block5.register_forward_hook(_make_hook("block5"))

    with torch.no_grad():
        model(inp)

    h1.remove(); h5.remove()

    def _plot_fmaps(act_tensor, title, n_maps=16):
        fmaps = act_tensor[0]             # [C, H, W]
        n     = min(n_maps, fmaps.size(0))
        cols  = 4
        rows  = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten() if rows * cols > 1 else [axes]
        for i in range(n):
            fm = fmaps[i].numpy()
            axes[i].imshow(fm, cmap="viridis")
            axes[i].axis("off")
        for i in range(n, len(axes)):
            axes[i].axis("off")
        fig.suptitle(title)
        plt.tight_layout()
        return fig

    fig1 = _plot_fmaps(activations.get("block1"), "Block-1 (1st conv layer) feature maps")
    fig5 = _plot_fmaps(activations.get("block5"), "Block-5 (last conv layer) feature maps")

    wandb_run.log({
        "feature_maps/block1": wandb.Image(fig1),
        "feature_maps/block5": wandb.Image(fig5),
        "feature_maps/input":  wandb.Image(pil_img),
    })
    plt.close("all")
    print("§2.4 – Feature maps logged.")


# ══════════════════════════════════════════════════════════════════════════════
# §2.5  Detection: Confidence & IoU table
# ══════════════════════════════════════════════════════════════════════════════

def _draw_bbox(ax, box_cxcywh, color, label=""):
    cx, cy, w, h = box_cxcywh
    x1, y1 = cx - w / 2, cy - h / 2
    rect = patches.Rectangle(
        (x1, y1), w, h, linewidth=2, edgecolor=color, facecolor="none"
    )
    ax.add_patch(rect)
    if label:
        ax.text(x1, y1 - 4, label, color=color, fontsize=7,
                fontweight="bold")


def evaluate_detection(det_ckpt: str, cls_ckpt: str,
                       wandb_run, n_samples: int = 10) -> None:
    """§2.5 – Log detection table with IoU and confidence score."""
    device = load_device()

    det_model = VGG11Localizer().to(device)
    det_model.load_state_dict(torch.load(det_ckpt, map_location="cpu"))
    det_model.eval()

    cls_model = VGG11Classifier(num_classes=37).to(device)
    cls_model.load_state_dict(torch.load(cls_ckpt, map_location="cpu"))
    cls_model.eval()

    dataset = OxfordIIITPetDataset(
        root=DATASET_ROOT, split="test", download=True, require_bbox=True
    )
    indices  = np.random.default_rng(0).choice(len(dataset), n_samples, replace=False)

    columns  = ["image", "breed", "confidence", "IoU", "failure?"]
    rows     = []
    img_logs = {}

    for i, idx in enumerate(indices):
        img_t, target = dataset[int(idx)]
        gt_bbox = target["bbox"]           # [4]
        label   = target["label"]

        inp = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_bbox = det_model(inp)[0].cpu()   # [4]
            logits    = cls_model(inp)[0].cpu()   # [37]

        probs      = torch.softmax(logits, dim=0)
        confidence = float(probs[label].item())
        iou        = iou_from_boxes(pred_bbox, gt_bbox)
        is_failure = (iou < 0.3) or (confidence < 0.3)

        # Visualise
        img_np = denorm(img_t)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_np)
        _draw_bbox(ax, gt_bbox.tolist(),   "lime",  "GT")
        _draw_bbox(ax, pred_bbox.tolist(), "red",   "Pred")
        ax.axis("off")
        ax.set_title(f"{BREED_NAMES[label]} | IoU={iou:.2f} | conf={confidence:.2f}",
                     fontsize=8)
        plt.tight_layout()

        key = f"detection/sample_{i}"
        img_logs[key] = wandb.Image(fig)
        plt.close(fig)

        rows.append([
            wandb.Image(Image.fromarray(img_np)),
            BREED_NAMES[label],
            round(confidence, 3),
            round(iou, 3),
            "YES" if is_failure else "",
        ])

    table = wandb.Table(columns=columns, data=rows)
    img_logs["detection/table"] = table
    wandb_run.log(img_logs)
    print(f"§2.5 – Detection table with {n_samples} samples logged.")


# ══════════════════════════════════════════════════════════════════════════════
# §2.6  Segmentation evaluation: Pixel Accuracy vs Dice Score
# ══════════════════════════════════════════════════════════════════════════════

def _mask_overlay(img_np: np.ndarray, mask: np.ndarray,
                  alpha: float = 0.5) -> np.ndarray:
    """Overlay a coloured trimap on a RGB image."""
    color_mask = TRIMAP_COLORS[mask.clip(0, 2)]
    return ((1 - alpha) * img_np + alpha * color_mask).astype(np.uint8)


def evaluate_segmentation(seg_ckpt: str, wandb_run,
                           n_samples: int = 5) -> None:
    """§2.6 – Log sample trimap predictions and track PA vs Dice."""
    device = load_device()
    model  = VGG11UNet(num_classes=3).to(device)
    model.load_state_dict(torch.load(seg_ckpt, map_location="cpu"))
    model.eval()

    dataset = OxfordIIITPetDataset(
        root=DATASET_ROOT, split="test", download=True
    )
    indices = np.random.default_rng(1).choice(len(dataset), n_samples, replace=False)

    columns = ["Original", "GT Trimap", "Predicted Trimap",
               "Pixel Accuracy", "Dice Score"]
    rows    = []
    log_dict = {}

    total_pix, total_dice, total_n = 0.0, 0.0, 0

    for i, idx in enumerate(indices):
        img_t, target = dataset[int(idx)]
        gt_mask = target["mask"]    # [224, 224]

        inp = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)         # [1, 3, 224, 224]

        pred_cls = logits.argmax(dim=1)[0].cpu()  # [224, 224]

        # Metrics
        pix  = (pred_cls == gt_mask).float().mean().item()
        eps  = 1e-6
        dice_per_cls = []
        for c in range(3):
            p = (pred_cls == c).float()
            t = (gt_mask  == c).float()
            dice_per_cls.append(
                (2.0 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
            )
        dice = torch.stack(dice_per_cls).mean().item()

        total_pix  += pix;  total_dice += dice;  total_n += 1

        img_np    = denorm(img_t)
        gt_arr    = gt_mask.numpy().astype(np.int32)
        pred_arr  = pred_cls.numpy().astype(np.int32)
        gt_vis    = _mask_overlay(img_np, gt_arr)
        pred_vis  = _mask_overlay(img_np, pred_arr)

        rows.append([
            wandb.Image(Image.fromarray(img_np)),
            wandb.Image(Image.fromarray(gt_vis)),
            wandb.Image(Image.fromarray(pred_vis)),
            round(pix, 3),
            round(dice, 3),
        ])

    # Log a few overlaid comparison images
    for i, row in enumerate(rows):
        log_dict[f"segmentation/sample_{i}/original"]  = row[0]
        log_dict[f"segmentation/sample_{i}/gt_mask"]   = row[1]
        log_dict[f"segmentation/sample_{i}/pred_mask"] = row[2]

    log_dict["segmentation/table"]       = wandb.Table(columns=columns, data=rows)
    log_dict["segmentation/mean_pix_acc"] = total_pix  / total_n
    log_dict["segmentation/mean_dice"]    = total_dice / total_n
    wandb_run.log(log_dict)
    print(f"§2.6 – Segmentation eval: mean pixel_acc={total_pix/total_n:.4f}"
          f"  mean_dice={total_dice/total_n:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# §2.7  Final pipeline showcase on novel internet images
# ══════════════════════════════════════════════════════════════════════════════

def run_final_pipeline(cls_ckpt: str, det_ckpt: str, seg_ckpt: str,
                       image_paths: list, wandb_run) -> None:
    """§2.7 – Run the unified pipeline on 3 novel images and log results."""
    device = load_device()

    cls_model = VGG11Classifier(num_classes=37).to(device)
    cls_model.load_state_dict(torch.load(cls_ckpt, map_location="cpu"))
    cls_model.eval()

    det_model = VGG11Localizer().to(device)
    det_model.load_state_dict(torch.load(det_ckpt, map_location="cpu"))
    det_model.eval()

    seg_model = VGG11UNet(num_classes=3).to(device)
    seg_model.load_state_dict(torch.load(seg_ckpt, map_location="cpu"))
    seg_model.eval()

    log_dict = {}

    for i, path in enumerate(image_paths):
        pil_img = Image.open(path).convert("RGB")
        inp     = preprocess_pil(pil_img).to(device)
        img_np  = np.array(pil_img.resize((_IMG_SIZE, _IMG_SIZE), Image.BILINEAR))

        with torch.no_grad():
            cls_logits = cls_model(inp)[0].cpu()
            pred_bbox  = det_model(inp)[0].cpu()
            seg_logits = seg_model(inp)

        probs     = torch.softmax(cls_logits, dim=0)
        top5      = torch.topk(probs, 5)
        breed_idx = int(top5.indices[0])
        breed     = BREED_NAMES[breed_idx] if breed_idx < len(BREED_NAMES) else str(breed_idx)
        conf      = float(top5.values[0])

        pred_mask = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

        # ── Figure: 3 panels (original + bbox, segmentation overlay, top-5 bar) ──
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Panel 1: original + predicted bbox
        axes[0].imshow(img_np)
        _draw_bbox(axes[0], pred_bbox.tolist(), "red", f"{breed} {conf:.2f}")
        axes[0].set_title("Detection")
        axes[0].axis("off")

        # Panel 2: segmentation overlay
        seg_vis = _mask_overlay(img_np, pred_mask, alpha=0.4)
        axes[1].imshow(seg_vis)
        axes[1].set_title("Segmentation")
        axes[1].axis("off")

        # Panel 3: top-5 breed probabilities
        names = [BREED_NAMES[j] if j < len(BREED_NAMES) else str(j)
                 for j in top5.indices.tolist()]
        vals  = [float(v) for v in top5.values.tolist()]
        axes[2].barh(names[::-1], vals[::-1])
        axes[2].set_xlim(0, 1)
        axes[2].set_title("Top-5 breeds")
        axes[2].set_xlabel("Probability")

        plt.suptitle(f"Novel image {i+1}: {os.path.basename(path)}")
        plt.tight_layout()

        log_dict[f"pipeline/novel_image_{i+1}"] = wandb.Image(fig)
        plt.close(fig)

    wandb_run.log(log_dict)
    print(f"§2.7 – Pipeline showcase logged for {len(image_paths)} images.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DA6401 Assignment-2 inference / evaluation")
    p.add_argument("--mode", required=True,
                   choices=["feat_maps", "detection", "segmentation", "pipeline"])
    p.add_argument("--wandb_project", default="da6401-assignment2")
    p.add_argument("--cls_ckpt",   default="checkpoints/classifier.pth")
    p.add_argument("--det_ckpt",   default="checkpoints/localizer.pth")
    p.add_argument("--seg_ckpt",   default="checkpoints/unet.pth")
    # For feat_maps
    p.add_argument("--image_path", default="", help="Path to a single image (§2.4)")
    # For pipeline showcase
    p.add_argument("--image_paths", nargs="+", default=[],
                   help="Paths to 3 novel internet images (§2.7)")
    return p.parse_args()


def main():
    args = parse_args()
    run  = wandb.init(project=args.wandb_project, name=f"inference_{args.mode}",
                      job_type="inference")

    if args.mode == "feat_maps":
        assert args.image_path, "--image_path required for feat_maps mode"
        visualize_feature_maps(args.cls_ckpt, args.image_path, run)

    elif args.mode == "detection":
        evaluate_detection(args.det_ckpt, args.cls_ckpt, run, n_samples=10)

    elif args.mode == "segmentation":
        evaluate_segmentation(args.seg_ckpt, run, n_samples=5)

    elif args.mode == "pipeline":
        assert len(args.image_paths) >= 3, \
            "Provide at least 3 image paths for pipeline showcase"
        run_final_pipeline(args.cls_ckpt, args.det_ckpt, args.seg_ckpt,
                           args.image_paths[:3], run)

    wandb.finish()


if __name__ == "__main__":
    main()
