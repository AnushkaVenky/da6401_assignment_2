"""Training entry-point for all four tasks.

Usage examples
--------------
# Task 1 – classification (with BN, dropout p=0.5)
python train.py --task cls --epochs 30 --dropout_p 0.5 --use_bn

# Task 1 – ablation: no BN
python train.py --task cls --epochs 30 --dropout_p 0.5 --no_bn --run_name vgg11_no_bn

# Task 1 – ablation: no dropout
python train.py --task cls --epochs 30 --dropout_p 0.0 --run_name vgg11_no_dropout

# Task 2 – object localisation
python train.py --task det --epochs 30

# Task 3 – segmentation, strict freeze (transfer learning)
python train.py --task seg --epochs 30 --freeze_strategy strict \
    --pretrained_backbone checkpoints/classifier.pth

# Task 3 – partial fine-tuning
python train.py --task seg --epochs 30 --freeze_strategy partial \
    --pretrained_backbone checkpoints/classifier.pth

# Task 3 – full fine-tuning
python train.py --task seg --epochs 30 --freeze_strategy full \
    --pretrained_backbone checkpoints/classifier.pth

# Task 4 – unified multi-task training (joint fine-tune from individual ckpts)
python train.py --task multi --epochs 20 \
    --classifier_ckpt checkpoints/classifier.pth \
    --localizer_ckpt  checkpoints/localizer.pth \
    --unet_ckpt       checkpoints/unet.pth
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import wandb

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_ROOT  = "./data/oxford_pet"
CKPT_DIR      = "./checkpoints"
VAL_FRACTION  = 0.1
NUM_WORKERS   = 2
_IMG_SIZE     = 224


# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def dice_loss(pred: torch.Tensor, target: torch.Tensor,
              num_classes: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss averaged over all classes.

    Args:
        pred:   [B, C, H, W] raw logits.
        target: [B, H, W]    long tensor with class indices.
    """
    pred_soft = torch.softmax(pred, dim=1)                # [B, C, H, W]
    target_oh = F.one_hot(target, num_classes)            # [B, H, W, C]
    target_oh = target_oh.permute(0, 3, 1, 2).float()    # [B, C, H, W]

    intersection = (pred_soft * target_oh).sum(dim=(2, 3))   # [B, C]
    denominator  = pred_soft.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))

    dice = (2.0 * intersection + eps) / (denominator + eps)  # [B, C]
    return 1.0 - dice.mean()


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_cls = pred.argmax(dim=1)
    return (pred_cls == target).float().mean().item()


def dice_score(pred: torch.Tensor, target: torch.Tensor,
               num_classes: int = 3, eps: float = 1e-6) -> float:
    pred_cls = pred.argmax(dim=1)
    scores = []
    for c in range(num_classes):
        p = (pred_cls == c).float()
        t = (target == c).float()
        scores.append((2.0 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps))
    return torch.stack(scores).mean().item()


def compute_iou_metric(pred_boxes: torch.Tensor,
                       tgt_boxes: torch.Tensor, eps: float = 1e-6) -> float:
    """Mean IoU metric (not loss) over a batch."""
    iou_loss_fn = IoULoss(eps=eps, reduction="none")
    per_sample  = 1.0 - iou_loss_fn(pred_boxes, tgt_boxes)
    return per_sample.mean().item()


def log_activation_histogram(model: nn.Module, sample_batch: torch.Tensor,
                              layer_name: str, device: torch.device) -> None:
    """Hook into a named layer and log its output histogram to W&B."""
    activations = {}

    def _hook(module, inp, out):
        activations["act"] = out.detach().cpu()

    # Find the target layer by name
    target = dict(model.named_modules()).get(layer_name)
    if target is None:
        return
    handle = target.register_forward_hook(_hook)
    model.eval()
    with torch.no_grad():
        model(sample_batch.to(device))
    handle.remove()

    if "act" in activations:
        vals = activations["act"].numpy().flatten()
        wandb.log({f"activation_hist/{layer_name}": wandb.Histogram(vals)})


def make_data_loaders(task: str, batch_size: int, require_bbox: bool = False):
    """Build train and val DataLoaders for the given task."""
    full_dataset = OxfordIIITPetDataset(
        root=DATASET_ROOT,
        split="trainval",
        download=True,
        require_bbox=require_bbox,
    )
    n_val   = max(1, int(len(full_dataset) * VAL_FRACTION))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader


def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  → Saved checkpoint: {path}")


def apply_freeze_strategy(model: VGG11UNet, strategy: str) -> None:
    """Apply encoder freeze strategy for transfer learning (Task 3 / §2.3)."""
    if strategy == "strict":
        # Freeze entire encoder – only decoder trains
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("  Freeze strategy: STRICT (encoder fully frozen)")

    elif strategy == "partial":
        # Freeze early (generic) blocks; unfreeze later (semantic) blocks
        frozen_attrs = ["block1", "pool1", "block2", "pool2", "block3", "pool3"]
        for attr in frozen_attrs:
            block = getattr(model.encoder, attr, None)
            if block is not None:
                for p in block.parameters():
                    p.requires_grad = False
        print("  Freeze strategy: PARTIAL (blocks 1-3 frozen, 4-5 trainable)")

    else:  # "full"
        # All parameters trainable
        for p in model.parameters():
            p.requires_grad = True
        print("  Freeze strategy: FULL (entire network fine-tuned)")


def load_encoder_weights(model: nn.Module, classifier_path: str) -> None:
    """Load VGG11Encoder weights from a saved VGG11Classifier checkpoint."""
    state = torch.load(classifier_path, map_location="cpu")
    enc_state = {k[len("encoder."):]: v
                 for k, v in state.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc_state, strict=True)
    print(f"  Loaded pretrained encoder from {classifier_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Task 1 – Classification
# ══════════════════════════════════════════════════════════════════════════════

def train_classifier(args):
    run_name = args.run_name or (
        f"cls_bn{int(args.use_bn)}_dp{args.dropout_p}"
    )
    wandb.init(project=args.wandb_project, name=run_name,
               config=vars(args), reinit=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = VGG11Classifier(
        num_classes=37, dropout_p=args.dropout_p, use_bn=args.use_bn
    ).to(device)

    train_loader, val_loader = make_data_loaders("cls", args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # Grab a fixed sample batch for activation histograms (§2.1)
    sample_batch, _ = next(iter(val_loader))

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, targets in train_loader:
            imgs   = imgs.to(device)
            labels = targets["label"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * imgs.size(0)
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total   += imgs.size(0)

        scheduler.step()
        train_loss = t_loss / t_total
        train_acc  = t_correct / t_total

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs   = imgs.to(device)
                labels = targets["label"].to(device)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                v_loss    += loss.item() * imgs.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)

        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        log_dict = {
            "epoch": epoch,
            "train/loss": train_loss, "train/acc": train_acc,
            "val/loss":   val_loss,   "val/acc":   val_acc,
            "lr": scheduler.get_last_lr()[0],
        }

        # §2.1 – log activation distribution of 3rd conv layer every 5 epochs
        if epoch % 5 == 0:
            log_activation_histogram(
                model, sample_batch,
                "encoder.block3",   # first layer of block3 ≡ 3rd conv overall
                device,
            )

        wandb.log(log_dict)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, os.path.join(CKPT_DIR, "classifier.pth"))

    print(f"Best val accuracy: {best_val_acc:.4f}")
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Task 2 – Object Localisation
# ══════════════════════════════════════════════════════════════════════════════

def train_localizer(args):
    run_name = args.run_name or "det_localizer"
    wandb.init(project=args.wandb_project, name=run_name,
               config=vars(args), reinit=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = VGG11Localizer(dropout_p=args.dropout_p).to(device)

    # If pretrained backbone is available, initialise encoder from it
    if args.pretrained_backbone and os.path.exists(args.pretrained_backbone):
        load_encoder_weights(model, args.pretrained_backbone)

    # Only use images that have XML bounding-box annotations
    train_loader, val_loader = make_data_loaders(
        "det", args.batch_size, require_bbox=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    mse_fn  = nn.MSELoss()
    iou_fn  = IoULoss(reduction="mean")

    best_val_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t_loss, t_iou, t_total = 0.0, 0.0, 0
        for imgs, targets in train_loader:
            imgs  = imgs.to(device)
            bboxes = targets["bbox"].to(device)

            optimizer.zero_grad()
            pred  = model(imgs)
            mse   = mse_fn(pred, bboxes)
            iou_l = iou_fn(pred, bboxes)
            loss  = mse + iou_l
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                iou_m = compute_iou_metric(pred, bboxes)
            t_loss  += loss.item() * imgs.size(0)
            t_iou   += iou_m * imgs.size(0)
            t_total += imgs.size(0)

        scheduler.step()
        train_loss = t_loss / t_total
        train_iou  = t_iou  / t_total

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        v_loss, v_iou, v_total = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs   = imgs.to(device)
                bboxes = targets["bbox"].to(device)
                pred   = model(imgs)
                mse    = mse_fn(pred, bboxes)
                iou_l  = iou_fn(pred, bboxes)
                loss   = mse + iou_l
                iou_m  = compute_iou_metric(pred, bboxes)
                v_loss  += loss.item() * imgs.size(0)
                v_iou   += iou_m * imgs.size(0)
                v_total += imgs.size(0)

        val_loss = v_loss / v_total
        val_iou  = v_iou  / v_total

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} iou={train_iou:.4f} | "
              f"val_loss={val_loss:.4f} iou={val_iou:.4f}")

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss, "train/iou": train_iou,
            "val/loss":   val_loss,   "val/iou":   val_iou,
            "lr": scheduler.get_last_lr()[0],
        })

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_checkpoint(model, os.path.join(CKPT_DIR, "localizer.pth"))

    print(f"Best val IoU: {best_val_iou:.4f}")
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Task 3 – Semantic Segmentation (+ transfer learning ablation §2.3)
# ══════════════════════════════════════════════════════════════════════════════

def train_segmentation(args):
    strategy  = args.freeze_strategy
    run_name  = args.run_name or f"seg_{strategy}"
    wandb.init(project=args.wandb_project, name=run_name,
               config=vars(args), reinit=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(device)

    # Optionally initialise encoder from a pretrained classifier
    if args.pretrained_backbone and os.path.exists(args.pretrained_backbone):
        load_encoder_weights(model, args.pretrained_backbone)

    # Apply the transfer-learning freeze strategy
    apply_freeze_strategy(model, strategy)

    train_loader, val_loader = make_data_loaders("seg", args.batch_size)

    # Only optimise parameters that require gradients
    params     = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.Adam(params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    ce_fn      = nn.CrossEntropyLoss()

    best_val_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t_loss, t_pix, t_dice, t_n = 0.0, 0.0, 0.0, 0
        for imgs, targets in train_loader:
            imgs  = imgs.to(device)
            masks = targets["mask"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            ce     = ce_fn(logits, masks)
            dl     = dice_loss(logits, masks, num_classes=3)
            loss   = ce + dl
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pix  = pixel_accuracy(logits, masks)
                dice = dice_score(logits, masks, num_classes=3)
            t_loss += loss.item() * imgs.size(0)
            t_pix  += pix  * imgs.size(0)
            t_dice += dice * imgs.size(0)
            t_n    += imgs.size(0)

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        v_loss, v_pix, v_dice, v_n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs  = imgs.to(device)
                masks = targets["mask"].to(device)
                logits = model(imgs)
                ce     = ce_fn(logits, masks)
                dl     = dice_loss(logits, masks, num_classes=3)
                loss   = ce + dl
                pix    = pixel_accuracy(logits, masks)
                dice   = dice_score(logits, masks, num_classes=3)
                v_loss += loss.item() * imgs.size(0)
                v_pix  += pix  * imgs.size(0)
                v_dice += dice * imgs.size(0)
                v_n    += imgs.size(0)

        t_loss_e = t_loss / t_n; t_pix_e = t_pix / t_n; t_dice_e = t_dice / t_n
        v_loss_e = v_loss / v_n; v_pix_e = v_pix / v_n; v_dice_e = v_dice / v_n

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss={t_loss_e:.4f} pix={t_pix_e:.4f} dice={t_dice_e:.4f} | "
              f"val loss={v_loss_e:.4f} pix={v_pix_e:.4f} dice={v_dice_e:.4f}")

        wandb.log({
            "epoch": epoch,
            "train/loss": t_loss_e, "train/pixel_acc": t_pix_e, "train/dice": t_dice_e,
            "val/loss":   v_loss_e, "val/pixel_acc":   v_pix_e, "val/dice":   v_dice_e,
            "lr": scheduler.get_last_lr()[0],
        })

        if v_dice_e > best_val_dice:
            best_val_dice = v_dice_e
            save_checkpoint(model, os.path.join(CKPT_DIR, "unet.pth"))

    print(f"Best val Dice: {best_val_dice:.4f}")
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Task 4 – Unified Multi-Task Training
# ══════════════════════════════════════════════════════════════════════════════

def train_multitask(args):
    """Joint fine-tune of a shared VGG11 backbone with three task heads."""
    run_name = args.run_name or "multitask_joint"
    wandb.init(project=args.wandb_project, name=run_name,
               config=vars(args), reinit=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialise individual models
    clf = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
    loc = VGG11Localizer(dropout_p=args.dropout_p).to(device)
    seg = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(device)

    # Load pre-trained weights if provided
    if args.classifier_ckpt and os.path.exists(args.classifier_ckpt):
        clf.load_state_dict(torch.load(args.classifier_ckpt, map_location="cpu"))
        print(f"  Loaded classifier from {args.classifier_ckpt}")
    if args.localizer_ckpt and os.path.exists(args.localizer_ckpt):
        loc.load_state_dict(torch.load(args.localizer_ckpt, map_location="cpu"))
        print(f"  Loaded localizer from {args.localizer_ckpt}")
    if args.unet_ckpt and os.path.exists(args.unet_ckpt):
        seg.load_state_dict(torch.load(args.unet_ckpt, map_location="cpu"))
        print(f"  Loaded UNet from {args.unet_ckpt}")

    # Share backbone: overwrite seg/loc encoder with clf encoder weights
    seg.encoder.load_state_dict(clf.encoder.state_dict())
    loc.encoder.load_state_dict(clf.encoder.state_dict())

    # Pool all parameters under one optimizer
    all_params = (
        list(clf.parameters()) +
        list(loc.regressor.parameters()) +
        list(seg.decoder.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=args.lr * 0.1,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    ce_fn  = nn.CrossEntropyLoss()
    mse_fn = nn.MSELoss()
    iou_fn = IoULoss(reduction="mean")

    train_loader, val_loader = make_data_loaders("cls", args.batch_size)

    lam_cls, lam_det, lam_seg = 1.0, 1.0, 1.0

    for epoch in range(1, args.epochs + 1):
        clf.train(); loc.train(); seg.train()
        t_loss, t_n = 0.0, 0

        for imgs, targets in train_loader:
            imgs   = imgs.to(device)
            labels = targets["label"].to(device)
            bboxes = targets["bbox"].to(device)
            masks  = targets["mask"].to(device)

            # Shared backbone forward (use clf's encoder, return skip features)
            bottleneck, feats = clf.encoder(imgs, return_features=True)

            # Three head forwards
            cls_logits = clf.classifier(bottleneck)
            raw_loc    = loc.regressor(bottleneck)
            bbox_pred  = torch.sigmoid(raw_loc) * 224.0
            seg_logits = seg.decoder(bottleneck, feats)

            # Combined loss
            l_cls = ce_fn(cls_logits, labels)
            l_det = mse_fn(bbox_pred, bboxes) + iou_fn(bbox_pred, bboxes)
            l_seg = ce_fn(seg_logits, masks) + dice_loss(seg_logits, masks)
            loss  = lam_cls * l_cls + lam_det * l_det + lam_seg * l_seg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss += loss.item() * imgs.size(0)
            t_n    += imgs.size(0)

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        clf.eval(); loc.eval(); seg.eval()
        v_cls_acc, v_iou, v_dice, v_n = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs   = imgs.to(device)
                labels = targets["label"].to(device)
                bboxes = targets["bbox"].to(device)
                masks  = targets["mask"].to(device)

                bottleneck, feats = clf.encoder(imgs, return_features=True)
                cls_logits = clf.classifier(bottleneck)
                raw_loc    = loc.regressor(bottleneck)
                bbox_pred  = torch.sigmoid(raw_loc) * 224.0
                seg_logits = seg.decoder(bottleneck, feats)

                v_cls_acc += (cls_logits.argmax(1) == labels).float().sum().item()
                v_iou     += compute_iou_metric(bbox_pred, bboxes) * imgs.size(0)
                v_dice    += dice_score(seg_logits, masks) * imgs.size(0)
                v_n       += imgs.size(0)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={t_loss/t_n:.4f} | "
              f"val cls_acc={v_cls_acc/v_n:.4f} "
              f"iou={v_iou/v_n:.4f} dice={v_dice/v_n:.4f}")

        wandb.log({
            "epoch": epoch,
            "train/loss": t_loss / t_n,
            "val/cls_acc": v_cls_acc / v_n,
            "val/iou":    v_iou / v_n,
            "val/dice":   v_dice / v_n,
            "lr": scheduler.get_last_lr()[0],
        })

    # Save all three checkpoints after joint training
    save_checkpoint(clf, os.path.join(CKPT_DIR, "classifier.pth"))
    save_checkpoint(loc, os.path.join(CKPT_DIR, "localizer.pth"))
    save_checkpoint(seg, os.path.join(CKPT_DIR, "unet.pth"))
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing & main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DA6401 Assignment-2 training script")

    p.add_argument("--task", choices=["cls", "det", "seg", "multi"],
                   required=True, help="Which task to train")

    # ── Common hyper-parameters ───────────────────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout_p",    type=float, default=0.5)

    # ── BatchNorm ablation (§2.1) ─────────────────────────────────────────────
    bn_group = p.add_mutually_exclusive_group()
    bn_group.add_argument("--use_bn",  dest="use_bn", action="store_true",  default=True)
    bn_group.add_argument("--no_bn",   dest="use_bn", action="store_false")

    # ── Transfer learning (§2.3) ──────────────────────────────────────────────
    p.add_argument("--freeze_strategy",
                   choices=["strict", "partial", "full"], default="full",
                   help="Encoder freeze strategy for segmentation")
    p.add_argument("--pretrained_backbone", type=str, default="",
                   help="Path to classifier.pth to initialise encoder")

    # ── Multi-task checkpoints (task=multi) ───────────────────────────────────
    p.add_argument("--classifier_ckpt", type=str,
                   default=os.path.join(CKPT_DIR, "classifier.pth"))
    p.add_argument("--localizer_ckpt",  type=str,
                   default=os.path.join(CKPT_DIR, "localizer.pth"))
    p.add_argument("--unet_ckpt",       type=str,
                   default=os.path.join(CKPT_DIR, "unet.pth"))

    # ── W&B ───────────────────────────────────────────────────────────────────
    p.add_argument("--wandb_project", type=str, default="da6401-assignment2")
    p.add_argument("--run_name",      type=str, default="",
                   help="W&B run name (auto-generated if empty)")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(CKPT_DIR, exist_ok=True)

    dispatch = {
        "cls":   train_classifier,
        "det":   train_localizer,
        "seg":   train_segmentation,
        "multi": train_multitask,
    }
    dispatch[args.task](args)


if __name__ == "__main__":
    main()
