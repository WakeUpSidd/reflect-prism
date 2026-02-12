import os
import math
import time
import csv
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm


# ----------------------------
# Config
# ----------------------------
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/Siddhant/data"))
OUT_ROOT  = Path(os.environ.get("OUT_ROOT",  "/workspace/Siddhant"))
RUN_NAME  = os.environ.get("RUN_NAME", "reflector_vit_baseline")

TRAIN_DIR = DATA_ROOT / "trans10k_export_bin" / "train"
VAL_DIR   = DATA_ROOT / "trans10k_export_bin" / "validation"

IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
LR = float(os.environ.get("LR", "3e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.05"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
SEED = int(os.environ.get("SEED", "42"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_DIR = OUT_ROOT / "checkpoints" / RUN_NAME
LOG_DIR  = OUT_ROOT / "logs" / RUN_NAME
VIS_DIR  = OUT_ROOT / "outputs" / RUN_NAME
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Dataset
# ----------------------------
class Trans10KBinary(Dataset):
    def __init__(self, root: Path, img_size: int, augment: bool):
        self.root = root
        self.img_dir = root / "rgb"
        self.mask_dir = root / "mask"
        self.img_size = img_size
        self.augment = augment
        self.files = sorted([p.name for p in self.img_dir.glob("*.png")])

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.files)

    def _resize(self, pil_img, is_mask: bool):
        if is_mask:
            return pil_img.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        return pil_img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(self.img_dir / name).convert("RGB")
        msk = Image.open(self.mask_dir / name).convert("L")  # 0 or 255

        # simple aug: horizontal flip
        if self.augment and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)

        img = self._resize(img, is_mask=False)
        msk = self._resize(msk, is_mask=True)

        img = np.asarray(img).astype(np.float32) / 255.0  # HWC
        img = torch.from_numpy(img).permute(2, 0, 1)      # CHW

        msk = (np.asarray(msk) > 0).astype(np.float32)    # HW in {0,1}
        msk = torch.from_numpy(msk)[None, :, :]           # 1HW

        return img, msk, name


# ----------------------------
# Model: ViT encoder + simple upsampling decoder
# ----------------------------
class ViTSeg(nn.Module):
    def __init__(self, backbone="vit_small_patch16_224", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool=""
        )
        embed_dim = self.backbone.num_features  # token dim

        # For 224 with patch16: 14x14 patch grid
        # Decoder upsamples 14->28->56->112->224
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        # ViT forward_features returns tokens, first is cls token
        tokens = self.backbone.forward_features(x)  # (B, 1+N, C) or (B,N,C) depending on timm
        if tokens.dim() != 3:
            raise RuntimeError(f"Unexpected token shape: {tokens.shape}")

        # Handle cls token if present
        if tokens.shape[1] == 1 + (IMG_SIZE // 16) * (IMG_SIZE // 16):
            tokens = tokens[:, 1:, :]

        B, N, C = tokens.shape
        grid = int(math.sqrt(N))
        feat = tokens.transpose(1, 2).contiguous().view(B, C, grid, grid)  # (B,C,14,14)
        logits = self.decoder(feat)  # (B,1,224,224)
        return logits


# ----------------------------
# Loss + metrics
# ----------------------------
def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    dice = 1 - (num + eps) / den
    return dice.mean()

@torch.no_grad()
def iou_score(logits, targets, thr=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3)) + eps
    iou = (inter + eps) / union
    return iou.mean().item()


def save_viz(img_tensor, gt_mask, pred_mask, out_path: Path):
    # img_tensor: (3,H,W) in [0,1], gt/pred: (H,W) in {0,1}
    img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gt = (gt_mask.cpu().numpy() * 255).astype(np.uint8)
    pr = (pred_mask.cpu().numpy() * 255).astype(np.uint8)

    # create 3-panel image: rgb | gt | pred
    rgb = Image.fromarray(img)
    gtI = Image.fromarray(gt, mode="L").convert("RGB")
    prI = Image.fromarray(pr, mode="L").convert("RGB")

    W, H = rgb.size
    canvas = Image.new("RGB", (W * 3, H))
    canvas.paste(rgb, (0, 0))
    canvas.paste(gtI, (W, 0))
    canvas.paste(prI, (W * 2, 0))
    canvas.save(out_path)


# ----------------------------
# Train / Val loops
# ----------------------------
def run_one_epoch(model, loader, optimizer, scaler, epoch, train: bool):
    model.train(train)
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0

    for imgs, masks, names in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(imgs)
                loss = bce(logits, masks) + 0.5 * dice_loss_with_logits(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item()
        total_iou += iou_score(logits.detach(), masks.detach())
        n_batches += 1

    return total_loss / max(1, n_batches), total_iou / max(1, n_batches)


def main():
    seed_everything(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_ds = Trans10KBinary(TRAIN_DIR, IMG_SIZE, augment=True)
    val_ds = Trans10KBinary(VAL_DIR, IMG_SIZE, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    model = ViTSeg(backbone="vit_small_patch16_224", pretrained=True).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    # CSV log
    log_path = LOG_DIR / "metrics.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_iou", "val_loss", "val_iou"])

    best_val = -1.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_iou = run_one_epoch(model, train_loader, optimizer, scaler, epoch, train=True)
        va_loss, va_iou = run_one_epoch(model, val_loader, optimizer, scaler, epoch, train=False)

        # save a small viz from first val batch
        model.eval()
        with torch.no_grad():
            imgs, masks, names = next(iter(val_loader))
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()

            save_viz(
                imgs[0].cpu(),
                masks[0, 0].cpu(),
                pred[0, 0].cpu(),
                VIS_DIR / f"epoch_{epoch:03d}_{names[0]}"
            )

        # checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "val_iou": va_iou,
        }
        torch.save(ckpt, CKPT_DIR / "last.pt")

        if va_iou > best_val:
            best_val = va_iou
            torch.save(ckpt, CKPT_DIR / "best.pt")

        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, tr_loss, tr_iou, va_loss, va_iou])

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {tr_loss:.4f} iou {tr_iou:.4f} | "
            f"val loss {va_loss:.4f} iou {va_iou:.4f} | "
            f"time {dt:.1f}s"
        )

    print("Done. Best val IoU:", best_val)
    print("Checkpoints:", CKPT_DIR)
    print("Visuals:", VIS_DIR)
    print("Logs:", log_path)


if __name__ == "__main__":
    main()
PY
