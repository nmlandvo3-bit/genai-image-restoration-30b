from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageEnhance
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from torchvision.transforms import functional as TF


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXP7_ROOT = (
    PROJECT_ROOT
    / "data"
    / "experiment_7"
)

TRAIN_DAMAGED_DIR = (
    EXP7_ROOT
    / "train"
    / "damaged"
)

TRAIN_MASK_DIR = (
    EXP7_ROOT
    / "train"
    / "masks"
)

VAL_DAMAGED_DIR = (
    EXP7_ROOT
    / "validation"
    / "damaged"
)

VAL_MASK_DIR = (
    EXP7_ROOT
    / "validation"
    / "masks"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "experiment_11c_damage_detector"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "experiment_11c"
)

BEST_MODEL_PATH = (
    MODEL_ROOT
    / "best_unet.pt"
)

FINAL_MODEL_PATH = (
    MODEL_ROOT
    / "final_unet.pt"
)

TRAIN_LOG_PATH = (
    RESULTS_ROOT
    / "training_log.csv"
)

THRESHOLD_RESULTS_PATH = (
    RESULTS_ROOT
    / "validation_threshold_sweep.csv"
)

CONFIG_PATH = (
    RESULTS_ROOT
    / "experiment_config.json"
)

IMAGE_SIZE = 512

BASE_CHANNELS = 48

BATCH_SIZE = 4

NUM_WORKERS = 2

MAX_EPOCHS = 40

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 8

MIN_EPOCHS_BEFORE_STOP = 20

SEED = 42


# ============================================================
# Loss configuration
# ============================================================

BCE_WEIGHT = 0.40
TVERSKY_WEIGHT = 0.60

TVERSKY_ALPHA = 0.35
TVERSKY_BETA = 0.65
TVERSKY_GAMMA = 0.75


# ============================================================
# Validation threshold sweep
# ============================================================

THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


# ============================================================
# Reproducibility
# ============================================================


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ============================================================
# Dataset
# ============================================================


class DamageDataset(Dataset):
    def __init__(
        self,
        damaged_dir: Path,
        mask_dir: Path,
        augment: bool,
    ) -> None:
        self.damaged_dir = damaged_dir
        self.mask_dir = mask_dir
        self.augment = augment

        if not damaged_dir.exists():
            raise FileNotFoundError(
                f"Damaged directory not found: {damaged_dir}"
            )

        if not mask_dir.exists():
            raise FileNotFoundError(
                f"Mask directory not found: {mask_dir}"
            )

        self.images = sorted(
            damaged_dir.glob("*.png")
        )

        if not self.images:
            raise RuntimeError(
                f"No PNG images found in {damaged_dir}"
            )

        for image_path in self.images:
            mask_path = (
                self.mask_dir
                / image_path.name
            )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing mask: {mask_path}"
                )

    def __len__(self) -> int:
        return len(self.images)

    def get_mask_ratio(
        self,
        index: int,
    ) -> float:
        image_path = self.images[index]

        mask_path = (
            self.mask_dir
            / image_path.name
        )

        mask = (
            Image.open(mask_path)
            .convert("L")
            .resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                Image.Resampling.NEAREST,
            )
        )

        array = np.asarray(
            mask,
            dtype=np.uint8,
        )

        return float(
            np.mean(
                array >= 127
            )
        )

    def __getitem__(
        self,
        index: int,
    ):
        image_path = self.images[index]

        mask_path = (
            self.mask_dir
            / image_path.name
        )

        image = (
            Image.open(image_path)
            .convert("RGB")
            .resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                Image.Resampling.LANCZOS,
            )
        )

        mask = (
            Image.open(mask_path)
            .convert("L")
            .resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                Image.Resampling.NEAREST,
            )
        )

        if self.augment:
            image, mask = self.apply_augmentation(
                image,
                mask,
            )

        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        image_array = (
            image_array
            / 255.0
        )

        image_tensor = (
            torch.from_numpy(
                image_array
            )
            .permute(
                2,
                0,
                1,
            )
        )

        mask_array = np.asarray(
            mask,
            dtype=np.float32,
        )

        mask_array = (
            mask_array >= 127
        ).astype(
            np.float32
        )

        mask_tensor = (
            torch.from_numpy(
                mask_array
            )
            .unsqueeze(0)
        )

        return (
            image_tensor,
            mask_tensor,
        )

    def apply_augmentation(
        self,
        image: Image.Image,
        mask: Image.Image,
    ):
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.25:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        rotation_choice = random.choice(
            [
                0,
                0,
                0,
                90,
                180,
                270,
            ]
        )

        if rotation_choice != 0:
            image = image.rotate(
                rotation_choice
            )

            mask = mask.rotate(
                rotation_choice
            )

        if random.random() < 0.30:
            brightness = random.uniform(
                0.85,
                1.15,
            )

            image = (
                ImageEnhance.Brightness(
                    image
                )
                .enhance(
                    brightness
                )
            )

        if random.random() < 0.30:
            contrast = random.uniform(
                0.85,
                1.15,
            )

            image = (
                ImageEnhance.Contrast(
                    image
                )
                .enhance(
                    contrast
                )
            )

        return (
            image,
            mask,
        )


# ============================================================
# Thin-mask-aware sampler
# ============================================================


def create_sample_weights(
    dataset: DamageDataset,
) -> list[float]:
    weights = []

    ratios = []

    print()
    print(
        "Calculating training mask-area weights..."
    )

    for index in range(
        len(dataset)
    ):
        ratio = dataset.get_mask_ratio(
            index
        )

        ratios.append(
            ratio
        )

        # Small/thin masks receive larger sampling weights.
        #
        # Large region:
        #     normal weighting
        #
        # Medium region:
        #     moderate weighting
        #
        # Very thin/small region:
        #     much higher chance of being sampled
        if ratio < 0.01:
            weight = 4.0

        elif ratio < 0.025:
            weight = 3.0

        elif ratio < 0.05:
            weight = 2.0

        elif ratio < 0.10:
            weight = 1.5

        else:
            weight = 1.0

        weights.append(
            weight
        )

    ratios_array = np.asarray(
        ratios,
        dtype=np.float64,
    )

    print(
        f"Mean mask ratio : "
        f"{ratios_array.mean():.4f}"
    )

    print(
        f"Min mask ratio  : "
        f"{ratios_array.min():.4f}"
    )

    print(
        f"Max mask ratio  : "
        f"{ratios_array.max():.4f}"
    )

    print()
    print(
        "Sampling weight distribution:"
    )

    unique_weights = sorted(
        set(weights)
    )

    for weight in unique_weights:
        count = sum(
            1
            for value in weights
            if value == weight
        )

        print(
            f"  {weight:.1f}x : "
            f"{count} samples"
        )

    return weights


# ============================================================
# U-Net
# ============================================================


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.SiLU(
                inplace=True
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.pool = nn.MaxPool2d(
            2
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv(
            self.pool(x)
        )


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            out_channels
            + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)

        diff_y = (
            skip.size(2)
            - x.size(2)
        )

        diff_x = (
            skip.size(3)
            - x.size(3)
        )

        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat(
            [
                skip,
                x,
            ],
            dim=1,
        )

        return self.conv(x)


class DamageUNetImproved(nn.Module):
    def __init__(
        self,
        base_channels: int,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.input = DoubleConv(
            3,
            c1,
        )

        self.down1 = Down(
            c1,
            c2,
        )

        self.down2 = Down(
            c2,
            c3,
        )

        self.down3 = Down(
            c3,
            c4,
        )

        self.down4 = Down(
            c4,
            c5,
        )

        self.up1 = Up(
            c5,
            c4,
            c4,
        )

        self.up2 = Up(
            c4,
            c3,
            c3,
        )

        self.up3 = Up(
            c3,
            c2,
            c2,
        )

        self.up4 = Up(
            c2,
            c1,
            c1,
        )

        self.output = nn.Conv2d(
            c1,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x1 = self.input(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x = self.up1(
            x5,
            x4,
        )

        x = self.up2(
            x,
            x3,
        )

        x = self.up3(
            x,
            x2,
        )

        x = self.up4(
            x,
            x1,
        )

        return self.output(x)


# ============================================================
# Focal Tversky Loss
# ============================================================


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(
            logits
        )

        dims = (
            1,
            2,
            3,
        )

        tp = (
            probabilities
            * targets
        ).sum(
            dim=dims
        )

        fp = (
            probabilities
            * (
                1.0
                - targets
            )
        ).sum(
            dim=dims
        )

        fn = (
            (
                1.0
                - probabilities
            )
            * targets
        ).sum(
            dim=dims
        )

        tversky = (
            tp
            + self.smooth
        ) / (
            tp
            + self.alpha
            * fp
            + self.beta
            * fn
            + self.smooth
        )

        loss = (
            1.0
            - tversky
        ) ** self.gamma

        return loss.mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.bce = (
            nn.BCEWithLogitsLoss()
        )

        self.tversky = (
            FocalTverskyLoss(
                alpha=TVERSKY_ALPHA,
                beta=TVERSKY_BETA,
                gamma=TVERSKY_GAMMA,
            )
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce_loss = self.bce(
            logits,
            targets,
        )

        tversky_loss = (
            self.tversky(
                logits,
                targets,
            )
        )

        return (
            BCE_WEIGHT
            * bce_loss
            + TVERSKY_WEIGHT
            * tversky_loss
        )


# ============================================================
# Segmentation metrics
# ============================================================


def batch_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    predictions = (
        probabilities
        >= threshold
    )

    targets = (
        targets
        >= 0.5
    )

    tp = (
        predictions
        & targets
    ).sum().item()

    fp = (
        predictions
        & ~targets
    ).sum().item()

    fn = (
        ~predictions
        & targets
    ).sum().item()

    epsilon = 1e-8

    iou = (
        tp
        / (
            tp
            + fp
            + fn
            + epsilon
        )
    )

    dice = (
        2.0
        * tp
        / (
            2.0
            * tp
            + fp
            + fn
            + epsilon
        )
    )

    precision = (
        tp
        / (
            tp
            + fp
            + epsilon
        )
    )

    recall = (
        tp
        / (
            tp
            + fn
            + epsilon
        )
    )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
    }


# ============================================================
# Training epoch
# ============================================================


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()

    losses = []

    for images, masks in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        losses.append(
            loss.item()
        )

    return float(
        np.mean(losses)
    )


# ============================================================
# Validation
# ============================================================


@torch.no_grad()
def validation_pass(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
):
    model.eval()

    losses = []

    all_probabilities = []
    all_targets = []

    for images, masks in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

        probabilities = torch.sigmoid(
            logits
        )

        losses.append(
            loss.item()
        )

        all_probabilities.append(
            probabilities.float().cpu()
        )

        all_targets.append(
            masks.float().cpu()
        )

    probabilities = torch.cat(
        all_probabilities,
        dim=0,
    )

    targets = torch.cat(
        all_targets,
        dim=0,
    )

    metrics = batch_metrics(
        probabilities,
        targets,
        threshold,
    )

    return (
        float(
            np.mean(losses)
        ),
        metrics,
        probabilities,
        targets,
    )


# ============================================================
# Threshold search
# ============================================================


def find_best_threshold(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
):
    best_threshold = None
    best_metrics = None
    rows = []

    for threshold in THRESHOLDS:
        metrics = batch_metrics(
            probabilities,
            targets,
            threshold,
        )

        row = {
            "threshold": threshold,
            **metrics,
        }

        rows.append(row)

        if (
            best_metrics is None
            or metrics["dice"]
            > best_metrics["dice"]
        ):
            best_threshold = threshold
            best_metrics = metrics

    return (
        best_threshold,
        best_metrics,
        rows,
    )


# ============================================================
# CSV helpers
# ============================================================


def initialize_training_log() -> None:
    with TRAIN_LOG_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "epoch",
                "learning_rate",
                "train_loss",
                "val_loss",
                "val_iou_050",
                "val_dice_050",
                "val_precision_050",
                "val_recall_050",
                "epoch_seconds",
                "peak_vram_gb",
            ]
        )


def append_training_log(
    epoch: int,
    learning_rate: float,
    train_loss: float,
    val_loss: float,
    val_metrics: dict,
    epoch_seconds: float,
    peak_vram_gb: float,
) -> None:
    with TRAIN_LOG_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                epoch,
                learning_rate,
                train_loss,
                val_loss,
                val_metrics["iou"],
                val_metrics["dice"],
                val_metrics["precision"],
                val_metrics["recall"],
                epoch_seconds,
                peak_vram_gb,
            ]
        )


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 94)
    print(
        "EXPERIMENT 11C - IMPROVED AUTOMATIC DAMAGE DETECTOR"
    )
    print("=" * 94)

    set_seed()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"GPU : "
        f"{torch.cuda.get_device_name(0)}"
    )

    total_vram = (
        torch.cuda
        .get_device_properties(0)
        .total_memory
        / 1024**3
    )

    print(
        f"VRAM: {total_vram:.2f} GB"
    )

    # ========================================================
    # Datasets
    # ========================================================

    train_dataset = DamageDataset(
        damaged_dir=TRAIN_DAMAGED_DIR,
        mask_dir=TRAIN_MASK_DIR,
        augment=True,
    )

    validation_dataset = DamageDataset(
        damaged_dir=VAL_DAMAGED_DIR,
        mask_dir=VAL_MASK_DIR,
        augment=False,
    )

    print()
    print(
        f"Training samples   : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples : "
        f"{len(validation_dataset)}"
    )

    # ========================================================
    # Weighted sampling
    # ========================================================

    sample_weights = (
        create_sample_weights(
            train_dataset
        )
    )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double,
        ),
        num_samples=len(
            sample_weights
        ),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    # ========================================================
    # Model
    # ========================================================

    model = DamageUNetImproved(
        base_channels=BASE_CHANNELS
    ).to(
        device
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        f"Base channels        : "
        f"{BASE_CHANNELS}"
    )

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    print(
        f"Batch size           : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Maximum epochs       : "
        f"{MAX_EPOCHS}"
    )

    print(
        f"Learning rate        : "
        f"{LEARNING_RATE}"
    )

    # ========================================================
    # Optimizer / scheduler
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS,
            eta_min=1e-6,
        )
    )

    criterion = CombinedSegmentationLoss()

    scaler = torch.amp.GradScaler(
        "cuda"
    )

    initialize_training_log()

    # ========================================================
    # Save experiment config
    # ========================================================

    config = {
        "image_size": IMAGE_SIZE,
        "base_channels": BASE_CHANNELS,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "bce_weight": BCE_WEIGHT,
        "tversky_weight": TVERSKY_WEIGHT,
        "tversky_alpha": TVERSKY_ALPHA,
        "tversky_beta": TVERSKY_BETA,
        "tversky_gamma": TVERSKY_GAMMA,
        "seed": SEED,
    }

    with CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )

    # ========================================================
    # Training
    # ========================================================

    best_dice = -1.0
    best_epoch = -1

    epochs_without_improvement = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        epoch_start = (
            time.perf_counter()
        )

        torch.cuda.reset_peak_memory_stats()

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
        )

        (
            val_loss,
            val_metrics,
            _,
            _,
        ) = validation_pass(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            threshold=0.50,
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        peak_vram_gb = (
            torch.cuda
            .max_memory_allocated()
            / 1024**3
        )

        print()
        print(
            f"Epoch "
            f"{epoch:02d}/{MAX_EPOCHS}"
        )

        print(
            f"LR         : "
            f"{current_lr:.8f}"
        )

        print(
            f"Train loss : "
            f"{train_loss:.6f}"
        )

        print(
            f"Val loss   : "
            f"{val_loss:.6f}"
        )

        print(
            f"IoU @0.50  : "
            f"{val_metrics['iou']:.4f}"
        )

        print(
            f"Dice @0.50 : "
            f"{val_metrics['dice']:.4f}"
        )

        print(
            f"Precision  : "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"Recall     : "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"Peak VRAM  : "
            f"{peak_vram_gb:.2f} GB"
        )

        print(
            f"Time       : "
            f"{epoch_seconds:.1f}s"
        )

        append_training_log(
            epoch=epoch,
            learning_rate=current_lr,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
            epoch_seconds=epoch_seconds,
            peak_vram_gb=peak_vram_gb,
        )

        if (
            val_metrics["dice"]
            > best_dice
        ):
            best_dice = (
                val_metrics["dice"]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "base_channels": (
                        BASE_CHANNELS
                    ),
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "val_metrics": (
                        val_metrics
                    ),
                },
                BEST_MODEL_PATH,
            )

            print(
                "New best detector saved."
            )

        else:
            epochs_without_improvement += 1

        scheduler.step()

        if (
            epoch
            >= MIN_EPOCHS_BEFORE_STOP
            and epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print()
            print(
                "Early stopping triggered."
            )

            break

    # ========================================================
    # Save final model
    # ========================================================

    final_epoch = epoch

    torch.save(
        {
            "epoch": final_epoch,
            "base_channels": (
                BASE_CHANNELS
            ),
            "model_state_dict": (
                model.state_dict()
            ),
        },
        FINAL_MODEL_PATH,
    )

    # ========================================================
    # Reload best model
    # ========================================================

    print()
    print(
        "Reloading best detector for "
        "threshold optimization..."
    )

    best_checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    # ========================================================
    # Validation probability collection
    # ========================================================

    (
        _,
        _,
        probabilities,
        targets,
    ) = validation_pass(
        model=model,
        loader=validation_loader,
        criterion=criterion,
        device=device,
        threshold=0.50,
    )

    (
        best_threshold,
        best_threshold_metrics,
        threshold_rows,
    ) = find_best_threshold(
        probabilities,
        targets,
    )

    # ========================================================
    # Save threshold sweep
    # ========================================================

    with THRESHOLD_RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "threshold",
                "iou",
                "dice",
                "precision",
                "recall",
            ],
        )

        writer.writeheader()

        writer.writerows(
            threshold_rows
        )

    # ========================================================
    # Update best checkpoint with threshold
    # ========================================================

    best_checkpoint[
        "recommended_threshold"
    ] = best_threshold

    best_checkpoint[
        "threshold_metrics"
    ] = best_threshold_metrics

    torch.save(
        best_checkpoint,
        BEST_MODEL_PATH,
    )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 94)
    print(
        "EXPERIMENT 11C TRAINING COMPLETE"
    )
    print("=" * 94)

    print()
    print(
        f"Best epoch       : "
        f"{best_epoch}"
    )

    print(
        f"Best Dice @0.50  : "
        f"{best_dice:.4f}"
    )

    print()
    print(
        "VALIDATION THRESHOLD OPTIMIZATION"
    )

    print(
        f"Best threshold   : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Threshold IoU    : "
        f"{best_threshold_metrics['iou']:.4f}"
    )

    print(
        f"Threshold Dice   : "
        f"{best_threshold_metrics['dice']:.4f}"
    )

    print(
        f"Threshold Precision: "
        f"{best_threshold_metrics['precision']:.4f}"
    )

    print(
        f"Threshold Recall : "
        f"{best_threshold_metrics['recall']:.4f}"
    )

    print()
    print(
        "Best model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print()
    print(
        "Training log:"
    )

    print(
        TRAIN_LOG_PATH
    )

    print()
    print(
        "Threshold sweep:"
    )

    print(
        THRESHOLD_RESULTS_PATH
    )


if __name__ == "__main__":
    main()