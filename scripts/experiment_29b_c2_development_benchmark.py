from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT = Path.cwd()

RESTORMER_ROOT = (
    ROOT
    / "external"
    / "Restormer"
)

RESTORMER_ARCH_PATH = (
    RESTORMER_ROOT
    / "basicsr"
    / "models"
    / "archs"
    / "restormer_arch.py"
)

PRETRAINED_MOTION_WEIGHT = (
    RESTORMER_ROOT
    / "Motion_Deblurring"
    / "pretrained_models"
    / "motion_deblurring.pth"
)

FINETUNED_CHECKPOINT = (
    ROOT
    / "models"
    / "experiment_29b_deblur_specialist"
    / "best_motion_gaussian_finetune.pt"
)

C1_FREEZE = (
    ROOT
    / "results"
    / "experiment_29b_c1"
    / "29b_c1_candidate_freeze.json"
)

DEV_MANIFEST = (
    ROOT
    / "data"
    / "experiment_29b"
    / "development_manifest.csv"
)

HOLDOUT_MANIFEST = (
    ROOT
    / "data"
    / "experiment_29b"
    / "holdout_manifest.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "experiment_29b_c2"
)

PRETRAINED_RAW_ROOT = (
    OUTPUT_ROOT
    / "pretrained_motion"
    / "raw"
)

PRETRAINED_LOCALIZED_ROOT = (
    OUTPUT_ROOT
    / "pretrained_motion"
    / "localized"
)

FINETUNED_RAW_ROOT = (
    OUTPUT_ROOT
    / "finetuned_motion"
    / "raw"
)

FINETUNED_LOCALIZED_ROOT = (
    OUTPUT_ROOT
    / "finetuned_motion"
    / "localized"
)

PER_SAMPLE_CSV = (
    OUTPUT_ROOT
    / "per_sample_metrics.csv"
)

SUMMARY_JSON = (
    OUTPUT_ROOT
    / "summary.json"
)


# =============================================================================
# FROZEN HASHES
# =============================================================================

EXPECTED_C1_FREEZE_SHA256 = (
    "3cf4e5446e7a27dfe0711ee7923a0dcd"
    "fbc4819ecc351a951bfaaafdc3d83b2c"
)

EXPECTED_FINETUNED_SHA256 = (
    "71754c8deda6519604f640a18c6f0685"
    "bb45f5c8031419a11b90406adeb503eb"
)

EXPECTED_PRETRAINED_SHA256 = (
    "194e38fb5b607c9dc5a5b3e08e65b2e7"
    "9ee2bf0ef5048e0612f6b2ff2f79da31"
)

EXPECTED_DEV_MANIFEST_SHA256 = (
    "35a4cc2d41dcf84377f24e39d8f8e5e0"
    "ee3e8fa92d73f4cb5dea6a39456cc7f7"
)

EXPECTED_HOLDOUT_MANIFEST_SHA256 = (
    "23d60076826663499f469a35213f3ad37"
    "5613a3aee8fa50ebced8fe694c675ec"
)

EXPECTED_DEV_ROWS = 93

USE_AMP = True


# =============================================================================
# RESTORMER CONFIG
# =============================================================================

RESTORMER_CONFIG = {
    "inp_channels": 3,
    "out_channels": 3,
    "dim": 48,
    "num_blocks": [4, 6, 6, 8],
    "num_refinement_blocks": 4,
    "heads": [1, 2, 4, 8],
    "ffn_expansion_factor": 2.66,
    "bias": False,
    "LayerNorm_type": "WithBias",
    "dual_pixel_task": False,
}


# =============================================================================
# HASHING
# =============================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        while True:

            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def verify_hash(
    path: Path,
    expected: str,
    label: str,
) -> str:

    if not path.exists():

        raise FileNotFoundError(
            f"{label} missing:\n{path}"
        )

    actual = sha256_file(path)

    if actual.lower() != expected.lower():

        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    print(
        f"[PASS] {label} SHA256"
    )

    return actual


# =============================================================================
# MANIFEST
# =============================================================================

def read_manifest(
    path: Path,
) -> list[dict[str, str]]:

    if not path.exists():

        raise FileNotFoundError(
            f"Manifest missing:\n{path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        rows = list(
            csv.DictReader(handle)
        )

    if not rows:

        raise RuntimeError(
            f"Manifest is empty:\n{path}"
        )

    return rows


def resolve_path(
    text: str,
) -> Path:

    path = Path(text)

    if not path.is_absolute():

        path = (
            ROOT
            / path
        )

    return path


# =============================================================================
# RESTORMER IMPORT
# =============================================================================

def import_restormer_class():

    if not RESTORMER_ARCH_PATH.exists():

        raise FileNotFoundError(
            "Restormer architecture missing:\n"
            f"{RESTORMER_ARCH_PATH}"
        )

    spec = (
        importlib.util.spec_from_file_location(
            "restormer_arch_29b_c2",
            RESTORMER_ARCH_PATH,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            "Could not import Restormer architecture."
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module.Restormer


# =============================================================================
# STATE-DICT HELPERS
# =============================================================================

def clean_state_dict(
    state_dict: dict,
) -> dict:

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    return cleaned


def load_pretrained_state() -> dict:

    checkpoint = torch.load(
        PRETRAINED_MOTION_WEIGHT,
        map_location="cpu",
        weights_only=False,
    )

    if "params" not in checkpoint:

        raise RuntimeError(
            "Pretrained Motion checkpoint "
            "does not contain 'params'."
        )

    return clean_state_dict(
        checkpoint["params"]
    )


def load_finetuned_state() -> dict:

    checkpoint = torch.load(
        FINETUNED_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if (
        checkpoint.get("experiment")
        != "29B-C1"
    ):

        raise RuntimeError(
            "Fine-tuned checkpoint experiment "
            "identifier mismatch."
        )

    if (
        checkpoint.get("epoch")
        != 12
    ):

        raise RuntimeError(
            "Frozen candidate is not epoch 12."
        )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Fine-tuned checkpoint does not "
            "contain model_state_dict."
        )

    if checkpoint.get(
        "holdout_accessed"
    ) is not False:

        raise RuntimeError(
            "Fine-tuned checkpoint does not "
            "record sealed holdout status."
        )

    return clean_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


# =============================================================================
# MODEL
# =============================================================================

def build_model(
    state_dict: dict,
    device: torch.device,
):

    Restormer = (
        import_restormer_class()
    )

    model = Restormer(
        **RESTORMER_CONFIG
    )

    result = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if result.missing_keys:

        raise RuntimeError(
            f"Missing model keys: "
            f"{result.missing_keys}"
        )

    if result.unexpected_keys:

        raise RuntimeError(
            f"Unexpected model keys: "
            f"{result.unexpected_keys}"
        )

    model = model.to(device)

    model.eval()

    return model


# =============================================================================
# IMAGE HELPERS
# =============================================================================

def load_rgb(
    path: Path,
) -> np.ndarray:

    with Image.open(path) as image:

        return np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )


def load_mask(
    path: Path,
) -> np.ndarray:

    with Image.open(path) as image:

        return (
            np.asarray(
                image.convert("L")
            )
            > 0
        )


def save_rgb(
    array: np.ndarray,
    path: Path,
) -> None:

    Image.fromarray(
        array.astype(np.uint8),
        mode="RGB",
    ).save(
        path
    )


# =============================================================================
# INFERENCE
# =============================================================================

def infer(
    model,
    image: np.ndarray,
    device: torch.device,
) -> np.ndarray:

    tensor = (
        torch.from_numpy(
            image.copy()
        )
        .permute(
            2,
            0,
            1,
        )
        .unsqueeze(0)
        .float()
        .div(255.0)
        .to(
            device,
            non_blocking=True,
        )
    )

    with torch.inference_mode():

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):

            restored = model(
                tensor
            )

    restored = torch.clamp(
        restored,
        0.0,
        1.0,
    )

    array = (
        restored[
            0
        ]
        .float()
        .cpu()
        .permute(
            1,
            2,
            0,
        )
        .numpy()
    )

    return np.clip(
        np.rint(
            array
            * 255.0
        ),
        0,
        255,
    ).astype(
        np.uint8
    )


# =============================================================================
# METRICS
# =============================================================================

def mse(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:

    difference = (
        prediction.astype(
            np.float64
        )
        - target.astype(
            np.float64
        )
    )

    return float(
        np.mean(
            difference
            * difference
        )
    )


def psnr(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:

    value = mse(
        prediction,
        target,
    )

    if value <= 0.0:
        return 100.0

    return float(
        10.0
        * math.log10(
            255.0 ** 2
            / value
        )
    )


def masked_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:

    if not np.any(mask):

        raise RuntimeError(
            "Empty mask in masked metric."
        )

    difference = (
        prediction[
            mask
        ].astype(
            np.float64
        )
        - target[
            mask
        ].astype(
            np.float64
        )
    )

    return float(
        np.mean(
            difference
            * difference
        )
    )


def masked_psnr(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:

    value = masked_mse(
        prediction,
        target,
        mask,
    )

    if value <= 0.0:
        return 100.0

    return float(
        10.0
        * math.log10(
            255.0 ** 2
            / value
        )
    )


def masked_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:

    if not np.any(mask):
        return 0.0

    difference = np.abs(
        prediction[
            mask
        ].astype(
            np.float64
        )
        - target[
            mask
        ].astype(
            np.float64
        )
    )

    return float(
        np.mean(
            difference
        )
    )


def ssim_rgb(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:

    prediction = (
        prediction.astype(
            np.float64
        )
    )

    target = (
        target.astype(
            np.float64
        )
    )

    c1 = (
        0.01
        * 255.0
    ) ** 2

    c2 = (
        0.03
        * 255.0
    ) ** 2

    scores = []

    for channel in range(3):

        x = prediction[
            :,
            :,
            channel
        ]

        y = target[
            :,
            :,
            channel
        ]

        mu_x = cv2.GaussianBlur(
            x,
            (11, 11),
            1.5,
        )

        mu_y = cv2.GaussianBlur(
            y,
            (11, 11),
            1.5,
        )

        mu_x_sq = (
            mu_x
            * mu_x
        )

        mu_y_sq = (
            mu_y
            * mu_y
        )

        mu_xy = (
            mu_x
            * mu_y
        )

        sigma_x_sq = (
            cv2.GaussianBlur(
                x * x,
                (11, 11),
                1.5,
            )
            - mu_x_sq
        )

        sigma_y_sq = (
            cv2.GaussianBlur(
                y * y,
                (11, 11),
                1.5,
            )
            - mu_y_sq
        )

        sigma_xy = (
            cv2.GaussianBlur(
                x * y,
                (11, 11),
                1.5,
            )
            - mu_xy
        )

        numerator = (
            (
                2.0
                * mu_xy
                + c1
            )
            * (
                2.0
                * sigma_xy
                + c2
            )
        )

        denominator = (
            (
                mu_x_sq
                + mu_y_sq
                + c1
            )
            * (
                sigma_x_sq
                + sigma_y_sq
                + c2
            )
        )

        score = (
            numerator
            / np.maximum(
                denominator,
                1e-12,
            )
        )

        scores.append(
            float(
                np.mean(score)
            )
        )

    return float(
        np.mean(scores)
    )


# =============================================================================
# LOCALIZED MERGE
# =============================================================================

def localized_merge(
    damaged: np.ndarray,
    restored: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:

    output = damaged.copy()

    output[
        mask
    ] = restored[
        mask
    ]

    return output


# =============================================================================
# METRIC BLOCK
# =============================================================================

def calculate_metrics(
    output: np.ndarray,
    clean: np.ndarray,
    mask: np.ndarray,
) -> dict:

    outside_mask = (
        ~mask
    )

    return {
        "psnr":
            psnr(
                output,
                clean,
            ),

        "ssim":
            ssim_rgb(
                output,
                clean,
            ),

        "inside_psnr":
            masked_psnr(
                output,
                clean,
                mask,
            ),

        "inside_mae":
            masked_mae(
                output,
                clean,
                mask,
            ),

        "outside_mae":
            masked_mae(
                output,
                clean,
                outside_mask,
            ),
    }


# =============================================================================
# SUMMARY HELPERS
# =============================================================================

def summarize(
    rows: list[dict],
    key: str,
) -> dict:

    values = np.asarray(
        [
            float(
                row[key]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    return {
        "mean":
            float(
                np.mean(values)
            ),

        "median":
            float(
                np.median(values)
            ),

        "min":
            float(
                np.min(values)
            ),

        "max":
            float(
                np.max(values)
            ),
    }


def mean_value(
    rows,
    key,
) -> float:

    return float(
        np.mean(
            [
                float(
                    row[key]
                )
                for row
                in rows
            ]
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 100)
    print("EXPERIMENT 29B-C2")
    print("FROZEN DEBLUR CANDIDATE DEVELOPMENT BENCHMARK")
    print("=" * 100)
    print()

    print("POLICY")
    print(
        "  Evaluation set: 29B development only."
    )
    print(
        "  Candidate: frozen 29B-C1 epoch-12 checkpoint."
    )
    print(
        "  Final 29B holdout pixels are NOT accessed."
    )
    print()

    # =========================================================================
    # ENVIRONMENT
    # =========================================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"PyTorch        : {torch.__version__}"
    )

    print(
        f"GPU            : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print()

    # =========================================================================
    # HASH VERIFICATION
    # =========================================================================

    verify_hash(
        C1_FREEZE,
        EXPECTED_C1_FREEZE_SHA256,
        "29B-C1 candidate freeze",
    )

    verify_hash(
        FINETUNED_CHECKPOINT,
        EXPECTED_FINETUNED_SHA256,
        "29B-C1 fine-tuned candidate",
    )

    verify_hash(
        PRETRAINED_MOTION_WEIGHT,
        EXPECTED_PRETRAINED_SHA256,
        "Pretrained Motion Restormer",
    )

    verify_hash(
        DEV_MANIFEST,
        EXPECTED_DEV_MANIFEST_SHA256,
        "29B development manifest",
    )

    # Only hash the holdout manifest.
    # Do not parse or access holdout image paths.
    verify_hash(
        HOLDOUT_MANIFEST,
        EXPECTED_HOLDOUT_MANIFEST_SHA256,
        "29B sealed holdout manifest",
    )

    # =========================================================================
    # DEVELOPMENT DATA
    # =========================================================================

    rows = read_manifest(
        DEV_MANIFEST
    )

    if len(rows) != EXPECTED_DEV_ROWS:

        raise RuntimeError(
            "Unexpected development row count.\n"
            f"Expected: {EXPECTED_DEV_ROWS}\n"
            f"Actual:   {len(rows)}"
        )

    print()
    print(
        f"Development samples : "
        f"{len(rows)}"
    )

    print(
        f"Development sources : "
        f"{len({row['source_id'] for row in rows})}"
    )

    # =========================================================================
    # OUTPUT
    # =========================================================================

    if OUTPUT_ROOT.exists():

        shutil.rmtree(
            OUTPUT_ROOT
        )

    for directory in [
        PRETRAINED_RAW_ROOT,
        PRETRAINED_LOCALIZED_ROOT,
        FINETUNED_RAW_ROOT,
        FINETUNED_LOCALIZED_ROOT,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================================
    # LOAD MODEL STATES
    # =========================================================================

    pretrained_state = (
        load_pretrained_state()
    )

    finetuned_state = (
        load_finetuned_state()
    )

    # =========================================================================
    # PRETRAINED MODEL
    # =========================================================================

    print()
    print("=" * 100)
    print("LOADING PRETRAINED MOTION MODEL")
    print("=" * 100)
    print()

    model = build_model(
        pretrained_state,
        device,
    )

    print(
        "[PASS] Pretrained Motion model loaded."
    )

    pretrained_outputs = {}

    start = time.time()

    for index, row in enumerate(
        rows,
        start=1,
    ):

        damaged_path = resolve_path(
            row["damaged_path"]
        )

        damaged = load_rgb(
            damaged_path
        )

        restored = infer(
            model,
            damaged,
            device,
        )

        pretrained_outputs[
            row["sample_id"]
        ] = restored

        save_rgb(
            restored,
            PRETRAINED_RAW_ROOT
            / damaged_path.name,
        )

        if (
            index == 1
            or index % 10 == 0
            or index == len(rows)
        ):

            print(
                f"Pretrained "
                f"{index:3d}/{len(rows)}"
            )

    pretrained_seconds = (
        time.time()
        - start
    )

    # Release GPU model before loading candidate.
    del model

    torch.cuda.empty_cache()

    # =========================================================================
    # FINE-TUNED MODEL
    # =========================================================================

    print()
    print("=" * 100)
    print("LOADING FINE-TUNED 29B-C1 MODEL")
    print("=" * 100)
    print()

    model = build_model(
        finetuned_state,
        device,
    )

    print(
        "[PASS] Frozen fine-tuned model loaded."
    )

    finetuned_outputs = {}

    start = time.time()

    for index, row in enumerate(
        rows,
        start=1,
    ):

        damaged_path = resolve_path(
            row["damaged_path"]
        )

        damaged = load_rgb(
            damaged_path
        )

        restored = infer(
            model,
            damaged,
            device,
        )

        finetuned_outputs[
            row["sample_id"]
        ] = restored

        save_rgb(
            restored,
            FINETUNED_RAW_ROOT
            / damaged_path.name,
        )

        if (
            index == 1
            or index % 10 == 0
            or index == len(rows)
        ):

            print(
                f"Fine-tuned "
                f"{index:3d}/{len(rows)}"
            )

    finetuned_seconds = (
        time.time()
        - start
    )

    del model

    torch.cuda.empty_cache()

    # =========================================================================
    # METRICS
    # =========================================================================

    print()
    print("=" * 100)
    print("CALCULATING APPLES-TO-APPLES METRICS")
    print("=" * 100)
    print()

    metric_rows = []

    for index, row in enumerate(
        rows,
        start=1,
    ):

        sample_id = row[
            "sample_id"
        ]

        damaged_path = resolve_path(
            row["damaged_path"]
        )

        clean_path = resolve_path(
            row["clean_target_path"]
        )

        mask_path = resolve_path(
            row["mask_path"]
        )

        damaged = load_rgb(
            damaged_path
        )

        clean = load_rgb(
            clean_path
        )

        mask = load_mask(
            mask_path
        )

        pretrained_raw = (
            pretrained_outputs[
                sample_id
            ]
        )

        finetuned_raw = (
            finetuned_outputs[
                sample_id
            ]
        )

        pretrained_localized = (
            localized_merge(
                damaged,
                pretrained_raw,
                mask,
            )
        )

        finetuned_localized = (
            localized_merge(
                damaged,
                finetuned_raw,
                mask,
            )
        )

        save_rgb(
            pretrained_localized,
            PRETRAINED_LOCALIZED_ROOT
            / damaged_path.name,
        )

        save_rgb(
            finetuned_localized,
            FINETUNED_LOCALIZED_ROOT
            / damaged_path.name,
        )

        baseline = calculate_metrics(
            damaged,
            clean,
            mask,
        )

        pretrained_raw_metrics = (
            calculate_metrics(
                pretrained_raw,
                clean,
                mask,
            )
        )

        pretrained_localized_metrics = (
            calculate_metrics(
                pretrained_localized,
                clean,
                mask,
            )
        )

        finetuned_raw_metrics = (
            calculate_metrics(
                finetuned_raw,
                clean,
                mask,
            )
        )

        finetuned_localized_metrics = (
            calculate_metrics(
                finetuned_localized,
                clean,
                mask,
            )
        )

        result = {
            "sample_id":
                sample_id,

            "source_id":
                row["source_id"],

            "radius_512":
                float(
                    row["radius_512"]
                ),

            "mask_fraction":
                float(
                    np.mean(mask)
                ),

            # -------------------------------------------------------------
            # BASELINE
            # -------------------------------------------------------------

            "baseline_psnr":
                baseline["psnr"],

            "baseline_ssim":
                baseline["ssim"],

            "baseline_inside_psnr":
                baseline["inside_psnr"],

            "baseline_inside_mae":
                baseline["inside_mae"],

            "baseline_outside_mae":
                baseline["outside_mae"],

            # -------------------------------------------------------------
            # PRETRAINED RAW
            # -------------------------------------------------------------

            "pretrained_raw_psnr":
                pretrained_raw_metrics["psnr"],

            "pretrained_raw_ssim":
                pretrained_raw_metrics["ssim"],

            "pretrained_raw_inside_psnr":
                pretrained_raw_metrics[
                    "inside_psnr"
                ],

            "pretrained_raw_inside_mae":
                pretrained_raw_metrics[
                    "inside_mae"
                ],

            "pretrained_raw_outside_mae":
                pretrained_raw_metrics[
                    "outside_mae"
                ],

            # -------------------------------------------------------------
            # PRETRAINED LOCALIZED
            # -------------------------------------------------------------

            "pretrained_localized_psnr":
                pretrained_localized_metrics[
                    "psnr"
                ],

            "pretrained_localized_ssim":
                pretrained_localized_metrics[
                    "ssim"
                ],

            "pretrained_localized_inside_psnr":
                pretrained_localized_metrics[
                    "inside_psnr"
                ],

            "pretrained_localized_inside_mae":
                pretrained_localized_metrics[
                    "inside_mae"
                ],

            "pretrained_localized_outside_mae":
                pretrained_localized_metrics[
                    "outside_mae"
                ],

            # -------------------------------------------------------------
            # FINE-TUNED RAW
            # -------------------------------------------------------------

            "finetuned_raw_psnr":
                finetuned_raw_metrics["psnr"],

            "finetuned_raw_ssim":
                finetuned_raw_metrics["ssim"],

            "finetuned_raw_inside_psnr":
                finetuned_raw_metrics[
                    "inside_psnr"
                ],

            "finetuned_raw_inside_mae":
                finetuned_raw_metrics[
                    "inside_mae"
                ],

            "finetuned_raw_outside_mae":
                finetuned_raw_metrics[
                    "outside_mae"
                ],

            # -------------------------------------------------------------
            # FINE-TUNED LOCALIZED
            # -------------------------------------------------------------

            "finetuned_localized_psnr":
                finetuned_localized_metrics[
                    "psnr"
                ],

            "finetuned_localized_ssim":
                finetuned_localized_metrics[
                    "ssim"
                ],

            "finetuned_localized_inside_psnr":
                finetuned_localized_metrics[
                    "inside_psnr"
                ],

            "finetuned_localized_inside_mae":
                finetuned_localized_metrics[
                    "inside_mae"
                ],

            "finetuned_localized_outside_mae":
                finetuned_localized_metrics[
                    "outside_mae"
                ],

            # -------------------------------------------------------------
            # DELTAS
            # -------------------------------------------------------------

            "pretrained_inside_psnr_delta":
                (
                    pretrained_raw_metrics[
                        "inside_psnr"
                    ]
                    - baseline[
                        "inside_psnr"
                    ]
                ),

            "finetuned_inside_psnr_delta":
                (
                    finetuned_raw_metrics[
                        "inside_psnr"
                    ]
                    - baseline[
                        "inside_psnr"
                    ]
                ),

            "finetuned_vs_pretrained_inside_psnr_delta":
                (
                    finetuned_raw_metrics[
                        "inside_psnr"
                    ]
                    - pretrained_raw_metrics[
                        "inside_psnr"
                    ]
                ),

            "pretrained_inside_improved":
                int(
                    pretrained_raw_metrics[
                        "inside_psnr"
                    ]
                    > baseline[
                        "inside_psnr"
                    ]
                ),

            "finetuned_inside_improved":
                int(
                    finetuned_raw_metrics[
                        "inside_psnr"
                    ]
                    > baseline[
                        "inside_psnr"
                    ]
                ),

            "finetuned_beats_pretrained":
                int(
                    finetuned_raw_metrics[
                        "inside_psnr"
                    ]
                    > pretrained_raw_metrics[
                        "inside_psnr"
                    ]
                ),
        }

        metric_rows.append(
            result
        )

        if (
            index == 1
            or index % 10 == 0
            or index == len(rows)
        ):

            print(
                f"{index:3d}/"
                f"{len(rows)} | "
                f"{sample_id} | "
                f"baseline="
                f"{baseline['inside_psnr']:.3f} | "
                f"pre="
                f"{pretrained_raw_metrics['inside_psnr']:.3f} | "
                f"fine="
                f"{finetuned_raw_metrics['inside_psnr']:.3f}"
            )

    # =========================================================================
    # WRITE PER-SAMPLE CSV
    # =========================================================================

    with PER_SAMPLE_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                metric_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            metric_rows
        )

    # =========================================================================
    # COUNTS
    # =========================================================================

    count = len(
        metric_rows
    )

    pretrained_improved = sum(
        row[
            "pretrained_inside_improved"
        ]
        for row in metric_rows
    )

    finetuned_improved = sum(
        row[
            "finetuned_inside_improved"
        ]
        for row in metric_rows
    )

    fine_beats_pretrained = sum(
        row[
            "finetuned_beats_pretrained"
        ]
        for row in metric_rows
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    summary = {
        "experiment":
            "29B-C2",

        "status":
            "FROZEN_CANDIDATE_DEVELOPMENT_BENCHMARK_COMPLETE",

        "evaluation_set":
            "29B development only",

        "samples":
            count,

        "sources":
            len(
                {
                    row["source_id"]
                    for row in metric_rows
                }
            ),

        "holdout_accessed":
            False,

        "frozen_artifacts": {
            "c1_candidate_freeze_sha256":
                EXPECTED_C1_FREEZE_SHA256,

            "finetuned_checkpoint_sha256":
                EXPECTED_FINETUNED_SHA256,

            "pretrained_motion_sha256":
                EXPECTED_PRETRAINED_SHA256,

            "development_manifest_sha256":
                EXPECTED_DEV_MANIFEST_SHA256,

            "holdout_manifest_sha256":
                EXPECTED_HOLDOUT_MANIFEST_SHA256,
        },

        "baseline": {
            "psnr":
                summarize(
                    metric_rows,
                    "baseline_psnr",
                ),

            "ssim":
                summarize(
                    metric_rows,
                    "baseline_ssim",
                ),

            "inside_psnr":
                summarize(
                    metric_rows,
                    "baseline_inside_psnr",
                ),

            "inside_mae":
                summarize(
                    metric_rows,
                    "baseline_inside_mae",
                ),

            "outside_mae":
                summarize(
                    metric_rows,
                    "baseline_outside_mae",
                ),
        },

        "pretrained_motion_raw": {
            "psnr":
                summarize(
                    metric_rows,
                    "pretrained_raw_psnr",
                ),

            "ssim":
                summarize(
                    metric_rows,
                    "pretrained_raw_ssim",
                ),

            "inside_psnr":
                summarize(
                    metric_rows,
                    "pretrained_raw_inside_psnr",
                ),

            "inside_mae":
                summarize(
                    metric_rows,
                    "pretrained_raw_inside_mae",
                ),

            "outside_mae":
                summarize(
                    metric_rows,
                    "pretrained_raw_outside_mae",
                ),

            "mean_inside_psnr_delta":
                mean_value(
                    metric_rows,
                    "pretrained_inside_psnr_delta",
                ),

            "median_inside_psnr_delta":
                float(
                    np.median(
                        [
                            row[
                                "pretrained_inside_psnr_delta"
                            ]
                            for row
                            in metric_rows
                        ]
                    )
                ),

            "inside_improved_count":
                pretrained_improved,

            "inside_improved_fraction":
                pretrained_improved
                / count,
        },

        "pretrained_motion_localized": {
            "psnr":
                summarize(
                    metric_rows,
                    "pretrained_localized_psnr",
                ),

            "ssim":
                summarize(
                    metric_rows,
                    "pretrained_localized_ssim",
                ),

            "inside_psnr":
                summarize(
                    metric_rows,
                    "pretrained_localized_inside_psnr",
                ),

            "inside_mae":
                summarize(
                    metric_rows,
                    "pretrained_localized_inside_mae",
                ),

            "outside_mae":
                summarize(
                    metric_rows,
                    "pretrained_localized_outside_mae",
                ),
        },

        "finetuned_motion_raw": {
            "psnr":
                summarize(
                    metric_rows,
                    "finetuned_raw_psnr",
                ),

            "ssim":
                summarize(
                    metric_rows,
                    "finetuned_raw_ssim",
                ),

            "inside_psnr":
                summarize(
                    metric_rows,
                    "finetuned_raw_inside_psnr",
                ),

            "inside_mae":
                summarize(
                    metric_rows,
                    "finetuned_raw_inside_mae",
                ),

            "outside_mae":
                summarize(
                    metric_rows,
                    "finetuned_raw_outside_mae",
                ),

            "mean_inside_psnr_delta":
                mean_value(
                    metric_rows,
                    "finetuned_inside_psnr_delta",
                ),

            "median_inside_psnr_delta":
                float(
                    np.median(
                        [
                            row[
                                "finetuned_inside_psnr_delta"
                            ]
                            for row
                            in metric_rows
                        ]
                    )
                ),

            "inside_improved_count":
                finetuned_improved,

            "inside_improved_fraction":
                finetuned_improved
                / count,

            "beats_pretrained_count":
                fine_beats_pretrained,

            "beats_pretrained_fraction":
                fine_beats_pretrained
                / count,

            "mean_inside_psnr_gain_over_pretrained":
                mean_value(
                    metric_rows,
                    "finetuned_vs_pretrained_inside_psnr_delta",
                ),
        },

        "finetuned_motion_localized": {
            "psnr":
                summarize(
                    metric_rows,
                    "finetuned_localized_psnr",
                ),

            "ssim":
                summarize(
                    metric_rows,
                    "finetuned_localized_ssim",
                ),

            "inside_psnr":
                summarize(
                    metric_rows,
                    "finetuned_localized_inside_psnr",
                ),

            "inside_mae":
                summarize(
                    metric_rows,
                    "finetuned_localized_inside_mae",
                ),

            "outside_mae":
                summarize(
                    metric_rows,
                    "finetuned_localized_outside_mae",
                ),
        },

        "runtime_seconds": {
            "pretrained":
                pretrained_seconds,

            "finetuned":
                finetuned_seconds,
        },

        "per_sample_csv":
            str(
                PER_SAMPLE_CSV
            ),

        "policy":
            (
                "29B-C2 characterizes the already "
                "frozen 29B-C1 candidate using only "
                "the development split. No model, "
                "training, checkpoint-selection, "
                "or holdout decision is made here."
            ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    per_sample_sha = (
        sha256_file(
            PER_SAMPLE_CSV
        )
    )

    summary_sha = (
        sha256_file(
            SUMMARY_JSON
        )
    )

    # =========================================================================
    # REPORT
    # =========================================================================

    baseline = summary[
        "baseline"
    ]

    pretrained = summary[
        "pretrained_motion_raw"
    ]

    pretrained_localized = summary[
        "pretrained_motion_localized"
    ]

    finetuned = summary[
        "finetuned_motion_raw"
    ]

    finetuned_localized = summary[
        "finetuned_motion_localized"
    ]

    print()
    print("=" * 100)
    print("29B-C2 DEVELOPMENT RESULTS")
    print("=" * 100)
    print()

    print("DAMAGED BASELINE")
    print(
        f"Mean whole PSNR          : "
        f"{baseline['psnr']['mean']:.4f} dB"
    )
    print(
        f"Mean whole SSIM          : "
        f"{baseline['ssim']['mean']:.6f}"
    )
    print(
        f"Mean inside PSNR         : "
        f"{baseline['inside_psnr']['mean']:.4f} dB"
    )
    print(
        f"Mean inside MAE          : "
        f"{baseline['inside_mae']['mean']:.4f}"
    )
    print()

    print("PRETRAINED MOTION — RAW")
    print(
        f"Mean whole PSNR          : "
        f"{pretrained['psnr']['mean']:.4f} dB"
    )
    print(
        f"Mean whole SSIM          : "
        f"{pretrained['ssim']['mean']:.6f}"
    )
    print(
        f"Mean inside PSNR         : "
        f"{pretrained['inside_psnr']['mean']:.4f} dB"
    )
    print(
        f"Mean inside MAE          : "
        f"{pretrained['inside_mae']['mean']:.4f}"
    )
    print(
        f"Mean outside MAE         : "
        f"{pretrained['outside_mae']['mean']:.4f}"
    )
    print(
        f"Mean inside delta        : "
        f"{pretrained['mean_inside_psnr_delta']:+.4f} dB"
    )
    print(
        f"Inside improved          : "
        f"{pretrained['inside_improved_count']}/"
        f"{count} "
        f"("
        f"{pretrained['inside_improved_fraction'] * 100:.2f}%"
        f")"
    )
    print()

    print("PRETRAINED MOTION — LOCALIZED")
    print(
        f"Mean whole PSNR          : "
        f"{pretrained_localized['psnr']['mean']:.4f} dB"
    )
    print(
        f"Mean whole SSIM          : "
        f"{pretrained_localized['ssim']['mean']:.6f}"
    )
    print(
        f"Mean outside MAE         : "
        f"{pretrained_localized['outside_mae']['mean']:.6f}"
    )
    print()

    print("FINE-TUNED MOTION — RAW")
    print(
        f"Mean whole PSNR          : "
        f"{finetuned['psnr']['mean']:.4f} dB"
    )
    print(
        f"Mean whole SSIM          : "
        f"{finetuned['ssim']['mean']:.6f}"
    )
    print(
        f"Mean inside PSNR         : "
        f"{finetuned['inside_psnr']['mean']:.4f} dB"
    )
    print(
        f"Mean inside MAE          : "
        f"{finetuned['inside_mae']['mean']:.4f}"
    )
    print(
        f"Mean outside MAE         : "
        f"{finetuned['outside_mae']['mean']:.4f}"
    )
    print(
        f"Mean inside delta        : "
        f"{finetuned['mean_inside_psnr_delta']:+.4f} dB"
    )
    print(
        f"Median inside delta      : "
        f"{finetuned['median_inside_psnr_delta']:+.4f} dB"
    )
    print(
        f"Inside improved          : "
        f"{finetuned['inside_improved_count']}/"
        f"{count} "
        f"("
        f"{finetuned['inside_improved_fraction'] * 100:.2f}%"
        f")"
    )
    print(
        f"Beats pretrained         : "
        f"{finetuned['beats_pretrained_count']}/"
        f"{count} "
        f"("
        f"{finetuned['beats_pretrained_fraction'] * 100:.2f}%"
        f")"
    )
    print(
        f"Gain over pretrained     : "
        f"{finetuned['mean_inside_psnr_gain_over_pretrained']:+.4f} dB"
    )
    print()

    print("FINE-TUNED MOTION — LOCALIZED")
    print(
        f"Mean whole PSNR          : "
        f"{finetuned_localized['psnr']['mean']:.4f} dB"
    )
    print(
        f"Mean whole SSIM          : "
        f"{finetuned_localized['ssim']['mean']:.6f}"
    )
    print(
        f"Mean inside PSNR         : "
        f"{finetuned_localized['inside_psnr']['mean']:.4f} dB"
    )
    print(
        f"Mean inside MAE          : "
        f"{finetuned_localized['inside_mae']['mean']:.4f}"
    )
    print(
        f"Mean outside MAE         : "
        f"{finetuned_localized['outside_mae']['mean']:.6f}"
    )
    print()

    print(
        f"Per-sample SHA256        : "
        f"{per_sample_sha}"
    )

    print(
        f"Summary SHA256           : "
        f"{summary_sha}"
    )

    print()
    print(
        "[PASS] 29B-C2 development benchmark complete."
    )
    print()
    print("IMPORTANT:")
    print(
        "29B holdout remains UNTOUCHED."
    )


if __name__ == "__main__":
    main()