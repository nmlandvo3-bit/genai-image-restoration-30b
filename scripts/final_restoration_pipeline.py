"""
Final Frozen Image Restoration Pipeline

Validated 28J production candidate:

    Input image
        -> DamageUNetImproved
        -> native tiled detector inference
             detector input = 512
             native tile = 640
             overlap = 0.125
             tile batch = 8
             uniform probability-average merge
        -> predicted binary damage mask
             threshold = 0.50
        -> predicted native-resolution mask fraction
        -> frozen 28J routing gate
             threshold = 0.0025
             comparison = >=
             preserve input, OR
             Restormer Gaussian_Color_Denoising
             predicted-mask localization
             Gaussian feather radius = 4.0
        -> final restored image

Detector candidate:
    28J-AE Epoch 2

Candidate selection:
    28J-AG

Final untouched clean confirmation:
    28J-AH
    0 / 250 false routes
    PASS_FINAL_CLEAN_CONFIRMATION

Do not tune the detector, tile configuration,
mask threshold, or routing gate from inference images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DETECTOR_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "experiment_11c_train_improved_detector.py"
)

DETECTOR_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "experiment_28j_release"
    / "damage_detector_28j_final.pt"
)

EXPECTED_DETECTOR_SHA256 = (
    "1f32439e6e86ec75d07c7e29c73f0b4590670ec8c44a83f16bb5764b97da300e"
)

RESTORMER_ROOT = (
    PROJECT_ROOT
    / "external"
    / "Restormer"
)

RESTORMER_DEMO = (
    RESTORMER_ROOT
    / "demo.py"
)

RESTORMER_WEIGHT = (
    RESTORMER_ROOT
    / "Denoising"
    / "pretrained_models"
    / "gaussian_color_denoising_blind.pth"
)


# =============================================================================
# FROZEN CONFIGURATION
# =============================================================================

IMAGE_SIZE = 512

DETECTOR_TILE_SIZE = 640

DETECTOR_TILE_OVERLAP_FRACTION = 0.125

DETECTOR_TILE_BATCH_SIZE = 8

DETECTOR_MASK_THRESHOLD = 0.50

GATE_THRESHOLD = 0.0025

FEATHER_RADIUS = 4.0

RESTORMER_TASK = "Gaussian_Color_Denoising"

RESTORMER_TILE = 512
RESTORMER_TILE_OVERLAP = 32


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def extract_state_dict(
    checkpoint,
) -> dict[str, torch.Tensor]:

    if isinstance(checkpoint, dict):

        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):

            value = checkpoint.get(key)

            if isinstance(value, dict):
                return value

        if (
            checkpoint
            and all(
                isinstance(value, torch.Tensor)
                for value in checkpoint.values()
            )
        ):
            return checkpoint

    raise RuntimeError(
        "Could not extract detector state_dict "
        "from checkpoint."
    )


def clean_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned[new_key] = value

    return cleaned


def infer_base_channels(
    state_dict: dict[str, torch.Tensor],
) -> int:

    candidate_keys = (
        "enc1.block.0.weight",
        "enc1.conv1.weight",
        "enc1.0.weight",
    )

    for key in candidate_keys:

        if key in state_dict:

            tensor = state_dict[key]

            if tensor.ndim >= 1:
                return int(tensor.shape[0])

    for key, tensor in state_dict.items():

        if (
            tensor.ndim == 4
            and tensor.shape[1] == 3
        ):
            return int(tensor.shape[0])

    raise RuntimeError(
        "Could not infer detector base_channels "
        "from checkpoint."
    )


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_detector_architecture():

    if not DETECTOR_SCRIPT.exists():
        raise FileNotFoundError(
            f"Detector architecture missing:\n"
            f"{DETECTOR_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "experiment_11c_architecture",
        DETECTOR_SCRIPT,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Could not load Experiment 11C "
            "architecture module."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    if not hasattr(
        module,
        "DamageUNetImproved",
    ):
        raise RuntimeError(
            "DamageUNetImproved was not found "
            "in Experiment 11C script."
        )

    return module.DamageUNetImproved


def file_sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def load_detector(
    device: torch.device,
):

    detector_sha256 = file_sha256(
        DETECTOR_CHECKPOINT
    )

    if (
        detector_sha256
        != EXPECTED_DETECTOR_SHA256
    ):

        raise RuntimeError(
            "Detector checkpoint SHA256 mismatch.\n"
            f"Expected: {EXPECTED_DETECTOR_SHA256}\n"
            f"Actual:   {detector_sha256}"
        )

    DamageUNetImproved = (
        load_detector_architecture()
    )

    checkpoint = torch.load(
        DETECTOR_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    base_channels = infer_base_channels(
        state_dict
    )

    model = DamageUNetImproved(
        base_channels=base_channels,
    ).to(
        device
    )

    incompatible = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if incompatible.missing_keys:
        raise RuntimeError(
            "Detector checkpoint has missing keys."
        )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Detector checkpoint has unexpected keys."
        )

    model.eval()

    return model, base_channels


# =============================================================================
# DETECTOR TILED INFERENCE
# =============================================================================

def load_input_image(
    path: Path,
) -> Image.Image:

    with Image.open(
        path
    ) as opened:

        image = opened.convert(
            "RGB"
        )

    return image


def detector_tile_positions(
    length: int,
) -> list[int]:

    overlap_pixels = int(
        round(
            DETECTOR_TILE_SIZE
            * DETECTOR_TILE_OVERLAP_FRACTION
        )
    )

    stride = (
        DETECTOR_TILE_SIZE
        - overlap_pixels
    )

    if length <= DETECTOR_TILE_SIZE:

        return [0]

    values = list(
        range(
            0,
            length - DETECTOR_TILE_SIZE + 1,
            stride,
        )
    )

    final_position = (
        length
        - DETECTOR_TILE_SIZE
    )

    if (
        values[-1]
        != final_position
    ):

        values.append(
            final_position
        )

    return values


def detector_image_to_tensor(
    image: Image.Image,
) -> torch.Tensor:

    resized = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        Image.Resampling.LANCZOS,
    )

    array = np.asarray(
        resized,
        dtype=np.float32,
    )

    array /= 255.0

    return (
        torch.from_numpy(
            array
        )
        .permute(
            2,
            0,
            1,
        )
        .contiguous()
    )


def predict_mask(
    model,
    image: Image.Image,
    device: torch.device,
) -> tuple[
    Image.Image,
    float,
]:

    width, height = image.size

    xs = detector_tile_positions(
        width
    )

    ys = detector_tile_positions(
        height
    )

    tiles = []

    for y in ys:

        for x in xs:

            right = min(
                x + DETECTOR_TILE_SIZE,
                width,
            )

            bottom = min(
                y + DETECTOR_TILE_SIZE,
                height,
            )

            tiles.append(
                (
                    x,
                    y,
                    right,
                    bottom,
                    image.crop(
                        (
                            x,
                            y,
                            right,
                            bottom,
                        )
                    ),
                )
            )

    probability_sum = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.float32,
    )

    coverage = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.float32,
    )

    for start in range(
        0,
        len(tiles),
        DETECTOR_TILE_BATCH_SIZE,
    ):

        batch_tiles = tiles[
            start:
            start + DETECTOR_TILE_BATCH_SIZE
        ]

        batch = torch.stack(
            [
                detector_image_to_tensor(
                    tile[4]
                )
                for tile in batch_tiles
            ]
        ).to(
            device,
            non_blocking=True,
        )

        with torch.inference_mode():

            probabilities = (
                torch.sigmoid(
                    model(
                        batch
                    )
                )
                .squeeze(1)
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

        for (
            tile,
            probability,
        ) in zip(
            batch_tiles,
            probabilities,
        ):

            (
                x,
                y,
                right,
                bottom,
                _,
            ) = tile

            native_width = (
                right - x
            )

            native_height = (
                bottom - y
            )

            native_probability = np.asarray(
                Image.fromarray(
                    probability,
                    mode="F",
                ).resize(
                    (
                        native_width,
                        native_height,
                    ),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            )

            probability_sum[
                y:bottom,
                x:right,
            ] += native_probability

            coverage[
                y:bottom,
                x:right,
            ] += 1.0

    if np.any(
        coverage <= 0
    ):

        raise RuntimeError(
            "Detector tiled inference "
            "produced uncovered pixels."
        )

    probability = (
        probability_sum
        / coverage
    )

    binary = (
        probability
        >= DETECTOR_MASK_THRESHOLD
    ).astype(
        np.uint8
    )

    mask_fraction = float(
        np.mean(
            binary
        )
    )

    mask = Image.fromarray(
        binary * 255,
        mode="L",
    )

    return (
        mask,
        mask_fraction,
    )


# =============================================================================
# RESTORMER
# =============================================================================

def find_restormer_output(
    directory: Path,
    stem: str,
) -> Path:

    direct = (
        directory
        / f"{stem}.png"
    )

    if direct.exists():
        return direct

    candidates = list(
        directory.glob(
            f"{stem}.*"
        )
    )

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        "Could not uniquely locate "
        f"Restormer output for {stem}."
    )


def run_restormer(
    input_image: Image.Image,
    sample_name: str,
    working_root: Path,
) -> Image.Image:

    input_dir = (
        working_root
        / "input"
    )

    result_root = (
        working_root
        / "restormer_results"
    )

    input_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = (
        input_dir
        / f"{sample_name}.png"
    )

    input_image.save(
        input_path
    )

    command = [
        sys.executable,
        RESTORMER_DEMO.name,
        "--task",
        RESTORMER_TASK,
        "--input_dir",
        str(input_dir),
        "--result_dir",
        str(result_root),
        "--tile",
        str(RESTORMER_TILE),
        "--tile_overlap",
        str(RESTORMER_TILE_OVERLAP),
    ]

    subprocess.run(
        command,
        cwd=str(
            RESTORMER_ROOT
        ),
        check=True,
    )

    raw_output_dir = (
        result_root
        / RESTORMER_TASK
    )

    restored_path = (
        find_restormer_output(
            raw_output_dir,
            sample_name,
        )
    )

    return Image.open(
        restored_path
    ).convert(
        "RGB"
    )


# =============================================================================
# LOCAL RESTORATION
# =============================================================================

def localize_restoration(
    original: Image.Image,
    restored: Image.Image,
    predicted_mask: Image.Image,
) -> Image.Image:

    if restored.size != original.size:

        restored = restored.resize(
            original.size,
            Image.Resampling.BICUBIC,
        )

    if predicted_mask.size != original.size:

        predicted_mask = predicted_mask.resize(
            original.size,
            Image.Resampling.NEAREST,
        )

    blend_mask = (
        predicted_mask.filter(
            ImageFilter.GaussianBlur(
                radius=FEATHER_RADIUS
            )
        )
    )

    return Image.composite(
        restored,
        original,
        blend_mask,
    )


# =============================================================================
# REPORT
# =============================================================================

def save_report(
    path: Path,
    report: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Frozen automatic image "
            "restoration pipeline."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input image.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Final output image.",
    )

    parser.add_argument(
        "--mask-output",
        type=Path,
        default=None,
        help=(
            "Optional path for the predicted "
            "damage mask."
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Optional JSON decision report."
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    print("=" * 100)
    print("FINAL FROZEN IMAGE RESTORATION PIPELINE")
    print("=" * 100)
    print()

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    required = (
        args.input,
        DETECTOR_SCRIPT,
        DETECTOR_CHECKPOINT,
        RESTORMER_ROOT,
        RESTORMER_DEMO,
        RESTORMER_WEIGHT,
    )

    for path in required:

        if not path.exists():
            raise FileNotFoundError(
                f"Required path missing:\n{path}"
            )

    # =========================================================================
    # DEVICE
    # =========================================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Input                 : {args.input}"
    )

    print(
        f"Output                : {args.output}"
    )

    print(
        f"Device                : {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU                   : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        "Detector threshold    : "
        f"{DETECTOR_MASK_THRESHOLD:.3f}"
    )

    print(
        "Frozen gate threshold : "
        f"{GATE_THRESHOLD:.3f}"
    )

    print(
        "Feather radius        : "
        f"{FEATHER_RADIUS:.1f}"
    )

    # =========================================================================
    # DETECTOR
    # =========================================================================

    print()
    print("Loading frozen detector...")

    model, base_channels = (
        load_detector(
            device
        )
    )

    print(
        "Detector loaded with strict=True."
    )

    print(
        f"Base channels         : "
        f"{base_channels}"
    )

    original = load_input_image(
        args.input
    )

    predicted_mask, mask_fraction = (
        predict_mask(
            model,
            original,
            device,
        )
    )

    print()
    print(
        "Predicted mask fraction: "
        f"{mask_fraction:.9f}"
    )

    # =========================================================================
    # FROZEN ROUTING DECISION
    # =========================================================================

    route_restormer = (
        mask_fraction
        >= GATE_THRESHOLD
    )

    if route_restormer:
        decision = "restormer"
    else:
        decision = "preserve"

    print(
        f"Routing decision       : "
        f"{decision.upper()}"
    )

    # =========================================================================
    # RESTORATION
    # =========================================================================

    if route_restormer:

        print()
        print(
            "Running Restormer specialist..."
        )

        with tempfile.TemporaryDirectory(
            prefix="frozen_restoration_"
        ) as temp_directory:

            working_root = Path(
                temp_directory
            )

            restored = run_restormer(
                original,
                args.input.stem,
                working_root,
            )

            final_image = (
                localize_restoration(
                    original,
                    restored,
                    predicted_mask,
                )
            )

    else:

        print()
        print(
            "Safety gate rejected specialist."
        )

        print(
            "Preserving original input."
        )

        final_image = original.copy()

    # =========================================================================
    # SAVE
    # =========================================================================

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_image.save(
        args.output
    )

    if args.mask_output is not None:

        args.mask_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        predicted_mask.save(
            args.mask_output
        )

    report = {
        "pipeline": (
            "frozen_restormer_safety_gate"
        ),
        "input": str(
            args.input
        ),
        "output": str(
            args.output
        ),
        "device": str(
            device
        ),
        "detector": (
            "Experiment 11C DamageUNetImproved"
        ),
        "detector_checkpoint": str(
            DETECTOR_CHECKPOINT
        ),
        "detector_mask_threshold": (
            DETECTOR_MASK_THRESHOLD
        ),
        "predicted_mask_fraction": (
            mask_fraction
        ),
        "gate_feature": (
            "predicted_mask_fraction"
        ),
        "gate_threshold": (
            GATE_THRESHOLD
        ),
        "decision": decision,
        "specialist": (
            "Restormer Gaussian_Color_Denoising"
        ),
        "restormer_applied": (
            route_restormer
        ),
        "feather_radius": (
            FEATHER_RADIUS
        ),
        "policy_status": "FROZEN",
    }

    if args.report is not None:

        save_report(
            args.report,
            report,
        )

    print()
    print("=" * 100)
    print("FINAL RESULT")
    print("=" * 100)
    print()

    print(
        f"Predicted mask fraction : "
        f"{mask_fraction:.9f}"
    )

    print(
        f"Frozen threshold        : "
        f"{GATE_THRESHOLD:.9f}"
    )

    print(
        f"Decision                : "
        f"{decision}"
    )

    print(
        f"Final image             : "
        f"{args.output}"
    )

    if args.mask_output is not None:

        print(
            f"Predicted mask          : "
            f"{args.mask_output}"
        )

    if args.report is not None:

        print(
            f"Decision report         : "
            f"{args.report}"
        )

    print()
    print(
        "[PASS] Frozen restoration pipeline "
        "completed successfully."
    )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()


