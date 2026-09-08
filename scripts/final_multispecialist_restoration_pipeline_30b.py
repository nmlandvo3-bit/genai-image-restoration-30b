from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch


# =============================================================================
# PROJECT
# =============================================================================

PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

RESTORMER_ROOT = (
    PROJECT_ROOT
    / "external"
    / "Restormer"
)

if str(RESTORMER_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(RESTORMER_ROOT),
    )

from basicsr.models.archs.restormer_arch import Restormer


FROZEN_28J_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "final_restoration_pipeline.py"
)

CLASSIFIER_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "experiment_29a_4_predicted_mask_validation.py"
)

DEBLUR_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "experiment_29b_c2_development_benchmark.py"
)


DETECTOR_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "experiment_28j_release"
    / "damage_detector_28j_final.pt"
)

CLASSIFIER_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "experiment_29a_damage_classifier"
    / "best_mask_aware_resnet18.pt"
)

DENOISER_WEIGHT = (
    PROJECT_ROOT
    / "models"
    / "experiment_30b_noise_specialist"
    / "best_noise_specialist.pt"
)

DEBLUR_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "experiment_29b_deblur_specialist"
    / "best_motion_gaussian_finetune.pt"
)


# =============================================================================
# FROZEN HASHES
# =============================================================================

EXPECTED_28J_SCRIPT_SHA256 = (
    "44864649e62077aa716e93ecf366e3b9"
    "5e61d11f3ead8bf0ff176e7042a5a9bc"
)

EXPECTED_DETECTOR_SHA256 = (
    "1f32439e6e86ec75d07c7e29c73f0b4"
    "590670ec8c44a83f16bb5764b97da300e"
)

EXPECTED_CLASSIFIER_SHA256 = (
    "2c7f4e796fd75616c55352c6a6d6a272"
    "a4936fc0875acf7fed08b123c14dde37"
)

EXPECTED_DEBLUR_SHA256 = (
    "71754c8deda6519604f640a18c6f0685"
    "bb45f5c8031419a11b90406adeb503eb"
)


# =============================================================================
# ROUTING
# =============================================================================

GATE_THRESHOLD = 0.0025

CLASS_NAMES = [
    "clean",
    "noise",
    "blur",
    "scratch",
    "missing",
    "irregular",
]

CLASS_TO_SPECIALIST = {
    "clean": "preserve",
    "noise": "denoise",
    "blur": "deblur",
    "scratch": "scratch_inpaint",
    "missing": "inpaint",
    "irregular": "inpaint",
}


# =============================================================================
# CLASSICAL SPECIALISTS
# =============================================================================

SCRATCH_RADIUS = 7
INPAINT_RADIUS = 15


# =============================================================================
# DEBLUR TILING
# =============================================================================

DEBLUR_TILE_SIZE = 512
DEBLUR_TILE_OVERLAP = 32
DEBLUR_TILE_STRIDE = (
    DEBLUR_TILE_SIZE
    - DEBLUR_TILE_OVERLAP
)


# =============================================================================
# HASH
# =============================================================================

def file_sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

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

    actual = file_sha256(
        path
    )

    if (
        actual.lower()
        != expected.lower()
    ):

        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    return actual


# =============================================================================
# IMPORT
# =============================================================================

def import_script(
    name: str,
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Python module missing:\n{path}"
        )

    spec = (
        importlib.util.spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            f"Could not import:\n{path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


# =============================================================================
# IMAGE HELPERS
# =============================================================================

def load_image(
    path: Path,
) -> Image.Image:

    if not path.exists():

        raise FileNotFoundError(
            f"Input image missing:\n{path}"
        )

    with Image.open(
        path
    ) as image:

        return image.convert(
            "RGB"
        )


def pil_to_rgb_array(
    image: Image.Image,
) -> np.ndarray:

    return np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    )


def mask_to_bool(
    mask: Image.Image,
) -> np.ndarray:

    return (
        np.asarray(
            mask.convert("L")
        )
        >= 128
    )


def array_to_pil(
    array: np.ndarray,
) -> Image.Image:

    return Image.fromarray(
        array.astype(
            np.uint8
        ),
        mode="RGB",
    )


# =============================================================================
# CLASSICAL RESTORATION
# =============================================================================

def opencv_inpaint(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    radius: int,
    flag: int,
) -> np.ndarray:

    mask = mask.astype(
        bool,
        copy=False,
    )

    if not np.any(
        mask
    ):

        return image_rgb.copy()

    mask_u8 = (
        mask.astype(
            np.uint8
        )
        * 255
    )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    restored_bgr = cv2.inpaint(
        image_bgr,
        mask_u8,
        float(radius),
        flag,
    )

    return cv2.cvtColor(
        restored_bgr,
        cv2.COLOR_BGR2RGB,
    )


def restore_scratch(
    image_rgb: np.ndarray,
    predicted_mask: np.ndarray,
) -> np.ndarray:

    return opencv_inpaint(
        image_rgb,
        predicted_mask,
        SCRATCH_RADIUS,
        cv2.INPAINT_NS,
    )


def restore_missing(
    image_rgb: np.ndarray,
    predicted_mask: np.ndarray,
) -> np.ndarray:

    return opencv_inpaint(
        image_rgb,
        predicted_mask,
        INPAINT_RADIUS,
        cv2.INPAINT_TELEA,
    )


# =============================================================================
# TILE POSITIONS
# =============================================================================

def tile_positions(
    length: int,
    tile_size: int,
    stride: int,
) -> list[int]:

    if length <= tile_size:
        return [0]

    positions = list(
        range(
            0,
            length - tile_size + 1,
            stride,
        )
    )

    final_position = (
        length
        - tile_size
    )

    if (
        positions[-1]
        != final_position
    ):

        positions.append(
            final_position
        )

    return positions


# =============================================================================
# 29B DEBLUR
# =============================================================================

def pad_tile_to_512(
    tile: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
    int,
]:

    height, width = (
        tile.shape[:2]
    )

    pad_bottom = (
        DEBLUR_TILE_SIZE
        - height
    )

    pad_right = (
        DEBLUR_TILE_SIZE
        - width
    )

    if (
        pad_bottom == 0
        and pad_right == 0
    ):

        return (
            tile,
            height,
            width,
        )

    padded = np.pad(
        tile,
        (
            (
                0,
                pad_bottom,
            ),
            (
                0,
                pad_right,
            ),
            (
                0,
                0,
            ),
        ),
        mode="reflect",
    )

    return (
        padded,
        height,
        width,
    )


def deblur_native_image(
    deblur_module,
    deblur_model,
    image_rgb: np.ndarray,
    device: torch.device,
) -> np.ndarray:

    height, width = (
        image_rgb.shape[:2]
    )

    y_positions = (
        tile_positions(
            height,
            DEBLUR_TILE_SIZE,
            DEBLUR_TILE_STRIDE,
        )
    )

    x_positions = (
        tile_positions(
            width,
            DEBLUR_TILE_SIZE,
            DEBLUR_TILE_STRIDE,
        )
    )

    accumulator = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.float64,
    )

    weights = np.zeros(
        (
            height,
            width,
            1,
        ),
        dtype=np.float64,
    )

    for top in y_positions:

        for left in x_positions:

            bottom = min(
                top
                + DEBLUR_TILE_SIZE,
                height,
            )

            right = min(
                left
                + DEBLUR_TILE_SIZE,
                width,
            )

            tile = image_rgb[
                top:bottom,
                left:right,
            ]

            (
                model_tile,
                original_height,
                original_width,
            ) = pad_tile_to_512(
                tile
            )

            restored_tile = (
                deblur_module.infer(
                    deblur_model,
                    model_tile,
                    device,
                )
            )

            restored_tile = (
                restored_tile[
                    :original_height,
                    :original_width,
                ]
            )

            accumulator[
                top:bottom,
                left:right,
            ] += (
                restored_tile.astype(
                    np.float64
                )
            )

            weights[
                top:bottom,
                left:right,
            ] += 1.0

    if np.any(
        weights == 0
    ):

        raise RuntimeError(
            "Deblur tiling produced "
            "uncovered image pixels."
        )

    restored = (
        accumulator
        / weights
    )

    return np.clip(
        np.rint(
            restored
        ),
        0,
        255,
    ).astype(
        np.uint8
    )


def localized_hard_merge(
    original: np.ndarray,
    restored: np.ndarray,
    predicted_mask: np.ndarray,
) -> np.ndarray:

    output = original.copy()

    output[
        predicted_mask
    ] = restored[
        predicted_mask
    ]

    return output


# =============================================================================
# 30B NOISE SPECIALIST
# =============================================================================

def build_30b_noise_model(
    device: torch.device,
):

    if not DENOISER_WEIGHT.exists():

        raise FileNotFoundError(
            f"30B noise checkpoint missing:\n"
            f"{DENOISER_WEIGHT}"
        )

    checkpoint = torch.load(
        DENOISER_WEIGHT,
        map_location="cpu",
        weights_only=False,
    )

    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[
            4,
            6,
            6,
            8,
        ],
        num_refinement_blocks=4,
        heads=[
            1,
            2,
            4,
            8,
        ],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="BiasFree",
        dual_pixel_task=False,
    )

    if (
        "model_state_dict"
        not in checkpoint
    ):

        raise RuntimeError(
            "30B checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model = model.to(
        device
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


@torch.no_grad()
def run_30b_noise_model(
    model,
    image: Image.Image,
    device: torch.device,
) -> Image.Image:

    array = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    ) / 255.0

    original_height, original_width = (
        array.shape[:2]
    )

    # Restormer downsamples three times,
    # so H and W must be divisible by 8.
    multiple = 8

    padded_height = (
        math.ceil(
            original_height
            / multiple
        )
        * multiple
    )

    padded_width = (
        math.ceil(
            original_width
            / multiple
        )
        * multiple
    )

    pad_bottom = (
        padded_height
        - original_height
    )

    pad_right = (
        padded_width
        - original_width
    )

    if (
        pad_bottom > 0
        or pad_right > 0
    ):

        array = np.pad(
            array,
            (
                (
                    0,
                    pad_bottom,
                ),
                (
                    0,
                    pad_right,
                ),
                (
                    0,
                    0,
                ),
            ),
            mode="reflect",
        )

    tensor = (
        torch.from_numpy(
            np.ascontiguousarray(
                array.transpose(
                    2,
                    0,
                    1,
                )
            )
        )
        .unsqueeze(0)
        .to(
            device
        )
    )

    with torch.amp.autocast(
        device_type="cuda",
        enabled=(
            device.type
            == "cuda"
        ),
    ):

        restored = model(
            tensor
        )

    restored = torch.clamp(
        restored,
        0.0,
        1.0,
    )

    output = (
        restored[
            0
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
        .transpose(
            1,
            2,
            0,
        )
    )

    output = output[
        :original_height,
        :original_width,
    ]

    output = np.clip(
        np.rint(
            output
            * 255.0
        ),
        0,
        255,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        output,
        mode="RGB",
    )


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    path: Path,
    report: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "30B-updated six-route image restoration pipeline."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input image path.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Restored output image path.",
    )

    parser.add_argument(
        "--mask-output",
        type=Path,
        default=None,
        help="Optional predicted-mask output path.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    print("=" * 100)

    print(
        "30B-UPDATED MULTI-SPECIALIST "
        "RESTORATION PIPELINE"
    )

    print("=" * 100)
    print()

    # =========================================================================
    # VERIFY ARTIFACTS
    # =========================================================================

    print(
        "Verifying frozen artifacts..."
    )

    production_script_sha = (
        verify_hash(
            FROZEN_28J_SCRIPT,
            EXPECTED_28J_SCRIPT_SHA256,
            "Frozen 28J production script",
        )
    )

    detector_sha = (
        verify_hash(
            DETECTOR_CHECKPOINT,
            EXPECTED_DETECTOR_SHA256,
            "28J detector",
        )
    )

    classifier_sha = (
        verify_hash(
            CLASSIFIER_CHECKPOINT,
            EXPECTED_CLASSIFIER_SHA256,
            "29A classifier",
        )
    )

    deblur_sha = (
        verify_hash(
            DEBLUR_CHECKPOINT,
            EXPECTED_DEBLUR_SHA256,
            "29B deblur",
        )
    )

    if not DENOISER_WEIGHT.exists():

        raise FileNotFoundError(
            f"30B denoiser missing:\n"
            f"{DENOISER_WEIGHT}"
        )

    denoiser_sha = (
        file_sha256(
            DENOISER_WEIGHT
        )
    )

    print(
        "[PASS] Frozen 29F dependency "
        "hashes verified."
    )

    print(
        f"30B denoiser SHA256 : "
        f"{denoiser_sha}"
    )

    # =========================================================================
    # DEVICE
    # =========================================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required for "
            "the learned specialists."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"GPU : "
        f"{torch.cuda.get_device_name(0)}"
    )

    # =========================================================================
    # IMPORT IMPLEMENTATIONS
    # =========================================================================

    production = import_script(
        "frozen_28j_production_for_30b",
        FROZEN_28J_SCRIPT,
    )

    classifier_module = import_script(
        "frozen_29a_classifier_for_30b",
        CLASSIFIER_SCRIPT,
    )

    deblur_module = import_script(
        "frozen_29b_deblur_for_30b",
        DEBLUR_SCRIPT,
    )

    classifier_module.validate_production_pipeline(
        production
    )

    if not math.isclose(
        float(
            production.GATE_THRESHOLD
        ),
        GATE_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):

        raise RuntimeError(
            "Frozen detector gate mismatch."
        )

    if (
        classifier_module.CLASS_NAMES
        != CLASS_NAMES
    ):

        raise RuntimeError(
            "29A class ordering mismatch."
        )

    if (
        classifier_module.CLASS_TO_SPECIALIST
        != CLASS_TO_SPECIALIST
    ):

        raise RuntimeError(
            "29A specialist mapping mismatch."
        )

    # =========================================================================
    # LOAD INPUT
    # =========================================================================

    input_path = (
        args.input.resolve()
    )

    output_path = (
        args.output.resolve()
    )

    image = load_image(
        input_path
    )

    image_rgb = (
        pil_to_rgb_array(
            image
        )
    )

    width, height = (
        image.size
    )

    print(
        f"Input : {input_path}"
    )

    print(
        f"Size  : "
        f"{width} x {height}"
    )

    # =========================================================================
    # LOAD DETECTOR + CLASSIFIER
    # =========================================================================

    print()
    print(
        "Loading frozen 28J detector..."
    )

    detector_result = (
        production.load_detector(
            device
        )
    )

    if not isinstance(
        detector_result,
        tuple,
    ):

        raise RuntimeError(
            "Unexpected 28J detector loader result."
        )

    detector = (
        detector_result[
            0
        ]
    )

    print(
        "[PASS] Detector loaded."
    )

    print(
        "Loading frozen 29A classifier..."
    )

    (
        classifier,
        classifier_payload,
    ) = (
        classifier_module.load_classifier(
            device
        )
    )

    print(
        "[PASS] Classifier loaded."
    )

    # =========================================================================
    # DETECTOR
    # =========================================================================

    print()
    print(
        "Running damage detector..."
    )

    (
        predicted_mask_pil,
        predicted_mask_fraction,
    ) = (
        production.predict_mask(
            detector,
            image,
            device,
        )
    )

    predicted_mask = (
        mask_to_bool(
            predicted_mask_pil
        )
    )

    detector_route = bool(
        predicted_mask_fraction
        >= GATE_THRESHOLD
    )

    print(
        f"Predicted mask fraction : "
        f"{predicted_mask_fraction:.8f}"
    )

    print(
        f"Detector route          : "
        f"{'DAMAGE' if detector_route else 'PRESERVE'}"
    )

    # =========================================================================
    # DEFAULT ROUTE
    # =========================================================================

    predicted_class = (
        "clean"
    )

    classifier_confidence = (
        1.0
    )

    probabilities = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    specialist = (
        "preserve"
    )

    restored_rgb = (
        image_rgb.copy()
    )

    noise_checkpoint_epoch = (
        None
    )

    noise_checkpoint_psnr = (
        None
    )

    # =========================================================================
    # CLASSIFIER
    # =========================================================================

    if detector_route:

        print()
        print(
            "Running damage classifier..."
        )

        (
            predicted_class,
            classifier_confidence,
            probabilities,
        ) = (
            classifier_module.classify(
                classifier,
                image,
                predicted_mask_pil,
                device,
            )
        )

        specialist = (
            CLASS_TO_SPECIALIST[
                predicted_class
            ]
        )

        print(
            f"Predicted class : "
            f"{predicted_class}"
        )

        print(
            f"Confidence      : "
            f"{classifier_confidence:.6f}"
        )

        print(
            f"Specialist      : "
            f"{specialist}"
        )

        if specialist == "preserve":

            restored_rgb = (
                image_rgb.copy()
            )

        elif specialist == "denoise":

            print()
            print(
                "Loading 30B fine-tuned "
                "noise specialist..."
            )

            (
                noise_model,
                noise_checkpoint,
            ) = (
                build_30b_noise_model(
                    device
                )
            )

            noise_checkpoint_epoch = (
                noise_checkpoint.get(
                    "epoch"
                )
            )

            noise_checkpoint_psnr = (
                noise_checkpoint.get(
                    "val_psnr"
                )
            )

            print(
                "Running 30B noise specialist..."
            )

            raw_restored = (
                run_30b_noise_model(
                    noise_model,
                    image,
                    device,
                )
            )

            localized = (
                production.localize_restoration(
                    image,
                    raw_restored,
                    predicted_mask_pil,
                )
            )

            restored_rgb = (
                pil_to_rgb_array(
                    localized
                )
            )

            print(
                f"30B checkpoint epoch : "
                f"{noise_checkpoint_epoch}"
            )

            if (
                noise_checkpoint_psnr
                is not None
            ):

                print(
                    f"30B validation PSNR  : "
                    f"{float(noise_checkpoint_psnr):.3f} dB"
                )

            del noise_model

            torch.cuda.empty_cache()

        elif specialist == "deblur":

            print()
            print(
                "Loading frozen 29B "
                "deblur model..."
            )

            deblur_state = (
                deblur_module.load_finetuned_state()
            )

            deblur_model = (
                deblur_module.build_model(
                    deblur_state,
                    device,
                )
            )

            print(
                "Running tiled 29B "
                "deblur inference..."
            )

            full_restored = (
                deblur_native_image(
                    deblur_module,
                    deblur_model,
                    image_rgb,
                    device,
                )
            )

            restored_rgb = (
                localized_hard_merge(
                    image_rgb,
                    full_restored,
                    predicted_mask,
                )
            )

            del deblur_model

            torch.cuda.empty_cache()

        elif (
            specialist
            == "scratch_inpaint"
        ):

            print()
            print(
                "Running Navier-Stokes "
                f"scratch inpainting "
                f"r{SCRATCH_RADIUS}..."
            )

            restored_rgb = (
                restore_scratch(
                    image_rgb,
                    predicted_mask,
                )
            )

        elif specialist == "inpaint":

            print()
            print(
                "Running Telea shared "
                f"inpainting "
                f"r{INPAINT_RADIUS}..."
            )

            restored_rgb = (
                restore_missing(
                    image_rgb,
                    predicted_mask,
                )
            )

        else:

            raise RuntimeError(
                f"Unknown specialist: "
                f"{specialist}"
            )

    # =========================================================================
    # SAVE
    # =========================================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    array_to_pil(
        restored_rgb
    ).save(
        output_path
    )

    if (
        args.mask_output
        is not None
    ):

        mask_output = (
            args
            .mask_output
            .resolve()
        )

        mask_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        predicted_mask_pil.save(
            mask_output
        )

    # =========================================================================
    # REPORT
    # =========================================================================

    report = {
        "pipeline":
            "final_multispecialist_restoration_pipeline_30b",

        "status":
            "COMPLETE",

        "input_path":
            str(
                input_path
            ),

        "output_path":
            str(
                output_path
            ),

        "image_width":
            width,

        "image_height":
            height,

        "detector": {
            "predicted_mask_fraction":
                float(
                    predicted_mask_fraction
                ),

            "gate_threshold":
                GATE_THRESHOLD,

            "routed_as_damage":
                detector_route,
        },

        "classifier": {
            "executed":
                detector_route,

            "predicted_class":
                predicted_class,

            "confidence":
                float(
                    classifier_confidence
                ),

            "class_names":
                CLASS_NAMES,

            "probabilities":
                {
                    name:
                        float(
                            probabilities[
                                index
                            ]
                        )

                    for index, name
                    in enumerate(
                        CLASS_NAMES
                    )
                },
        },

        "routing": {
            "specialist":
                specialist,

            "class_to_specialist":
                CLASS_TO_SPECIALIST,
        },

        "specialist_configuration": {
            "noise": {
                "type":
                    "Restormer_30B_finetuned",

                "checkpoint":
                    str(
                        DENOISER_WEIGHT
                    ),

                "checkpoint_epoch":
                    noise_checkpoint_epoch,

                "checkpoint_validation_psnr":
                    (
                        float(
                            noise_checkpoint_psnr
                        )
                        if (
                            noise_checkpoint_psnr
                            is not None
                        )
                        else None
                    ),

                "localization":
                    "frozen_28j_feathered_predicted_mask",
            },

            "blur": {
                "type":
                    "Restormer",

                "checkpoint":
                    str(
                        DEBLUR_CHECKPOINT
                    ),

                "tile_size":
                    DEBLUR_TILE_SIZE,

                "tile_overlap":
                    DEBLUR_TILE_OVERLAP,

                "localization":
                    "hard_binary_predicted_mask",
            },

            "scratch": {
                "type":
                    "OpenCV",

                "method":
                    "Navier-Stokes",

                "radius":
                    SCRATCH_RADIUS,

                "localization":
                    "predicted_mask",
            },

            "missing_irregular": {
                "type":
                    "OpenCV",

                "method":
                    "Telea",

                "radius":
                    INPAINT_RADIUS,

                "localization":
                    "predicted_mask",
            },
        },

        "artifacts": {
            "28j_production_script_sha256":
                production_script_sha,

            "detector_sha256":
                detector_sha,

            "classifier_sha256":
                classifier_sha,

            "30b_denoiser_sha256":
                denoiser_sha,

            "deblur_sha256":
                deblur_sha,
        },
    }

    if (
        args.report
        is not None
    ):

        write_report(
            args.report.resolve(),
            report,
        )

    del detector
    del classifier

    torch.cuda.empty_cache()

    print()
    print("=" * 100)
    print(
        "RESTORATION COMPLETE"
    )
    print("=" * 100)

    print(
        f"Output            : "
        f"{output_path}"
    )

    print(
        f"Final class       : "
        f"{predicted_class}"
    )

    print(
        f"Final specialist  : "
        f"{specialist}"
    )


if __name__ == "__main__":
    main()