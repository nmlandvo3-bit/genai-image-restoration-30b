from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import resnet18


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path.cwd()

PIPELINE_PATH = (
    ROOT
    / "scripts"
    / "final_restoration_pipeline.py"
)

VALIDATION_MANIFEST = (
    ROOT
    / "data"
    / "experiment_29a"
    / "validation_manifest.csv"
)

CLASSIFIER_CHECKPOINT = (
    ROOT
    / "models"
    / "experiment_29a_damage_classifier"
    / "best_mask_aware_resnet18.pt"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "experiment_29a_4"
)

PREDICTED_MASK_ROOT = (
    OUTPUT_ROOT
    / "predicted_masks"
)

PREDICTIONS_CSV = (
    OUTPUT_ROOT
    / "predictions.csv"
)

SUMMARY_JSON = (
    OUTPUT_ROOT
    / "summary.json"
)


# =============================================================================
# FROZEN 29A-3 CANDIDATE
# =============================================================================

EXPECTED_CLASSIFIER_SHA256 = (
    "2c7f4e796fd75616c55352c6a6d6a272"
    "a4936fc0875acf7fed08b123c14dde37"
)

CLASSIFIER_IMAGE_SIZE = 320


CLASS_NAMES = [
    "clean",
    "noise",
    "blur",
    "scratch",
    "missing",
    "irregular",
]


CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(
        CLASS_NAMES
    )
}


# =============================================================================
# SPECIALIST ROUTES
# =============================================================================

SPECIALIST_NAMES = [
    "preserve",
    "denoise",
    "deblur",
    "scratch_inpaint",
    "inpaint",
]


CLASS_TO_SPECIALIST = {
    "clean":
        "preserve",

    "noise":
        "denoise",

    "blur":
        "deblur",

    "scratch":
        "scratch_inpaint",

    "missing":
        "inpaint",

    "irregular":
        "inpaint",
}


# =============================================================================
# IMAGENET NORMALIZATION
# =============================================================================

RGB_MEAN = np.asarray(
    [
        0.485,
        0.456,
        0.406,
    ],
    dtype=np.float32,
).reshape(
    1,
    1,
    3,
)

RGB_STD = np.asarray(
    [
        0.229,
        0.224,
        0.225,
    ],
    dtype=np.float32,
).reshape(
    1,
    1,
    3,
)


# =============================================================================
# HELPERS
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

            digest.update(
                block
            )

    return digest.hexdigest()


def load_csv(
    path: Path,
) -> list[dict[str, str]]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing CSV:\n{path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        rows = list(
            csv.DictReader(
                handle
            )
        )

    if not rows:

        raise RuntimeError(
            f"CSV contains no rows:\n{path}"
        )

    return rows


def import_module(
    path: Path,
    name: str,
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
# VERIFY FROZEN PRODUCTION DETECTOR
# =============================================================================

def validate_production_pipeline(
    pipeline,
) -> None:

    expected = {
        "IMAGE_SIZE":
            512,

        "DETECTOR_TILE_SIZE":
            640,

        "DETECTOR_TILE_OVERLAP_FRACTION":
            0.125,

        "DETECTOR_TILE_BATCH_SIZE":
            8,

        "DETECTOR_MASK_THRESHOLD":
            0.50,

        "GATE_THRESHOLD":
            0.0025,
    }

    for key, expected_value in (
        expected.items()
    ):

        if not hasattr(
            pipeline,
            key,
        ):

            raise RuntimeError(
                f"Production pipeline missing "
                f"constant: {key}"
            )

        actual = getattr(
            pipeline,
            key,
        )

        if actual != expected_value:

            raise RuntimeError(
                "Frozen detector configuration "
                "mismatch.\n"
                f"{key}\n"
                f"Expected: {expected_value}\n"
                f"Actual:   {actual}"
            )

    required_functions = [
        "load_detector",
        "load_input_image",
        "predict_mask",
    ]

    for name in required_functions:

        if not hasattr(
            pipeline,
            name,
        ):

            raise RuntimeError(
                "Production pipeline missing "
                f"function: {name}"
            )


# =============================================================================
# LOAD 29A CLASSIFIER
# =============================================================================

def load_classifier(
    device: torch.device,
) -> tuple[
    nn.Module,
    dict,
]:

    if not CLASSIFIER_CHECKPOINT.exists():

        raise FileNotFoundError(
            "29A-3 classifier checkpoint "
            f"missing:\n{CLASSIFIER_CHECKPOINT}"
        )

    actual_sha = file_sha256(
        CLASSIFIER_CHECKPOINT
    )

    if (
        actual_sha
        != EXPECTED_CLASSIFIER_SHA256
    ):

        raise RuntimeError(
            "29A-3 checkpoint SHA256 mismatch.\n"
            f"Expected: "
            f"{EXPECTED_CLASSIFIER_SHA256}\n"
            f"Actual:   {actual_sha}"
        )

    checkpoint = torch.load(
        CLASSIFIER_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_classes = (
        checkpoint.get(
            "class_names"
        )
    )

    if (
        checkpoint_classes
        != CLASS_NAMES
    ):

        raise RuntimeError(
            "Classifier class ordering "
            "does not match 29A."
        )

    if (
        int(
            checkpoint.get(
                "input_channels",
                -1,
            )
        )
        != 4
    ):

        raise RuntimeError(
            "Classifier is not the expected "
            "4-channel model."
        )

    if (
        int(
            checkpoint.get(
                "image_size",
                -1,
            )
        )
        != CLASSIFIER_IMAGE_SIZE
    ):

        raise RuntimeError(
            "Classifier image-size mismatch."
        )

    model = resnet18(
        weights=None
    )

    old_conv = model.conv1

    model.conv1 = nn.Conv2d(
        in_channels=4,
        out_channels=
            old_conv.out_channels,
        kernel_size=
            old_conv.kernel_size,
        stride=
            old_conv.stride,
        padding=
            old_conv.padding,
        bias=False,
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(
            CLASS_NAMES
        ),
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    incompatible = (
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    )

    if incompatible.missing_keys:

        raise RuntimeError(
            "Classifier missing keys:\n"
            f"{incompatible.missing_keys}"
        )

    if incompatible.unexpected_keys:

        raise RuntimeError(
            "Classifier unexpected keys:\n"
            f"{incompatible.unexpected_keys}"
        )

    model = model.to(
        device
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


# =============================================================================
# CLASSIFIER INPUT
# =============================================================================

def make_classifier_tensor(
    image: Image.Image,
    mask: Image.Image,
) -> torch.Tensor:

    resized_image = (
        image
        .convert(
            "RGB"
        )
        .resize(
            (
                CLASSIFIER_IMAGE_SIZE,
                CLASSIFIER_IMAGE_SIZE,
            ),
            Image.Resampling.LANCZOS,
        )
    )

    resized_mask = (
        mask
        .convert(
            "L"
        )
        .resize(
            (
                CLASSIFIER_IMAGE_SIZE,
                CLASSIFIER_IMAGE_SIZE,
            ),
            Image.Resampling.NEAREST,
        )
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32,
    ) / 255.0

    mask_array = np.asarray(
        resized_mask,
        dtype=np.float32,
    ) / 255.0

    mask_array = (
        mask_array >= 0.5
    ).astype(
        np.float32
    )

    image_array = (
        image_array
        - RGB_MEAN
    ) / RGB_STD

    mask_array = np.expand_dims(
        mask_array,
        axis=2,
    )

    four_channel = np.concatenate(
        [
            image_array,
            mask_array,
        ],
        axis=2,
    )

    tensor = (
        torch.from_numpy(
            four_channel
        )
        .permute(
            2,
            0,
            1,
        )
        .unsqueeze(
            0
        )
        .contiguous()
        .float()
    )

    return tensor


def classify(
    model: nn.Module,
    image: Image.Image,
    mask: Image.Image,
    device: torch.device,
) -> tuple[
    str,
    float,
    list[float],
]:

    tensor = make_classifier_tensor(
        image,
        mask,
    ).to(
        device,
        non_blocking=True,
    )

    with torch.inference_mode():

        logits = model(
            tensor
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )
            .squeeze(
                0
            )
            .detach()
            .cpu()
            .numpy()
        )

    prediction_index = int(
        np.argmax(
            probabilities
        )
    )

    prediction_class = (
        CLASS_NAMES[
            prediction_index
        ]
    )

    confidence = float(
        probabilities[
            prediction_index
        ]
    )

    return (
        prediction_class,
        confidence,
        [
            float(
                value
            )
            for value
            in probabilities
        ],
    )


# =============================================================================
# SEGMENTATION METRICS
# =============================================================================

def mask_metrics(
    predicted: Image.Image,
    target: Image.Image,
) -> dict[str, float]:

    if target.size != predicted.size:

        target = target.resize(
            predicted.size,
            Image.Resampling.NEAREST,
        )

    predicted_array = (
        np.asarray(
            predicted.convert(
                "L"
            ),
            dtype=np.uint8,
        )
        >= 128
    )

    target_array = (
        np.asarray(
            target.convert(
                "L"
            ),
            dtype=np.uint8,
        )
        >= 128
    )

    tp = int(
        np.logical_and(
            predicted_array,
            target_array,
        ).sum()
    )

    fp = int(
        np.logical_and(
            predicted_array,
            ~target_array,
        ).sum()
    )

    fn = int(
        np.logical_and(
            ~predicted_array,
            target_array,
        ).sum()
    )

    predicted_count = int(
        predicted_array.sum()
    )

    target_count = int(
        target_array.sum()
    )

    union = (
        tp
        + fp
        + fn
    )

    if union > 0:

        iou = (
            tp
            / union
        )

    else:

        iou = 1.0

    denominator = (
        2 * tp
        + fp
        + fn
    )

    if denominator > 0:

        dice = (
            2 * tp
            / denominator
        )

    else:

        dice = 1.0

    predicted_fraction = float(
        predicted_array.mean()
    )

    target_fraction = float(
        target_array.mean()
    )

    return {
        "iou":
            float(
                iou
            ),

        "dice":
            float(
                dice
            ),

        "predicted_fraction":
            predicted_fraction,

        "target_fraction":
            target_fraction,

        "predicted_pixels":
            predicted_count,

        "target_pixels":
            target_count,
    }


# =============================================================================
# CONFUSION MATRIX / CLASSIFICATION METRICS
# =============================================================================

def confusion_matrix(
    true_values: list[str],
    predicted_values: list[str],
    names: list[str],
) -> np.ndarray:

    index = {
        name: i
        for i, name
        in enumerate(
            names
        )
    }

    matrix = np.zeros(
        (
            len(names),
            len(names),
        ),
        dtype=np.int64,
    )

    for true_name, predicted_name in zip(
        true_values,
        predicted_values,
    ):

        matrix[
            index[
                true_name
            ],
            index[
                predicted_name
            ],
        ] += 1

    return matrix


def classification_metrics(
    matrix: np.ndarray,
    names: list[str],
) -> dict:

    total = int(
        matrix.sum()
    )

    correct = int(
        np.trace(
            matrix
        )
    )

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    per_class = {}

    f1_values = []

    recall_values = []

    for index, name in enumerate(
        names
    ):

        tp = int(
            matrix[
                index,
                index,
            ]
        )

        fp = int(
            matrix[
                :,
                index,
            ].sum()
            - tp
        )

        fn = int(
            matrix[
                index,
                :,
            ].sum()
            - tp
        )

        support = int(
            matrix[
                index,
                :
            ].sum()
        )

        precision = (
            tp
            / (
                tp
                + fp
            )
            if (
                tp
                + fp
            ) > 0
            else 0.0
        )

        recall = (
            tp
            / (
                tp
                + fn
            )
            if (
                tp
                + fn
            ) > 0
            else 0.0
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            ) > 0
            else 0.0
        )

        per_class[
            name
        ] = {
            "precision":
                float(
                    precision
                ),

            "recall":
                float(
                    recall
                ),

            "f1":
                float(
                    f1
                ),

            "support":
                support,
        }

        if support > 0:

            f1_values.append(
                f1
            )

            recall_values.append(
                recall
            )

    return {
        "accuracy":
            float(
                accuracy
            ),

        "macro_f1":
            float(
                np.mean(
                    f1_values
                )
                if f1_values
                else 0.0
            ),

        "balanced_accuracy":
            float(
                np.mean(
                    recall_values
                )
                if recall_values
                else 0.0
            ),

        "per_class":
            per_class,

        "confusion_matrix":
            matrix.tolist(),
    }


# =============================================================================
# OUTPUT
# =============================================================================

def write_predictions(
    rows: list[dict],
) -> None:

    if not rows:
        return

    with PREDICTIONS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def print_matrix(
    matrix: np.ndarray,
    names: list[str],
) -> None:

    print(
        "             "
        + " ".join(
            f"{name[:8]:>8}"
            for name in names
        )
    )

    for index, name in enumerate(
        names
    ):

        print(
            f"{name[:12]:>12} "
            + " ".join(
                f"{int(value):8d}"
                for value
                in matrix[
                    index
                ]
            )
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print(
        "=" * 110
    )

    print(
        "EXPERIMENT 29A-4"
    )

    print(
        "PREDICTED-MASK PRODUCTION "
        "REALISM VALIDATION"
    )

    print(
        "=" * 110
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTED_MASK_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required for 29A-4."
        )

    device = torch.device(
        "cuda"
    )

    print()

    print(
        f"GPU                    : "
        f"{torch.cuda.get_device_name(0)}"
    )

    # =========================================================================
    # LOAD PRODUCTION PIPELINE
    # =========================================================================

    pipeline = import_module(
        PIPELINE_PATH,
        "production_28j_pipeline",
    )

    validate_production_pipeline(
        pipeline
    )

    pipeline_sha = file_sha256(
        PIPELINE_PATH
    )

    print(
        f"Production pipeline SHA: "
        f"{pipeline_sha}"
    )

    print(
        f"Detector checkpoint     : "
        f"{pipeline.DETECTOR_CHECKPOINT}"
    )

    print(
        f"Detector SHA expected   : "
        f"{pipeline.EXPECTED_DETECTOR_SHA256}"
    )

    print(
        f"Detector tile           : "
        f"{pipeline.DETECTOR_TILE_SIZE}"
    )

    print(
        f"Detector overlap        : "
        f"{pipeline.DETECTOR_TILE_OVERLAP_FRACTION}"
    )

    print(
        f"Detector batch          : "
        f"{pipeline.DETECTOR_TILE_BATCH_SIZE}"
    )

    print(
        f"Mask threshold          : "
        f"{pipeline.DETECTOR_MASK_THRESHOLD}"
    )

    print(
        f"Routing gate            : "
        f"{pipeline.GATE_THRESHOLD}"
    )

    print()

    print(
        "Loading frozen 28J detector..."
    )

    detector, base_channels = (
        pipeline.load_detector(
            device
        )
    )

    print(
        f"Detector base channels  : "
        f"{base_channels}"
    )

    print(
        "[PASS] Frozen production detector loaded."
    )

    # =========================================================================
    # CLASSIFIER
    # =========================================================================

    print()

    print(
        "Loading 29A-3 classifier..."
    )

    classifier, classifier_payload = (
        load_classifier(
            device
        )
    )

    print(
        f"Classifier checkpoint   : "
        f"{CLASSIFIER_CHECKPOINT}"
    )

    print(
        f"Classifier SHA256       : "
        f"{EXPECTED_CLASSIFIER_SHA256}"
    )

    print(
        f"Classifier best epoch   : "
        f"{classifier_payload.get('epoch')}"
    )

    print(
        "[PASS] 29A-3 classifier loaded."
    )

    # =========================================================================
    # VALIDATION DATA ONLY
    # =========================================================================

    manifest = load_csv(
        VALIDATION_MANIFEST
    )

    print()

    print(
        f"Validation samples      : "
        f"{len(manifest)}"
    )

    print(
        "Test manifest           : NOT ACCESSED"
    )

    print(
        "External 28J-N          : NOT ACCESSED"
    )

    validation_classes = Counter(
        row[
            "class_name"
        ]
        for row in manifest
    )

    print()

    print(
        "Validation class counts:"
    )

    for class_name in CLASS_NAMES:

        print(
            f"  {class_name:<12}: "
            f"{validation_classes[class_name]}"
        )

    # =========================================================================
    # METRIC STORAGE
    # =========================================================================

    routing_total = Counter()

    routing_positive = Counter()

    mask_iou = defaultdict(
        list
    )

    mask_dice = defaultdict(
        list
    )

    true_classifier_all = []

    predicted_classifier_all = []

    true_end_to_end = []

    predicted_end_to_end = []

    true_specialist = []

    predicted_specialist = []

    routed_damaged_true = []

    routed_damaged_predicted = []

    result_rows = []

    start_time = time.time()

    # =========================================================================
    # EVALUATION
    # =========================================================================

    print()

    print(
        "=" * 110
    )

    print(
        "RUNNING FROZEN DETECTOR + "
        "29A CLASSIFIER"
    )

    print(
        "=" * 110
    )

    for sample_index, row in enumerate(
        manifest,
        start=1,
    ):

        sample_id = row[
            "sample_id"
        ]

        true_class = row[
            "class_name"
        ]

        if true_class not in CLASS_NAMES:

            raise RuntimeError(
                f"Unknown class in validation "
                f"manifest: {true_class}"
            )

        image_path = Path(
            row[
                "image_path"
            ]
        )

        gt_mask_path = Path(
            row[
                "mask_path"
            ]
        )

        if not image_path.is_absolute():

            image_path = (
                ROOT
                / image_path
            )

        if not gt_mask_path.is_absolute():

            gt_mask_path = (
                ROOT
                / gt_mask_path
            )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image missing:\n{image_path}"
            )

        if not gt_mask_path.exists():

            raise FileNotFoundError(
                f"GT mask missing:\n"
                f"{gt_mask_path}"
            )

        # ---------------------------------------------------------------------
        # EXACT PRODUCTION DETECTOR
        # ---------------------------------------------------------------------

        image = (
            pipeline.load_input_image(
                image_path
            )
        )

        predicted_mask, \
            predicted_fraction = (
                pipeline.predict_mask(
                    detector,
                    image,
                    device,
                )
            )

        route_positive = (
            predicted_fraction
            >= pipeline.GATE_THRESHOLD
        )

        routing_total[
            true_class
        ] += 1

        if route_positive:

            routing_positive[
                true_class
            ] += 1

        # Save predicted mask so this expensive
        # detector pass is auditable and reusable.
        predicted_mask_path = (
            PREDICTED_MASK_ROOT
            / f"{sample_id}.png"
        )

        predicted_mask.save(
            predicted_mask_path
        )

        # ---------------------------------------------------------------------
        # GT MASK ONLY FOR SCORING
        # ---------------------------------------------------------------------

        with Image.open(
            gt_mask_path
        ) as opened:

            gt_mask = opened.convert(
                "L"
            )

        segmentation = mask_metrics(
            predicted_mask,
            gt_mask,
        )

        if true_class != "clean":

            mask_iou[
                true_class
            ].append(
                segmentation[
                    "iou"
                ]
            )

            mask_dice[
                true_class
            ].append(
                segmentation[
                    "dice"
                ]
            )

        # ---------------------------------------------------------------------
        # CLASSIFIER WITH PREDICTED MASK
        # ---------------------------------------------------------------------

        predicted_class, \
            confidence, \
            probabilities = (
                classify(
                    classifier,
                    image,
                    predicted_mask,
                    device,
                )
            )

        true_classifier_all.append(
            true_class
        )

        predicted_classifier_all.append(
            predicted_class
        )

        # ---------------------------------------------------------------------
        # END-TO-END SIX-CLASS DECISION
        #
        # If 28J does not route, the system
        # effectively decides CLEAN/PRESERVE.
        # ---------------------------------------------------------------------

        if route_positive:

            end_to_end_class = (
                predicted_class
            )

        else:

            end_to_end_class = (
                "clean"
            )

        true_end_to_end.append(
            true_class
        )

        predicted_end_to_end.append(
            end_to_end_class
        )

        # ---------------------------------------------------------------------
        # SPECIALIST ROUTING
        # ---------------------------------------------------------------------

        true_route = (
            CLASS_TO_SPECIALIST[
                true_class
            ]
        )

        if route_positive:

            predicted_route = (
                CLASS_TO_SPECIALIST[
                    predicted_class
                ]
            )

        else:

            predicted_route = (
                "preserve"
            )

        true_specialist.append(
            true_route
        )

        predicted_specialist.append(
            predicted_route
        )

        # ---------------------------------------------------------------------
        # CLASSIFIER PERFORMANCE ONLY ON
        # DAMAGED SAMPLES THAT THE DETECTOR
        # ACTUALLY ROUTED.
        # ---------------------------------------------------------------------

        if (
            true_class != "clean"
            and route_positive
        ):

            routed_damaged_true.append(
                true_class
            )

            routed_damaged_predicted.append(
                predicted_class
            )

        result_row = {
            "sample_id":
                sample_id,

            "source_id":
                row[
                    "source_id"
                ],

            "true_class":
                true_class,

            "predicted_mask_fraction":
                predicted_fraction,

            "gate_threshold":
                pipeline.GATE_THRESHOLD,

            "detector_routed":
                int(
                    route_positive
                ),

            "mask_iou":
                segmentation[
                    "iou"
                ],

            "mask_dice":
                segmentation[
                    "dice"
                ],

            "gt_mask_fraction":
                segmentation[
                    "target_fraction"
                ],

            "classifier_prediction":
                predicted_class,

            "classifier_confidence":
                confidence,

            "end_to_end_class":
                end_to_end_class,

            "true_specialist_route":
                true_route,

            "predicted_specialist_route":
                predicted_route,

            "prob_clean":
                probabilities[
                    CLASS_TO_INDEX[
                        "clean"
                    ]
                ],

            "prob_noise":
                probabilities[
                    CLASS_TO_INDEX[
                        "noise"
                    ]
                ],

            "prob_blur":
                probabilities[
                    CLASS_TO_INDEX[
                        "blur"
                    ]
                ],

            "prob_scratch":
                probabilities[
                    CLASS_TO_INDEX[
                        "scratch"
                    ]
                ],

            "prob_missing":
                probabilities[
                    CLASS_TO_INDEX[
                        "missing"
                    ]
                ],

            "prob_irregular":
                probabilities[
                    CLASS_TO_INDEX[
                        "irregular"
                    ]
                ],

            "image_path":
                str(
                    image_path
                ),

            "predicted_mask_path":
                str(
                    predicted_mask_path
                ),
        }

        result_rows.append(
            result_row
        )

        if (
            sample_index == 1
            or sample_index % 50 == 0
            or sample_index == len(
                manifest
            )
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"{sample_index:4d}/"
                f"{len(manifest)} | "
                f"{true_class:<9} | "
                f"mask={predicted_fraction:.6f} | "
                f"route={'YES' if route_positive else 'NO ':<3} | "
                f"class={predicted_class:<9} | "
                f"elapsed={elapsed / 60.0:.1f} min"
            )

    # =========================================================================
    # WRITE PER-SAMPLE OUTPUT
    # =========================================================================

    write_predictions(
        result_rows
    )

    # =========================================================================
    # DETECTOR ROUTING METRICS
    # =========================================================================

    routing_summary = {}

    damaged_total = 0
    damaged_routed = 0

    for class_name in CLASS_NAMES:

        total = int(
            routing_total[
                class_name
            ]
        )

        routed = int(
            routing_positive[
                class_name
            ]
        )

        rate = (
            routed / total
            if total
            else 0.0
        )

        routing_summary[
            class_name
        ] = {
            "total":
                total,

            "routed":
                routed,

            "route_rate":
                float(
                    rate
                ),
        }

        if class_name != "clean":

            damaged_total += (
                total
            )

            damaged_routed += (
                routed
            )

    detector_damage_recall = (
        damaged_routed
        / damaged_total
        if damaged_total
        else 0.0
    )

    clean_total = (
        routing_total[
            "clean"
        ]
    )

    clean_false_routes = (
        routing_positive[
            "clean"
        ]
    )

    clean_false_route_rate = (
        clean_false_routes
        / clean_total
        if clean_total
        else 0.0
    )

    # =========================================================================
    # CLASSIFIER USING PREDICTED MASKS
    # =========================================================================

    classifier_matrix = (
        confusion_matrix(
            true_classifier_all,
            predicted_classifier_all,
            CLASS_NAMES,
        )
    )

    classifier_metrics = (
        classification_metrics(
            classifier_matrix,
            CLASS_NAMES,
        )
    )

    # =========================================================================
    # END-TO-END 6-CLASS
    # =========================================================================

    end_to_end_matrix = (
        confusion_matrix(
            true_end_to_end,
            predicted_end_to_end,
            CLASS_NAMES,
        )
    )

    end_to_end_metrics = (
        classification_metrics(
            end_to_end_matrix,
            CLASS_NAMES,
        )
    )

    # =========================================================================
    # SPECIALIST ROUTING
    #
    # Missing + irregular are deliberately
    # merged into one INPAINT route.
    # =========================================================================

    specialist_matrix = (
        confusion_matrix(
            true_specialist,
            predicted_specialist,
            SPECIALIST_NAMES,
        )
    )

    specialist_metrics = (
        classification_metrics(
            specialist_matrix,
            SPECIALIST_NAMES,
        )
    )

    # =========================================================================
    # CLASSIFIER ON ROUTED DAMAGED ONLY
    # =========================================================================

    routed_damaged_correct = sum(
        true_value
        == predicted_value
        for true_value, predicted_value
        in zip(
            routed_damaged_true,
            routed_damaged_predicted,
        )
    )

    routed_damaged_accuracy = (
        routed_damaged_correct
        / len(
            routed_damaged_true
        )
        if routed_damaged_true
        else 0.0
    )

    routed_damaged_specialist_correct = sum(
        CLASS_TO_SPECIALIST[
            true_value
        ]
        ==
        CLASS_TO_SPECIALIST[
            predicted_value
        ]
        for true_value, predicted_value
        in zip(
            routed_damaged_true,
            routed_damaged_predicted,
        )
    )

    routed_damaged_specialist_accuracy = (
        routed_damaged_specialist_correct
        / len(
            routed_damaged_true
        )
        if routed_damaged_true
        else 0.0
    )

    # =========================================================================
    # MASK QUALITY
    # =========================================================================

    mask_quality = {}

    for class_name in CLASS_NAMES:

        if class_name == "clean":
            continue

        ious = mask_iou[
            class_name
        ]

        dices = mask_dice[
            class_name
        ]

        mask_quality[
            class_name
        ] = {
            "count":
                len(
                    ious
                ),

            "mean_iou":
                float(
                    np.mean(
                        ious
                    )
                    if ious
                    else 0.0
                ),

            "mean_dice":
                float(
                    np.mean(
                        dices
                    )
                    if dices
                    else 0.0
                ),

            "median_iou":
                float(
                    np.median(
                        ious
                    )
                    if ious
                    else 0.0
                ),

            "median_dice":
                float(
                    np.median(
                        dices
                    )
                    if dices
                    else 0.0
                ),
        }

    total_seconds = (
        time.time()
        - start_time
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    summary = {
        "experiment":
            "29A-4",

        "status":
            "PREDICTED_MASK_VALIDATION_COMPLETE",

        "evaluation_set":
            "29A validation only",

        "validation_samples":
            len(
                manifest
            ),

        "production_pipeline":
            str(
                PIPELINE_PATH
            ),

        "production_pipeline_sha256":
            pipeline_sha,

        "detector_checkpoint":
            str(
                pipeline.DETECTOR_CHECKPOINT
            ),

        "detector_checkpoint_sha256":
            pipeline.EXPECTED_DETECTOR_SHA256,

        "detector_configuration": {
            "input_size":
                pipeline.IMAGE_SIZE,

            "tile_size":
                pipeline.DETECTOR_TILE_SIZE,

            "overlap_fraction":
                pipeline.DETECTOR_TILE_OVERLAP_FRACTION,

            "tile_batch_size":
                pipeline.DETECTOR_TILE_BATCH_SIZE,

            "mask_threshold":
                pipeline.DETECTOR_MASK_THRESHOLD,

            "routing_gate":
                pipeline.GATE_THRESHOLD,
        },

        "classifier_checkpoint":
            str(
                CLASSIFIER_CHECKPOINT
            ),

        "classifier_sha256":
            EXPECTED_CLASSIFIER_SHA256,

        "classifier_best_epoch":
            classifier_payload.get(
                "epoch"
            ),

        "classifier_input":
            (
                "RGB + frozen 28J "
                "predicted binary mask"
            ),

        "detector_routing":
            routing_summary,

        "detector_damaged_route_recall":
            float(
                detector_damage_recall
            ),

        "clean_false_route_rate":
            float(
                clean_false_route_rate
            ),

        "classifier_predicted_mask_metrics":
            classifier_metrics,

        "end_to_end_six_class_metrics":
            end_to_end_metrics,

        "specialist_route_metrics":
            specialist_metrics,

        "routed_damaged_classifier_accuracy":
            float(
                routed_damaged_accuracy
            ),

        "routed_damaged_specialist_accuracy":
            float(
                routed_damaged_specialist_accuracy
            ),

        "mask_quality_by_damage_class":
            mask_quality,

        "specialist_mapping":
            CLASS_TO_SPECIALIST,

        "important_note":
            (
                "Ground-truth masks were used "
                "only for segmentation scoring. "
                "Classifier inference used only "
                "frozen 28J predicted masks."
            ),

        "holdout_status": {
            "29a_test":
                "UNTOUCHED",

            "28j_n_external":
                "UNTOUCHED",
        },

        "predictions_csv":
            str(
                PREDICTIONS_CSV
            ),

        "total_seconds":
            float(
                total_seconds
            ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # CONSOLE REPORT
    # =========================================================================

    print()

    print(
        "=" * 110
    )

    print(
        "29A-4 RESULTS"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "DETECTOR ROUTING"
    )

    for class_name in CLASS_NAMES:

        item = routing_summary[
            class_name
        ]

        print(
            f"  {class_name:<10} "
            f"{item['routed']:4d}/"
            f"{item['total']:4d} "
            f"= "
            f"{item['route_rate'] * 100:6.2f}%"
        )

    print()

    print(
        f"Damaged routing recall : "
        f"{detector_damage_recall * 100:.2f}%"
    )

    print(
        f"Clean false-route rate : "
        f"{clean_false_route_rate * 100:.2f}%"
    )

    print()

    print(
        "CLASSIFIER WITH PREDICTED MASKS"
    )

    print(
        f"  Accuracy            : "
        f"{classifier_metrics['accuracy']:.6f}"
    )

    print(
        f"  Macro F1            : "
        f"{classifier_metrics['macro_f1']:.6f}"
    )

    print(
        f"  Balanced accuracy   : "
        f"{classifier_metrics['balanced_accuracy']:.6f}"
    )

    print()

    print(
        "Classifier confusion matrix:"
    )

    print_matrix(
        classifier_matrix,
        CLASS_NAMES,
    )

    print()

    print(
        "END-TO-END 6-CLASS SYSTEM"
    )

    print(
        f"  Accuracy            : "
        f"{end_to_end_metrics['accuracy']:.6f}"
    )

    print(
        f"  Macro F1            : "
        f"{end_to_end_metrics['macro_f1']:.6f}"
    )

    print(
        f"  Balanced accuracy   : "
        f"{end_to_end_metrics['balanced_accuracy']:.6f}"
    )

    print()

    print(
        "End-to-end confusion matrix:"
    )

    print_matrix(
        end_to_end_matrix,
        CLASS_NAMES,
    )

    print()

    print(
        "SPECIALIST ROUTING"
    )

    print(
        f"  Accuracy            : "
        f"{specialist_metrics['accuracy']:.6f}"
    )

    print(
        f"  Macro F1            : "
        f"{specialist_metrics['macro_f1']:.6f}"
    )

    print(
        f"  Balanced accuracy   : "
        f"{specialist_metrics['balanced_accuracy']:.6f}"
    )

    print()

    print(
        "Specialist confusion matrix:"
    )

    print_matrix(
        specialist_matrix,
        SPECIALIST_NAMES,
    )

    print()

    print(
        "ROUTED DAMAGED IMAGES ONLY"
    )

    print(
        f"  Exact 6-class accuracy : "
        f"{routed_damaged_accuracy:.6f}"
    )

    print(
        f"  Specialist accuracy    : "
        f"{routed_damaged_specialist_accuracy:.6f}"
    )

    print()

    print(
        "PREDICTED MASK QUALITY"
    )

    for class_name in (
        "noise",
        "blur",
        "scratch",
        "missing",
        "irregular",
    ):

        item = mask_quality[
            class_name
        ]

        print(
            f"  {class_name:<10} "
            f"IoU={item['mean_iou']:.4f} "
            f"Dice={item['mean_dice']:.4f}"
        )

    print()

    print(
        f"Predictions CSV       : "
        f"{PREDICTIONS_CSV}"
    )

    print(
        f"Summary JSON          : "
        f"{SUMMARY_JSON}"
    )

    print(
        f"Predicted masks       : "
        f"{PREDICTED_MASK_ROOT}"
    )

    print(
        f"Evaluation time       : "
        f"{total_seconds / 60.0:.2f} min"
    )

    print()

    print(
        "[PASS] 29A-4 production-realism "
        "validation complete."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "29A test and external 28J-N "
        "remain untouched."
    )


if __name__ == "__main__":
    main()
