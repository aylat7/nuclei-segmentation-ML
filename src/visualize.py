"""Color-coded overlay visualizations for instance-level segmentation evaluation."""

from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label


def count_nuclei(binary_mask: np.ndarray) -> int:
    """Count the number of nuclei (connected components) in a binary mask.

    Args:
        binary_mask: 2D binary or boolean array of shape (H, W).

    Returns:
        Number of connected components found in the mask.

    Example:
        >>> import numpy as np
        >>> mask = np.zeros((20, 20), dtype=bool)
        >>> mask[0:5, 0:5] = True
        >>> mask[10:15, 10:15] = True
        >>> count_nuclei(mask)
        2
    """
    _, count = label(binary_mask.astype(bool))
    return int(count)


def extract_instances(binary_mask: np.ndarray) -> List[np.ndarray]:
    """Find connected components in a binary mask, one array per nucleus.

    Args:
        binary_mask: 2D boolean or 0/1 array of shape (H, W).

    Returns:
        List of boolean (H, W) arrays, one per connected component.
        Empty if no foreground pixels exist.

    Example:
        >>> mask = np.zeros((10, 10), dtype=bool)
        >>> mask[1:3, 1:3] = True
        >>> mask[7:9, 7:9] = True
        >>> len(extract_instances(mask))
        2
    """
    labeled, num_components = label(binary_mask.astype(bool))
    instances = []
    for component_id in range(1, num_components + 1):
        instances.append(labeled == component_id)
    return instances


def _instance_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two boolean mask arrays."""
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return intersection / union if union > 0 else 0.0


def match_instances(
    pred_instances: List[np.ndarray],
    gt_instances: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Greedily match predicted instances to ground truth instances by IoU.

    A predicted instance is a True Positive (TP) if it overlaps with some
    ground truth instance at IoU >= iou_threshold. Unmatched predictions are
    False Positives (FP); unmatched ground truths are False Negatives (FN).

    Args:
        pred_instances: Predicted connected-component masks.
        gt_instances: Ground truth connected-component masks.
        iou_threshold: Minimum IoU to count as a match.

    Returns:
        Tuple of (true_positives, false_positives, false_negatives) where each
        element is a list of boolean (H, W) arrays.
    """
    if not pred_instances:
        return [], [], list(gt_instances)
    if not gt_instances:
        return [], list(pred_instances), []

    # Build IoU matrix: rows = predictions, cols = GT
    iou_matrix = np.zeros((len(pred_instances), len(gt_instances)))
    for i, pred in enumerate(pred_instances):
        for j, gt in enumerate(gt_instances):
            iou_matrix[i, j] = _instance_iou(pred, gt)

    matched_pred = set()
    matched_gt = set()

    # Greedy matching: pick highest IoU pair repeatedly
    while True:
        if iou_matrix.max() < iou_threshold:
            break
        i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        matched_pred.add(i)
        matched_gt.add(j)
        iou_matrix[i, :] = -1
        iou_matrix[:, j] = -1

    true_positives = [pred_instances[i] for i in matched_pred]
    false_positives = [pred_instances[i] for i in range(len(pred_instances)) if i not in matched_pred]
    false_negatives = [gt_instances[j] for j in range(len(gt_instances)) if j not in matched_gt]

    return true_positives, false_positives, false_negatives


def create_overlay(
    image: np.ndarray,
    instances_tp: List[np.ndarray],
    instances_fp: List[np.ndarray],
    instances_fn: List[np.ndarray],
    alpha: float = 0.45,
) -> np.ndarray:
    """Composite color overlays onto the original RGB image.

    Colors:
    - Green  (0, 255, 0): True Positives
    - Red    (255, 0, 0): False Positives
    - Blue   (0, 0, 255): False Negatives

    Args:
        image: Original RGB image in [0, 1], shape (H, W, 3).
        instances_tp: TP boolean masks.
        instances_fp: FP boolean masks.
        instances_fn: FN boolean masks.
        alpha: Overlay transparency (0 = invisible, 1 = opaque).

    Returns:
        Composited RGB image as uint8, shape (H, W, 3).
    """
    # Work in float [0, 1] to avoid clipping on blending
    if image.dtype != np.float32 and image.dtype != np.float64:
        canvas = image.astype(np.float32) / 255.0
    else:
        canvas = image.copy()

    color_map = [
        ((0.0, 1.0, 0.0), instances_tp),   # green
        ((1.0, 0.0, 0.0), instances_fp),   # red
        ((0.0, 0.0, 1.0), instances_fn),   # blue
    ]

    for color, instances in color_map:
        for inst in instances:
            for c, val in enumerate(color):
                canvas[:, :, c] = np.where(
                    inst,
                    canvas[:, :, c] * (1 - alpha) + val * alpha,
                    canvas[:, :, c],
                )

    return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)


def plot_evaluation_panels(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    save_path: str,
    iou_threshold: float = 0.5,
) -> None:
    """Generate and save a 4-panel color-coded evaluation figure.

    Panel layout:
        [Input Image]                             [Instance Correctness - Prediction]
        [Instance Correctness - Annotation]       [Pixel Correctness]

    Args:
        image: Original RGB image in [0, 1], shape (H, W, 3).
        pred_mask: Predicted binary mask, shape (H, W).
        gt_mask: Ground truth binary mask, shape (H, W).
        save_path: File path to save the figure.
        iou_threshold: IoU threshold for TP/FP/FN matching.
    """
    pred_instances = extract_instances(pred_mask)
    gt_instances = extract_instances(gt_mask)
    tp, fp, fn = match_instances(pred_instances, gt_instances, iou_threshold)

    # Panel 1: plain input
    panel_input = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    # Panel 2: instance correctness overlaid on predicted mask boundaries
    pred_rgb = np.stack([pred_mask * 255] * 3, axis=-1).astype(np.uint8)
    panel_pred = create_overlay(
        pred_rgb.astype(np.float32) / 255.0, tp, fp, fn
    )

    # Panel 3: instance correctness overlaid on GT mask boundaries
    gt_rgb = np.stack([gt_mask * 255] * 3, axis=-1).astype(np.uint8)
    panel_gt = create_overlay(
        gt_rgb.astype(np.float32) / 255.0, tp, [], fn
    )

    # Panel 4: pixel-level TP/FP/FN on original image
    pred_bin = pred_mask > 0.5
    gt_bin = gt_mask > 0.5
    tp_mask = pred_bin & gt_bin
    fp_mask = pred_bin & ~gt_bin
    fn_mask = ~pred_bin & gt_bin

    tp_instances_pixel = [tp_mask] if tp_mask.any() else []
    fp_instances_pixel = [fp_mask] if fp_mask.any() else []
    fn_instances_pixel = [fn_mask] if fn_mask.any() else []

    panel_pixel = create_overlay(
        image.astype(np.float32) if image.max() <= 1 else image.astype(np.float32) / 255.0,
        tp_instances_pixel,
        fp_instances_pixel,
        fn_instances_pixel,
    )

    # Compose 2x2 figure with legend
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    titles = [
        "Input Image",
        f"Instance Correctness - Prediction ({len(pred_instances)} detected)",
        f"Instance Correctness - Annotation ({len(gt_instances)} nuclei)",
        "Pixel Correctness",
    ]
    panels = [panel_input, panel_pred, panel_gt, panel_pixel]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (row, col), title, panel in zip(positions, titles, panels):
        axes[row, col].imshow(panel)
        axes[row, col].set_title(title, fontsize=11, fontweight="bold")
        axes[row, col].axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=(0, 1, 0), label="True Positive (IoU >= 0.5)"),
        mpatches.Patch(color=(1, 0, 0), label="False Positive (IoU < 0.5)"),
        mpatches.Patch(color=(0, 0, 1), label="False Negative (IoU < 0.5)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_instance_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute instance-level precision, recall, and F1.

    Args:
        pred_mask: Predicted binary mask, shape (H, W).
        gt_mask: Ground truth binary mask, shape (H, W).
        iou_threshold: IoU threshold for TP matching.

    Returns:
        Dictionary with keys "precision", "recall", "f1".

    Example:
        >>> import numpy as np
        >>> pred = np.zeros((10, 10)); pred[0:3, 0:3] = 1
        >>> gt = np.zeros((10, 10)); gt[0:3, 0:3] = 1
        >>> m = compute_instance_metrics(pred, gt)
        >>> m["f1"]
        1.0
    """
    pred_instances = extract_instances(pred_mask)
    gt_instances = extract_instances(gt_mask)
    tp, fp, fn = match_instances(pred_instances, gt_instances, iou_threshold)

    n_tp = len(tp)
    n_fp = len(fp)
    n_fn = len(fn)

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}
