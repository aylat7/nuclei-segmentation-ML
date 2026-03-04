"""Gradio web demo for nuclei segmentation with U-Net and Attention U-Net."""

import tempfile
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.attention_unet import AttentionUNet
from src.evaluate import dice_coefficient, iou_score
from src.model import UNet
from src.utils import load_config, get_device
from src.visualize import (
    compute_instance_metrics,
    create_overlay,
    extract_instances,
    match_instances,
    plot_evaluation_panels,
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

CFG = load_config("configs/config.yaml")
DEVICE = get_device(CFG["device"])
OUTPUT_DIR = Path(CFG["output_dir"])
IMAGE_SIZE = CFG["image_size"]

_model_cache: dict = {}


def _load_model(model_name: str) -> Optional[torch.nn.Module]:
    """Load a model from checkpoint, caching it in memory."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    ckpt_map = {
        "U-Net": OUTPUT_DIR / "best_unet.pth",
        "Attention U-Net": OUTPUT_DIR / "best_attention_unet.pth",
    }
    ckpt = ckpt_map.get(model_name)
    if ckpt is None or not ckpt.exists():
        return None

    if model_name == "Attention U-Net":
        model = AttentionUNet(
            in_channels=CFG["in_channels"],
            out_channels=CFG["out_channels"],
            features=CFG["features"],
        )
    else:
        model = UNet(
            in_channels=CFG["in_channels"],
            out_channels=CFG["out_channels"],
            features=CFG["features"],
        )

    state = torch.load(str(ckpt), map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    _model_cache[model_name] = model
    return model


def _available_models() -> list[str]:
    """Return list of model names with existing checkpoints."""
    options = []
    if (OUTPUT_DIR / "best_attention_unet.pth").exists():
        options.append("Attention U-Net")
    if (OUTPUT_DIR / "best_unet.pth").exists():
        options.append("U-Net")
    return options or ["U-Net"]  # Fallback label even if no checkpoint


# ---------------------------------------------------------------------------
# Preprocessing / Postprocessing
# ---------------------------------------------------------------------------

def _preprocess_image(pil_image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Resize, convert to tensor, and return original size for postprocessing."""
    original_size = pil_image.size  # (W, H)
    pil_resized = pil_image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
    )
    arr = np.array(pil_resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor, original_size


def _predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    threshold: float,
    original_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model inference and return (pred_mask_resized, image_np_normalized).

    Returns:
        pred_mask: Binary mask at original resolution, shape (H_orig, W_orig).
        img_np: Normalized image array [0, 1], shape (IMAGE_SIZE, IMAGE_SIZE, 3).
    """
    with torch.no_grad():
        logit = model(image_tensor.to(DEVICE))
        prob = torch.sigmoid(logit)  # (1, 1, H, W)
        # Resize back to original resolution
        prob_orig = F.interpolate(
            prob, size=(original_size[1], original_size[0]), mode="bilinear", align_corners=False
        )
        pred_mask = (prob_orig.squeeze().cpu().numpy() >= threshold).astype(np.float32)

    img_np = image_tensor.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 3) in [0,1]
    return pred_mask, img_np


# ---------------------------------------------------------------------------
# Gradio inference function
# ---------------------------------------------------------------------------

def run_inference(
    input_image: np.ndarray,
    gt_mask_image: Optional[np.ndarray],
    model_name: str,
    threshold: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], str, np.ndarray]:
    """Run segmentation and return all output tab contents.

    Args:
        input_image: Uploaded RGB image as numpy array (H, W, 3).
        gt_mask_image: Optional ground truth mask (H, W) or (H, W, 3).
        model_name: Name of the model to use.
        threshold: Confidence threshold for binary prediction.

    Returns:
        Tuple of:
        - segmentation_overlay: RGB image with green nuclei overlay.
        - evaluation_panels: 4-panel figure (None if no GT provided).
        - metrics_text: Formatted metrics string.
        - raw_mask: Binary mask as uint8 image.
    """
    model = _load_model(model_name)
    if model is None:
        msg = f"No trained checkpoint found for {model_name}. Run train_model.py first."
        blank = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        return blank, None, msg, blank

    pil_image = Image.fromarray(input_image.astype(np.uint8))
    image_tensor, original_size = _preprocess_image(pil_image)
    pred_mask, img_np_small = _predict(model, image_tensor, threshold, original_size)

    # Resize image to original size for overlays
    img_orig = np.array(pil_image.convert("RGB"), dtype=np.float32) / 255.0

    # --- Tab 1: Segmentation overlay (green nuclei) ---
    pred_instances = extract_instances(pred_mask)
    seg_overlay = create_overlay(img_orig, pred_instances, [], [])

    # --- Tab 4: Raw mask ---
    raw_mask_img = (pred_mask * 255).astype(np.uint8)
    raw_mask_rgb = np.stack([raw_mask_img] * 3, axis=-1)

    # --- Metrics ---
    # Pixel-level metrics require GT
    metrics_lines = []

    if gt_mask_image is not None:
        # Convert GT to binary float mask at original resolution
        if gt_mask_image.ndim == 3:
            gt_gray = gt_mask_image.mean(axis=-1)
        else:
            gt_gray = gt_mask_image.astype(np.float32)
        gt_binary = (gt_gray > 127).astype(np.float32)

        # Resize GT to match pred if different
        if gt_binary.shape != pred_mask.shape:
            gt_pil = Image.fromarray((gt_binary * 255).astype(np.uint8)).resize(
                (pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST
            )
            gt_binary = np.array(gt_pil, dtype=np.float32) / 255.0

        pred_t = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0)
        gt_t = torch.from_numpy(gt_binary).unsqueeze(0).unsqueeze(0)

        dice = dice_coefficient(pred_t, gt_t).item()
        iou = iou_score(pred_t, gt_t).item()
        inst_metrics = compute_instance_metrics(pred_mask, gt_binary)

        metrics_lines = [
            f"Pixel-level Metrics:",
            f"  Dice:       {dice:.4f}",
            f"  IoU:        {iou:.4f}",
            f"",
            f"Instance-level Metrics (IoU threshold = 0.5):",
            f"  Precision:  {inst_metrics['precision']:.4f}",
            f"  Recall:     {inst_metrics['recall']:.4f}",
            f"  F1:         {inst_metrics['f1']:.4f}",
        ]

        # --- Tab 2: Evaluation panels ---
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        plot_evaluation_panels(img_orig, pred_mask, gt_binary, tmp_path)
        eval_panel_img = np.array(Image.open(tmp_path))
        Path(tmp_path).unlink(missing_ok=True)
    else:
        metrics_lines = [
            "Upload a ground truth mask to see pixel-level and instance-level metrics.",
            "",
            f"Detected nuclei: {len(pred_instances)}",
        ]
        eval_panel_img = None

    metrics_text = "\n".join(metrics_lines)
    return seg_overlay, eval_panel_img, metrics_text, raw_mask_rgb


# ---------------------------------------------------------------------------
# Build Gradio interface
# ---------------------------------------------------------------------------

def build_interface() -> gr.Blocks:
    available = _available_models()

    with gr.Blocks(title="Nuclei Segmentation Demo") as demo:
        gr.Markdown("# Cell Nuclei Segmentation")
        gr.Markdown(
            "Upload a microscopy image to segment cell nuclei. "
            "Optionally upload a ground truth mask for full evaluation."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input Microscopy Image", type="numpy")
                gt_mask = gr.Image(label="Ground Truth Mask (optional)", type="numpy")
                model_selector = gr.Dropdown(
                    choices=available,
                    value=available[0],
                    label="Model",
                )
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold",
                )
                run_btn = gr.Button("Run Segmentation", variant="primary")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Segmentation"):
                        seg_output = gr.Image(label="Prediction Overlay (green = nuclei)")
                    with gr.Tab("Evaluation Panels"):
                        eval_output = gr.Image(
                            label="4-Panel Evaluation (requires ground truth)"
                        )
                    with gr.Tab("Metrics"):
                        metrics_output = gr.Textbox(
                            label="Metrics", lines=10, interactive=False
                        )
                    with gr.Tab("Raw Mask"):
                        mask_output = gr.Image(label="Binary Prediction Mask")

        run_btn.click(
            fn=run_inference,
            inputs=[input_image, gt_mask, model_selector, threshold_slider],
            outputs=[seg_output, eval_output, metrics_output, mask_output],
        )

        gr.Markdown(
            "_Upload any microscopy image. For full evaluation panels, also upload "
            "the corresponding binary ground truth mask._"
        )

    return demo


demo = build_interface()

if __name__ == "__main__":
    demo.launch()
