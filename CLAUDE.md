# Medical Image Segmentation with Attention U-Net

## Project Summary

Portfolio-quality medical image segmentation using PyTorch. Trains standard U-Net and Attention U-Net on the 2018 Data Science Bowl cell nuclei dataset, compares both architectures, and serves predictions through a Gradio web demo. Dockerized for reproducibility.

## Tech Stack

- Python 3.10+, PyTorch, torchvision
- scipy (connected component labeling for instance segmentation evaluation)
- Gradio (interactive demo UI)
- matplotlib, NumPy, Pillow (visualization and image processing)
- PyYAML (config), tqdm (progress bars)
- Docker and Docker Compose
- pytest (unit and integration tests)

## Project Structure

```
medical-segmentation/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yaml
├── .dockerignore
├── configs/
│   └── config.yaml              # All hyperparameters in one place
├── src/
│   ├── __init__.py
│   ├── model.py                 # Standard U-Net architecture
│   ├── attention_unet.py        # Attention U-Net (attention gates on skip connections)
│   ├── dataset.py               # NucleiDataset class + transforms + mask combining
│   ├── train.py                 # Training loop with validation and checkpointing
│   ├── evaluate.py              # Metrics (Dice, IoU) and prediction visualization
│   ├── visualize.py             # Advanced overlays: color-coded TP/FP/FN on original image
│   └── utils.py                 # Config loading, device setup, plotting helpers
├── tests/
│   ├── __init__.py
│   ├── test_model.py            # Model shape tests and forward pass verification
│   ├── test_attention_unet.py   # Attention gate and architecture tests
│   ├── test_dataset.py          # Dataset loading and mask combining tests
│   ├── test_metrics.py          # Dice and IoU correctness tests
│   ├── test_visualize.py        # Overlay generation and instance matching tests
│   └── test_training.py         # Integration test: 1-epoch training on tiny data
├── train_model.py               # Main entry: trains both models
├── compare_models.py            # Side-by-side architecture comparison visualizations
├── app.py                       # Gradio web demo
└── outputs/                     # Created at runtime
```

## Build Phases

Work through these phases in order. After each phase, run the relevant tests to verify before moving on.

### Phase 1: Core U-Net and Dataset

Build the foundational model and data pipeline.

**Files to create:** `src/model.py`, `src/dataset.py`, `src/utils.py`, `configs/config.yaml`, `requirements.txt`

**src/model.py -- Standard U-Net:**
- DoubleConv block: Conv2d(3x3, padding=1) -> BatchNorm2d -> ReLU, repeated twice
- Encoder: 4 levels with features [64, 128, 256, 512], MaxPool2d(2) between each
- Bottleneck: DoubleConv with 1024 features
- Decoder: ConvTranspose2d(2x2) to upsample, concatenate skip connection, DoubleConv
- Final: Conv2d(1x1) mapping to output channels
- Input (B, 3, 128, 128) -> Output (B, 1, 128, 128)

**src/dataset.py -- NucleiDataset:**
- Expects Data Science Bowl 2018 structure: each sample is a folder with `images/` (one PNG) and `masks/` (multiple PNGs, one per nucleus)
- Combine individual masks into single binary mask using np.maximum across all mask files
- Resize to configurable image_size (default 128)
- Normalize images to [0, 1], binarize masks at threshold 0.5
- Use NEAREST interpolation for masks, BILINEAR for images
- Data augmentation: RandomHorizontalFlip + RandomVerticalFlip applied identically to image and mask (stack as 4-channel tensor, transform, then split)

**configs/config.yaml:**
```yaml
data_dir: "data/stage1_train"
image_size: 128
train_split: 0.8
in_channels: 3
out_channels: 1
features: [64, 128, 256, 512]
batch_size: 16
learning_rate: 0.001
num_epochs: 25
device: "auto"
output_dir: "outputs"
```

**Verify Phase 1:**
```bash
pytest tests/test_model.py tests/test_dataset.py -v
```

### Phase 2: Training Pipeline and Metrics

Build the training loop and evaluation metrics.

**Files to create:** `src/train.py`, `src/evaluate.py`, `train_model.py`

**src/train.py:**
- Loss function: BCEWithLogitsLoss (numerically stable, model outputs raw logits)
- Optimizer: Adam
- train_one_epoch(): forward pass, loss, backward, optimizer step
- validate(): no_grad context, compute loss + Dice + IoU per batch, return averages
- train_model(): full loop with epoch logging, saves best model by validation Dice score
- Print format per epoch: `Epoch [X/N]  Train Loss: X.XXXX  Val Loss: X.XXXX  Val Dice: X.XXXX  Val IoU: X.XXXX`

**src/evaluate.py:**
- dice_coefficient(): 2*intersection / (sum_pred + sum_target), with smooth=1e-6
- iou_score(): intersection / union, with smooth=1e-6
- plot_predictions(): grid of [Input Image | Ground Truth | Prediction] for N samples
- load_model(): load checkpoint into fresh UNet, return in eval mode

**train_model.py (main entry point):**
1. Load config from configs/config.yaml
2. Create NucleiDataset, split into train/val with random_split (seed=42)
3. Build UNet, train, save best model
4. Generate training_curve.png and predictions.png in outputs/

**Verify Phase 2:**
```bash
pytest tests/test_metrics.py tests/test_training.py -v
```

### Phase 3: Advanced Visualization and Instance Evaluation

Build color-coded overlay visualizations that show model accuracy directly on the original image, plus instance-level (per-nucleus) evaluation.

**Files to create:** `src/visualize.py`

**src/visualize.py -- Color-Coded Overlay Visualization:**

This module produces a 4-panel figure for each sample, similar to research-grade segmentation evaluation:

Panel layout:
```
[Input Image]                          [Instance Correctness, Prediction]
[Instance Correctness, Annotation]     [Pixel Correctness]
```

With a legend: Green = True Positive (overlap >= 50%), Red = False Positive (overlap < 50%), Blue = False Negative (overlap < 50%)

**Implementation details:**

1. `extract_instances(binary_mask) -> list[np.ndarray]`:
   - Use scipy.ndimage.label() to find connected components in a binary mask
   - Each connected component is one nucleus instance
   - Return list of boolean masks, one per instance

2. `match_instances(pred_instances, gt_instances, iou_threshold=0.5) -> tuple[list, list, list]`:
   - For each predicted instance, compute IoU with every ground truth instance
   - Match predicted to ground truth using greedy matching (highest IoU first)
   - A match counts as True Positive if IoU >= threshold
   - Unmatched predictions = False Positives
   - Unmatched ground truths = False Negatives
   - Return (true_positives, false_positives, false_negatives) as lists of instance masks

3. `create_overlay(image, instances_tp, instances_fp, instances_fn, alpha=0.45) -> np.ndarray`:
   - Start with the original RGB image
   - Overlay green (0, 255, 0) on True Positive regions with transparency alpha
   - Overlay red (255, 0, 0) on False Positive regions
   - Overlay blue (0, 0, 255) on False Negative regions
   - Return the composited image

4. `plot_evaluation_panels(image, pred_mask, gt_mask, save_path, iou_threshold=0.5)`:
   - Generate the 4-panel figure:
     - Top-left: Original input image (no overlay)
     - Top-right: Instance correctness on prediction (TP green, FP red on predicted nuclei)
     - Bottom-left: Instance correctness on annotation (TP green, FN blue on ground truth nuclei)
     - Bottom-right: Pixel correctness (TP green, FP red, FN blue combined view)
   - Add legend below panels showing color meanings
   - Save to save_path

5. `compute_instance_metrics(pred_mask, gt_mask, iou_threshold=0.5) -> dict`:
   - Returns: precision, recall, F1 at the instance level
   - precision = TP / (TP + FP)
   - recall = TP / (TP + FN)
   - F1 = 2 * precision * recall / (precision + recall)

**Important: the dataset stores individual nucleus masks separately in the masks/ folder. For instance-level evaluation during validation, load the individual masks (do not use the combined binary mask). Add a method to NucleiDataset or a utility function that returns the list of individual mask arrays for a given sample.**

**Update evaluate.py:**
- After generating basic predictions.png, also generate evaluation_panels.png using plot_evaluation_panels for the best/worst performing validation samples
- Print instance-level metrics (precision, recall, F1) alongside pixel-level metrics (Dice, IoU)

**Update app.py (Gradio demo, in Phase 5):**
- Add the color-coded overlay as an additional output tab/panel
- When a user uploads an image and optionally a ground truth mask, show all 4 panels
- When no ground truth is provided, show just the prediction overlay (green regions on original)

**Verify Phase 3:**
```bash
pytest tests/test_visualize.py -v
```

### Phase 4: Attention U-Net

Add the Attention U-Net variant for architecture comparison.

**Files to create:** `src/attention_unet.py`, `compare_models.py`

**src/attention_unet.py -- Attention U-Net:**
- Same encoder/bottleneck/decoder structure as standard U-Net
- Add AttentionGate module at each skip connection in the decoder
- AttentionGate mechanism:
  - Takes gating signal (from decoder) and skip connection (from encoder)
  - W_g: Conv2d(1x1) on gating signal
  - W_x: Conv2d(1x1) on skip connection
  - psi: Conv2d(1x1) after ReLU(W_g + W_x), followed by Sigmoid
  - Output: skip_connection * psi (element-wise attention weighting)
- This lets the model learn WHICH parts of the skip connection are important

**compare_models.py:**
- Train both standard U-Net and Attention U-Net using the same train/val split
- Generate comparison visualization: side-by-side predictions from both models on same images
- Generate comparison metrics table printed to console
- Save outputs/comparison.png

**Update train_model.py:**
- Add `--model` flag: "unet" (default) or "attention_unet"
- Or train both sequentially if no flag given

**Verify Phase 4:**
```bash
pytest tests/test_attention_unet.py -v
```

### Phase 5: Gradio Demo

Interactive web interface for predictions.

**Files to create:** `app.py`

**app.py:**
- Load best trained model (default to Attention U-Net if available, else standard)
- Gradio interface with:
  - Image upload input
  - Optional ground truth mask upload (for evaluation panels)
  - Model selector dropdown (U-Net / Attention U-Net) if both checkpoints exist
  - Confidence threshold slider (default 0.5)
  - Output tabs:
    - Tab 1 "Segmentation": predicted mask overlaid on original image (green regions)
    - Tab 2 "Evaluation Panels" (only when ground truth provided): the 4-panel color-coded figure (input, instance correctness prediction, instance correctness annotation, pixel correctness) with TP=green, FP=red, FN=blue
    - Tab 3 "Metrics": Dice, IoU, instance precision, recall, F1 displayed as text
    - Tab 4 "Raw Mask": binary black/white mask
- Preprocessing: resize uploaded image to model's expected size, normalize to [0,1]
- Postprocessing: sigmoid on logits, threshold, resize mask back to original image size
- Launch on port 7860

**Verify Phase 5:**
```bash
python app.py
# Open http://localhost:7860 in browser
# Upload any microscopy image and verify mask output appears
# Upload with ground truth mask to see 4-panel color-coded evaluation
```

### Phase 6: Docker

Containerize the full project.

**Files to create:** `Dockerfile`, `docker-compose.yaml`, `.dockerignore`

**Dockerfile:**
- Base image: python:3.10-slim
- Install system deps if needed (libgl1-mesa-glx for OpenCV if used)
- Copy requirements.txt, pip install
- Copy project source
- Expose port 7860 for Gradio
- Default CMD: python app.py (serve the demo)
- Also support training: docker run ... python train_model.py

**docker-compose.yaml:**
- Service `demo`: builds from Dockerfile, maps port 7860, mounts data/ and outputs/ as volumes
- Service `train`: same image, overrides command to python train_model.py, mounts data/ and outputs/
- GPU support: include deploy.resources.reservations for nvidia GPU if available

**.dockerignore:**
- outputs/, data/, __pycache__/, .git/, *.pyc, .venv/

**Verify Phase 6:**
```bash
docker compose build
docker compose run train    # Should complete training
docker compose up demo      # Should serve Gradio at localhost:7860
```

### Phase 7: Tests and README

Finalize test suite and documentation.

**Tests to create:**

**tests/test_model.py:**
- test_unet_output_shape: input (2, 3, 128, 128) -> output (2, 1, 128, 128)
- test_unet_different_sizes: verify works with 64x64 and 256x256
- test_unet_single_image: batch size 1 works
- test_unet_parameter_count: sanity check param count is in expected range (~31M)

**tests/test_attention_unet.py:**
- test_attention_unet_output_shape: same shape tests as standard
- test_attention_gate_output_shape: verify gate produces correct tensor shape
- test_attention_vs_standard_different_weights: ensure they are distinct models

**tests/test_dataset.py:**
- test_dataset_item_shapes: image is (3, H, W), mask is (1, H, W)
- test_mask_is_binary: mask values are only 0.0 or 1.0
- test_image_normalized: image values in [0, 1]
- test_mask_combining: given folder with 3 individual masks, combined mask has all nuclei
- Use a small fixture directory with 2-3 synthetic samples for tests (create in conftest.py)

**tests/test_metrics.py:**
- test_dice_perfect_overlap: identical tensors -> Dice ~1.0
- test_dice_no_overlap: disjoint tensors -> Dice ~0.0
- test_iou_perfect_overlap: identical tensors -> IoU ~1.0
- test_iou_no_overlap: disjoint tensors -> IoU ~0.0
- test_dice_known_value: hand-calculated example

**tests/test_visualize.py:**
- test_extract_instances_count: mask with 3 separate blobs -> 3 instances
- test_extract_instances_single: one connected region -> 1 instance
- test_match_instances_perfect: identical masks -> all TP, no FP/FN
- test_match_instances_no_overlap: completely different masks -> all FP and FN
- test_overlay_shape: output image has same HxWx3 shape as input
- test_overlay_modifies_image: overlay output differs from plain input
- test_instance_metrics_perfect: precision=1, recall=1, F1=1
- test_instance_metrics_partial: known setup with 2 TP, 1 FP, 1 FN -> verify precision=0.667, recall=0.667

**tests/test_training.py (integration):**
- test_one_epoch_reduces_loss: create tiny synthetic dataset (10 samples of random noise), run 2 epochs, verify epoch 2 loss < epoch 1 loss
- test_model_saves_checkpoint: train 1 epoch, verify .pth file exists
- Use small image size (32x32) and batch_size=2 for speed

**README.md should include:**
- Project title and one-line description
- Architecture diagram (ASCII art showing U-Net structure and attention gates)
- Quick start instructions (pip install, download data, train, demo)
- Docker instructions (build, train, serve)
- Expected results section with descriptions of what output images show
- Comparison section: standard U-Net vs Attention U-Net with metric expectations
- Technologies used
- Project structure tree
- "What I Learned" section

**Run full test suite:**
```bash
pytest tests/ -v --tb=short
```

## Code Style

- Type hints on all function signatures
- Docstrings on all public classes and functions (include Args, Returns, Example where useful)
- Comments explaining WHY, not WHAT (the code should be readable on its own)
- No unused imports
- Use pathlib.Path instead of os.path where possible
- f-strings for formatting

## Important Notes

- Never use em dashes in any text output, comments, or documentation
- The dataset uses a unique structure where each sample folder contains multiple individual mask PNGs that must be combined into one binary mask
- Use NEAREST interpolation when resizing masks to avoid introducing non-binary values
- BCEWithLogitsLoss expects raw logits (no sigmoid in the model's forward pass). Apply sigmoid only during inference and metric computation
- When applying augmentations, stack image and mask as a single tensor so the same random flip is applied to both
- Pin random seed (42) for dataset splits to ensure reproducibility
- All generated files (models, plots) go in outputs/ directory
- The Gradio app should work with any uploaded microscopy image, not just images from the dataset

## Expected Results

After 25 epochs of training on the full dataset:

**Pixel-level metrics:**
- Standard U-Net: Dice ~0.82-0.88, IoU ~0.72-0.80
- Attention U-Net: Dice ~0.85-0.91, IoU ~0.75-0.83

**Instance-level metrics (at IoU threshold 0.5):**
- Standard U-Net: Precision ~0.75-0.85, Recall ~0.70-0.80, F1 ~0.73-0.82
- Attention U-Net: Precision ~0.78-0.88, Recall ~0.73-0.83, F1 ~0.76-0.85

**Visual outputs:**
- training_curve.png: smooth loss decrease, Dice increase, no major spikes
- predictions.png: basic side-by-side masks closely matching ground truth
- evaluation_panels.png: 4-panel color-coded overlays on original images. Most nuclei should be green (TP). A few red (FP) or blue (FN) on difficult overlapping or edge nuclei is normal
- comparison.png: Attention U-Net should show cleaner boundaries, especially on clustered nuclei
- Gradio demo: upload image, see green overlay on detected nuclei instantly. Upload with ground truth to see full 4-panel evaluation

## Testing Checklist

Run these commands to verify the project works end to end:

```bash
# 1. Unit tests pass
pytest tests/ -v

# 2. Training completes (use fewer epochs for quick check)
python train_model.py  # Or: edit config.yaml to num_epochs: 3 for a smoke test

# 3. Output files generated
ls outputs/
# Expected: best_model.pth, training_curve.png, predictions.png, evaluation_panels.png

# 4. Comparison script works (after training both models)
python compare_models.py
# Expected: outputs/comparison.png

# 5. Gradio demo launches
python app.py
# Expected: opens at http://localhost:7860
# Test 1: upload microscopy image only -> green overlay on detected nuclei
# Test 2: upload image + ground truth mask -> full 4-panel evaluation with TP/FP/FN colors

# 6. Docker works
docker compose build
docker compose up demo
# Expected: Gradio accessible at localhost:7860
```
