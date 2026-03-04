# Medical Image Segmentation with Attention U-Net

Portfolio-quality cell nuclei segmentation using standard U-Net and Attention U-Net, trained on the 2018 Data Science Bowl dataset.

---

## Architecture

### Standard U-Net

```
Input (3, 128, 128)
        |
  [DoubleConv] -> skip_1 (64 ch)
        |
    [MaxPool]
        |
  [DoubleConv] -> skip_2 (128 ch)
        |
    [MaxPool]
        |
  [DoubleConv] -> skip_3 (256 ch)
        |
    [MaxPool]
        |
  [DoubleConv] -> skip_4 (512 ch)
        |
    [MaxPool]
        |
  [Bottleneck]  (1024 ch)
        |
  [ConvTranspose2d] <- skip_4
  [DoubleConv]
        |
  [ConvTranspose2d] <- skip_3
  [DoubleConv]
        |
  [ConvTranspose2d] <- skip_2
  [DoubleConv]
        |
  [ConvTranspose2d] <- skip_1
  [DoubleConv]
        |
  [Conv 1x1] -> Output (1, 128, 128) logits
```

### Attention Gates (Attention U-Net extension)

At each decoder skip connection, an AttentionGate is inserted:

```
Gating signal (g) ----> [W_g Conv 1x1] ---+
                                           |--> ReLU --> [psi Conv 1x1] --> Sigmoid --> alpha
Skip connection (x) --> [W_x Conv 1x1] ---+

Output = x * alpha   (element-wise spatial weighting)
```

Attention gates suppress irrelevant background regions, focusing the decoder on foreground nuclei.

---

## Quick Start

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Download data
# Place the 2018 Data Science Bowl stage1_train folder at:
#   data/stage1_train/<sample_id>/images/<id>.png
#                                /masks/<mask_id>.png ...

# 3. Train (both models, 25 epochs)
python train_model.py

# 4. Compare architectures
python compare_models.py

# 5. Launch Gradio demo
python app.py
# Open http://localhost:7860
```

Train a single model:

```bash
python train_model.py --model unet
python train_model.py --model attention_unet
```

---

## Docker

```bash
# Build image
docker compose build

# Train both models (results in outputs/)
docker compose run train

# Serve Gradio demo at localhost:7860
docker compose up demo
```

GPU training (NVIDIA): uncomment the `deploy.resources.reservations` section in `docker-compose.yaml`.

---

## Expected Results

After 25 epochs on the full Data Science Bowl dataset:

### Pixel-level Metrics

| Model           | Dice        | IoU         |
|-----------------|-------------|-------------|
| U-Net           | 0.82 - 0.88 | 0.72 - 0.80 |
| Attention U-Net | 0.85 - 0.91 | 0.75 - 0.83 |

### Instance-level Metrics (IoU threshold 0.5)

| Model           | Precision   | Recall      | F1          |
|-----------------|-------------|-------------|-------------|
| U-Net           | 0.75 - 0.85 | 0.70 - 0.80 | 0.73 - 0.82 |
| Attention U-Net | 0.78 - 0.88 | 0.73 - 0.83 | 0.76 - 0.85 |

### Output Files

- `outputs/training_curve_*.png`: smooth loss decrease and Dice/IoU increase over epochs
- `outputs/predictions_*.png`: side-by-side [Input | GT | Prediction] grid
- `outputs/evaluation_panels_*.png`: 4-panel color-coded overlays (green TP / red FP / blue FN)
- `outputs/comparison.png`: U-Net vs Attention U-Net predictions on the same images

---

## Model Comparison

Attention U-Net typically shows:
- Cleaner nucleus boundaries, especially for clustered or touching nuclei
- Fewer false positives in dense regions
- Slightly better instance-level recall

The attention maps learn to suppress non-nucleus background, leading to more precise skip connection features.

---

## Gradio Demo

Upload any microscopy image to see:

- **Segmentation tab**: predicted nuclei highlighted green on the original image
- **Evaluation Panels tab**: 4-panel color-coded analysis (upload ground truth mask to enable)
- **Metrics tab**: Dice, IoU, precision, recall, F1
- **Raw Mask tab**: binary black/white prediction

Use the model selector dropdown to switch between U-Net and Attention U-Net checkpoints.

---

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
│   └── config.yaml
├── src/
│   ├── model.py            # Standard U-Net
│   ├── attention_unet.py   # Attention U-Net + AttentionGate
│   ├── dataset.py          # NucleiDataset
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Metrics and visualization helpers
│   ├── visualize.py        # Color-coded TP/FP/FN overlays
│   └── utils.py            # Config, device, plotting
├── tests/
│   ├── conftest.py
│   ├── test_model.py
│   ├── test_attention_unet.py
│   ├── test_dataset.py
│   ├── test_metrics.py
│   ├── test_visualize.py
│   └── test_training.py
├── train_model.py
├── compare_models.py
└── app.py
```

---

## Technologies

- **PyTorch** - model definition, training, inference
- **torchvision** - image transforms
- **scipy** - connected component labeling for instance segmentation
- **Gradio** - interactive web demo
- **matplotlib / NumPy / Pillow** - visualization and image processing
- **PyYAML** - configuration management
- **Docker / Docker Compose** - reproducible containerized deployment
- **pytest** - unit and integration testing

---

## What I Learned

- **Attention mechanisms**: How soft spatial attention gates focus skip connections on salient regions, improving boundary precision without significantly increasing inference cost
- **Instance vs pixel metrics**: Pixel-level Dice/IoU can look good even when individual nuclei are merged; instance-level precision/recall at IoU 0.5 is a more faithful measure for counting applications
- **BCEWithLogitsLoss**: Combining sigmoid and BCE in one numerically stable operation is important when logits can be very large or very small
- **Mask interpolation**: Using NEAREST interpolation for binary masks and BILINEAR for images prevents label leakage at resize boundaries
- **Data augmentation coupling**: Stacking image and mask into a single tensor before applying random flips ensures both receive the exact same geometric transform
- **Gradio**: Building interactive ML demos requires careful preprocessing/postprocessing to match training-time conventions (resize, normalize, threshold)
