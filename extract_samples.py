"""Extract 4 diverse sample image-mask pairs for the Gradio demo examples."""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as scipy_label

DATA_DIR = Path("data/stage1_train")
OUT_DIR = Path("samples")
MAX_SIZE = 256


def _get_brightness(sample_dir: Path) -> float:
    img_path = next((sample_dir / "images").glob("*.png"))
    return float(np.array(Image.open(img_path).convert("L")).mean())


def _load_and_resize(sample_dir: Path):
    """Return (image PIL, mask PIL) both resized to MAX_SIZE."""
    img_path = next((sample_dir / "images").glob("*.png"))
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)
    target_size = img.size  # (W, H)

    combined = None
    for mask_path in sorted((sample_dir / "masks").glob("*.png")):
        arr = np.array(Image.open(mask_path).convert("L"))
        combined = arr if combined is None else np.maximum(combined, arr)

    if combined is None:
        combined = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)

    mask_pil = Image.fromarray(combined).resize(target_size, Image.NEAREST)
    return img, mask_pil


def _count_nuclei(mask_pil: Image.Image) -> int:
    _, n = scipy_label(np.array(mask_pil) > 127)
    return int(n)


def main() -> None:
    sample_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"Scanning {len(sample_dirs)} samples...")

    infos = []
    for d in sample_dirs:
        masks = list((d / "masks").glob("*.png"))
        imgs = list((d / "images").glob("*.png"))
        if not masks or not imgs:
            continue
        infos.append({
            "dir": d,
            "n_masks": len(masks),         # fast proxy for nucleus count
            "brightness": _get_brightness(d),
        })

    infos.sort(key=lambda x: x["n_masks"])
    n = len(infos)

    # Pick 4 samples spanning the distribution
    few = infos[max(0, n // 20)]            # ~5th percentile: few nuclei
    many = infos[min(n - 1, int(n * 0.92))]  # ~92nd percentile: many nuclei

    # From the middle half, pick the darkest and brightest backgrounds
    middle = infos[n // 4 : 3 * n // 4]
    middle_by_brightness = sorted(middle, key=lambda x: x["brightness"])
    dark_bg = middle_by_brightness[0]
    light_bg = middle_by_brightness[-1]

    selected = [few, dark_bg, light_bg, many]
    labels = ["few nuclei", "dark background", "light background", "many nuclei"]

    OUT_DIR.mkdir(exist_ok=True)

    print("\nSelected samples:")
    for i, (info, label) in enumerate(zip(selected, labels), start=1):
        img, mask = _load_and_resize(info["dir"])
        count = _count_nuclei(mask)

        img_path = OUT_DIR / f"sample_{i}_image.png"
        mask_path = OUT_DIR / f"sample_{i}_mask.png"
        img.save(img_path)
        mask.save(mask_path)

        print(
            f"  Sample {i} ({label}): {info['dir'].name}\n"
            f"    Nuclei: {count}  |  "
            f"Brightness: {info['brightness']:.1f}  |  "
            f"Size: {img.size}"
        )

    print(f"\nSaved 4 pairs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
