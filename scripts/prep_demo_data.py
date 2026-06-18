"""
Prepare a tiny dataset for smoke-testing the edge-conditioning pipeline.

Two sources:
  * synthetic (default): crisp geometric shapes -> very clean Canny edges, so you
    can visually confirm that generation follows the edge structure;
  * --src_dir DIR: center-crop/resize real images from DIR instead.

Filenames are written as ``{class_id}_{idx}.png`` so the class-conditional data
loader (which derives the label from the filename prefix) works out of the box.

Example:
  python scripts/prep_demo_data.py --out_dir data/demo64 --n 300 --size 64
  python scripts/prep_demo_data.py --out_dir data/demo64 --src_dir /my/images --size 64
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageDraw


def random_shape_image(size, rng):
    bg = tuple(int(c) for c in rng.integers(0, 256, 3))
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(2, 5))):
        kind = int(rng.integers(0, 3))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        x0, x1 = sorted(int(v) for v in rng.integers(0, size, 2))
        y0, y1 = sorted(int(v) for v in rng.integers(0, size, 2))
        if kind == 0:
            draw.ellipse([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))
        elif kind == 1:
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))
        else:
            draw.line([x0, y0, x1, y1], fill=color, width=int(rng.integers(1, 4)))
    return img


def center_crop_resize(pil_image, size):
    pil_image = pil_image.convert("RGB")
    while min(*pil_image.size) >= 2 * size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    cy = (arr.shape[0] - size) // 2
    cx = (arr.shape[1] - size) // 2
    return Image.fromarray(arr[cy : cy + size, cx : cx + size])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=300, help="#images (synthetic mode)")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--class_id", type=int, default=0)
    ap.add_argument("--src_dir", default="", help="use real images from here instead")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.src_dir:
        files = []
        for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.JPEG"):
            files += glob.glob(os.path.join(args.src_dir, "**", e), recursive=True)
        files = sorted(set(files))
        for i, f in enumerate(files):
            try:
                img = center_crop_resize(Image.open(f), args.size)
            except Exception as ex:  # noqa
                print(f"skip {f}: {ex}")
                continue
            img.save(os.path.join(args.out_dir, f"{args.class_id}_{i:05d}.png"))
        print(f"wrote {len(files)} resized images to {args.out_dir}")
    else:
        rng = np.random.default_rng(args.seed)
        for i in range(args.n):
            img = random_shape_image(args.size, rng)
            img.save(os.path.join(args.out_dir, f"{args.class_id}_{i:05d}.png"))
        print(f"wrote {args.n} synthetic images to {args.out_dir}")


if __name__ == "__main__":
    main()
