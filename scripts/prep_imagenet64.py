"""
Download (a subset of) ImageNet-64 and write it as image files for the
class-conditional data loader.

Source: https://hf.co/datasets/ChocolateDave/imagenet-64  (the standard Chrabaszcz
downsampled ImageNet; labels are the canonical ImageNet class ids 0-999, which
match the consistency-model backbone's class conditioning exactly).

Files are written as ``{label}_{idx}.png``. Our data loader detects the numeric
prefix and uses it as the literal class id, so a subset that omits some classes
still gets correct labels.

Setup (in the cm env):   pip install datasets
Examples:
  # 100k random images (recommended balance for a ControlNet demo)
  HF_HOME=/data1/<you>/hf_cache python scripts/prep_imagenet64.py \
      --out_dir data/imagenet64 --n 100000
  # everything (1.28M images -- large!)
  python scripts/prep_imagenet64.py --out_dir data/imagenet64 --n 0
  # low-disk: stream instead of downloading the full 10.6GB parquet first
  python scripts/prep_imagenet64.py --out_dir data/imagenet64 --n 50000 --streaming
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=100000, help="#images to save (0 = all)")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--streaming", action="store_true",
                    help="stream (no full 10.6GB download) -- approximate shuffle")
    ap.add_argument("--repo", default="ChocolateDave/imagenet-64")
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out_dir, exist_ok=True)

    if args.streaming:
        ds = load_dataset(args.repo, split=args.split, streaming=True)
        ds = ds.shuffle(seed=args.seed, buffer_size=20000)
        iterator = ds
    else:
        ds = load_dataset(args.repo, split=args.split)  # downloads + caches parquet
        ds = ds.shuffle(seed=args.seed)
        if args.n > 0:
            ds = ds.select(range(min(args.n, len(ds))))
        iterator = ds

    n = 0
    seen_classes = set()
    for ex in iterator:
        label = int(ex["label"])
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(os.path.join(args.out_dir, f"{label}_{n:07d}.png"))
        seen_classes.add(label)
        n += 1
        if n % 5000 == 0:
            print(f"  saved {n} images ({len(seen_classes)} classes seen)")
        if args.streaming and args.n and n >= args.n:
            break

    print(f"done: {n} images, {len(seen_classes)}/1000 classes, in {args.out_dir}")
    if len(seen_classes) < 1000:
        print("note: not all 1000 classes present; that's fine -- numeric labels "
              "are used verbatim, so conditioning stays correct.")


if __name__ == "__main__":
    main()
