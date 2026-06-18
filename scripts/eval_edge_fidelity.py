"""
Edge-following metric for edge-conditioned generation.

We re-extract Canny edges from each *generated* image and compare them to the
*conditioning* edge map. Because exact pixel overlap is too strict for thin
edges, we match with a small dilation tolerance and report precision / recall /
F1 (edge agreement) and a dilated IoU.

Input: the ``edge_samples_rank*.npz`` files written by
``scripts/image_sample.py`` in ControlNet mode, each holding:
  * ``samples``: (N, H, W, 3) uint8 generated images
  * ``edges``:   (N, H, W)    uint8 conditioning edge maps ({0,255})

Example:
  python scripts/eval_edge_fidelity.py --npz_glob "out/edge_samples_rank*.npz" \
      --canny_low 100 --canny_high 200 --tol 2
"""
import argparse
import glob

import numpy as np


def _dilate(mask_uint8, r):
    import cv2

    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return cv2.dilate(mask_uint8, k)


def edge_metrics(pred_edge, gt_edge, tol=2):
    """pred_edge, gt_edge: uint8 {0,255} HxW. Returns (precision, recall, f1, iou)."""
    pred = (pred_edge > 0).astype(np.uint8)
    gt = (gt_edge > 0).astype(np.uint8)
    gt_d = _dilate(gt, tol) > 0
    pred_d = _dilate(pred, tol) > 0
    predb, gtb = pred > 0, gt > 0

    tp = np.logical_and(predb, gt_d).sum()
    fp = np.logical_and(predb, ~gt_d).sum()
    fn = np.logical_and(gtb, ~pred_d).sum()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    inter = np.logical_and(pred_d, gt_d).sum()
    union = np.logical_or(pred_d, gt_d).sum()
    iou = inter / (union + 1e-9)
    return prec, rec, f1, iou


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_glob", required=True, help="glob for edge_samples_rank*.npz")
    ap.add_argument("--canny_low", type=int, default=100)
    ap.add_argument("--canny_high", type=int, default=200)
    ap.add_argument("--tol", type=int, default=2, help="dilation tolerance (px)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.npz_glob))
    if not files:
        raise SystemExit(f"no files match {args.npz_glob}")

    ps, rs, fs, ious = [], [], [], []
    n = 0
    for fp in files:
        d = np.load(fp)
        samples, edges = d["samples"], d["edges"]  # (N,H,W,3), (N,H,W)
        for i in range(samples.shape[0]):
            gray = cv2.cvtColor(np.ascontiguousarray(samples[i]), cv2.COLOR_RGB2GRAY)
            pred = cv2.Canny(gray, args.canny_low, args.canny_high)
            p, r, f1, iou = edge_metrics(pred, edges[i], tol=args.tol)
            ps.append(p); rs.append(r); fs.append(f1); ious.append(iou)
            n += 1

    print(f"evaluated {n} generated images from {len(files)} file(s)")
    print(f"  edge precision : {np.mean(ps):.4f}")
    print(f"  edge recall    : {np.mean(rs):.4f}")
    print(f"  edge F1        : {np.mean(fs):.4f}   <-- higher = better edge-following")
    print(f"  dilated IoU    : {np.mean(ious):.4f}")


if __name__ == "__main__":
    main()
