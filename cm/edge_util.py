"""
Shared helpers for binary-edge-map conditioning.

Used by ``scripts/image_sample.py`` (batch edge-conditional sampling),
``scripts/eval_edge_fidelity.py`` (edge-following metric) and the interactive
demo. Deliberately free of mpi4py / blobfile so the demo stays lightweight.
"""
import io

import numpy as np
from PIL import Image


def load_image_any(source):
    """
    Load an image from a local path, an http(s) URL, raw bytes, or a PIL image,
    and return a PIL RGB image. Robust to grayscale / RGBA / palette inputs --
    exactly what an arbitrary picture grabbed from the web might be.
    """
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    elif isinstance(source, str) and source.startswith(("http://", "https://")):
        import requests

        resp = requests.get(
            source, timeout=20, headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
    else:
        img = Image.open(source)
    img.load()
    return img.convert("RGB")


def center_crop_resize(pil_image, image_size):
    """
    Center-crop to a square and resize to ``image_size`` using high-quality
    downsampling (BOX at powers of two, then BICUBIC), matching the training
    preprocessing in ``cm/image_datasets.py``. Returns a uint8 HWC RGB array.
    """
    pil_image = pil_image.convert("RGB")
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    cy = (arr.shape[0] - image_size) // 2
    cx = (arr.shape[1] - image_size) // 2
    return arr[cy : cy + image_size, cx : cx + image_size]


def canny_edge(image, image_size, low=100, high=200, already_cropped=False):
    """
    Compute a binary Canny edge map.

    :param image: a PIL image OR a uint8 HWC RGB array.
    :param image_size: target square resolution.
    :param low/high: Canny hysteresis thresholds.
    :param already_cropped: set True if ``image`` is already a cropped uint8
        array of the right size (skips the crop/resize).
    :return: ``(edge, arr)`` where ``edge`` is a float32 ``(1, H, W)`` map in
        [0, 1] and ``arr`` is the uint8 HWC image the edge was computed from.
    """
    import cv2

    if already_cropped:
        arr = np.asarray(image)
    elif isinstance(image, Image.Image):
        arr = center_crop_resize(image, image_size)
    else:
        arr = np.asarray(image)
    arr = np.ascontiguousarray(arr)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, int(low), int(high))  # uint8 {0, 255}
    edge = (edges.astype(np.float32) / 255.0)[None, :, :]
    return edge, arr


def edge_to_uint8_rgb(edge_chw):
    """(1, H, W) float in [0, 1] -> (H, W, 3) uint8 for display."""
    e = (np.clip(edge_chw[0], 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.stack([e, e, e], axis=-1)


def sample_to_uint8(sample_chw):
    """A model sample in [-1, 1], shape (3, H, W) -> (H, W, 3) uint8."""
    s = ((sample_chw + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return np.transpose(s, (1, 2, 0))


def make_triptych(orig_uint8, edge_chw, gen_uint8, labels=True):
    """
    Concatenate ``[original | edge | generated]`` horizontally into one PIL
    image (with optional captions). All inputs are at the model resolution.
    """
    edge_rgb = edge_to_uint8_rgb(edge_chw)
    panels = [np.asarray(orig_uint8), edge_rgb, np.asarray(gen_uint8)]
    grid = np.concatenate(panels, axis=1)
    img = Image.fromarray(grid)
    if labels:
        try:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(img)
            w = panels[0].shape[1]
            for i, name in enumerate(["input", "canny edge", "generated"]):
                draw.text((i * w + 2, 2), name, fill=(255, 0, 0))
        except Exception:
            pass
    return img
