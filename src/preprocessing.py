"""
Module 1: Preprocessing
------------------------
Turns a messy color photo into clean edges the computer can reason about.

Pipeline: load -> resize (track scale ratio) -> grayscale -> blur -> Canny edges
"""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk. Raises ValueError if the file can't be
    read as an image (missing file, corrupt file, unsupported format, etc.)."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read '{path}' as an image (missing, corrupt, or unsupported format).")
    return img


def resize_image(img: np.ndarray, target_dim: int = 900):
    """Resize so the longer side is ~target_dim px (default 900, within the
    recommended 800-1000px range). Full phone-camera resolution (3000+ px)
    makes contour-finding slow and noisy.

    Returns (resized_image, scale_ratio) where scale_ratio maps resized
    coordinates back to full-resolution coordinates:
        original_coord = resized_coord * scale_ratio
    """
    h, w = img.shape[:2]
    longer_side = max(h, w)
    if longer_side <= target_dim:
        # Already small enough; no upscaling needed.
        return img.copy(), 1.0

    scale = target_dim / float(longer_side)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # scale_ratio = original_height / resized_height (equivalent to original_width/resized_width
    # since aspect ratio is preserved) -- used later to map detected corners back to full res.
    scale_ratio = h / float(new_h)
    return resized, scale_ratio


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale. Color is irrelevant for finding edges."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def blur(gray: np.ndarray, ksize=(5, 5)) -> np.ndarray:
    """Gaussian blur to suppress small noise/texture (like paper grain) so
    the edge detector doesn't pick up thousands of tiny irrelevant edges."""
    return cv2.GaussianBlur(gray, ksize, 0)


def detect_edges(blurred: np.ndarray, low_threshold: int = 75, high_threshold: int = 200) -> np.ndarray:
    """Canny edge detection. Too low a threshold -> noise; too high -> lose
    faint edges. 75/200 is a reasonable starting point; tune per image set."""
    return cv2.Canny(blurred, low_threshold, high_threshold)


def preprocess(path: str, target_dim: int = 900, low_threshold: int = 75, high_threshold: int = 200):
    """Run the full preprocessing pipeline on an image file.

    Returns:
        resized_original: the resized color image (kept for later warping)
        edged: the binary edge map (used for contour detection)
        scale_ratio: multiplier to map resized coords back to full resolution
    """
    img = load_image(path)
    resized_original, scale_ratio = resize_image(img, target_dim)
    gray = to_grayscale(resized_original)
    blurred = blur(gray)
    edged = detect_edges(blurred, low_threshold, high_threshold)
    return resized_original, edged, scale_ratio
