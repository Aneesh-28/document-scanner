"""
Module 2: Contour detection & perspective warp
------------------------------------------------
The actual "computer vision" core: find the document's outline in the edge
map, order its 4 corners consistently, and warp it flat into a top-down
rectangle.
"""

import cv2
import numpy as np


class DocumentNotFoundError(Exception):
    """Raised when no plausible document contour can be found at all."""
    pass


def find_candidate_contours(edged: np.ndarray, top_n: int = 5):
    """Find every closed shape in the edge map, keep the top_n largest by
    area. The document should be one of the largest shapes in frame -- this
    is why background contrast matters: if the page blends into the desk,
    its contour won't be strong."""
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise DocumentNotFoundError("No contours found at all in the edge map.")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_n]
    return contours


def find_document_corners(edged: np.ndarray, top_n: int = 5):
    """Look through the top_n largest contours for one that approximates to
    a 4-point polygon (our document). If none has exactly 4 points, fall
    back to the single largest contour's bounding rectangle -- the
    error-handling path for messy backgrounds.

    Returns a (4, 2) array of corner points (unordered) plus a bool
    indicating whether the fallback path was used.
    """
    contours = find_candidate_contours(edged, top_n)

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32"), False

    # Fallback: largest contour's bounding rectangle.
    largest = contours[0]
    x, y, w, h = cv2.boundingRect(largest)
    corners = np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype="float32",
    )
    return corners, True


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left].

    approxPolyDP/boundingRect give 4 points but not in a predictable order,
    and warpPerspective needs them ordered consistently.

    Trick:
        top-left     -> smallest sum of (x + y)
        bottom-right -> largest sum of (x + y)
        top-right    -> smallest difference (x - y)
        bottom-left  -> largest difference (x - y)
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def compute_output_dimensions(rect: np.ndarray):
    """Compute the pixel size of the flattened output rectangle from the
    4 ordered source points."""
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    return max(max_width, 1), max(max_height, 1)


def warp_perspective(original_full_res: np.ndarray, ordered_src_points: np.ndarray):
    """Warp the (full-resolution) source image so the given 4 ordered
    corners become a perfect flat rectangle."""
    max_width, max_height = compute_output_dimensions(ordered_src_points)

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    m = cv2.getPerspectiveTransform(ordered_src_points, dst)
    warped = cv2.warpPerspective(original_full_res, m, (max_width, max_height))
    return warped


def detect_and_warp(resized_original: np.ndarray, edged: np.ndarray, scale_ratio: float,
                     full_res_original: np.ndarray = None):
    """Full Module 2 pipeline: find corners in the (small) edge map, order
    them, scale them back to full resolution, and warp the full-resolution
    original into a flattened document image.

    If full_res_original is not provided, warps the resized image itself
    (useful when you don't need to keep the original full-resolution file
    around).
    """
    corners, used_fallback = find_document_corners(edged)
    ordered = order_points(corners)

    if full_res_original is not None:
        ordered_full_res = ordered * scale_ratio
        warped = warp_perspective(full_res_original, ordered_full_res)
    else:
        warped = warp_perspective(resized_original, ordered)

    return warped, used_fallback
