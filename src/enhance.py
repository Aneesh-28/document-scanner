"""
Module 3: Enhancement
-----------------------
Takes the warped (flattened) document image and makes it look "scanned"
rather than just photographed, using adaptive thresholding so it copes
with uneven lighting across the page.
"""

import cv2
import numpy as np


def enhance(warped: np.ndarray) -> np.ndarray:
    """Convert the warped color image to a crisp black-text-on-white-background
    scanned look using adaptive Gaussian thresholding, which adapts to
    lighting variation across the page rather than using one global
    threshold."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        10,
    )
    return scanned
