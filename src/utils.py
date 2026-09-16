"""
Shared utility helpers used across modules.
"""

import os
import time
import functools

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def has_supported_extension(path: str) -> bool:
    """Check whether a file's extension is one we know how to read."""
    _, ext = os.path.splitext(path)
    return ext.lower() in SUPPORTED_EXTENSIONS


def ensure_dir(path: str) -> None:
    """Create a directory (and parents) if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def timed(func):
    """Decorator that returns (result, elapsed_seconds) instead of just result."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    return wrapper


def list_images_in_dir(folder: str):
    """Return sorted list of full paths to supported image files in a folder."""
    entries = sorted(os.listdir(folder))
    return [
        os.path.join(folder, f) for f in entries
        if has_supported_extension(f) and os.path.isfile(os.path.join(folder, f))
    ]
