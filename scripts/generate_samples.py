#!/usr/bin/env python3
"""
generate_samples.py
--------------------
Synthesizes stand-in test images for data/samples/, since this environment
has no phone camera to shoot real photos with (Step 1 of the project plan).

Each image simulates: a white printed page with text-like lines and a
receipt-style page, photographed at a slight angle, on a dark contrasting
desk background, with mild blur/noise to mimic a real phone photo.

If you have real phone photos, just drop them into data/samples/ instead --
this script only exists to make the rest of the pipeline runnable and
testable in a sandboxed environment.
"""

import os
import numpy as np
import cv2

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "samples")


def make_page(width=1000, height=1300, kind="document"):
    """Draw a clean white 'page' with black text-like lines / a border."""
    page = np.full((height, width, 3), 255, dtype=np.uint8)

    # Border, like a printed page margin.
    cv2.rectangle(page, (20, 20), (width - 20, height - 20), (0, 0, 0), 2)

    if kind == "document":
        # Title bar
        cv2.putText(page, "INVOICE #4471", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
        cv2.line(page, (60, 130), (width - 60, 130), (0, 0, 0), 2)
        # Body "text" lines of varying length to look like paragraphs
        rng = np.random.default_rng(42)
        y = 200
        while y < height - 150:
            line_width = int(rng.uniform(0.4, 0.9) * (width - 120))
            cv2.line(page, (60, y), (60 + line_width, y), (30, 30, 30), 6)
            y += 40
        # Footer
        cv2.line(page, (60, height - 100), (width - 60, height - 100), (0, 0, 0), 2)
        cv2.putText(page, "Total: $128.50", (60, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    else:  # receipt-like, narrower content
        cv2.putText(page, "CORNER STORE", (width // 2 - 180, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
        rng = np.random.default_rng(7)
        y = 160
        while y < height - 200:
            line_width = int(rng.uniform(0.3, 0.8) * (width - 120))
            cv2.line(page, (60, y), (60 + line_width, y), (20, 20, 20), 5)
            y += 34
        cv2.putText(page, "TOTAL   $23.47", (60, height - 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    return page


def photograph(page, canvas_size=(1600, 1800), angle_deg=8, skew=0.06,
               bg_color=(35, 35, 40), noise_level=6, blur_ksize=3, seed=0):
    """Place the page onto a larger dark canvas with a perspective skew and
    slight rotation to simulate a hand-held phone photo taken at an angle,
    then add mild blur + noise."""
    ch, cw = canvas_size
    ph, pw = page.shape[:2]

    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
    canvas[:, :] = bg_color
    # subtle desk texture/gradient so it's not perfectly flat
    grad = np.tile(np.linspace(0, 25, cw, dtype=np.uint8), (ch, 1))
    for c in range(3):
        canvas[:, :, c] = cv2.add(canvas[:, :, c], grad)

    # Source corners of the page (its own 4 corners)
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])

    # Destination corners: place page roughly centered in canvas, with a
    # perspective skew + rotation to simulate camera angle.
    offset_x = (cw - pw) / 2
    offset_y = (ch - ph) / 2
    rng = np.random.default_rng(seed)
    jitter = lambda: rng.uniform(-skew, skew)

    def pt(x, y):
        return [x + offset_x, y + offset_y]

    tl = pt(pw * jitter() * 2, ph * (0.02 + abs(jitter())))
    tr = pt(pw * (1 - abs(jitter())), ph * jitter() * 2)
    br = pt(pw * (1 - abs(jitter()) * 1.5), ph * (1 - 0.02 - abs(jitter())))
    bl = pt(pw * abs(jitter()) * 1.5, ph * (1 - abs(jitter()) * 1.2))

    dst = np.float32([tl, tr, br, bl])

    m = cv2.getPerspectiveTransform(src, dst)
    warped_page = cv2.warpPerspective(page, m, (cw, ch), borderValue=bg_color)

    # mask so we only composite the page pixels, not the border fill areas
    mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)
    mask_3c = cv2.merge([mask, mask, mask])

    composite = np.where(mask_3c > 0, warped_page, canvas)

    # Slight overall rotation of the whole "photo" to mimic hand-held angle
    center = (cw // 2, ch // 2)
    rot_m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    composite = cv2.warpAffine(composite, rot_m, (cw, ch), borderValue=bg_color)

    # Mild blur + sensor noise to look like a real photo, not a render
    composite = cv2.GaussianBlur(composite, (blur_ksize, blur_ksize), 0)
    noise = rng.normal(0, noise_level, composite.shape).astype(np.int16)
    composite = np.clip(composite.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return composite


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    configs = [
        dict(name="photo1.jpg", kind="document", angle_deg=6, skew=0.05, seed=1),
        dict(name="photo2.jpg", kind="document", angle_deg=-9, skew=0.08, seed=2),
        dict(name="photo3.jpg", kind="receipt", angle_deg=4, skew=0.04, seed=3),
        dict(name="photo4.jpg", kind="document", angle_deg=12, skew=0.10, seed=4),
        dict(name="photo5.jpg", kind="receipt", angle_deg=-5, skew=0.06, seed=5),
    ]

    for cfg in configs:
        page = make_page(kind=cfg["kind"])
        photo = photograph(
            page,
            angle_deg=cfg["angle_deg"],
            skew=cfg["skew"],
            seed=cfg["seed"],
        )
        out_path = os.path.join(OUT_DIR, cfg["name"])
        cv2.imwrite(out_path, photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"Wrote {out_path}")

    # One deliberately "hard" case: low background contrast, to exercise the
    # Step 4b fallback path (no clean 4-point contour).
    page = make_page(kind="document")
    hard_photo = photograph(page, bg_color=(230, 228, 222), noise_level=10, angle_deg=15, skew=0.12, seed=9)
    out_path = os.path.join(OUT_DIR, "photo6_low_contrast.jpg")
    cv2.imwrite(out_path, hard_photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Wrote {out_path}")

    # One genuinely invalid "image" file, to exercise error handling.
    bad_path = os.path.join(OUT_DIR, "not_an_image.jpg")
    with open(bad_path, "w") as f:
        f.write("this is not image data")
    print(f"Wrote {bad_path} (intentionally invalid, for error-handling tests)")


if __name__ == "__main__":
    main()
