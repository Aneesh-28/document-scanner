# Project Statement: Document Scanner

## 1. Objective

Build a command-line tool that takes a photo of a printed document or
receipt — taken at an angle, on an ordinary desk, with a normal phone
camera — and produces a clean, flattened, high-contrast image that looks
like it came from a real scanner, not a photo.

## 2. Problem Statement

A phone photo of a document has three problems a scanner output doesn't:
perspective distortion (the page isn't shot dead-on), uneven lighting
across the page, and background clutter around the page's edges. The tool
needs to detect the page within the frame, undo the perspective distortion,
and normalize the lighting/contrast — automatically, for a batch of images,
without manual cropping.

## 3. Approach / Methodology

The problem splits cleanly into three sequential computer-vision stages,
implemented as three modules with a thin CLI wrapper on top:

1. **Preprocessing** — reduce the image to information the next stage
   actually needs (edges), while discarding what it doesn't (color, noise,
   resolution it can't use).
2. **Contour detection & perspective warp** — the core geometric problem:
   find the page's outline, and mathematically "unwarp" it into a flat
   rectangle.
3. **Enhancement** — cosmetic normalization so the output looks like a
   scan rather than a cropped photo.

This mirrors a classic OpenCV document-scanner pipeline (Canny edges →
contour approximation → four-point perspective transform → adaptive
threshold), built from first principles rather than a pre-built library.

## 4. Architecture

```
Image Input → Preprocessing → Contour Detection → Perspective Warp → Enhancement → Output + CSV Log
```

See `docs/architecture_diagram.png` for the visual version.

## 5. Module 1 — Preprocessing

Implemented in `src/preprocessing.py`.

- `load_image()` reads the file with `cv2.imread()`, raising a clear
  `ValueError` (not a crash) if the file is missing, corrupt, or an
  unsupported format.
- `resize_image()` shrinks the longer side to ~900px. Full phone-camera
  resolution (often 3000+ px) makes contour-finding slow and noisy without
  improving accuracy. The function returns a `scale_ratio` so corners found
  on the small image can be mapped back to full resolution before the
  final warp — this keeps the *output* sharp while keeping *detection*
  fast.
- `to_grayscale()` and `blur()` strip color (irrelevant for edge-finding)
  and suppress fine texture noise like paper grain.
- `detect_edges()` runs Canny edge detection (thresholds 75/200 as a
  starting point) to produce a binary edge map.

## 6. Module 2 — Contour Detection & Perspective Warp

Implemented in `src/detect_and_warp.py`. This was the hardest part of the
project, as expected.

- `find_candidate_contours()` pulls every closed shape out of the edge map
  and keeps the 5 largest by area — the document should be one of the
  largest shapes in frame, provided the background contrasts with it.
- `find_document_corners()` approximates each candidate to a polygon via
  `cv2.approxPolyDP`. If one approximates to exactly 4 points, that's the
  document. If none do (typical on a messy or low-contrast background),
  it falls back to the largest contour's bounding rectangle — a graceful
  degradation rather than a hard failure.
- `order_points()` sorts the 4 corners into a consistent
  [top-left, top-right, bottom-right, bottom-left] order using the
  sum/difference trick (`x+y` and `x-y`), since `approxPolyDP` doesn't
  guarantee any particular order and `warpPerspective` requires one.
- `compute_output_dimensions()` derives the output rectangle's width and
  height from the actual corner distances, so the flattened result has
  correct, undistorted proportions rather than an arbitrary fixed size.
- `warp_perspective()` computes the transform matrix with
  `cv2.getPerspectiveTransform()` and applies it with
  `cv2.warpPerspective()`, using corners scaled back up to full resolution
  so the final output isn't limited to the small working resolution.

## 7. Module 3 — Enhancement

Implemented in `src/enhance.py`. `cv2.adaptiveThreshold()` (Gaussian,
block size 11, constant 10) converts the warped color image into binary
black-on-white text. Adaptive (rather than global) thresholding matters
because desk lighting is rarely uniform across the whole page — a single
global threshold would blow out one corner or crush another.

## 8. CLI Design

Implemented in `src/scan.py` using `argparse`, with `--input` (file or
folder), `--output` (default `./output`), and `--debug` (also saves the
intermediate edge map per image). The CLI loops over one or many images,
runs all three modules per image, and writes both per-file console status
lines and a `results.csv` log (`filename, status, processing_time_s,
output_path, error`) so a batch run leaves an auditable record.

## 9. Error Handling

Handled explicitly, per file, without ever crashing the whole batch:

- File doesn't exist → caught before attempting to read it.
- File isn't a valid image (`cv2.imread` returns `None`) → raised as a
  `ValueError` with a clear message.
- No 4-point contour found → handled by the Module 2 fallback path; a
  genuinely empty edge map raises a dedicated `DocumentNotFoundError`.
- Unsupported file extension → checked against a known-extension list
  before any image-reading is attempted.
- A catch-all `except Exception` in `process_single_image()` ensures one
  unexpectedly bad file logs an error row and lets the rest of the batch
  continue, rather than stopping the whole run.

## 10. Testing

12 `pytest` tests in `tests/test_pipeline.py`, covering:

- Preprocessing output shape/type/dtype correctness.
- Aspect-ratio preservation on resize.
- A known-good sample producing a warped output of the expected (portrait)
  dimensions.
- Point-ordering correctness on a hand-constructed shuffled point set.
- The fallback path (`DocumentNotFoundError`) on a blank edge map.
- A non-image file raising a handled `ValueError`, not crashing.
- Enhancement output being strictly binary (0/255).
- An end-to-end smoke test across every real sample image in
  `data/samples/`.

All 12 tests pass (`pytest tests/ -v`).

## 11. Test Data

Since this project was built in a sandboxed environment with no camera,
`scripts/generate_samples.py` synthesizes stand-in photos: a rendered page
with a border, title, and paragraph-like lines, composited onto a dark
desk-colored canvas with a perspective skew, slight rotation, blur, and
sensor noise to mimic a real hand-held shot. This includes one
deliberately low-contrast case (to exercise the fallback contour path) and
one deliberately invalid file (to exercise error handling). Real phone
photos, once available, drop into `data/samples/` and work identically —
nothing in the pipeline is specific to the synthetic images.

## 12. Results

Running `python src/scan.py --input data/samples/ --output output/ --debug`
processes all 7 sample files: 6 succeed (one via the fallback contour
path), and the 1 intentionally invalid file fails cleanly with a logged
error, all recorded in `output/results.csv`. See `docs/before_after.png`
for a visual before/after comparison on `photo1.jpg`.

## 13. Challenges & Solutions

- **Mapping corners back to full resolution.** Detecting on a resized
  image but warping the original meant every corner coordinate needed the
  `scale_ratio` multiplier applied before `warpPerspective` — easy to get
  wrong by warping the small image instead and losing output quality.
- **Corner ordering.** `approxPolyDP` returns points in whatever order the
  contour happened to be traced, not a usable top-left/top-right/etc.
  order. The sum/difference trick solves this without needing to know
  anything about the page's actual rotation.
- **Low-contrast backgrounds.** When the page barely contrasts with the
  desk, Canny edges around the page boundary are weak or broken, so no
  contour approximates cleanly to 4 points. The bounding-rectangle fallback
  keeps the pipeline from hard-failing on these, at the cost of some
  accuracy on rotated pages.

## 14. Limitations & Future Work

- The fallback bounding-rectangle path doesn't correct rotation, only
  cropping — a genuinely rotated page on a low-contrast background will
  come out cropped but still skewed.
- Canny thresholds are static; an adaptive or auto-tuned threshold
  (e.g. based on image statistics) would generalize better across lighting
  conditions than the current fixed 75/200.
- No automatic orientation/rotation correction (e.g. detecting that a page
  was photographed upside-down or sideways).
- Could be extended to multi-page PDF assembly from a folder of scans.

## 15. Conclusion

The three-module pipeline — preprocess, detect & warp, enhance — reliably
turns an angled desk photo into a flattened, high-contrast scan, with a
graceful fallback for harder cases and a CLI that logs every outcome for
batch runs. The hardest and most instructive part was Module 2: getting
corner ordering and the resize/full-resolution coordinate mapping correct
was where nearly all of the debugging time went, exactly as expected going
in.
