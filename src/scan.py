#!/usr/bin/env python3
"""
scan.py - CLI entry point for the document scanner.

Usage:
    python src/scan.py --input data/samples/photo1.jpg --output output/
    python src/scan.py --input data/samples/ --output output/ --debug
"""

import argparse
import csv
import os
import sys
import time

import cv2

# Allow running this script directly (python src/scan.py ...) by making sure
# its own directory is importable regardless of the caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocess, load_image, resize_image, to_grayscale, blur, detect_edges
from detect_and_warp import detect_and_warp, DocumentNotFoundError
from enhance import enhance
from utils import ensure_dir, has_supported_extension, list_images_in_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan photos of documents into clean, flattened, high-contrast scans."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a single image file, or a folder of images."
    )
    parser.add_argument(
        "--output", default="./output",
        help="Output directory (default: ./output)."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Also save the intermediate edge-map image for each input."
    )
    return parser.parse_args()


def process_single_image(path: str, output_dir: str, debug: bool = False):
    """Run the full pipeline on one image.

    Returns a dict describing the outcome, suitable for a results.csv row:
        {filename, status, processing_time_s, output_path, error}
    """
    filename = os.path.basename(path)
    start = time.perf_counter()
    row = {
        "filename": filename,
        "status": "fail",
        "processing_time_s": 0.0,
        "output_path": "",
        "error": "",
    }

    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File does not exist: {path}")

        if not has_supported_extension(path):
            raise ValueError(f"Unsupported file extension: {path}")

        # --- Module 1: preprocessing ---
        # Load full-resolution image separately so we can warp at full res.
        full_res = load_image(path)  # raises ValueError if unreadable
        resized_original, scale_ratio = resize_image(full_res)
        gray = to_grayscale(resized_original)
        blurred = blur(gray)
        edged = detect_edges(blurred)

        if debug:
            ensure_dir(output_dir)
            debug_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_edges.png")
            cv2.imwrite(debug_path, edged)

        # --- Module 2: contour detection & warp ---
        warped, used_fallback = detect_and_warp(
            resized_original, edged, scale_ratio, full_res_original=full_res
        )

        # --- Module 3: enhancement ---
        scanned = enhance(warped)

        ensure_dir(output_dir)
        out_name = f"{os.path.splitext(filename)[0]}_scanned.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, scanned)

        row["status"] = "success" if not used_fallback else "success (fallback contour)"
        row["output_path"] = out_path

    except (FileNotFoundError, ValueError, DocumentNotFoundError) as e:
        row["error"] = str(e)
    except Exception as e:  # noqa: BLE001 - last-resort guard so one bad file doesn't crash a batch
        row["error"] = f"Unexpected error: {e}"
    finally:
        row["processing_time_s"] = round(time.perf_counter() - start, 4)

    return row


def write_results_csv(rows, output_dir: str):
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "results.csv")
    fieldnames = ["filename", "status", "processing_time_s", "output_path", "error"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def main():
    args = parse_args()
    ensure_dir(args.output)

    if os.path.isdir(args.input):
        image_paths = list_images_in_dir(args.input)
        if not image_paths:
            print(f"No supported images found in folder: {args.input}")
            sys.exit(1)
    elif os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        print(f"Error: input path does not exist: {args.input}")
        sys.exit(1)

    results = []
    for path in image_paths:
        row = process_single_image(path, args.output, debug=args.debug)
        results.append(row)
        status_symbol = "OK  " if row["status"].startswith("success") else "FAIL"
        detail = row["output_path"] if row["status"].startswith("success") else row["error"]
        print(f"[{status_symbol}] {row['filename']:30s} ({row['processing_time_s']:>6.3f}s)  {detail}")

    csv_path = write_results_csv(results, args.output)

    successes = sum(1 for r in results if r["status"].startswith("success"))
    print()
    print(f"Summary: {successes}/{len(results)} succeeded. Log written to {csv_path}")


if __name__ == "__main__":
    main()
