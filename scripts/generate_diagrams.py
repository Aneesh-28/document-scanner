#!/usr/bin/env python3
"""Renders the three docs/ diagrams: architecture, workflow, and class/component."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BOX_STYLE = dict(boxstyle="round,pad=0.4,rounding_size=0.08", linewidth=1.5)


def box(ax, x, y, w, h, text, facecolor="#eef3fb", edgecolor="#2c5f8a", fontsize=11, fontweight="normal"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                        linewidth=1.6, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, fontweight=fontweight, wrap=True)
    return b


def arrow(ax, x1, y1, x2, y2, color="#2c5f8a", style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                         linewidth=1.6, color=color)
    ax.add_patch(a)


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# 1. Architecture diagram
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13.5, 3.2))
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 3)
ax.axis("off")
ax.set_title("Architecture: data flow through the scanner pipeline", fontsize=13, fontweight="bold", pad=14)

stages = ["Image\nInput", "Preprocessing", "Contour\nDetection", "Perspective\nWarp", "Enhancement", "Output +\nCSV Log"]
colors = ["#fde9d9", "#eaf2fb", "#eaf2fb", "#eaf2fb", "#eaf2fb", "#e3f6e5"]
n = len(stages)
box_w, gap = 1.7, 0.45
total_w = n * box_w + (n - 1) * gap
x = (13.5 - total_w) / 2
y, h = 1.0, 1.0
centers = []
for label, c in zip(stages, colors):
    box(ax, x, y, box_w, h, label, facecolor=c, fontsize=10.5)
    centers.append((x + box_w / 2, x + box_w))
    x += box_w + gap

for i in range(n - 1):
    x1 = centers[i][1]
    x2_center = centers[i + 1][0]
    x2 = x2_center - box_w / 2
    arrow(ax, x1, y + h / 2, x2, y + h / 2)

fig.tight_layout()
save(fig, "docs/architecture_diagram.png")
plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Workflow diagram
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7.5))
ax.set_xlim(0, 9)
ax.set_ylim(0, 12.5)
ax.axis("off")
ax.set_title("Workflow: CLI run, from invocation to summary", fontsize=13, fontweight="bold", pad=14)

steps = [
    ("User runs CLI\n(python src/scan.py --input ... --output ...)", "#fde9d9"),
    ("Parse args: --input, --output, --debug", "#eaf2fb"),
    ("Is --input a file or a folder?", "#fff6d8"),
    ("Build list of image paths\n(single file, or every supported file in folder)", "#eaf2fb"),
    ("For each image:\n  preprocess -> detect & warp -> enhance -> save PNG", "#eaf2fb"),
    ("On error (missing file, bad image,\nno contour, bad extension): log & continue", "#fde2e2"),
    ("Append row to output/results.csv\n(filename, status, time, path)", "#eaf2fb"),
    ("Print per-file status line", "#eaf2fb"),
    ("Print final summary\n(N/M succeeded, CSV path)", "#e3f6e5"),
]

box_h = 1.05
y = 12.5 - box_h - 0.2
xw = 7.6
xstart = (9 - xw) / 2
box_positions = []
for label, c in steps:
    box(ax, xstart, y, xw, box_h, label, facecolor=c, fontsize=9.5)
    box_positions.append(y)
    y -= box_h + 0.35

for i in range(len(box_positions) - 1):
    y1 = box_positions[i]
    y2 = box_positions[i + 1] + box_h
    arrow(ax, xstart + xw / 2, y1, xstart + xw / 2, y2)

# loop-back arrow from the per-image block area to indicate iteration
fig.tight_layout()
save(fig, "docs/workflow_diagram.png")
plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Class / component diagram (actual file structure + call graph)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 11)
ax.set_ylim(0, 9)
ax.axis("off")
ax.set_title("Components: files and call relationships", fontsize=13, fontweight="bold", pad=14)

modules = {
    "scan.py\n(CLI entry point)\n\n- parse_args()\n- process_single_image()\n- write_results_csv()\n- main()": (0.4, 5.6, 3.0, 3.0, "#fde9d9"),
    "preprocessing.py\n\n- load_image()\n- resize_image()\n- to_grayscale()\n- blur()\n- detect_edges()\n- preprocess()": (4.2, 6.0, 3.0, 2.6, "#eaf2fb"),
    "detect_and_warp.py\n\n- find_candidate_contours()\n- find_document_corners()\n- order_points()\n- compute_output_dimensions()\n- warp_perspective()\n- detect_and_warp()": (7.6, 5.4, 3.0, 3.2, "#eaf2fb"),
    "enhance.py\n\n- enhance()": (4.2, 3.0, 3.0, 1.6, "#eaf2fb"),
    "utils.py\n\n- has_supported_extension()\n- ensure_dir()\n- list_images_in_dir()\n- timed()": (0.4, 1.4, 3.0, 2.4, "#f3eaf9"),
    "tests/test_pipeline.py": (7.6, 1.4, 3.0, 1.4, "#e3f6e5"),
}

for label, (x, y, w, h, c) in modules.items():
    box(ax, x, y, w, h, label, facecolor=c, fontsize=8.7)

def center(name):
    x, y, w, h, _ = modules[name]
    return x + w / 2, y + h / 2

def edge_point(name, towards):
    x, y, w, h, _ = modules[name]
    cx, cy = x + w / 2, y + h / 2
    tx, ty = towards
    return cx, cy  # simple center-to-center; matplotlib arrow trims visually enough at this scale

arrow(ax, *center("scan.py\n(CLI entry point)\n\n- parse_args()\n- process_single_image()\n- write_results_csv()\n- main()"),
      *center("preprocessing.py\n\n- load_image()\n- resize_image()\n- to_grayscale()\n- blur()\n- detect_edges()\n- preprocess()"))
arrow(ax, *center("scan.py\n(CLI entry point)\n\n- parse_args()\n- process_single_image()\n- write_results_csv()\n- main()"),
      *center("detect_and_warp.py\n\n- find_candidate_contours()\n- find_document_corners()\n- order_points()\n- compute_output_dimensions()\n- warp_perspective()\n- detect_and_warp()"))
arrow(ax, *center("scan.py\n(CLI entry point)\n\n- parse_args()\n- process_single_image()\n- write_results_csv()\n- main()"),
      *center("enhance.py\n\n- enhance()"))
arrow(ax, *center("scan.py\n(CLI entry point)\n\n- parse_args()\n- process_single_image()\n- write_results_csv()\n- main()"),
      *center("utils.py\n\n- has_supported_extension()\n- ensure_dir()\n- list_images_in_dir()\n- timed()"))
arrow(ax, *center("tests/test_pipeline.py"),
      *center("preprocessing.py\n\n- load_image()\n- resize_image()\n- to_grayscale()\n- blur()\n- detect_edges()\n- preprocess()"))
arrow(ax, *center("tests/test_pipeline.py"),
      *center("detect_and_warp.py\n\n- find_candidate_contours()\n- find_document_corners()\n- order_points()\n- compute_output_dimensions()\n- warp_perspective()\n- detect_and_warp()"))
arrow(ax, *center("tests/test_pipeline.py"),
      *center("enhance.py\n\n- enhance()"))

fig.tight_layout()
save(fig, "docs/class_component_diagram.png")
plt.close(fig)
