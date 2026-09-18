#!/usr/bin/env python3
"""Generates the Use Case and Sequence diagrams for the project report."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse, Circle
import numpy as np
import os

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")


def box(ax, x, y, w, h, text, facecolor="#eef3fb", edgecolor="#2c5f8a",
        fontsize=10, fontweight="normal", linestyle="-"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                        linewidth=1.6, edgecolor=edgecolor, facecolor=facecolor,
                        linestyle=linestyle)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, fontweight=fontweight, wrap=True)
    return b


def arrow(ax, x1, y1, x2, y2, color="#2c5f8a", style="-|>", lw=1.4):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                         linewidth=lw, color=color)
    ax.add_patch(a)


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# 1. Use Case Diagram
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 8))
ax.set_xlim(0, 11)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("Use Case Diagram", fontsize=14, fontweight="bold", pad=16)

# System boundary
system_box = FancyBboxPatch((3.0, 0.4), 5.5, 7.2,
                             boxstyle="round,pad=0.1,rounding_size=0.15",
                             linewidth=2, edgecolor="#2c5f8a", facecolor="#f8faff",
                             linestyle="--")
ax.add_patch(system_box)
ax.text(5.75, 7.35, "Document Scanner System", ha="center", va="center",
         fontsize=12, fontweight="bold", color="#2c5f8a")

# Actor (stick figure)
head = Circle((1.2, 5.0), 0.22, linewidth=1.5, edgecolor="#2c5f8a", facecolor="white")
ax.add_patch(head)
ax.plot([1.2, 1.2], [4.78, 4.15], color="#2c5f8a", linewidth=1.5)     # body
ax.plot([0.75, 1.65], [4.55, 4.55], color="#2c5f8a", linewidth=1.5)   # arms
ax.plot([1.2, 0.85], [4.15, 3.55], color="#2c5f8a", linewidth=1.5)    # left leg
ax.plot([1.2, 1.55], [4.15, 3.55], color="#2c5f8a", linewidth=1.5)    # right leg
ax.text(1.2, 3.3, "User", fontsize=11, ha="center", va="center", fontweight="bold")

# Use cases (ellipses)
use_cases = [
    (5.75, 6.5, "Scan Single Image"),
    (5.75, 5.5, "Scan Folder of Images"),
    (5.75, 4.5, "View Debug Edge Maps"),
    (5.75, 3.5, "Get Processing Results\n(CSV Log)"),
    (5.75, 2.5, "Detect Document\nContour"),
    (5.75, 1.5, "Flatten & Enhance\nDocument"),
]

for (cx, cy, label) in use_cases:
    ellipse = Ellipse((cx, cy), 3.8, 0.75, linewidth=1.5,
                       edgecolor="#2c5f8a", facecolor="#eaf2fb")
    ax.add_patch(ellipse)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=9)

# Lines from actor to use cases
for (cx, cy, _) in use_cases:
    ax.plot([1.7, cx - 1.9], [4.0, cy], color="#2c5f8a", linewidth=1.2, alpha=0.6)

# <<include>> relationships
ax.annotate("", xy=(5.75, 2.1), xytext=(5.75, 2.9),
            arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2, linestyle="--"))
ax.text(6.9, 2.5, "«include»", fontsize=7.5, color="#888", fontstyle="italic")

ax.annotate("", xy=(5.75, 1.1), xytext=(5.75, 1.9),
            arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2, linestyle="--"))
ax.text(6.9, 1.5, "«include»", fontsize=7.5, color="#888", fontstyle="italic")

fig.tight_layout()
save(fig, os.path.join(DOCS_DIR, "use_case_diagram.png"))
plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Sequence Diagram
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 13)
ax.axis("off")
ax.set_title("Sequence Diagram: Processing a Single Image", fontsize=14, fontweight="bold", pad=16)

# Lifeline positions (x)
lifelines = {
    "User / CLI": 1.5,
    "scan.py": 3.5,
    "preprocessing.py": 5.8,
    "detect_and_warp.py": 8.3,
    "enhance.py": 10.5,
    "Output / CSV": 12.5,
}

# Draw lifeline headers and dashed lines
for name, x in lifelines.items():
    box(ax, x - 0.85, 12.0, 1.7, 0.7, name, facecolor="#eaf2fb",
        edgecolor="#2c5f8a", fontsize=8.5, fontweight="bold")
    ax.plot([x, x], [12.0, 0.3], color="#aaa", linewidth=1, linestyle="--", zorder=0)

# Sequence messages (from_x, to_x, y, label, return_label)
y = 11.3
msgs = [
    ("User / CLI", "scan.py", "python scan.py --input photo.jpg", None),
    ("scan.py", "scan.py", "parse_args(), validate input", None),
    ("scan.py", "preprocessing.py", "load_image(path)", None),
    ("preprocessing.py", "scan.py", "", "full_res image"),
    ("scan.py", "preprocessing.py", "resize_image(img)", None),
    ("preprocessing.py", "scan.py", "", "(resized, scale_ratio)"),
    ("scan.py", "preprocessing.py", "to_grayscale() → blur() → detect_edges()", None),
    ("preprocessing.py", "scan.py", "", "edge map"),
    ("scan.py", "detect_and_warp.py", "detect_and_warp(resized, edged, ratio, full_res)", None),
    ("detect_and_warp.py", "detect_and_warp.py", "findContours → approxPolyDP → order_points", None),
    ("detect_and_warp.py", "detect_and_warp.py", "getPerspectiveTransform → warpPerspective", None),
    ("detect_and_warp.py", "scan.py", "", "warped image"),
    ("scan.py", "enhance.py", "enhance(warped)", None),
    ("enhance.py", "scan.py", "", "scanned (binary) image"),
    ("scan.py", "Output / CSV", "cv2.imwrite(scanned) + append CSV row", None),
    ("Output / CSV", "scan.py", "", "saved"),
    ("scan.py", "User / CLI", "", "status + summary"),
]

step = 0.58
for (src, dst, label, ret_label) in msgs:
    sx = lifelines[src]
    dx = lifelines[dst]

    if ret_label is not None:
        # Return (dashed)
        ax.annotate("", xy=(dx, y), xytext=(sx, y),
                    arrowprops=dict(arrowstyle="-|>", color="#d46a00", lw=1.3, linestyle="--"))
        mid = (sx + dx) / 2
        ax.text(mid, y + 0.12, ret_label, ha="center", va="bottom",
                fontsize=7.5, color="#d46a00", fontstyle="italic")
    elif src == dst:
        # Self-call
        ax.annotate("", xy=(sx + 0.3, y - 0.15), xytext=(sx + 0.3, y + 0.15),
                    arrowprops=dict(arrowstyle="-|>", color="#2c5f8a", lw=1.3,
                                   connectionstyle="arc3,rad=0.4"))
        ax.text(sx + 0.5, y + 0.05, label, ha="left", va="center", fontsize=7.5, color="#2c5f8a")
    else:
        # Normal call
        ax.annotate("", xy=(dx, y), xytext=(sx, y),
                    arrowprops=dict(arrowstyle="-|>", color="#2c5f8a", lw=1.3))
        mid = (sx + dx) / 2
        ax.text(mid, y + 0.12, label, ha="center", va="bottom", fontsize=7.5, color="#2c5f8a")

    y -= step

fig.tight_layout()
save(fig, os.path.join(DOCS_DIR, "sequence_diagram.png"))
plt.close(fig)

print("\nDone — both diagrams generated in docs/")
