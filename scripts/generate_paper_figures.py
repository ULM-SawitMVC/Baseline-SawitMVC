"""Generate publication-oriented figures for the paper draft."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "paper"
IMAGES = ROOT / "images"
MPL_CACHE = ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.10)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#2b2b2b",
            "axes.linewidth": 0.8,
        }
    )


def crop_around_box(img: Image.Image, box: list[int], crop_size: int = 520) -> tuple[Image.Image, list[int]]:
    width, height = img.size
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    left = int(round(cx - crop_size / 2))
    top = int(round(cy - crop_size / 2))
    left = max(0, min(left, width - crop_size))
    top = max(0, min(top, height - crop_size))
    right = min(width, left + crop_size)
    bottom = min(height, top + crop_size)
    crop = img.crop((left, top, right, bottom))
    shifted_box = [x1 - left, y1 - top, x2 - left, y2 - top]
    return crop, shifted_box


def figure_cross_view_linking() -> None:
    tree_id = "DAMIMAS_A21B_0847"
    sides = [1, 2, 3]
    boxes = {
        1: [304, 531, 409, 681],
        2: [292, 503, 485, 682],
        3: [537, 435, 683, 600],
    }

    fig = plt.figure(figsize=(7.4, 2.65))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.82], wspace=0.08)

    for idx, side in enumerate(sides):
        ax = fig.add_subplot(grid[0, idx])
        path = IMAGES / f"{tree_id}_{side}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"Missing source image: {path}")

        img = Image.open(path).convert("RGB")
        crop, shifted = crop_around_box(img, boxes[side])
        draw = ImageDraw.Draw(crop)
        x1, y1, x2, y2 = shifted
        for offset in range(4):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=(230, 60, 45))
        draw.rectangle([x1, y1 - 25, x1 + 58, y1], fill=(178, 30, 53))
        draw.text((x1 + 6, y1 - 21), "B3", fill=(255, 255, 255))

        ax.imshow(crop)
        ax.set_title(f"Side {side}", pad=4, fontweight="bold")
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, linewidth=0.9, edgecolor="#d6d6d6"))

    ax_note = fig.add_subplot(grid[0, 3])
    ax_note.set_xlim(0, 1)
    ax_note.set_ylim(0, 1)
    ax_note.axis("off")
    ax_note.text(0.05, 0.86, "Same physical\nbunch", fontsize=10.5, fontweight="bold", color="#222222", va="top")
    ax_note.text(0.05, 0.64, "visible in three\nadjacent side views", fontsize=8.8, color="#444444", va="top")

    summary_box = FancyBboxPatch(
        (0.05, 0.14),
        0.88,
        0.34,
        boxstyle="round,pad=0.025,rounding_size=0.018",
        linewidth=1.0,
        edgecolor="#b21e35",
        facecolor="#fff7f7",
    )
    ax_note.add_patch(summary_box)
    ax_note.text(0.12, 0.40, "Naive\nappearances", fontsize=7.8, color="#555555", va="center")
    ax_note.text(0.84, 0.40, "3", fontsize=13, fontweight="bold", color="#b21e35", ha="right", va="center")
    ax_note.text(0.12, 0.23, "Tree-level\ncount", fontsize=7.8, color="#555555", va="center")
    ax_note.text(0.84, 0.23, "1", fontsize=13, fontweight="bold", color="#2f6b4f", ha="right", va="center")

    save(fig, "fig01_cross_view_linking")


def draw_pipeline_row(
    ax: plt.Axes,
    y: float,
    label: str,
    steps: list[str],
    accent: str,
    note: str,
    highlight_index: int,
) -> None:
    if len(steps) == 4:
        x_positions = [0.15, 0.39, 0.63, 0.82]
        widths = [0.145, 0.145, 0.115, 0.135]
    else:
        x_positions = [0.06, 0.25, 0.44, 0.63, 0.82]
        widths = [0.13, 0.13, 0.13, 0.11, 0.135]
    height = 0.16
    ax.text(0.06, y + height + 0.045, label, ha="left", va="center", fontsize=9.6, fontweight="bold", color=accent)

    for idx, (x, width, text) in enumerate(zip(x_positions, widths, steps)):
        is_highlight = idx == highlight_index
        face = "#f7fbf9" if is_highlight else "#fafafa"
        edge = accent if is_highlight else "#bdbdbd"
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=1.4 if is_highlight else 0.9,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=8.8, color="#222222")

    for idx in range(len(steps) - 1):
        x0 = x_positions[idx] + widths[idx] + 0.012
        x1 = x_positions[idx + 1] - 0.012
        ax.add_patch(FancyArrowPatch((x0, y + height / 2), (x1, y + height / 2), arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#444444"))

    ax.text(0.06, y - 0.065, note, ha="left", va="center", fontsize=8.5, color="#555555")


def figure_detection_conditions() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 2.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.94, "Two detection conditions isolate detector error from counter error", fontsize=11.5, fontweight="bold", color="#222222")

    draw_pipeline_row(
        ax,
        0.62,
        "GT detection",
        ["GT\nannotations", "Tree-level\nfeatures", "Counter", "[B1, B2,\nB3, B4]"],
        "#2f6b4f",
        "Detector output quality is removed as an error source.",
        0,
    )
    draw_pipeline_row(
        ax,
        0.28,
        "Fixed-detector",
        ["Multi-view\nimages", "YOLOv26\noutputs", "Tree-level\nfeatures", "Counter", "[B1, B2,\nB3, B4]"],
        "#5b6fbc",
        "Detector misses and class confusions enter before counting.",
        1,
    )
    save(fig, "fig02_detection_conditions")


def figure_gap_bias() -> None:
    classes = ["B1", "B2", "B3", "B4"]
    gt_acc = [100.00, 99.29, 93.62, 99.29]
    fixed_acc = [96.45, 78.72, 56.03, 78.72]
    fixed_bias = [0.014, -0.078, -0.177, 0.071]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.15), gridspec_kw={"height_ratios": [1.18, 1.0], "hspace": 0.36})
    x = list(range(len(classes)))
    width = 0.34
    gt_bars = ax1.bar([v - width / 2 for v in x], gt_acc, width, label="GT detection", color="#2f6b4f")
    fixed_bars = ax1.bar([v + width / 2 for v in x], fixed_acc, width, label="Fixed-detector", color="#5b6fbc")
    ax1.set_title("Counting is near ceiling with GT detections, but drops under detector outputs", loc="left", pad=8, fontweight="bold")
    ax1.set_ylabel("Class +/-1 Acc (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_ylim(0, 110)
    ax1.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax1.legend(frameon=False, loc="upper right", ncol=2)
    for bars in [gt_bars, fixed_bars]:
        for bar in bars:
            value = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}", ha="center", va="bottom", fontsize=8.3, color="#333333")

    for idx, (g, f) in enumerate(zip(gt_acc, fixed_acc)):
        gap = g - f
        ax1.plot([idx + width / 2, idx + width / 2], [f + 3.5, g - 3.5], color="#9a9a9a", lw=0.8, alpha=0.8)
        if gap >= 15:
            ax1.text(idx + width / 2 + 0.06, (g + f) / 2, f"-{gap:.1f}", va="center", fontsize=8, color="#666666")

    colors = ["#2f6b4f" if v >= 0 else "#b21e35" for v in fixed_bias]
    bars = ax2.bar(classes, fixed_bias, color=colors, width=0.50)
    ax2.set_title("Fixed-detector bias is directional for B2 and B3", loc="left", pad=8, fontweight="bold")
    ax2.axhline(0, color="#222222", linewidth=1.0)
    ax2.set_ylabel("Mean signed error")
    ax2.set_ylim(-0.24, 0.13)
    ax2.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax2.text(3.55, -0.205, "negative = undercounting", ha="right", va="center", fontsize=8.4, color="#666666")
    for i, value in enumerate(fixed_bias):
        va = "bottom" if value >= 0 else "top"
        y = value + (0.012 if value >= 0 else -0.012)
        ax2.text(i, y, f"{value:+.3f}", ha="center", va=va, fontsize=8)

    save(fig, "fig03_gap_bias")


def main() -> None:
    set_style()
    figure_cross_view_linking()
    figure_detection_conditions()
    figure_gap_bias()
    print(f"Generated figures in {OUT}")


if __name__ == "__main__":
    main()
