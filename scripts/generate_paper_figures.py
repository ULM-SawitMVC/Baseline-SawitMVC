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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def figure_cross_view_linking() -> None:
    tree_id = "DAMIMAS_A21B_0847"
    sides = [1, 2, 3]
    boxes = {
        1: [304, 531, 409, 681],
        2: [292, 503, 485, 682],
        3: [537, 435, 683, 600],
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, side in zip(axes, sides):
        path = IMAGES / f"{tree_id}_{side}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"Missing source image: {path}")

        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = boxes[side]
        for offset in range(4):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=(230, 60, 45))

        ax.imshow(img)
        ax.set_title(f"Side {side}")
        ax.text(
            x1,
            max(0, y1 - 16),
            "Bunch 3, B3",
            color="white",
            fontsize=8,
            bbox={"facecolor": "#b21e35", "edgecolor": "none", "pad": 2},
        )
        ax.axis("off")

    fig.suptitle("Cross-view duplicate visibility of one physical bunch", y=1.02)
    save(fig, "fig01_cross_view_linking")


def pipeline_panel(ax: plt.Axes, title: str, steps: list[str], evidence_note: str, color: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, pad=8)

    box_width = 0.18 if len(steps) == 4 else 0.145
    x_positions = [0.02, 0.28, 0.54, 0.80] if len(steps) == 4 else [0.01, 0.215, 0.42, 0.625, 0.83]
    for x, text in zip(x_positions, steps):
        patch = FancyBboxPatch(
            (x, 0.36),
            box_width,
            0.28,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.2,
            edgecolor=color,
            facecolor="#f7f7f7",
        )
        ax.add_patch(patch)
        ax.text(x + box_width / 2, 0.50, text, ha="center", va="center", fontsize=8.4)

    arrow_pairs = [(x + box_width + 0.01, x_next - 0.01) for x, x_next in zip(x_positions, x_positions[1:])]
    for x0, x1 in arrow_pairs:
        ax.add_patch(FancyArrowPatch((x0, 0.50), (x1, 0.50), arrowstyle="-|>", mutation_scale=13, color="#333333"))

    ax.text(0.50, 0.18, evidence_note, ha="center", va="center", fontsize=8, color="#333333")


def figure_detection_conditions() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    pipeline_panel(
        axes[0],
        "GT detection setting",
        ["GT\nannotations", "Tree-level\nfeatures", "Counter", "[B1, B2,\nB3, B4]"],
        "Measures the ceiling of the tree-level counter.",
        "#2f6b4f",
    )
    pipeline_panel(
        axes[1],
        "Fixed-detector setting",
        ["Multi-view\nimages", "YOLOv26\noutputs", "Tree-level\nfeatures", "Counter", "[B1, B2,\nB3, B4]"],
        "Measures the deployed detector-plus-counter pipeline.",
        "#5b6fbc",
    )
    save(fig, "fig02_detection_conditions")


def figure_gap_bias() -> None:
    classes = ["B1", "B2", "B3", "B4"]
    gt_acc = [100.00, 99.29, 93.62, 99.29]
    fixed_acc = [96.45, 78.72, 56.03, 78.72]
    fixed_bias = [0.014, -0.078, -0.177, 0.071]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.0), gridspec_kw={"height_ratios": [1.2, 1.0]})
    x = list(range(len(classes)))
    width = 0.34
    ax1.bar([v - width / 2 for v in x], gt_acc, width, label="GT detection", color="#2f6b4f")
    ax1.bar([v + width / 2 for v in x], fixed_acc, width, label="Fixed-detector", color="#5b8ebc")
    ax1.set_ylabel("Class +/-1 Acc (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, loc="lower left")

    colors = ["#2f6b4f" if v >= 0 else "#b21e35" for v in fixed_bias]
    ax2.bar(classes, fixed_bias, color=colors, width=0.52)
    ax2.axhline(0, color="#333333", linewidth=0.8)
    ax2.set_ylabel("Fixed-detector bias")
    ax2.set_ylim(-0.24, 0.12)
    ax2.grid(axis="y", alpha=0.25)
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
