"""Generate publication-oriented figures for the paper draft."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "paper"
IMAGES = ROOT / "images"
MPL_CACHE = ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


GT_COLOR = "#2f6b4f"
FIXED_COLOR = "#3a4f9c"
ACCENT_RED = "#b21e35"
INK = "#1f1f1f"
MUTED = "#6b6b6b"


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.2,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#2b2b2b",
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
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


def draw_box_with_halo(crop: Image.Image, box: list[int]) -> Image.Image:
    """Draw a clearly visible bbox with a soft white halo so it reads on dark bunches."""
    x1, y1, x2, y2 = box
    overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    halo = ImageDraw.Draw(overlay)
    # White halo: draw a thick white rectangle outline then a coloured one on top
    for offset in range(6, 2, -1):
        halo.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline=(255, 255, 255, 180),
            width=1,
        )
    overlay = overlay.filter(ImageFilter.GaussianBlur(1.0))
    crop = crop.convert("RGBA")
    crop.alpha_composite(overlay)
    draw = ImageDraw.Draw(crop)
    for offset in range(3):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline=(255, 235, 59),  # bright yellow for high contrast
            width=1,
        )
    # Inner red core
    draw.rectangle([x1, y1, x2, y2], outline=(230, 30, 50), width=2)
    return crop.convert("RGB")


def figure_cross_view_linking() -> None:
    tree_id = "DAMIMAS_A21B_0847"
    sides = [1, 2, 3]
    boxes = {
        1: [304, 531, 409, 681],
        2: [292, 503, 485, 682],
        3: [537, 435, 683, 600],
    }
    ribbon_color = "#3a4f9c"

    fig = plt.figure(figsize=(7.4, 3.55))
    grid = fig.add_gridspec(
        2, 3,
        height_ratios=[0.13, 1.0],
        width_ratios=[1, 1, 1],
        hspace=0.04,
        wspace=0.05,
    )

    # Header ribbon — one ribbon per crop, indicating "Side n"
    for idx, side in enumerate(sides):
        ax_ribbon = fig.add_subplot(grid[0, idx])
        ax_ribbon.set_xlim(0, 1)
        ax_ribbon.set_ylim(0, 1)
        ax_ribbon.axis("off")
        ribbon = FancyBboxPatch(
            (0.04, 0.18), 0.92, 0.64,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=0, facecolor=ribbon_color,
        )
        ax_ribbon.add_patch(ribbon)
        ax_ribbon.text(
            0.5, 0.5, f"Side {side}",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white",
        )

    # Image crops
    crop_axes = []
    for idx, side in enumerate(sides):
        ax = fig.add_subplot(grid[1, idx])
        path = IMAGES / f"{tree_id}_{side}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"Missing source image: {path}")
        img = Image.open(path).convert("RGB")
        crop, shifted = crop_around_box(img, boxes[side])
        crop = draw_box_with_halo(crop, shifted)
        ax.imshow(crop)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#d6d6d6")
            spine.set_linewidth(0.8)
        crop_axes.append(ax)

    # Reserve space at the bottom for the connector + caption, then draw overlays.
    fig.subplots_adjust(bottom=0.18, top=0.97, left=0.025, right=0.975)
    fig.canvas.draw()

    line_ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    line_ax.set_xlim(0, 1)
    line_ax.set_ylim(0, 1)
    line_ax.set_xticks([])
    line_ax.set_yticks([])
    line_ax.patch.set_alpha(0)

    # Anchor points just below each crop axis
    anchors = []
    for ax in crop_axes:
        bbox = ax.get_position()
        cx = (bbox.x0 + bbox.x1) / 2
        cy = bbox.y0
        anchors.append((cx, cy))

    # Connector band Y position — well below the crop bottoms but above the caption.
    connector_y = anchors[0][1] - 0.07
    caption_y = connector_y - 0.05

    # Vertical stems from each crop bottom down to the connector line.
    for x, y in anchors:
        line_ax.plot(
            [x, x], [y - 0.005, connector_y],
            color=ACCENT_RED, linewidth=1.4, linestyle=(0, (2.5, 2.5)),
            solid_capstyle="round",
        )
        line_ax.plot(
            x, connector_y, marker="o",
            markersize=5, markerfacecolor=ACCENT_RED, markeredgecolor="white", markeredgewidth=1.0,
        )

    # Horizontal line connecting the three stems.
    line_ax.plot(
        [anchors[0][0], anchors[-1][0]],
        [connector_y, connector_y],
        color=ACCENT_RED, linewidth=1.6, linestyle="-",
        solid_capstyle="round",
    )

    cap_x = (anchors[0][0] + anchors[-1][0]) / 2
    line_ax.text(
        cap_x, caption_y,
        "same physical B3 bunch — three appearances, one tree-level count",
        ha="center", va="top",
        fontsize=9.4, color=ACCENT_RED, fontweight="bold",
    )

    fig.savefig(OUT / "fig01_cross_view_linking.png", dpi=320, facecolor="white")
    fig.savefig(OUT / "fig01_cross_view_linking.pdf", facecolor="white")
    plt.close(fig)


def draw_pipeline_row(
    ax: plt.Axes,
    y_center: float,
    height: float,
    label: str,
    steps: list[str],
    color: str,
    tint: str,
) -> None:
    """Right-edge aligned pipeline row with a tinted background band."""
    # Background band
    band = FancyBboxPatch(
        (0.005, y_center - height / 2 - 0.04),
        0.99, height + 0.08,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=0.8, edgecolor=color, facecolor=tint,
    )
    ax.add_patch(band)

    # Row label
    ax.text(
        0.025, y_center, label,
        ha="left", va="center",
        fontsize=10, fontweight="bold", color=color,
    )

    # Stage layout: right-edge aligned. Last stage is the output block.
    label_left_pad = 0.16
    stage_w = 0.135
    stage_gap = 0.018
    # Right edge at x=0.985; build from the right.
    right_edge = 0.985
    n = len(steps)
    xs = []
    for i in range(n - 1, -1, -1):
        x_right = right_edge - (n - 1 - i) * (stage_w + stage_gap)
        xs.append(x_right - stage_w)
    xs = list(reversed(xs))

    # Make sure first box does not collide with the row label.
    if xs[0] < label_left_pad + 0.02:
        shift = (label_left_pad + 0.02) - xs[0]
        xs = [x + shift for x in xs]

    for i, (x, text) in enumerate(zip(xs, steps)):
        is_output = i == n - 1
        face = color if is_output else "white"
        edge = color
        text_color = "white" if is_output else INK
        weight = "bold" if is_output else "normal"
        box = FancyBboxPatch(
            (x, y_center - height / 2),
            stage_w, height,
            boxstyle="round,pad=0.012,rounding_size=0.022",
            linewidth=1.2, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(box)
        ax.text(
            x + stage_w / 2, y_center, text,
            ha="center", va="center",
            fontsize=9.0, color=text_color, fontweight=weight,
        )

    for i in range(n - 1):
        x_from = xs[i] + stage_w + 0.001
        x_to = xs[i + 1] - 0.001
        ax.add_patch(FancyArrowPatch(
            (x_from, y_center), (x_to, y_center),
            arrowstyle="-|>", mutation_scale=12,
            lw=1.2, color="#3a3a3a",
        ))


def figure_detection_conditions() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 2.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_pipeline_row(
        ax,
        y_center=0.74,
        height=0.18,
        label="GT detection\nsetting",
        steps=["GT\nannotations", "Tree-level\nfeatures", "Counter", "[B1, B2,\nB3, B4]"],
        color=GT_COLOR,
        tint="#eef6f1",
    )
    draw_pipeline_row(
        ax,
        y_center=0.26,
        height=0.18,
        label="Fixed-detector\nsetting",
        steps=["Multi-view\nimages", "YOLOv26\noutputs", "Tree-level\nfeatures", "Counter", "[B1, B2,\nB3, B4]"],
        color=FIXED_COLOR,
        tint="#eef0f9",
    )

    # Vertical guide aligning the two output boxes
    ax.plot(
        [0.86, 0.86], [0.05, 0.95],
        linestyle=(0, (2, 3)), color="#bbbbbb", linewidth=0.8,
    )

    save(fig, "fig02_detection_conditions")


def figure_gap_bias() -> None:
    classes = ["B1", "B2", "B3", "B4"]
    # GT setting best: ElasticNet on GT detections, per-class Class +/-1 Acc
    gt_acc = [100.00, 99.29, 93.62, 99.29]
    # Fixed-detector best: Ridge + F_all, per-class Class +/-1 Acc (matches README Table 7 Ridge row)
    fixed_acc = [97.16, 82.98, 57.45, 72.34]
    # Fixed-detector best: Ridge + F_all, per-class signed bias
    fixed_bias = [0.014, -0.078, -0.177, 0.071]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(7.2, 5.2),
        gridspec_kw={"height_ratios": [1.18, 1.0], "hspace": 0.42},
    )

    # ---- Top panel: per-class accuracy gap ----
    x = list(range(len(classes)))
    width = 0.36
    gt_bars = ax1.bar(
        [v - width / 2 for v in x], gt_acc, width,
        label="GT detection (ElasticNet)", color=GT_COLOR, edgecolor=GT_COLOR,
    )
    fixed_bars = ax1.bar(
        [v + width / 2 for v in x], fixed_acc, width,
        label="Fixed-detector (Ridge + F$_{\\mathrm{all}}$)", color=FIXED_COLOR, edgecolor=FIXED_COLOR,
    )

    ax1.set_title(
        "Counting is near ceiling under GT detection, but drops under fixed-detector outputs",
        loc="left", pad=22, fontweight="bold",
    )
    ax1.set_ylabel("Class $\\pm$1 Acc (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_ylim(0, 108)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.axhline(100, color="#999999", linestyle=(0, (1.5, 2)), linewidth=0.8)
    ax1.text(3.55, 101.5, "ceiling", ha="right", va="bottom", fontsize=8.4, color=MUTED)
    ax1.grid(axis="y", alpha=0.20, linewidth=0.6)

    # Legend above plot, single row
    ax1.legend(
        frameon=False, loc="lower left",
        bbox_to_anchor=(0.0, 1.01), ncol=2, handlelength=1.6, columnspacing=1.2,
    )

    for bars in (gt_bars, fixed_bars):
        for bar in bars:
            v = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2, v + 1.4,
                f"{v:.1f}",
                ha="center", va="bottom",
                fontsize=8.6, color=INK,
            )

    # Annotate the largest gap (B3) with a clean bracket and label
    gap_b3 = gt_acc[2] - fixed_acc[2]
    bracket_x = 2 + 0.34  # just right of B3 group
    top_y = gt_acc[2]
    bot_y = fixed_acc[2]
    ax1.plot([bracket_x, bracket_x], [bot_y, top_y], color=ACCENT_RED, linewidth=1.2)
    ax1.plot([bracket_x - 0.04, bracket_x], [top_y, top_y], color=ACCENT_RED, linewidth=1.2)
    ax1.plot([bracket_x - 0.04, bracket_x], [bot_y, bot_y], color=ACCENT_RED, linewidth=1.2)
    ax1.text(
        bracket_x + 0.05, (top_y + bot_y) / 2,
        f"$-${gap_b3:.1f} pp",
        fontsize=9.2, color=ACCENT_RED, fontweight="bold",
        ha="left", va="center",
    )

    # ---- Bottom panel: per-class signed bias ----
    colors = [GT_COLOR if v >= 0 else ACCENT_RED for v in fixed_bias]
    bias_bars = ax2.bar(x, fixed_bias, color=colors, width=0.44, edgecolor=colors)
    ax2.set_title(
        "Fixed-detector under-counts B2 and B3 systematically",
        loc="left", pad=8, fontweight="bold",
    )
    ax2.axhline(0, color=INK, linewidth=1.0)
    ax2.set_ylabel("Mean signed error (bunches)")
    ax2.set_ylim(-0.24, 0.13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.grid(axis="y", alpha=0.20, linewidth=0.6)

    for i, value in enumerate(fixed_bias):
        va = "bottom" if value >= 0 else "top"
        y_offset = 0.010 if value >= 0 else -0.010
        ax2.text(
            i, value + y_offset, f"{value:+.3f}",
            ha="center", va=va, fontsize=8.6, color=INK,
        )

    # In-axis legend chip explaining sign
    ax2.text(
        3.42, -0.215, "negative = undercount",
        ha="right", va="center", fontsize=8.4, color=MUTED, style="italic",
    )

    save(fig, "fig03_gap_bias")


def main() -> None:
    set_style()
    figure_cross_view_linking()
    figure_detection_conditions()
    figure_gap_bias()
    print(f"Generated figures in {OUT}")


if __name__ == "__main__":
    main()
