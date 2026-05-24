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


def figure_detection_conditions() -> None:
    """Column-aligned two-row pipeline. Both rows share the downstream
    'features -> counter -> output' columns; only the detection-input zone
    differs (GT oracle box vs. images + YOLO sub-stages)."""
    fig, ax = plt.subplots(figsize=(7.9, 2.95))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    label_left = 0.015
    label_w = 0.195
    detect_w = 0.275
    stage_w = 0.128
    gap = 0.022
    row_h = 0.28
    band_pad = 0.05

    x_detect = label_left + label_w
    x_feat = x_detect + detect_w + gap
    x_cnt = x_feat + stage_w + gap
    x_out = x_cnt + stage_w + gap
    band_right = x_out + stage_w + 0.01

    def draw_stage(x, y, w, h, text, edge, *, fill="white", text_color=INK, weight="normal", fontsize=9.0):
        box = FancyBboxPatch(
            (x, y - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.022",
            linewidth=1.2, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y, text,
                ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight)

    def draw_arrow(x_from, y, x_to):
        ax.add_patch(FancyArrowPatch(
            (x_from + 0.002, y), (x_to - 0.002, y),
            arrowstyle="-|>", mutation_scale=12,
            lw=1.2, color="#3a3a3a",
        ))

    def draw_row(y, label, color, tint, detect_render):
        band = FancyBboxPatch(
            (label_left - 0.005, y - row_h / 2 - band_pad),
            band_right - (label_left - 0.005),
            row_h + 2 * band_pad,
            boxstyle="round,pad=0,rounding_size=0.018",
            linewidth=0.8, edgecolor=color, facecolor=tint,
        )
        ax.add_patch(band)
        ax.text(label_left + 0.008, y, label,
                ha="left", va="center",
                fontsize=10, fontweight="bold", color=color)
        detect_render(y, color)
        draw_arrow(x_detect + detect_w, y, x_feat)
        draw_stage(x_feat, y, stage_w, row_h, "Tree-level\nfeatures", color)
        draw_arrow(x_feat + stage_w, y, x_cnt)
        draw_stage(x_cnt, y, stage_w, row_h, "Counter", color)
        draw_arrow(x_cnt + stage_w, y, x_out)
        draw_stage(x_out, y, stage_w, row_h, "[B1, B2,\nB3, B4]", color,
                   fill=color, text_color="white", weight="bold")

    def gt_detect(y, color):
        draw_stage(x_detect, y, detect_w, row_h,
                   "GT boxes & classes\n(oracle annotations)", color)

    def fixed_detect(y, color):
        sub_gap = 0.018
        sub_w = (detect_w - sub_gap) / 2
        x1 = x_detect
        x2 = x_detect + sub_w + sub_gap
        draw_stage(x1, y, sub_w, row_h, "Multi-view\nimages", color, fontsize=8.8)
        draw_arrow(x1 + sub_w, y, x2)
        draw_stage(x2, y, sub_w, row_h, "YOLOv26\noutputs", color, fontsize=8.8)

    draw_row(0.72, "GT detection\nsetting",
             GT_COLOR, "#eef6f1", gt_detect)
    draw_row(0.24, "Fixed-detector\nsetting",
             FIXED_COLOR, "#eef0f9", fixed_detect)

    # Subtle shared-downstream marker spanning the three rightmost columns.
    shared_left = x_feat - 0.005
    shared_right = x_out + stage_w + 0.005
    bracket_y = 0.985
    ax.plot([shared_left, shared_right], [bracket_y, bracket_y],
            color="#9a9a9a", linewidth=0.8)
    ax.plot([shared_left, shared_left], [bracket_y - 0.012, bracket_y],
            color="#9a9a9a", linewidth=0.8)
    ax.plot([shared_right, shared_right], [bracket_y - 0.012, bracket_y],
            color="#9a9a9a", linewidth=0.8)
    ax.text((shared_left + shared_right) / 2, bracket_y + 0.018,
            "identical downstream pipeline",
            ha="center", va="bottom",
            fontsize=8.4, color=MUTED, fontstyle="italic")

    save(fig, "fig02_detection_conditions")


def figure_gap_bias() -> None:
    classes = ["B1", "B2", "B3", "B4"]
    # GT setting best: ElasticNet on GT detections, per-class Class +/-1 Acc
    gt_acc = [100.00, 99.29, 93.62, 99.29]
    # Fixed-detector best: Ridge + F_all, per-class Class +/-1 Acc (matches README Table 7 Ridge row)
    fixed_acc = [97.16, 82.98, 57.45, 72.34]
    # Fixed-detector best: Ridge + F_all, per-class signed bias
    fixed_bias = [0.014, -0.078, -0.177, 0.071]
    gaps = [gt - fx for gt, fx in zip(gt_acc, fixed_acc)]
    max_gap_idx = gaps.index(max(gaps))

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(7.4, 5.7),
        gridspec_kw={"height_ratios": [1.30, 0.95], "hspace": 0.50},
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
        "Counting is near ceiling under GT detection, but drops under the fixed detector",
        loc="left", pad=24, fontweight="bold",
    )
    ax1.set_ylabel("Class $\\pm$1 Acc (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_xlim(-0.62, 3.62)
    ax1.set_ylim(0, 128)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.axhline(100, color="#999999", linestyle=(0, (1.5, 2)), linewidth=0.8)
    ax1.annotate(
        "100% ceiling",
        xy=(3.62, 100), xycoords=("data", "data"),
        xytext=(6, 0), textcoords="offset points",
        ha="left", va="center",
        fontsize=8.0, color=MUTED, fontstyle="italic",
        annotation_clip=False,
    )
    ax1.grid(axis="y", alpha=0.20, linewidth=0.6)

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

    # Per-class GT - Fixed drop row, set above the bars so it never collides
    # with neighbouring bar value labels.
    gap_y = 118
    ax1.text(-0.58, gap_y, "drop:", ha="left", va="center",
             fontsize=8.4, color="#555", fontstyle="italic")
    for i, gap in enumerate(gaps):
        is_max = i == max_gap_idx
        color = ACCENT_RED if is_max else "#666666"
        weight = "bold" if is_max else "normal"
        size = 9.4 if is_max else 8.6
        face = "#fdecef" if is_max else "white"
        edge = ACCENT_RED if is_max else "#cccccc"
        ax1.text(
            i, gap_y, f"−{gap:.1f} pp",
            ha="center", va="center",
            fontsize=size, color=color, fontweight=weight,
            bbox=dict(boxstyle="round,pad=0.30", facecolor=face, edgecolor=edge, linewidth=0.7),
        )

    # ---- Bottom panel: per-class signed bias ----
    bar_w = 0.52
    colors = [GT_COLOR if v >= 0 else ACCENT_RED for v in fixed_bias]
    ax2.bar(x, fixed_bias, color=colors, width=bar_w, edgecolor=colors)
    ax2.set_title(
        "Fixed-detector under-counts B2 and especially B3",
        loc="left", pad=8, fontweight="bold",
    )
    ax2.axhline(0, color=INK, linewidth=1.0)
    ax2.set_ylabel("Mean signed error (bunches)")
    ax2.set_ylim(-0.24, 0.13)
    ax2.set_xlim(-0.62, 3.62)
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

    ax2.text(
        3.55, -0.215, "negative = undercount",
        ha="right", va="center", fontsize=8.4, color=MUTED, fontstyle="italic",
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
