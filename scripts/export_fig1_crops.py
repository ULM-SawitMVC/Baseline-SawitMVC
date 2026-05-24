"""Export the three Fig.1 crops (with bbox + halo) as standalone PNGs
so they can be styled externally (Figma, Gemini, etc.)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
OUT = ROOT / "figures" / "paper" / "src" / "fig01_crops"
OUT.mkdir(parents=True, exist_ok=True)

TREE_ID = "DAMIMAS_A21B_0847"
SIDES = [1, 2, 3]
BOXES = {
    1: [304, 531, 409, 681],
    2: [292, 503, 485, 682],
    3: [537, 435, 683, 600],
}
CROP_SIZE = 520


def crop_around_box(img: Image.Image, box: list[int], crop_size: int = CROP_SIZE):
    width, height = img.size
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    left = max(0, min(int(round(cx - crop_size / 2)), width - crop_size))
    top = max(0, min(int(round(cy - crop_size / 2)), height - crop_size))
    right = min(width, left + crop_size)
    bottom = min(height, top + crop_size)
    crop = img.crop((left, top, right, bottom))
    shifted = [x1 - left, y1 - top, x2 - left, y2 - top]
    return crop, shifted


def draw_box_with_halo(crop: Image.Image, box: list[int]) -> Image.Image:
    x1, y1, x2, y2 = box
    overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    halo = ImageDraw.Draw(overlay)
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
            outline=(255, 235, 59),
            width=1,
        )
    draw.rectangle([x1, y1, x2, y2], outline=(230, 30, 50), width=2)
    return crop.convert("RGB")


def main() -> None:
    for side in SIDES:
        src = IMAGES / f"{TREE_ID}_{side}.jpg"
        full = Image.open(src).convert("RGB")

        # Annotated full-resolution version (bbox drawn on the original image).
        full_annot = draw_box_with_halo(full.copy(), BOXES[side])
        full_annot.save(OUT / f"side{side}_full.png")

        # Cropped + annotated 520x520 version (what Fig.1 actually shows).
        crop, shifted = crop_around_box(full, BOXES[side])
        crop_annot = draw_box_with_halo(crop, shifted)
        crop_annot.save(OUT / f"side{side}_crop.png")

        # Plain crop without any annotation, for fully custom styling later.
        crop_plain, _ = crop_around_box(full, BOXES[side])
        crop_plain.save(OUT / f"side{side}_crop_plain.png")

        print(f"side {side}: bbox={BOXES[side]}  ->  3 files saved")

    print(f"\nAll outputs written to {OUT}")


if __name__ == "__main__":
    main()
