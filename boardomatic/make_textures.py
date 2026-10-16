"""Builds the face textures for the boardomatic from the original app icons.

icons/ holds the real app icons (uploaded to the Maksym project in Magnific).
For every rival app this writes three states of the slab face:
  <name>_clean.png, <name>_cracked.png, <name>_glitch.png
plus genora_top.png for the top of the Genora vehicle.

Run with system Python + Pillow:  python3 make_textures.py <icons_dir> <out_dir>
"""
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

SIZE = 1024
RIVALS = ["chatgpt", "claude", "gemini", "deepseek", "grok"]
DARK_ICONS = {"grok", "genora"}


def flat_icon(path):
    """Icon with its transparent rounded corners filled by its own background color."""
    icon = Image.open(path).convert("RGBA").resize((SIZE, SIZE), Image.LANCZOS)
    bg = icon.getpixel((SIZE // 2, 30))[:3]
    img = Image.new("RGBA", (SIZE, SIZE), bg + (255,))
    img.alpha_composite(icon)
    return img


def crack_layer(seed, dark):
    rnd = random.Random(seed)
    layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = (40, 30, 28, 235) if dark else (230, 225, 220, 235)
    origin = (SIZE * rnd.uniform(0.35, 0.65), SIZE * rnd.uniform(0.3, 0.6))
    for _ in range(9):
        ang = rnd.uniform(0, 2 * math.pi)
        x, y = origin
        w = 10
        for _ in range(rnd.randint(5, 9)):
            ang += rnd.uniform(-0.6, 0.6)
            step = rnd.uniform(50, 130)
            nx, ny = x + math.cos(ang) * step, y + math.sin(ang) * step
            d.line([(x, y), (nx, ny)], fill=col, width=max(2, int(w)))
            if rnd.random() < 0.35:
                b_ang = ang + rnd.choice([-1, 1]) * rnd.uniform(0.6, 1.2)
                d.line([(nx, ny), (nx + math.cos(b_ang) * step * 0.6,
                                   ny + math.sin(b_ang) * step * 0.6)],
                       fill=col, width=max(2, int(w * 0.6)))
            x, y = nx, ny
            w *= 0.82
    return layer


def glitch(img, seed):
    rnd = random.Random(seed)
    r, g, b, a = img.convert("RGBA").split()
    r = ImageChops.offset(r, 18, 0)
    b = ImageChops.offset(b, -14, 4)
    out = Image.merge("RGBA", (r, g, b, a))
    for _ in range(14):  # displaced horizontal slices
        y0 = rnd.randint(0, SIZE - 40)
        h = rnd.randint(8, 60)
        band = out.crop((0, y0, SIZE, y0 + h))
        out.paste(ImageChops.offset(band, rnd.randint(-90, 90), 0), (0, y0))
    lines = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lines)
    for y in range(0, SIZE, 8):
        ld.line([(0, y), (SIZE, y)], fill=(0, 0, 0, 40), width=2)
    out.alpha_composite(lines)
    d = ImageDraw.Draw(out)
    for _ in range(10):
        x0, y0 = rnd.randint(0, SIZE), rnd.randint(0, SIZE)
        d.rectangle([x0, y0, x0 + rnd.randint(30, 160), y0 + rnd.randint(6, 20)],
                    fill=rnd.choice([(0, 255, 255, 200), (255, 0, 200, 200), (255, 255, 255, 220)]))
    return out


def main():
    icons, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(RIVALS):
        clean = flat_icon(icons / f"{name}.png")
        clean.convert("RGB").save(out / f"{name}_clean.png")
        cracked = clean.copy()
        cracked.alpha_composite(crack_layer(100 + i, dark=name not in DARK_ICONS))
        cracked.convert("RGB").save(out / f"{name}_cracked.png")
        glitch(cracked, 200 + i).convert("RGB").save(out / f"{name}_glitch.png")
    flat_icon(icons / "genora.png").convert("RGB").save(out / "genora_top.png")


if __name__ == "__main__":
    main()
