"""Build the site icons in static/ from one source image.

    pip install pillow
    python tools/make_icons.py path/to/new-icon.png   # replace the icon
    python tools/make_icons.py                        # rebuild from static/icon.png

The source should be a square PNG of 256 px or more with a transparent background.
"""
import sys
from pathlib import Path

from PIL import Image

STATIC = Path(__file__).resolve().parent.parent / 'static'
SOURCE = STATIC / 'icon.png'


def crop_to_content(image, margin=0.02):
    """Trim transparent padding so the icon fills small sizes such as a 16 px tab icon."""
    alpha = image.getchannel('A').point(lambda a: 255 if a > 24 else 0)  # ignore faint halo noise
    left, top, right, bottom = alpha.getbbox()
    side = max(right - left, bottom - top)
    side += round(side * margin) * 2
    cx, cy = (left + right) / 2, (top + bottom) / 2
    box = (round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))
    square = Image.new('RGBA', (side, side), (0, 0, 0, 0))
    square.paste(image.crop(box), (0, 0))
    return square


def filled_square(icon):
    """For Apple touch icons: iOS rounds the corners itself, so fill the square edge to edge
    with the icon's own background colour instead of leaving transparent corners."""
    tight = crop_to_content(icon, margin=0)
    w, h = tight.size
    background = tight.getpixel((w // 2, max(1, h // 20)))  # inside the shape, near the top edge
    canvas = Image.new('RGBA', tight.size, background[:3] + (255,))
    canvas.alpha_composite(tight)
    return canvas.convert('RGB')


def main():
    STATIC.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        crop_to_content(Image.open(sys.argv[1]).convert('RGBA')).save(SOURCE, optimize=True)
    icon = Image.open(SOURCE).convert('RGBA')

    sizes = (16, 32, 48)
    frames = [icon.resize((s, s), Image.LANCZOS) for s in sizes]
    frames[-1].save(STATIC / 'favicon.ico', sizes=[(s, s) for s in sizes], append_images=frames[:-1])
    icon.resize((192, 192), Image.LANCZOS).save(STATIC / 'icon-192.png', optimize=True)
    filled_square(icon).resize((180, 180), Image.LANCZOS).save(STATIC / 'apple-touch-icon.png', optimize=True)

    for path in sorted(STATIC.glob('*')):
        print(f'{path.name:24} {path.stat().st_size:>7} bytes')


if __name__ == '__main__':
    main()
