"""Remove a solid-colour background and save a transparent PNG.

The Auto Animate cut-out tool, step 1. You generate a picture with AI
on a solid background (e.g. an insect on magenta), and this tool cuts
that background colour out, leaving the subject on transparency.

How it works (chroma key)
-------------------------
Every pixel's colour distance from the key colour is measured. Pixels
close to the key colour become transparent, pixels far from it stay
opaque, and pixels in between get a partial alpha so edges look soft
instead of jagged. AI "solid" backgrounds are never perfectly flat —
compression and shading leave noise — so `--tolerance` sets how close
a pixel must be to count as background, and `--feather` softens the
edge over a few colour-distance units. Isolated specks (a stray
background-coloured dot inside the subject, or a subject-coloured
speck in the background) are cleaned up by keeping the largest
connected piece of the subject.

Usage
-----
    python remove_background.py insect.png --color magenta --out insect_cut.png
    python remove_background.py photo.png --color "#00FF00" --tolerance 60

    from remove_background import remove_background
    cut = remove_background(img, key_color="magenta")  # BGRA image

Notes
-----
- `--color` accepts a name (magenta, green, blue, white, black, red,
  yellow, cyan) or a hex code (#FF00FF or FF00FF).
- Pick a background colour the subject does NOT contain: a magenta
  insect on a magenta background cannot be separated — that is a
  property of the picture, not the tool. That is why green and
  magenta screens are the usual choices.
- Output is always a PNG with an alpha channel (BGRA). JPEG cannot
  hold transparency, so the output format is fixed to PNG.
"""

import numpy as np

NAMED_COLORS = {
    # name -> (B, G, R), OpenCV order
    "magenta": (255, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
}


def parse_color(color):
    """A colour name or hex code -> (B, G, R) tuple of ints."""
    if isinstance(color, (tuple, list)) and len(color) == 3:
        return tuple(int(c) for c in color)
    if not isinstance(color, str):
        raise ValueError(f"cannot parse colour {color!r}")
    name = color.strip().lower()
    if name in NAMED_COLORS:
        return NAMED_COLORS[name]
    hexcode = name.lstrip("#")
    if len(hexcode) == 6:
        try:
            r = int(hexcode[0:2], 16)
            g = int(hexcode[2:4], 16)
            b = int(hexcode[4:6], 16)
        except ValueError:
            raise ValueError(f"bad hex colour {color!r}") from None
        return (b, g, r)
    raise ValueError(
        f"unknown colour {color!r}; use a name "
        f"({', '.join(sorted(NAMED_COLORS))}) or a hex code like #FF00FF")


def _keep_largest_component(mask):
    """Keep only the largest connected white region of a binary mask."""
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest)


def remove_background(img, key_color="magenta", tolerance=40.0,
                      feather=2.0, keep_largest=True):
    """Cut the key colour out of `img`, returning a BGRA image.

    `img` is a BGR (or BGRA — its alpha is ignored) image. `tolerance`
    is the colour distance below which a pixel counts as background;
    `feather` is the distance over which the alpha fades from 0 to
    255, for soft edges. Returns a BGRA image: the subject opaque,
    the background transparent.
    """
    import cv2

    bgr = img[:, :, :3].astype(np.float32)
    key = np.array(parse_color(key_color), dtype=np.float32)
    dist = np.sqrt(((bgr - key) ** 2).sum(axis=2))

    # Alpha: 0 below the tolerance, fading to 255 over `feather`.
    if feather <= 0:
        alpha = np.where(dist <= tolerance, 0.0, 255.0)
    else:
        alpha = np.clip((dist - tolerance) / feather, 0.0, 1.0) * 255.0
    alpha = alpha.astype(np.uint8)

    if keep_largest:
        subject = _keep_largest_component((alpha > 0).astype(np.uint8) * 255)
        # Specks outside the subject become fully transparent; the
        # subject's own alpha (including feathered edges) is kept.
        alpha = np.where(subject, alpha, 0).astype(np.uint8)

    # Despeckle the alpha edge slightly so compression noise in the
    # background does not leave a fringe of half-transparent dots.
    alpha = cv2.medianBlur(alpha, 3)

    bgra = np.dstack([img[:, :, 0], img[:, :, 1], img[:, :, 2], alpha])
    return bgra


def subject_fraction(bgra):
    """Fraction of pixels that are (mostly) opaque — for reporting."""
    alpha = bgra[:, :, 3]
    return float((alpha > 127).mean())


if __name__ == "__main__":
    import argparse

    import cv2

    ap = argparse.ArgumentParser(
        description="Remove a solid-colour background; save a "
                    "transparent PNG.")
    ap.add_argument("input", help="input image (any format)")
    ap.add_argument("--color", default="magenta",
                    help="background colour to remove: a name "
                         "(magenta, green, ...) or hex (#FF00FF). "
                         "Default: magenta")
    ap.add_argument("--tolerance", type=float, default=40.0,
                    help="colour distance below which a pixel counts as "
                         "background (default 40). Raise it if the "
                         "background is noisy; lower it if the subject "
                         "is being eaten.")
    ap.add_argument("--feather", type=float, default=2.0,
                    help="soft-edge width in colour-distance units "
                         "(default 2)")
    ap.add_argument("--no-keep-largest", action="store_true",
                    help="do not drop isolated specks outside the subject")
    ap.add_argument("--out", default=None,
                    help="output PNG; default <input>_cut.png")
    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {args.input}")
    cut = remove_background(img, key_color=args.color,
                            tolerance=args.tolerance, feather=args.feather,
                            keep_largest=not args.no_keep_largest)
    out = args.out
    if out is None:
        stem = args.input.rsplit(".", 1)[0]
        out = f"{stem}_cut.png"
    if not out.lower().endswith(".png"):
        out += ".png"
    cv2.imwrite(out, cut)
    print(f"background {args.color!r} removed -> {out}")
    print(f"subject covers {100 * subject_fraction(cut):.1f}% of the image")
