"""Render a list of image assets onto a screen — the asset renderer.

The Auto Animate stage. You have a list of assets (images), each
with a position and a z-factor, and this tool displays them all on
one big screen.

The coordinate system
---------------------
The screen is a grid with the origin at the BOTTOM-LEFT corner:
x grows to the right, y grows UP — the positive-positive part of a
Cartesian grid. This is the opposite of image pixels (where y grows
down), so the renderer flips it for you.

Each asset has a hook: the BOTTOM-LEFT corner of the image. The
hook is the point that sits at the asset's (x, y) on the grid, so
the image covers [x, x+width] by [y, y+height] on the screen.

Each asset also has a z-factor: the order in which assets are
drawn. Lowest z is drawn first, highest z last, so the highest z
appears on top. Assets with equal z keep their list order (later
in the list is on top).

Usage
-----
A scene is a JSON file with a list of assets:

    {
      "width": 1000, "height": 800,
      "assets": [
        {"image": "insect.png", "x": 50, "y": 50, "z": 0},
        {"image": "insect_rot45.png", "x": 350, "y": 150, "z": 2}
      ]
    }

    python render.py scene.json --out screen.png
    python render.py scene.json --width 1920 --height 1080 --background white

From Python, the scene is just a list:

    from render import Asset, render_assets
    screen = render_assets([
        Asset("insect.png", x=50, y=50, z=0),
        Asset("insect_rot45.png", x=350, y=150, z=2),
    ], width=1000, height=800)

Notes
-----
- Assets are clipped to the screen: an asset hanging off the edge
  (or at a negative position) is drawn only where it is visible.
- Images without transparency are treated as opaque; transparent
  PNGs keep their alpha, so overlapping assets blend correctly.
- The output is a PNG with an alpha channel unless `--background`
  fills it with a solid colour.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Asset:
    """One image on the screen.

    `image` is a file path or an image array (BGR or BGRA). `x`, `y`
    is where the hook — the bottom-left corner of the image — sits
    on the grid. `z` is the draw order: highest z ends up on top.
    """
    image: object
    x: int = 0
    y: int = 0
    z: int = 0

    def load(self):
        import cv2

        if isinstance(self.image, np.ndarray):
            img = self.image
        else:
            img = cv2.imread(str(self.image), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise SystemExit(f"could not read asset {self.image}")
        if img.shape[2] == 3:
            img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        return img


def _paste(canvas, img, left, top):
    """Alpha-blend `img` onto `canvas` at pixel (left, top).

    Clips to the canvas. Both are BGRA arrays.
    """
    ch, cw = canvas.shape[:2]
    h, w = img.shape[:2]
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + w, cw), min(top + h, ch)
    if x1 <= x0 or y1 <= y0:
        return  # entirely off-screen
    ix0, iy0 = x0 - left, y0 - top
    ix1, iy1 = ix0 + (x1 - x0), iy0 + (y1 - y0)
    src = img[iy0:iy1, ix0:ix1].astype(np.float32)
    dst = canvas[y0:y1, x0:x1].astype(np.float32)
    a = src[:, :, 3:4] / 255.0
    out = src * a + dst * (1.0 - a)
    # The result's alpha is the "over" of the two alphas.
    out[:, :, 3] = src[:, :, 3] + dst[:, :, 3] * (1.0 - a[:, :, 0])
    canvas[y0:y1, x0:x1] = out.astype(np.uint8)


def render_assets(assets, width, height, background=None):
    """Render a list of `Asset`s onto a width x height screen.

    Returns a BGRA image (or BGR if `background` is given). The
    origin is the bottom-left corner of the screen, y grows up.
    Assets are drawn in ascending z order, so the highest z is on
    top.
    """
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    ordered = sorted(assets, key=lambda a: a.z)  # stable: equal z keeps order
    for asset in ordered:
        img = asset.load()
        h, w = img.shape[:2]
        # Hook (bottom-left corner of the image) sits at (x, y) on
        # the grid; convert y-up to pixel rows (y-down).
        left = int(asset.x)
        top = height - int(asset.y) - h
        _paste(canvas, img, left, top)
    if background is not None:
        from add_outline import flatten

        return flatten(canvas, background)
    return canvas


def load_scene(path):
    """Load a scene JSON file -> (assets, width, height).

    Accepts either {"width", "height", "assets": [...]} or a bare
    list of assets (then width/height come from the caller).
    """
    import json

    data = json.loads(open(path).read())
    if isinstance(data, list):
        return [Asset(**a) for a in data], None, None
    assets = [Asset(**a) for a in data.get("assets", [])]
    return assets, data.get("width"), data.get("height")


if __name__ == "__main__":
    import argparse

    import cv2

    ap = argparse.ArgumentParser(
        description="Render a list of image assets onto a screen.")
    ap.add_argument("scene", help="scene JSON file (see render.py docs)")
    ap.add_argument("--width", type=int, default=None,
                    help="screen width in pixels (overrides the scene)")
    ap.add_argument("--height", type=int, default=None,
                    help="screen height in pixels (overrides the scene)")
    ap.add_argument("--background", default=None,
                    help="fill the screen with this solid colour "
                         "(name or hex) instead of transparency")
    ap.add_argument("--out", default="screen.png",
                    help="output PNG (default screen.png)")
    args = ap.parse_args()

    assets, sw, sh = load_scene(args.scene)
    width = args.width or sw or 1920
    height = args.height or sh or 1080
    screen = render_assets(assets, width, height, background=args.background)
    cv2.imwrite(args.out, screen)
    print(f"rendered {len(assets)} assets onto {width}x{height} -> {args.out}")
    for a in sorted(assets, key=lambda a: a.z):
        print(f"  z={a.z}: {a.image} at ({a.x}, {a.y})")
