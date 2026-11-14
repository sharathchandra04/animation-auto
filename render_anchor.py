"""Render a list of image assets onto a screen — the anchored render.

A second version of render.py. The first render hooks every image
by its bottom-left corner; this one lets you choose the hook —
the anchor point — for each image. The image's position (x, y) is
where its anchor point sits on the grid.

The six anchor options
----------------------
- "bottom-left": the hook is the bottom-left corner (same as the
  first render).
- "bottom-right": the bottom-right corner.
- "top-left": the top-left corner.
- "top-right": the top-right corner.
- "center": the centre of the image.
- "custom": a point you define, given as fractions of the image
  size from the bottom-left corner: anchor_fx across (0 = left
  edge, 1 = right edge) and anchor_fy up (0 = bottom edge,
  1 = top edge). For example (0.25, 0.75) is a point a quarter
  across and three quarters up. (In a future app interface this
  is the point you would click on the image; here you give the
  numbers.)

The screen is the same grid as the first render: origin at the
bottom-left corner, x to the right, y going up. Each asset also
has a z-factor, highest z on top, same as before.

Usage
-----
A scene is a JSON file with a list of assets:

    {
      "width": 1000, "height": 800,
      "assets": [
        {"image": "insect.png", "x": 500, "y": 400, "z": 0,
         "anchor": "center"},
        {"image": "insect.png", "x": 200, "y": 200, "z": 1,
         "anchor": "custom", "anchor_fx": 0.25, "anchor_fy": 0.75}
      ]
    }

    python render_anchor.py scene.json --out screen.png
    python render_anchor.py scene.json --background white

From Python:

    from render_anchor import AnchoredAsset, render_anchored_assets
    screen = render_anchored_assets([
        AnchoredAsset("insect.png", x=500, y=400, z=0, anchor="center"),
        AnchoredAsset("insect.png", x=200, y=200, z=1,
                      anchor="custom", anchor_fx=0.25, anchor_fy=0.75),
    ], width=1000, height=800)

Notes
-----
- An asset with no "anchor" in the scene is "bottom-left", so a
  scene made for the first render works here unchanged.
- Assets are clipped to the screen and transparent PNGs blend,
  same as the first render.
"""

from dataclasses import dataclass

import numpy as np

from render import Asset, _paste

ANCHORS = ("top-left", "top-right", "bottom-left", "bottom-right",
           "center", "custom")


@dataclass
class AnchoredAsset:
    """One image on the screen, placed by its anchor point.

    `image` is a file path or an image array (BGR or BGRA). `x`,
    `y` is where the anchor point sits on the grid. `z` is the
    draw order: highest z ends up on top. `anchor` is one of the
    six options; for "custom", `anchor_fx` and `anchor_fy` give
    the point as fractions of the image width/height from the
    bottom-left corner.
    """
    image: object
    x: int = 0
    y: int = 0
    z: int = 0
    anchor: str = "bottom-left"
    anchor_fx: float = None
    anchor_fy: float = None

    def load(self):
        return Asset(self.image).load()


def anchor_offset(anchor, w, h, fx=None, fy=None):
    """Offset of the anchor from the image's bottom-left corner.

    In grid units (pixels, y up): the image covers
    [x - ax, x + w - ax] by [y - ay, y + h - ay] when its anchor
    sits at (x, y).
    """
    if anchor == "bottom-left":
        return (0, 0)
    if anchor == "bottom-right":
        return (w, 0)
    if anchor == "top-left":
        return (0, h)
    if anchor == "top-right":
        return (w, h)
    if anchor == "center":
        return (w / 2.0, h / 2.0)
    if anchor == "custom":
        if fx is None or fy is None:
            raise ValueError(
                "a custom anchor needs anchor_fx and anchor_fy")
        if not (0 <= fx <= 1 and 0 <= fy <= 1):
            raise ValueError(
                "anchor_fx and anchor_fy must be between 0 and 1")
        return (fx * w, fy * h)
    raise ValueError(f"unknown anchor {anchor!r}; use one of {ANCHORS}")


def render_anchored_assets(assets, width, height, background=None):
    """Render a list of `AnchoredAsset`s onto a width x height screen.

    Returns a BGRA image (or BGR if `background` is given). The
    origin is the bottom-left corner of the screen, y grows up.
    Assets are drawn in ascending z order, so the highest z is on
    top. Each asset is placed so its anchor point sits at (x, y).
    """
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    ordered = sorted(assets, key=lambda a: a.z)  # stable: equal z keeps order
    for asset in ordered:
        img = asset.load()
        h, w = img.shape[:2]
        ax, ay = anchor_offset(asset.anchor, w, h,
                               asset.anchor_fx, asset.anchor_fy)
        # Anchor sits at (x, y); the image's bottom-left corner in
        # grid coords is (x - ax, y - ay); convert y-up to pixels.
        left = int(round(asset.x - ax))
        top = int(round(height - (asset.y - ay) - h))
        _paste(canvas, img, left, top)
    if background is not None:
        from add_outline import flatten

        return flatten(canvas, background)
    return canvas


def load_anchored_scene(path):
    """Load a scene JSON file -> (assets, width, height).

    Accepts either {"width", "height", "assets": [...]} or a bare
    list of assets (then width/height come from the caller).
    """
    import json

    data = json.loads(open(path).read())
    if isinstance(data, list):
        return [AnchoredAsset(**a) for a in data], None, None
    assets = [AnchoredAsset(**a) for a in data.get("assets", [])]
    return assets, data.get("width"), data.get("height")


if __name__ == "__main__":
    import argparse

    import cv2

    ap = argparse.ArgumentParser(
        description="Render a list of image assets onto a screen, "
                    "each placed by its anchor point.")
    ap.add_argument("scene", help="scene JSON file (see render_anchor.py)")
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

    assets, sw, sh = load_anchored_scene(args.scene)
    width = args.width or sw or 1920
    height = args.height or sh or 1080
    screen = render_anchored_assets(assets, width, height,
                                    background=args.background)
    cv2.imwrite(args.out, screen)
    print(f"rendered {len(assets)} assets onto {width}x{height} -> {args.out}")
    for a in sorted(assets, key=lambda a: a.z):
        extra = ""
        if a.anchor == "custom":
            extra = f" ({a.anchor_fx}, {a.anchor_fy})"
        print(f"  z={a.z}: {a.image} at ({a.x}, {a.y}) anchor={a.anchor}{extra}")
