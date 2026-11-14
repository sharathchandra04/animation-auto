"""Apply a common move to a group of assets — rotate, move, scale.

The Auto Animate group feature. You have several assets on a
screen, and you want to move them all together — as one piece —
around a common pivot point.

The idea
---------
Pick a pivot point P on the grid. Every asset is a distance and
an angle from that pivot, and the line from the pivot to the
asset's hook is its spoke. A common move changes every asset
the same way, measured from the pivot:

- Rotate by an angle: each asset orbits the pivot (its angle
  changes, its distance stays the same) AND the image itself
  spins by the same angle — the group turns as one rigid piece,
  like rotating a photograph. Positive angles are
  counter-clockwise, the same as rotate.py.
- Move by (dx, dy): the pivot and every asset shift together.
- Scale by a factor: every asset's distance from the pivot is
  multiplied by it, and the images are scaled too (unless
  --no-scale-images, which only spreads the assets out).

The pivot can be a point you give ("500,400"), the centre of the
screen ("center"), or the middle of the group ("group" — the
average of the assets' centres).

How the rigid rotation is done
------------------------------
For each asset, its centre C is rotated around the pivot with the
standard rotation matrix, and its image is rotated by the same
angle around its own centre (rotate.py). The rotated image is
then placed so its centre sits at the rotated centre. That is
exactly a rigid rotation of the whole group around the pivot.

Usage
-----
    python group.py scene.json --pivot 500,400 --rotate 30
    python group.py scene.json --pivot center --rotate -45 --out scene2.json
    python group.py scene.json --pivot group --rotate 90 --dx 50 --dy -20
    python group.py scene.json --pivot 500,400 --scale 1.5

The output is a new scene JSON (default <scene>_group.json) with
the assets' new positions, and the rotated/scaled images are
saved next to the originals (e.g. insect_rot30.png). The screen
size and z-factors are unchanged, and assets that rotate off the
screen are clipped by the renderer as usual.
"""

import math
import os

import numpy as np

from render import Asset, load_scene
from rotate import rotate_image
from scale import scale_image


def rotate_point(px, py, cx, cy, angle_deg):
    """Rotate point (px, py) around centre (cx, cy) by angle_deg.

    Positive angles are counter-clockwise on the grid (y up).
    Returns the new (x, y) as floats.
    """
    t = math.radians(angle_deg)
    dx, dy = px - cx, py - cy
    return (cx + dx * math.cos(t) - dy * math.sin(t),
            cy + dx * math.sin(t) + dy * math.cos(t))


def resolve_pivot(pivot, assets, width=None, height=None):
    """Turn a pivot spec into (px, py).

    Accepts "x,y", "center" (the middle of the screen), or "group"
    (the average of the assets' centres).
    """
    if pivot == "center":
        if width is None or height is None:
            raise ValueError("--pivot center needs a scene with width/height")
        return (width / 2.0, height / 2.0)
    if pivot == "group":
        centres = [asset_centre(a) for a in assets]
        return (sum(c[0] for c in centres) / len(centres),
                sum(c[1] for c in centres) / len(centres))
    try:
        x, y = pivot.split(",")
        return (float(x), float(y))
    except ValueError:
        raise ValueError(
            "--pivot must be 'x,y', 'center', or 'group'") from None


def asset_centre(asset):
    """The centre of an asset on the grid, from its hook (x, y)."""
    img = asset.load()
    h, w = img.shape[:2]
    return (asset.x + w / 2.0, asset.y + h / 2.0)


def transform_group(assets, pivot, angle=0.0, dx=0, dy=0, scale=1.0,
                    scale_images=True, out_dir=None):
    """Apply a common move to `assets` around `pivot`.

    `pivot` is (px, py). Returns new Asset objects: each asset's
    centre is moved to pivot + (dx, dy) + scale * R(angle) *
    (centre - pivot), and its image is rotated by `angle` and
    scaled by `scale` when needed. Images that changed are saved
    next to the originals (or in `out_dir`) and the new assets
    point at them.
    """
    px, py = pivot
    t = math.radians(angle)
    cos_t, sin_t = math.cos(t), math.sin(t)
    new_assets = []
    for asset in assets:
        img = asset.load()
        h, w = img.shape[:2]
        cx, cy = asset.x + w / 2.0, asset.y + h / 2.0
        # Centre after the common move: translate the pivot, scale
        # the offset from it, rotate the offset.
        ox, oy = cx - px, cy - py
        ncx = px + dx + scale * (ox * cos_t - oy * sin_t)
        ncy = py + dy + scale * (ox * sin_t + oy * cos_t)
        new_img, suffix = img, ""
        if angle:
            new_img = rotate_image(new_img, angle)
            suffix += f"_rot{angle:g}"
        if scale != 1.0 and scale_images:
            new_img = scale_image(new_img, factor=scale)
            suffix += f"_x{scale:g}"
        nh, nw = new_img.shape[:2]
        # The rotated/scaled image is placed by its centre.
        nx = int(round(ncx - nw / 2.0))
        ny = int(round(ncy - nh / 2.0))
        if suffix:
            src = asset.image if isinstance(asset.image, str) else None
            if src is None:
                # No file to save next to: keep the rotated image
                # in memory as an array.
                new_assets.append(Asset(new_img, x=nx, y=ny, z=asset.z))
                continue
            stem, _ = os.path.splitext(os.path.basename(src))
            name = f"{stem}{suffix.replace('.', 'p')}.png"
            folder = out_dir or os.path.dirname(src) or "."
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, name)
            import cv2

            cv2.imwrite(path, new_img)
            new_assets.append(Asset(path, x=nx, y=ny, z=asset.z))
        else:
            new_assets.append(Asset(asset.image, x=nx, y=ny, z=asset.z))
    return new_assets


if __name__ == "__main__":
    import argparse
    import json

    import cv2

    ap = argparse.ArgumentParser(
        description="Apply a common move to a group of assets "
                    "around a pivot point.")
    ap.add_argument("scene", help="scene JSON file (see render.py)")
    ap.add_argument("--pivot", required=True,
                    help="pivot point: 'x,y', 'center' (screen centre), "
                         "or 'group' (middle of the assets)")
    ap.add_argument("--rotate", type=float, default=0.0,
                    help="rotate the group by this many degrees "
                         "(positive = counter-clockwise)")
    ap.add_argument("--dx", type=int, default=0,
                    help="move the group right by this many pixels")
    ap.add_argument("--dy", type=int, default=0,
                    help="move the group up by this many pixels")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale the group's distances from the pivot "
                         "(above 1 spreads out, below 1 pulls in)")
    ap.add_argument("--no-scale-images", action="store_true",
                    help="scale only the distances, not the images")
    ap.add_argument("--out", default=None,
                    help="output scene JSON; default <scene>_group.json")
    args = ap.parse_args()

    assets, width, height = load_scene(args.scene)
    pivot = resolve_pivot(args.pivot, assets, width, height)
    moved = transform_group(
        assets, pivot, angle=args.rotate, dx=args.dx, dy=args.dy,
        scale=args.scale, scale_images=not args.no_scale_images)
    out = args.out
    if out is None:
        stem, _ = os.path.splitext(args.scene)
        out = f"{stem}_group.json"
    scene = {"width": width, "height": height,
             "assets": [{"image": os.path.relpath(
                 a.image, os.path.dirname(os.path.abspath(out)))
                 if isinstance(a.image, str) else a.image,
                 "x": a.x, "y": a.y, "z": a.z} for a in moved]}
    with open(out, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"pivot {pivot}, rotate {args.rotate} deg, move ({args.dx}, "
          f"{args.dy}), scale {args.scale} -> {out}")
    for a in moved:
        print(f"  z={a.z}: {a.image} at ({a.x}, {a.y})")
