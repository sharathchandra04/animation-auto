"""Tilt an image on an axis, like pulling a door towards you.

The Auto Animate tilt feature. You pick one edge of the image as
the hinge — left, right, top or bottom — and an angle, and the
image swings forward around that edge in 3D, the way a door opens
toward you.

How it works (real perspective, not a flat squash)
--------------------------------------------------
The image plane is rotated in 3D around the chosen edge, then
projected with a pinhole camera:

- The hinge edge does not move at all — it is the axle.
- The opposite (free) edge swings toward the viewer, so it looks
  bigger: it gets taller (for a left/right hinge) and its top and
  bottom edges converge toward the hinge. That is exactly what a
  real door looks like opening toward you.
- A negative angle pushes the free edge away instead, so it looks
  smaller — the door pushed away from you.

The camera distance (`--focal`, default twice the image size)
controls how strong the perspective is: a larger focal length
looks flatter, a smaller one more dramatic.

Usage
-----
    python tilt.py insect.png --axis left --angle 30
    python tilt.py insect.png --axis top --angle 45 --out insect_top.png
    python tilt.py insect.png --axis right --angle -20
    python tilt.py insect.png --axis bottom --angle 30 --background white

    # Chain tilts one after another on the same image, for a 3D
    # turn: left 45 degrees, then top 45 degrees on the result.
    python tilt.py insect.png --tilt left:45 --tilt top:45
    python tilt.py insect.png --tilt left:45 --tilt top:45 --focal 800

    from tilt import tilt_image, tilt_sequence
    tilted = tilt_image(bgra, axis="left", angle=30)  # BGRA
    turned = tilt_sequence(bgra, [("left", 45), ("top", 45)])

Notes
-----
- `--axis` is left, right, top or bottom. For left/right the hinge
  is a vertical edge; for top/bottom it is a horizontal edge.
- `--angle` is in degrees, from -89 to 89 (0 is flat). Positive
  pulls the free edge toward you; negative pushes it away.
- The canvas grows to fit the tilted image; empty corners are
  transparent (PNG with alpha) unless `--background` fills them.
"""

import numpy as np

AXES = ("left", "right", "top", "bottom")


def tilt_corners(w, h, axis, angle, focal):
    """Projected corners of a wxh image tilted around `axis`.

    Returns (dst_corners, canvas_w, canvas_h): the four corner
    positions after the 3D rotation and perspective projection,
    shifted so they are all positive, and the canvas size that
    fits them. Corners are in order top-left, top-right,
    bottom-right, bottom-left.
    """
    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = w / 2.0, h / 2.0
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    pts = np.zeros((4, 3))
    for i, (x, y) in enumerate(corners):
        # Centred coordinates: the camera looks at the image centre,
        # so left and right tilts are mirror images of each other.
        u, v = x - cx, y - cy
        if axis == "left":
            # Hinge is x=0 (u=-w/2); the right side swings toward
            # the viewer.
            U, V, Z = -cx + (u + cx) * cos_t, v, -(u + cx) * sin_t
        elif axis == "right":
            U, V, Z = cx - (cx - u) * cos_t, v, -(cx - u) * sin_t
        elif axis == "top":
            U, V, Z = u, -cy + (v + cy) * cos_t, -(v + cy) * sin_t
        elif axis == "bottom":
            U, V, Z = u, cy - (cy - v) * cos_t, -(cy - v) * sin_t
        else:
            raise ValueError(f"unknown axis {axis!r}; use {AXES}")
        pts[i] = (U, V, Z)
    # Pinhole projection: camera at z=-focal, plane at z=0, looking
    # at the image centre. Points with Z<0 are closer to the viewer,
    # so they are magnified.
    scale = focal / (focal + pts[:, 2])
    projected = np.column_stack([pts[:, 0] * scale + cx,
                                 pts[:, 1] * scale + cy])
    shift = -projected.min(axis=0)
    projected = projected + shift
    size = projected.max(axis=0)
    canvas_w, canvas_h = int(np.ceil(size[0])), int(np.ceil(size[1]))
    return projected.astype(np.float32), canvas_w, canvas_h


def tilt_image(img, axis, angle, focal=None, background=None):
    """Tilt `img` (BGR or BGRA) around `axis` by `angle` degrees.

    Positive angle pulls the free edge toward the viewer; negative
    pushes it away. Returns a BGRA image (transparent corners)
    unless `background` (a colour name or hex) is given, in which
    case the result is flattened onto that colour (BGR output).
    """
    import cv2

    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; use {AXES}")
    if not -89.0 < angle < 89.0:
        raise ValueError("angle must be between -89 and 89 degrees")
    h, w = img.shape[:2]
    if focal is None:
        focal = 2.0 * max(w, h)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst, new_w, new_h = tilt_corners(w, h, axis, angle, focal)
    m = cv2.getPerspectiveTransform(src, dst)
    if img.shape[2] == 3:
        img = np.dstack([img, np.full((h, w), 255, np.uint8)])
    out = cv2.warpPerspective(img, m, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))
    if background is not None:
        from add_outline import flatten

        return flatten(out, background)
    return out


def parse_step(spec):
    """Parse one chained tilt step, 'axis:angle' (e.g. 'left:45')."""
    try:
        axis, angle = spec.split(":")
        angle = float(angle)
    except ValueError:
        raise ValueError(
            f"--tilt takes 'axis:angle', e.g. 'left:45', got {spec!r}"
        ) from None
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; use {AXES}")
    return axis, angle


def tilt_sequence(img, steps, focal=None, background=None):
    """Tilt `img` through a series of (axis, angle) steps, in order.

    Each step tilts the result of the previous one, so the image
    turns in 3D step by step — left 45 then top 45 is a door
    opening sideways and then tipping back. Order matters: left
    then top is not the same as top then left, because 3D
    rotations do not commute. If `focal` is given it is shared by
    every step (one camera for the whole turn); otherwise each
    step uses twice its own image size. `background` fills the
    empty corners of the final image only.
    """
    out = img
    for axis, angle in steps:
        out = tilt_image(out, axis, angle, focal=focal)
    if background is not None:
        from add_outline import flatten

        return flatten(out, background)
    return out


if __name__ == "__main__":
    import argparse

    import cv2

    ap = argparse.ArgumentParser(
        description="Tilt an image on an axis, like pulling a door "
                    "towards you.")
    ap.add_argument("input", help="input image (any format)")
    ap.add_argument("--axis", default=None, choices=list(AXES),
                    help="the hinge edge: left, right, top or bottom "
                         "(single tilt)")
    ap.add_argument("--angle", type=float, default=None,
                    help="tilt angle in degrees (-89 to 89). Positive "
                         "pulls the free edge toward you; negative "
                         "pushes it away. (single tilt)")
    ap.add_argument("--tilt", action="append", default=None,
                    help="one step of a chained tilt, 'axis:angle' "
                         "(e.g. --tilt left:45 --tilt top:45). Repeat "
                         "for more steps; applied in order.")
    ap.add_argument("--focal", type=float, default=None,
                    help="camera distance in pixels (default: twice the "
                         "image size). Larger is flatter, smaller is "
                         "more dramatic. For a chained tilt it is "
                         "shared by every step.")
    ap.add_argument("--background", default=None,
                    help="fill the empty corners with this solid colour "
                         "(name or hex) instead of transparency")
    ap.add_argument("--out", default=None,
                    help="output PNG; default "
                         "<input>_tilt-<axis><angle>.png (or "
                         "<input>_tilt-<axis><angle>-<axis><angle>.png "
                         "for a chain)")
    args = ap.parse_args()

    if args.tilt:
        steps = [parse_step(s) for s in args.tilt]
    elif args.axis is not None and args.angle is not None:
        steps = [(args.axis, args.angle)]
    else:
        ap.error("give --axis and --angle for one tilt, or --tilt "
                 "'axis:angle' (repeatable) for a chained tilt")

    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"could not read {args.input}")
    if len(steps) == 1:
        out_img = tilt_image(img, *steps[0], focal=args.focal,
                             background=args.background)
    else:
        out_img = tilt_sequence(img, steps, focal=args.focal,
                                background=args.background)
    out = args.out
    if out is None:
        stem = args.input.rsplit(".", 1)[0]
        tag = "-".join(
            f"{ax}{ang:g}".replace("-", "m").replace(".", "p")
            for ax, ang in steps)
        out = f"{stem}_tilt-{tag}.png"
    if not out.lower().endswith(".png"):
        out += ".png"
    cv2.imwrite(out, out_img)
    print(f"tilted {' then '.join(f'{ax} {ang:g}' for ax, ang in steps)} "
          f"degrees ({img.shape[1]}x{img.shape[0]} -> "
          f"{out_img.shape[1]}x{out_img.shape[0]}) -> {out}")
