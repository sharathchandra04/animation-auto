"""Shape interpolation for Auto Animate.

Given two mouth shapes, generate the in-between shapes — the classic
animation "in-betweening". This is a standalone feature: it does not
need audio, video or face tracking. It works on the traced mouth
library (mouth_library_v2/manifest.json) or on any two point lists.

How it works
------------
Both shapes are resampled to the same 64 points, in the same order
around the loop (same orientation, starting at the leftmost point —
see lip_outline._resample_closed), so the points line up one-to-one.
The in-between at position t is then a point-by-point blend:

    points(t) = (1 - e) * points_a + e * points_b

where e is t passed through an easing function. With "smoothstep"
(the default, same as the lipsync morphing) the shape eases out of
A and into B; with "linear" it moves at a constant rate. The aspect
(chart height / width) is blended the same way.

Usage
-----
    from interpolate import interpolate_shapes, interpolate_points
    import lip_outline as lo

    lib = lo.load_mouth_library("mouth_library_v2/manifest.json")

    # All shapes from CLOSED to AH_I_2, 4 steps apart (5 shapes:
    # the two ends plus 3 in-betweens).
    frames = interpolate_shapes(lib, "CLOSED", "AH_I_2", steps=4)

    # One in-between, halfway, as raw point arrays.
    mid = interpolate_points(pts_a, pts_b, 0.5)

    # From the command line, draw a contact sheet of the in-betweens:
    #   python interpolate.py CLOSED AH_I_2 --steps 6 --out demo.png
"""

import numpy as np

import lip_outline as lo

N_POINTS = lo.LIPSYNC_N_POINTS  # 64, same as the lipsync morphing


def _ease(u, ease):
    """Map t in [0, 1] through the easing function."""
    u = min(1.0, max(0.0, float(u)))
    if ease == "linear":
        return u
    if ease == "smoothstep":
        return lo._smoothstep(u)
    raise ValueError(f"unknown ease {ease!r}; use 'linear' or 'smoothstep'")


def _as_points(shape, library):
    """Resolve a shape to a (64, 2) array in mouth-width units.

    `shape` is either a shape name from the library (e.g. "CLOSED")
    or a raw (N, 2) array of points in 0..1 coordinates.
    Returns (points, aspect).
    """
    if isinstance(shape, str):
        if library is None:
            raise ValueError("a library is needed to resolve a shape name")
        try:
            return library[shape]
        except KeyError:
            raise ValueError(
                f"unknown shape {shape!r}; known shapes: "
                + ", ".join(sorted(library))) from None
    pts = np.asarray(shape, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (N, 2) array")
    resampled = lo._resample_closed(pts, N_POINTS)
    # Aspect of a raw point list: its own height / width.
    w = pts[:, 0].max() - pts[:, 0].min()
    h = pts[:, 1].max() - pts[:, 1].min()
    aspect = h / w if w else 0.1
    return resampled * np.array([1.0, aspect]), aspect


def interpolate_points(pts_a, pts_b, t, ease="smoothstep"):
    """One in-between shape between two point lists, at position t.

    `pts_a`, `pts_b` are (N, 2) arrays (any N, any order — both are
    resampled to 64 points in the same order first). `t` is 0.0
    (shape A) to 1.0 (shape B). Returns a (64, 2) array.
    """
    a, _ = _as_points(pts_a, None)
    b, _ = _as_points(pts_b, None)
    e = _ease(t, ease)
    return (1.0 - e) * a + e * b


def interpolate_at(library, name_a, name_b, t, ease="smoothstep"):
    """One in-between shape between two library shapes, at position t.

    Returns (points, aspect): points is a (64, 2) array in
    mouth-width units, aspect is the blended chart height / width.
    """
    a, aspect_a = _as_points(name_a, library)
    b, aspect_b = _as_points(name_b, library)
    e = _ease(t, ease)
    pts = (1.0 - e) * a + e * b
    return pts, (1.0 - e) * aspect_a + e * aspect_b


def interpolate_shapes(library, name_a, name_b, steps, ease="smoothstep"):
    """All in-between shapes from `name_a` to `name_b`.

    `steps` is the number of intervals between the two shapes; the
    result has steps + 1 entries — the two ends plus the in-betweens:
    steps=1 gives [A, B], steps=4 gives [A, 0.25, 0.5, 0.75, B].
    Each entry is (points, aspect) as in `interpolate_at`.
    `name_a` / `name_b` may also be raw (N, 2) point arrays.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    a, aspect_a = _as_points(name_a, library)
    b, aspect_b = _as_points(name_b, library)
    out = []
    for i in range(steps + 1):
        e = _ease(i / steps, ease)
        pts = (1.0 - e) * a + e * b
        aspect = (1.0 - e) * aspect_a + e * aspect_b
        out.append((pts, aspect))
    return out


def draw_shape(points, aspect, size=(360, 240), thickness=3, color=(0, 0, 0)):
    """Draw one shape (points in mouth-width units) on a white canvas.

    Returns a BGR image of `size`. Used for previews and contact
    sheets; the same drawing is what draw_lipsync does on video.
    """
    import cv2

    h, w = size[1], size[0]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    # Fit the shape in the canvas with a margin.
    margin = 0.12
    scale = min(w, h / max(aspect, 1e-6)) * (1.0 - 2 * margin)
    cx = w / 2.0
    cy = h / 2.0
    px = cx + (points[:, 0] - 0.5) * scale
    py = cy + (points[:, 1] - aspect / 2.0) * scale
    poly = np.column_stack([px, py]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [poly], True, color, thickness,
                  lineType=cv2.LINE_AA)
    return canvas


def contact_sheet(frames, size=(360, 240)):
    """Side-by-side sheet of (points, aspect) frames. Returns an image."""
    import cv2

    imgs = [draw_shape(pts, aspect, size=size) for pts, aspect in frames]
    return np.hstack(imgs)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Interpolate between two mouth shapes and draw the "
                    "in-between sheet.")
    ap.add_argument("shape_a", help="first shape name, e.g. CLOSED")
    ap.add_argument("shape_b", help="second shape name, e.g. AH_I_2")
    ap.add_argument("--steps", type=int, default=6,
                    help="intervals between the shapes (default 6)")
    ap.add_argument("--ease", default="smoothstep",
                    choices=["smoothstep", "linear"])
    ap.add_argument("--manifest", default="mouth_library_v2/manifest.json")
    ap.add_argument("--out", default=None,
                    help="output PNG; default "
                         "<A>_to_<B>_interp.png")
    args = ap.parse_args()

    lib = lo.load_mouth_library(args.manifest)
    frames = interpolate_shapes(lib, args.shape_a, args.shape_b,
                                args.steps, ease=args.ease)
    out = args.out or f"{args.shape_a}_to_{args.shape_b}_interp.png"
    import cv2
    cv2.imwrite(out, contact_sheet(frames))
    print(f"{args.shape_a} -> {args.shape_b}: {len(frames)} shapes "
          f"({args.steps} steps, {args.ease}) -> {out}")
    print("shapes:", ", ".join(sorted(lib)))
