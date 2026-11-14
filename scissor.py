"""Cut an image into two pieces along a line — the scissor feature.

The Auto Animate cut feature. You place an image on a grid, propose
a line on that grid, and this tool cuts the image into two pieces
along the line, saving each piece as its own transparent asset.

How it works
------------
The line is the infinite line through the two points you give (or
from an equation you give). Every pixel is assigned to one side of
the line by the sign of the cross product — pixels on one side stay
in piece A, pixels on the other side stay in piece B, and the other
side of each piece is transparent. So the two pieces always add up
to the original image exactly: nothing is lost, nothing is doubled.
The line cuts across the whole image, however long the segment
between your two points is (its length is reported, for reference).

The grid is the image's coordinate system: x grows to the right, y
grows down, origin at the top-left corner. `--show-grid` draws the
grid on a preview copy so you can read coordinates off it.

Assets are named with a random ID and a timestamp, so every cut is
unique and you can tell when it was made:
    insect_outlined_A_a3f9c2_20261113_194500.png
    insect_outlined_B_a3f9c2_20261113_194500.png
The two pieces of one cut share the same ID and timestamp, so they
stay paired. The location is `--out-dir` (default `assets/cuts/`).

Usage
-----
    # Cut along the line through (100, 50) and (300, 350):
    python scissor.py insect_outlined.png --p1 100,50 --p2 300,350

    # Or give the line as an equation instead of two points:
    python scissor.py insect_outlined.png --equation "y = 0.5x + 10"
    python scissor.py insect_outlined.png --equation "x = 200"

    # Draw the grid preview first, to pick your points:
    python scissor.py insect_outlined.png --show-grid --grid-step 50

    from scissor import cut_image, line_from_points
    part_a, part_b = cut_image(bgra, (100, 50), (300, 350))
"""

import re
import secrets
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_point(text):
    """'x,y' -> (x, y) as floats."""
    try:
        x, y = text.split(",")
        return (float(x), float(y))
    except ValueError:
        raise ValueError(
            f"bad point {text!r}; give it as x,y, e.g. 100,50") from None


def line_from_points(p1, p2):
    """The line through two points, as (a, b, c) with ax + by + c = 0.

    Also returns the segment length between the points, for reference.
    """
    (x1, y1), (x2, y2) = p1, p2
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length == 0:
        raise ValueError("the two points must be different")
    # Normalised so (a, b) is a unit vector: |ax + by + c| is then the
    # perpendicular distance from (x, y) to the line.
    a, b = -dy / length, dx / length
    c = -(a * x1 + b * y1)
    return (a, b, c), length


def line_from_equation(eq):
    """Parse 'y = mx + b' or 'x = c' into two points on the line."""
    s = eq.replace(" ", "").lower()
    m = re.fullmatch(r"y=([+-]?[\d.]*)\*?x([+-][\d.]+)?", s)
    if m:
        slope = float(m.group(1)) if m.group(1) not in ("", "+", "-") else (
            -1.0 if m.group(1) == "-" else 1.0)
        intercept = float(m.group(2)) if m.group(2) else 0.0
        return (0.0, intercept), (1.0, slope + intercept)
    m = re.fullmatch(r"x=([+-]?[\d.]+)", s)
    if m:
        c = float(m.group(1))
        return (c, 0.0), (c, 1.0)
    raise ValueError(
        f"cannot parse equation {eq!r}; use 'y = mx + b' or 'x = c'")


def side_mask(shape, line):
    """Boolean mask of pixels on side A of the line (ax + by + c >= 0).

    `shape` is (height, width) of the image.
    """
    a, b, c = line
    h, w = shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    return (a * xs + b * ys + c) >= 0


def cut_image(img, p1, p2):
    """Cut `img` (BGR or BGRA) into two pieces along the line p1->p2.

    Returns (part_a, part_b) as BGRA images: each holds one side of
    the original, the other side transparent. Together they rebuild
    the original exactly.
    """
    line, length = line_from_points(p1, p2)
    if img.shape[2] == 3:
        bgra = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    else:
        bgra = img.copy()
    mask_a = side_mask(bgra.shape, line)
    part_a = bgra.copy()
    part_b = bgra.copy()
    part_a[~mask_a] = (0, 0, 0, 0)
    part_b[mask_a] = (0, 0, 0, 0)
    return part_a, part_b, length


def draw_grid(img, step=50, color=(200, 200, 200), label_color=(90, 90, 90)):
    """Draw a coordinate grid on a copy of `img` (BGR), labelled in px.

    Grid lines every `step` pixels; coordinates are written along the
    top (x) and left (y) edges so you can read points off the image.
    """
    import cv2

    out = img[:, :, :3].copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h - 1), color, 1)
        cv2.putText(out, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, label_color, 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w - 1, y), color, 1)
        cv2.putText(out, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, label_color, 1, cv2.LINE_AA)
    return out


def asset_name(base, side, rand_id=None, timestamp=None):
    """<base>_<side>_<random id>_<timestamp>.png for one cut piece."""
    rand_id = rand_id or secrets.token_hex(3)  # 6 hex chars
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{side}_{rand_id}_{timestamp}.png"


def scissor_pipeline(input_path, p1=None, p2=None, equation=None,
                     out_dir="assets/cuts"):
    """Cut one image and save the two pieces as assets.

    Give the line as two points (`p1`, `p2`) or as `--equation`.
    Returns (path_a, path_b, length): the two asset files and the
    length of the segment between the two points.
    """
    import cv2

    if equation is not None:
        p1, p2 = line_from_equation(equation)
    if p1 is None or p2 is None:
        raise ValueError("give the line as --p1 x,y --p2 x,y, or --equation")

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"could not read {input_path}")
    part_a, part_b, length = cut_image(img, p1, p2)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(input_path).stem
    rand_id = secrets.token_hex(3)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_a = out_dir / asset_name(base, "A", rand_id, timestamp)
    path_b = out_dir / asset_name(base, "B", rand_id, timestamp)
    cv2.imwrite(str(path_a), part_a)
    cv2.imwrite(str(path_b), part_b)
    return path_a, path_b, length


if __name__ == "__main__":
    import argparse

    import cv2

    ap = argparse.ArgumentParser(
        description="Cut an image into two pieces along a line; save "
                    "each piece as an asset.")
    ap.add_argument("input", help="input image (any format)")
    ap.add_argument("--p1", default=None, help="first point, as x,y")
    ap.add_argument("--p2", default=None, help="second point, as x,y")
    ap.add_argument("--equation", default=None,
                    help="line as an equation instead of two points: "
                         "'y = mx + b' or 'x = c'")
    ap.add_argument("--out-dir", default="assets/cuts",
                    help="folder for the two asset files "
                         "(default assets/cuts/)")
    ap.add_argument("--show-grid", action="store_true",
                    help="save a grid preview (<input>_grid.png) and stop")
    ap.add_argument("--grid-step", type=int, default=50,
                    help="grid spacing in pixels (default 50)")
    args = ap.parse_args()

    if args.show_grid:
        img = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"could not read {args.input}")
        grid = draw_grid(img, step=args.grid_step)
        out = Path(args.input).stem + "_grid.png"
        cv2.imwrite(out, grid)
        print(f"grid preview -> {out} "
              f"({img.shape[1]}x{img.shape[0]}, step {args.grid_step}px)")
        raise SystemExit(0)

    p1 = parse_point(args.p1) if args.p1 else None
    p2 = parse_point(args.p2) if args.p2 else None
    path_a, path_b, length = scissor_pipeline(
        args.input, p1=p1, p2=p2, equation=args.equation,
        out_dir=args.out_dir)
    print(f"line length between points: {length:.1f} px")
    print(f"piece A -> {path_a}")
    print(f"piece B -> {path_b}")
