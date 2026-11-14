"""Rotate an image by a given angle — the rotate feature.

The Auto Animate rotate tool. You give an image and an angle in
degrees; it rotates the image by that angle and saves the result.

How it works
------------
The rotation is done with an affine transform around the centre of
the image (or a centre you give). Positive angles rotate
counter-clockwise, negative angles clockwise — the standard
convention, the same as OpenCV and PIL.

By default the canvas grows (`--expand`) so nothing is cropped: a
45-degree rotation of a square image needs a canvas about 1.4x as
big, and the empty corners are transparent. With `--keep-size` the
canvas stays the same size and the corners are cropped off instead.
The empty corners are transparent by default (the output is a PNG
with an alpha channel); `--background` fills them with a solid
colour instead.

Usage
-----
    python rotate.py insect.png --angle 45
    python rotate.py insect.png --angle -30 --out insect_r-30.png
    python rotate.py insect.png --angle 90 --keep-size
    python rotate.py insect.png --angle 45 --background white

    from rotate import rotate_image
    rotated = rotate_image(bgra, angle=45)  # BGRA, canvas expanded

Notes
-----
- `--angle` is in degrees and can be any number (720 is the same as
  0; -90 is a quarter turn clockwise).
- `--center x,y` rotates around a point other than the image centre.
- `--interpolation` is nearest, linear (default) or cubic. Linear is
  the smooth choice for pictures; nearest keeps hard pixel edges
  (and rotates by multiples of 90 degrees exactly).
"""

import numpy as np


def rotate_image(img, angle, expand=True, background=None, center=None,
                 interpolation="linear"):
    """Rotate `img` (BGR or BGRA) by `angle` degrees counter-clockwise.

    Returns the rotated image. With `expand` (default) the canvas
    grows to fit the whole rotated image; otherwise it keeps the
    original size and crops. Empty corners are transparent (BGRA
    output) unless `background` (a colour name or hex) is given, in
    which case the result is flattened onto that colour (BGR output).
    """
    import cv2

    from remove_background import parse_color

    interp = {"nearest": cv2.INTER_NEAREST,
              "linear": cv2.INTER_LINEAR,
              "cubic": cv2.INTER_CUBIC}[interpolation]
    h, w = img.shape[:2]
    cx, cy = center if center is not None else (w / 2.0, h / 2.0)

    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    if expand:
        # Size of the bounding box of the rotated image.
        cos_a, sin_a = abs(m[0, 0]), abs(m[0, 1])
        new_w = int(round(h * sin_a + w * cos_a))
        new_h = int(round(h * cos_a + w * sin_a))
        # Shift the rotation centre to the middle of the new canvas.
        m[0, 2] += (new_w - w) / 2.0
        m[1, 2] += (new_h - h) / 2.0
    else:
        new_w, new_h = w, h

    if img.shape[2] == 3:
        img = np.dstack([img, np.full((h, w), 255, np.uint8)])
    out = cv2.warpAffine(img, m, (new_w, new_h), flags=interp,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0, 0))
    if background is not None:
        from add_outline import flatten

        return flatten(out, background)
    return out


if __name__ == "__main__":
    import argparse

    import cv2

    ap = argparse.ArgumentParser(
        description="Rotate an image by a given angle in degrees.")
    ap.add_argument("input", help="input image (any format)")
    ap.add_argument("--angle", type=float, required=True,
                    help="rotation angle in degrees; positive is "
                         "counter-clockwise, negative is clockwise")
    ap.add_argument("--keep-size", action="store_true",
                    help="keep the original canvas size (crop corners) "
                         "instead of expanding it")
    ap.add_argument("--center", default=None,
                    help="rotation centre as x,y (default: image centre)")
    ap.add_argument("--background", default=None,
                    help="fill the empty corners with this solid colour "
                         "(name or hex) instead of transparency")
    ap.add_argument("--interpolation", default="linear",
                    choices=["nearest", "linear", "cubic"])
    ap.add_argument("--out", default=None,
                    help="output PNG; default <input>_rot<angle>.png")
    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"could not read {args.input}")
    center = None
    if args.center:
        try:
            x, y = args.center.split(",")
            center = (float(x), float(y))
        except ValueError:
            raise SystemExit("--center must be x,y") from None
    out_img = rotate_image(img, args.angle, expand=not args.keep_size,
                           background=args.background, center=center,
                           interpolation=args.interpolation)
    out = args.out
    if out is None:
        stem = args.input.rsplit(".", 1)[0]
        angle_txt = f"{args.angle:g}".replace("-", "m").replace(".", "p")
        out = f"{stem}_rot{angle_txt}.png"
    if not out.lower().endswith(".png"):
        out += ".png"
    cv2.imwrite(out, out_img)
    print(f"rotated {args.angle:g} degrees "
          f"({img.shape[1]}x{img.shape[0]} -> "
          f"{out_img.shape[1]}x{out_img.shape[0]}) -> {out}")
