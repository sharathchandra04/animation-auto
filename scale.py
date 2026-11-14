"""Scale an image — zoom in or zoom out.

The Auto Animate scale feature. Give an image and a zoom factor
(or an exact size) and it returns the image scaled to that size.

How it works
------------
The image is resampled with OpenCV's resize. A factor above 1
zooms in (the image gets bigger), a factor below 1 zooms out (it
gets smaller). The aspect ratio is kept unless you give different
x and y factors, or an exact width and height together.

Interpolation matters for how the zoom looks: `linear` (default)
is smooth for photos and drawings; `cubic` and `lanczos` are
smoother still for zooming in; `nearest` keeps hard pixel edges
(and is exact for whole-number zooms — 2x then 0.5x gives back
the original image).

Usage
-----
    python scale.py insect.png --factor 2        # zoom in 2x
    python scale.py insect.png --factor 0.5      # zoom out to half
    python scale.py insect.png --factor 1.5 --out insect_big.png
    python scale.py insect.png --width 800       # width 800, height follows
    python scale.py insect.png --width 800 --height 600
    python scale.py insect.png --fx 2 --fy 0.5  # stretch x, squash y

    from scale import scale_image
    big = scale_image(bgra, factor=2.0)
    small = scale_image(bgra, width=200)

Notes
-----
- Transparency is preserved: the alpha channel is scaled along
  with the colours, so a cut-out stays a cut-out.
- The output is a PNG. Dimensions are rounded to whole pixels.
"""

import numpy as np

INTERPOLATIONS = {
    "nearest": 0,  # cv2.INTER_NEAREST
    "linear": 1,   # cv2.INTER_LINEAR
    "cubic": 2,    # cv2.INTER_CUBIC
    "lanczos": 4,  # cv2.INTER_LANCZOS4
}


def scale_image(img, factor=None, fx=None, fy=None, width=None,
                height=None, interpolation="linear"):
    """Scale `img` (BGR or BGRA) and return the scaled image.

    Give exactly one way to size it:
    - `factor`: uniform zoom (2.0 doubles, 0.5 halves).
    - `fx`, `fy`: separate x and y zoom factors.
    - `width` and/or `height`: exact size in pixels. One of them
      alone keeps the aspect ratio; both together set it exactly.
    """
    import cv2

    h, w = img.shape[:2]
    if factor is not None:
        if factor <= 0:
            raise ValueError("factor must be above 0")
        fx, fy = float(factor), float(factor)
    if width is not None or height is not None:
        if width is not None and height is not None:
            new_w, new_h = int(width), int(height)
        elif width is not None:
            new_w = int(width)
            new_h = max(1, round(h * new_w / w))
        else:
            new_h = int(height)
            new_w = max(1, round(w * new_h / h))
        return cv2.resize(img, (new_w, new_h),
                          interpolation=INTERPOLATIONS[interpolation])
    if fx is None and fy is None:
        raise ValueError("give a factor, fx/fy, or width/height")
    fx = float(fx) if fx is not None else float(fy)
    fy = float(fy) if fy is not None else float(fx)
    if fx <= 0 or fy <= 0:
        raise ValueError("scale factors must be above 0")
    new_w = max(1, round(w * fx))
    new_h = max(1, round(h * fy))
    return cv2.resize(img, (new_w, new_h),
                      interpolation=INTERPOLATIONS[interpolation])


if __name__ == "__main__":
    import argparse

    import cv2

    ap = argparse.ArgumentParser(
        description="Scale an image — zoom in or zoom out.")
    ap.add_argument("input", help="input image (any format)")
    ap.add_argument("--factor", type=float, default=None,
                    help="uniform zoom factor: 2 zooms in 2x, 0.5 zooms "
                         "out to half")
    ap.add_argument("--fx", type=float, default=None,
                    help="x zoom factor (with --fy for non-uniform scale)")
    ap.add_argument("--fy", type=float, default=None,
                    help="y zoom factor (with --fx for non-uniform scale)")
    ap.add_argument("--width", type=int, default=None,
                    help="exact output width in pixels (height follows "
                         "unless --height is also given)")
    ap.add_argument("--height", type=int, default=None,
                    help="exact output height in pixels (width follows "
                         "unless --width is also given)")
    ap.add_argument("--interpolation", default="linear",
                    choices=list(INTERPOLATIONS),
                    help="resampling: nearest, linear (default), cubic, "
                         "lanczos")
    ap.add_argument("--out", default=None,
                    help="output PNG; default <input>_x<factor>.png")
    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"could not read {args.input}")
    out_img = scale_image(img, factor=args.factor, fx=args.fx, fy=args.fy,
                          width=args.width, height=args.height,
                          interpolation=args.interpolation)
    out = args.out
    if out is None:
        stem = args.input.rsplit(".", 1)[0]
        if args.factor is not None:
            tag = f"x{args.factor:g}".replace(".", "p")
        elif args.width is not None or args.height is not None:
            tag = f"{out_img.shape[1]}x{out_img.shape[0]}"
        else:
            tag = f"{args.fx}x{args.fy}"
        out = f"{stem}_{tag}.png"
    if not out.lower().endswith(".png"):
        out += ".png"
    cv2.imwrite(out, out_img)
    print(f"scaled {img.shape[1]}x{img.shape[0]} -> "
          f"{out_img.shape[1]}x{out_img.shape[0]} -> {out}")
