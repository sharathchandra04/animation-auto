"""Add an outline of a chosen colour around a cut-out subject.

The Auto Animate cut-out tool, step 2 — the full pipeline in one
place:

    AI image (subject on a solid background)
      -> remove the background (remove_background.py)
      -> save the cut-out as an asset / icon (transparent PNG)
      -> draw an outline of a colour of your choice around it
      -> save the outlined asset (transparent PNG, ready to use)

The outline is drawn OUTSIDE the subject, so it never covers any
part of the picture itself: the subject mask is dilated by
`thickness` pixels, and the ring between the dilated mask and the
subject is filled with the outline colour. The subject is then
pasted back on top, so its own soft edge is kept.

Usage
-----
    # Whole pipeline from an AI image on magenta:
    python add_outline.py insect.png --key-color magenta \
        --outline-color black --thickness 6

    # From an already cut-out transparent PNG (background removal
    # is skipped when the input already has transparency):
    python add_outline.py insect_cut.png --outline-color "#FF0000"

    # Flatten onto a background colour instead of transparency:
    python add_outline.py insect.png --background white

    from add_outline import add_outline, outline_pipeline
    outlined = add_outline(cut_bgra, outline_color="black", thickness=6)

Notes
-----
- `--outline-color` accepts a name (black, red, white, ...) or a
  hex code (#FF0000). Default: black.
- `--thickness` is in pixels. A good starting point is about 1.5%
  of the image width (6 px on a 400 px image).
- The intermediate cut-out is always saved as `<input>_cut.png` —
  that is the asset/icon, reusable on any background. The outlined
  version is `<input>_outlined.png` unless `--out` is given.
"""

import numpy as np

from remove_background import parse_color, remove_background


def add_outline(bgra, outline_color="black", thickness=6):
    """Draw an outline around the subject of a BGRA image.

    `bgra` is a transparent cut-out (background alpha 0, subject
    alpha 255, possibly feathered edges). Returns a BGRA image of
    the same size: the same subject, with a ring of `thickness`
    pixels in `outline_color` around it, on transparency.
    """
    import cv2

    if bgra.shape[2] != 4:
        raise ValueError("add_outline needs a BGRA image with an alpha channel")
    if thickness < 1:
        return bgra.copy()

    color = parse_color(outline_color)  # (B, G, R)
    alpha = bgra[:, :, 3]
    subject = (alpha > 127).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * thickness + 1, 2 * thickness + 1))
    dilated = cv2.dilate(subject, kernel)
    ring = cv2.subtract(dilated, subject)  # outline band, outside only

    out = np.zeros_like(bgra)
    out[:, :, 0] = color[0]
    out[:, :, 1] = color[1]
    out[:, :, 2] = color[2]
    out[:, :, 3] = ring
    # Paste the original subject on top: its own colours and soft
    # edge are kept, and it covers the inner edge of the ring.
    cover = alpha > 0
    out[cover] = bgra[cover]
    return out


def flatten(bgra, background="white"):
    """Composite a BGRA image onto a solid background colour.

    Returns a BGR image (no transparency)."""
    color = np.array(parse_color(background), dtype=np.float32)
    alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
    comp = bgra[:, :, :3].astype(np.float32) * alpha + color * (1.0 - alpha)
    return comp.astype(np.uint8)


def outline_pipeline(input_path, key_color="magenta", tolerance=40.0,
                     feather=2.0, outline_color="black", thickness=6,
                     background=None, cut_path=None, out_path=None):
    """Run the whole pipeline on one image file.

    Returns (cut_bgra, outlined_bgra, cut_path, out_path). The
    cut-out asset and the outlined asset are both written to disk.
    If `input_path` is already a transparent PNG, background removal
    is skipped and the cut-out is the input itself.
    """
    import cv2

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"could not read {input_path}")

    stem = input_path.rsplit(".", 1)[0]
    cut_path = cut_path or f"{stem}_cut.png"
    out_path = out_path or f"{stem}_outlined.png"

    if img.shape[2] == 4 and (img[:, :, 3] < 255).any():
        # Already a cut-out: keep it as the asset.
        cut = img
        cv2.imwrite(cut_path, cut)
    else:
        bgr = img[:, :, :3] if img.shape[2] >= 3 else img
        cut = remove_background(bgr, key_color=key_color,
                                tolerance=tolerance, feather=feather)
        cv2.imwrite(cut_path, cut)

    outlined = add_outline(cut, outline_color=outline_color,
                           thickness=thickness)
    if background is not None:
        cv2.imwrite(out_path, flatten(outlined, background))
    else:
        cv2.imwrite(out_path, outlined)
    return cut, outlined, cut_path, out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Remove the background, save the cut-out asset, "
                    "and add an outline of your colour around it.")
    ap.add_argument("input", help="input image: AI picture on a solid "
                                  "background, or an already cut-out PNG")
    ap.add_argument("--key-color", default="magenta",
                    help="background colour to remove (name or hex). "
                         "Default: magenta")
    ap.add_argument("--tolerance", type=float, default=40.0,
                    help="passed to background removal (default 40)")
    ap.add_argument("--feather", type=float, default=2.0,
                    help="passed to background removal (default 2)")
    ap.add_argument("--outline-color", default="black",
                    help="outline colour: a name (black, red, ...) or "
                         "hex (#FF0000). Default: black")
    ap.add_argument("--thickness", type=int, default=6,
                    help="outline thickness in pixels (default 6)")
    ap.add_argument("--background", default=None,
                    help="optional: flatten the result onto this solid "
                         "background colour instead of transparency")
    ap.add_argument("--cut-out", default=None,
                    help="where to save the intermediate cut-out asset; "
                         "default <input>_cut.png")
    ap.add_argument("--out", default=None,
                    help="output file; default <input>_outlined.png")
    args = ap.parse_args()

    _, outlined, cut_path, out_path = outline_pipeline(
        args.input, key_color=args.key_color, tolerance=args.tolerance,
        feather=args.feather, outline_color=args.outline_color,
        thickness=args.thickness, background=args.background,
        cut_path=args.cut_out, out_path=args.out)
    print(f"cut-out asset -> {cut_path}")
    print(f"outlined ({args.outline_color}, {args.thickness}px) -> {out_path}")
