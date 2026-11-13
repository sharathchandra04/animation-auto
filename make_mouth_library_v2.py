#!/usr/bin/env python3
"""Mouth library v2 — from the 'Lip Sync Animation Theory' chart.

Replaces the clay-mouth library (v1). The chart's mouths are already
single dark-navy line drawings, so tracing is cleaner.
Method:
  1. Detect dark-navy strokes, keep only the left/right mouth columns
     (centre character and labels excluded), 12 mouths found.
  2. For each mouth, trace the largest dark contour (the outer line),
     smooth it (approx + Chaikin), draw it as ONE black line on white,
     normalised to 360x240.
  3. Closed mouths (closed, m/b/p) are a single stroke, not a loop:
     draw the centreline of the thin stroke instead of its outline.
Output: mouth_library_v2/ — one PNG per mouth, labelled contact sheet,
manifest.json with traced points (0-1 coords) for the video mapping.
"""
import json, math
from pathlib import Path
import cv2
import numpy as np

SRC = Path("/home/hatch/workspace/user/files/maxresdefault_1_1i72.jpg")
OUT = Path("/home/hatch/workspace/lip_outline/mouth_library_v2")
OUT.mkdir(exist_ok=True)

# (label, phonemes, side, order top-to-bottom)
MOUTHS = [
    ("CLOSED", "closed", "L", 0),
    ("S_C_Z_SH_CH", "s, c, z, sh, ch", "L", 1),
    ("AH_I_1", "ah, i (small)", "L", 2),
    ("AH_I_2", "ah, i (large)", "L", 3),
    ("EE_UH_EN_1", "ee, uh, en (upper)", "L", 4),
    ("EE_UH_EN_2", "ee, uh, en (lower)", "L", 5),
    ("M_B_P", "m, b, p", "R", 0),
    ("F_V", "f, v", "R", 1),
    ("OH", "oh", "R", 2),
    ("OO_U_W_R_1", "oo, u, w, r (upper)", "R", 3),
    ("OO_U_W_R_2", "oo, u, w, r (lower)", "R", 4),
    ("L_TH", "L, th", "R", 5),
]
LINE_MOUTHS = {"CLOSED", "M_B_P"}  # single open stroke, not a loop

def dark_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([100, 50, 0]), np.array([140, 255, 130]))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

def smooth_contour(cnt, eps_frac=0.01):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps_frac * peri, True)
    pts = approx.reshape(-1, 2).astype(float)
    for _ in range(2):
        out = []
        n = len(pts)
        for i in range(n):
            p, q = pts[i], pts[(i + 1) % n]
            out.append(0.75 * p + 0.25 * q)
            out.append(0.25 * p + 0.75 * q)
        pts = np.array(out)
    return pts

def centreline(pts, n=60):
    """Midline of a thin stroke: split loop at leftmost/rightmost
    points into two chains, resample both left->right, average."""
    pts = np.asarray(pts, float)
    i_left = int(np.argmin(pts[:, 0]))
    i_right = int(np.argmax(pts[:, 0]))
    if i_left < i_right:
        chain_a = pts[i_left:i_right + 1]
        chain_b = np.vstack([pts[i_right:], pts[:i_left + 1]])
    else:
        chain_a = pts[i_right:i_left + 1]
        chain_b = np.vstack([pts[i_left:], pts[:i_right + 1]])
    for ch_name in ("a", "b"):
        ch = chain_a if ch_name == "a" else chain_b
        if np.hypot(*(ch[0] - pts[i_right])) < np.hypot(*(ch[0] - pts[i_left])):
            if ch_name == "a":
                chain_a = ch[::-1]
            else:
                chain_b = ch[::-1]
    def resample(chain):
        d = np.sqrt(((np.diff(chain, axis=0)) ** 2).sum(axis=1))
        s = np.concatenate([[0], np.cumsum(d)])
        t = np.linspace(0, s[-1], n)
        return np.column_stack([np.interp(t, s, chain[:, 0]),
                                np.interp(t, s, chain[:, 1])])
    return (resample(chain_a) + resample(chain_b)) / 2

def main():
    im = cv2.imread(str(SRC))
    assert im is not None, f"cannot read {SRC}"
    H, W = im.shape[:2]
    mask = dark_mask(im)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        a = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w / 2
        side = "L" if 50 < cx < 190 else ("R" if 1110 < cx < 1230 else None)
        if side and w >= 18 and 5 <= h < 60 and a >= 80:
            boxes.append((side, y, x, w, h, c))
    assert len(boxes) == 12, f"expected 12 mouths, found {len(boxes)}"
    boxes.sort(key=lambda b: (b[0], b[1]))

    manifest = []
    for idx, (label, phonemes, side, order) in enumerate(MOUTHS, start=1):
        side_boxes = [b for b in boxes if b[0] == side]
        _, y, x, w, h, _ = side_boxes[order]
        pad = 6
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = im[y0:y1, x0:x1]
        cmask = dark_mask(crop)
        cnts, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer = max(cnts, key=cv2.contourArea)
        if label == "F_V":
            # the chart's f/v has two small ticks hanging below the lens;
            # the convex hull drops them and keeps the clean lens shape
            outer = cv2.convexHull(outer)
        eps = 0.02 if w < 35 else 0.01  # extra smoothing for tiny circles
        outer_pts = smooth_contour(outer, eps)
        ox, oy, ow, oh = cv2.boundingRect(outer)
        scale = min((360 - 40) / ow, (240 - 40) / oh)
        def norm(pts):
            return [((int((p[0] - ox) * scale) + (360 - int(ow * scale)) // 2,
                      int((p[1] - oy) * scale) + (240 - int(oh * scale)) // 2))
                    for p in pts]
        canvas = np.full((240, 360, 3), 255, np.uint8)
        drawn_as = "single_line" if label in LINE_MOUTHS else "single_loop"
        if label in LINE_MOUTHS:
            cl_n = norm(centreline(outer_pts))
            cv2.polylines(canvas, [np.array(cl_n, np.int32)], False, (0, 0, 0), 4, cv2.LINE_AA)
            drawn_pts = centreline(outer_pts)
        else:
            outer_n = norm(outer_pts)
            cv2.polylines(canvas, [np.array(outer_n, np.int32)], True, (0, 0, 0), 4, cv2.LINE_AA)
            drawn_pts = outer_pts
        fname = f"{idx:02d}_{label}.png"
        cv2.imwrite(str(OUT / fname), canvas)
        manifest.append({
            "file": fname, "label": label, "phonemes": phonemes,
            "drawn_as": drawn_as,
            "chart_box": [int(x), int(y), int(w), int(h)],
            "aspect_w_h": round(float(w) / h, 3),
            "points_01": [[round(float(p[0] - ox) / ow, 3),
                           round(float(p[1] - oy) / oh, 3)] for p in drawn_pts],
        })
        print(f"{fname:24s} {drawn_as:12s} box {w}x{h} pts {len(drawn_pts)}")
    cols, cell_w, cell_h, label_h = 4, 360, 240, 56
    rows = math.ceil(len(manifest) / cols)
    sheet = np.full((rows * (cell_h + label_h), cols * cell_w, 3), 255, np.uint8)
    for i, m in enumerate(manifest):
        r, c = divmod(i, cols)
        img = cv2.imread(str(OUT / m["file"]))
        sheet[r * (cell_h + label_h):r * (cell_h + label_h) + cell_h,
              c * cell_w:(c + 1) * cell_w] = img
        cv2.putText(sheet, f"{i+1:02d} {m['label']}",
                    (c * cell_w + 10, r * (cell_h + label_h) + cell_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"({m['phonemes']})",
                    (c * cell_w + 10, r * (cell_h + label_h) + cell_h + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "mouth_library_chart.png"), sheet)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest)} mouths + contact sheet to {OUT}")

if __name__ == "__main__":
    main()
