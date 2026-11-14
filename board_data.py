"""Data layer for the Auto Animate board (PySide6 GUI).

No Qt imports here — pure stdlib + numpy — so the data model,
validation and interpolation can be tested without a GUI.

The data model
--------------
One manifest file per keyframe. A keyframe file is a scene JSON:

    {"width": 1000, "height": 800, "t": 0.0, "assets": [...]}

`t` is the keyframe time in seconds. A folder of keyframe files is
the timeline: files are named frame_0001.json, frame_0002.json, ...
(zero-padded, so they sort in time order) and the renderer consumes
them all in order.

Assets are separate files referenced by relative paths in the
manifest — the manifest never embeds pixels, except for the
"image_data" form (base64 PNG) used by the mock data, because the
GitHub push is text-only.

Asset forms
-----------
Raster (an image):

    {"image": "insect.png", "x": 50, "y": 50, "z": 0,
     "scale": 1.0, "anchor": "bottom-left"}

    {"image_data": "<base64 PNG>", "x": 50, "y": 50, "z": 0}

Vector (a shape, drawn as a single line):

    {"label": "mouth", "shape": "CLOSED", "x": 450, "y": 600,
     "z": 2, "width": 200, "height": 29}

    {"label": "stroke_1", "points_01": [[0.0, 0.1], ...],
     "x": 100, "y": 100, "z": 1, "width": 120, "height": 80,
     "color": "black", "stroke_width": 3}

"shape" is a label from mouth_library_v2/manifest.json; "points_01"
is an inline shape (e.g. a brush stroke), normalised to 0..1 in
both axes with y = 0 at the TOP of the shape (chart convention).
"width"/"height" are the shape's size in board units. "closed"
(optional) says whether the path closes on itself — true for
library loops, false for brush strokes.

Coordinates
-----------
The board grid has its origin at the BOTTOM-LEFT corner, y up.
An asset's (x, y) is where its anchor point sits; the anchor is
one of the six options (default "bottom-left", the hook at the
image's bottom-left corner). See anchor_offset().
"""

import base64
import binascii
import copy
import json
import math
import re
import struct
from pathlib import Path

import numpy as np

ANCHORS = ("top-left", "top-right", "bottom-left", "bottom-right",
           "center", "custom")

N_POINTS = 64  # points every shape is resampled to before blending

DEFAULT_VECTOR_WIDTH = 200.0


# ----------------------------------------------------------------------
# Small math helpers
# ----------------------------------------------------------------------

def smoothstep(u):
    """Ease u in [0, 1] through smoothstep (0 at the ends, fast mid)."""
    u = min(1.0, max(0.0, float(u)))
    return u * u * (3.0 - 2.0 * u)


def _lerp(a, b, e):
    return (1.0 - e) * a + e * b


def _resample_closed(pts, n):
    """Resample a closed loop of points to exactly `n` points.

    Same orientation (signed area < 0) and start at the leftmost
    point, so two shapes resampled this way morph point-by-point
    without twisting. Reimplemented from lip_outline._resample_closed
    (board modules must not import the video tools).
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return np.tile(pts[0], (n, 1))
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1)
                        - np.roll(pts[:, 0], -1) * pts[:, 1])
    if area > 0:
        pts = pts[::-1]
    start = int(np.argmin(pts[:, 0]))
    pts = np.vstack([pts[start:], pts[:start]])
    closed = np.vstack([pts, pts[:1]])
    d = np.sqrt(((np.diff(closed, axis=0)) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, closed[:, 0]),
                            np.interp(t, s, closed[:, 1])])


def _resample_open(pts, n):
    """Resample an open polyline (a brush stroke) to exactly `n` points.

    Arc-length resampling that keeps the start and end points, so
    two strokes morph point-by-point along their length.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return np.tile(pts[0], (n, 1))
    d = np.sqrt(((np.diff(pts, axis=0)) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return np.tile(pts[0], (n, 1))
    t = np.linspace(0.0, s[-1], n, endpoint=True)
    return np.column_stack([np.interp(t, s, pts[:, 0]),
                            np.interp(t, s, pts[:, 1])])


# ----------------------------------------------------------------------
# Anchors
# ----------------------------------------------------------------------

def anchor_offset(anchor, w, h, fx=None, fy=None):
    """Offset of the anchor from the asset's bottom-left corner.

    In board units (y up): the asset covers
    [x - ax, x + w - ax] by [y - ay, y + h - ay] when its anchor
    sits at (x, y). Same convention as render_anchor.anchor_offset.
    """
    if anchor == "bottom-left":
        return (0.0, 0.0)
    if anchor == "bottom-right":
        return (float(w), 0.0)
    if anchor == "top-left":
        return (0.0, float(h))
    if anchor == "top-right":
        return (float(w), float(h))
    if anchor == "center":
        return (float(w) / 2.0, float(h) / 2.0)
    if anchor == "custom":
        if fx is None or fy is None:
            raise ValueError(
                "a custom anchor needs anchor_fx and anchor_fy")
        if not (0 <= fx <= 1 and 0 <= fy <= 1):
            raise ValueError(
                "anchor_fx and anchor_fy must be between 0 and 1")
        return (fx * float(w), fy * float(h))
    raise ValueError(f"unknown anchor {anchor!r}; use one of {ANCHORS}")


def asset_rect(asset, img_w, img_h):
    """Board rect (left, bottom, width, height) of an asset.

    `img_w`, `img_h` are the asset's natural size in board units
    (pixels for raster, width/height for vector) before `scale`.
    """
    scale = asset.get("scale", 1.0)
    w = float(img_w) * scale
    h = float(img_h) * scale
    ax, ay = anchor_offset(asset.get("anchor", "bottom-left"), w, h,
                           asset.get("anchor_fx"), asset.get("anchor_fy"))
    return (float(asset["x"]) - ax, float(asset["y"]) - ay, w, h)


# ----------------------------------------------------------------------
# Images (stdlib PNG size parsing — no cv2/Qt needed here)
# ----------------------------------------------------------------------

def png_size(data):
    """(width, height) of a PNG from its raw bytes, via the IHDR chunk."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG image")
    w, h = struct.unpack(">II", data[16:24])
    return w, h


def load_image_bytes(asset, base_dir=None):
    """Raw PNG bytes for a raster asset.

    From "image_data" (base64) or from the "image" file, resolved
    relative to `base_dir` (the manifest's folder).
    """
    if "image_data" in asset:
        try:
            return base64.b64decode(asset["image_data"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                f"asset {asset.get('label', asset.get('image', '?'))!r}: "
                f"image_data is not valid base64 ({exc})") from None
    path = asset["image"]
    full = Path(path) if base_dir is None else Path(base_dir) / path
    if not full.is_file():
        raise ValueError(f"asset image file not found: {full}")
    return full.read_bytes()


# ----------------------------------------------------------------------
# Mouth library
# ----------------------------------------------------------------------

def load_mouth_library(manifest_path):
    """Load the traced mouth shapes.

    Returns {label: {"points": (N, 2) array in [0, 1], y = 0 at top,
    "drawn_as": "single_line" | "single_loop", "aspect": height/width}}.
    """
    data = json.loads(Path(manifest_path).read_text())
    lib = {}
    for entry in data:
        pts = np.array(entry["points_01"], dtype=float)
        w, h = entry["chart_box"][2], entry["chart_box"][3]
        aspect = h / w if w else 0.1
        lib[entry["label"]] = {
            "points": pts,
            "drawn_as": entry["drawn_as"],
            "aspect": aspect,
        }
    return lib


def _library_loop(entry):
    """The closed loop a library shape morphs as.

    Single-line shapes (closed mouths) are stored as a flattened
    loop — along the centreline and back — so they morph like any
    loop, the same trick lip_outline.load_mouth_library uses.
    """
    pts = entry["points"]
    if entry["drawn_as"] == "single_line":
        return np.vstack([pts, pts[::-1]])
    return pts


def vector_size(asset, library=None):
    """(width, height) in board units of a vector asset."""
    if "width" in asset and "height" in asset:
        return float(asset["width"]), float(asset["height"])
    w = DEFAULT_VECTOR_WIDTH
    aspect = 1.0
    if "shape" in asset and library is not None:
        entry = library.get(asset["shape"])
        if entry is not None:
            aspect = entry["aspect"]
    return w, w * aspect


def asset_display_size(asset, base_dir=None, library=None):
    """(width, height) in board units of an asset, after `scale`."""
    scale = asset.get("scale", 1.0)
    if is_raster(asset):
        data = load_image_bytes(asset, base_dir)
        w, h = png_size(data)
        return w * scale, h * scale
    w, h = vector_size(asset, library)
    return w * scale, h * scale


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def is_raster(asset):
    return "image" in asset or "image_data" in asset


def is_vector(asset):
    return "shape" in asset or "points_01" in asset


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v)


def validate_asset(asset, base_dir=None, library=None):
    """Validate one asset dict; raises ValueError with a clear message."""
    if not isinstance(asset, dict):
        raise ValueError(f"asset must be an object, got {asset!r}")
    raster, vector = is_raster(asset), is_vector(asset)
    if raster and vector:
        raise ValueError(
            f"asset {asset.get('label', '?')!r}: has both image and shape")
    if not raster and not vector:
        raise ValueError(
            f"asset {asset.get('label', '?')!r}: needs 'image'/'image_data' "
            "or 'shape'/'points_01'")
    for field in ("x", "y"):
        if field not in asset:
            raise ValueError(f"asset {asset.get('label', '?')!r}: "
                             f"missing '{field}'")
        if not _is_number(asset[field]):
            raise ValueError(f"asset {asset.get('label', '?')!r}: "
                             f"'{field}' must be a number")
    if "z" in asset:
        if not isinstance(asset["z"], int) or isinstance(asset["z"], bool):
            raise ValueError(f"asset {asset.get('label', '?')!r}: "
                             "'z' must be an integer")
    scale = asset.get("scale", 1.0)
    if not _is_number(scale) or scale <= 0:
        raise ValueError(f"asset {asset.get('label', '?')!r}: "
                         "'scale' must be a number above 0")
    anchor = asset.get("anchor", "bottom-left")
    if anchor not in ANCHORS:
        raise ValueError(f"asset {asset.get('label', '?')!r}: "
                         f"unknown anchor {anchor!r}; use one of {ANCHORS}")
    if anchor == "custom":
        for f in ("anchor_fx", "anchor_fy"):
            if f not in asset:
                raise ValueError(f"asset {asset.get('label', '?')!r}: "
                                 f"a custom anchor needs '{f}'")
            if not _is_number(asset[f]) or not (0 <= asset[f] <= 1):
                raise ValueError(f"asset {asset.get('label', '?')!r}: "
                                 f"'{f}' must be between 0 and 1")
    if raster:
        if "image" in asset and "image_data" in asset:
            raise ValueError(f"asset {asset.get('label', '?')!r}: "
                             "give 'image' or 'image_data', not both")
        data = load_image_bytes(asset, base_dir)
        png_size(data)  # raises if the bytes are not a PNG
    if vector:
        label = asset.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("a vector asset needs a non-empty 'label'")
        if "shape" in asset and "points_01" in asset:
            raise ValueError(f"vector asset {label!r}: give 'shape' or "
                             "'points_01', not both")
        if "shape" in asset:
            if not isinstance(asset["shape"], str):
                raise ValueError(f"vector asset {label!r}: 'shape' must be "
                                 "a shape name")
            if library is not None and asset["shape"] not in library:
                raise ValueError(
                    f"vector asset {label!r}: unknown shape "
                    f"{asset['shape']!r}; known shapes: "
                    + ", ".join(sorted(library)))
        if "points_01" in asset:
            pts = asset["points_01"]
            if not isinstance(pts, list) or len(pts) < 2:
                raise ValueError(f"vector asset {label!r}: 'points_01' "
                                 "needs at least 2 points")
            for p in pts:
                if (not isinstance(p, (list, tuple)) or len(p) != 2
                        or not _is_number(p[0]) or not _is_number(p[1])
                        or not (0 <= p[0] <= 1 and 0 <= p[1] <= 1)):
                    raise ValueError(
                        f"vector asset {label!r}: every point in "
                        "'points_01' must be [x, y] in [0, 1]")
            for f in ("width", "height"):
                if f not in asset:
                    raise ValueError(f"vector asset {label!r}: an inline "
                                     f"shape needs '{f}'")
                if not _is_number(asset[f]) or asset[f] <= 0:
                    raise ValueError(f"vector asset {label!r}: '{f}' must "
                                     "be a number above 0")
        if "stroke_width" in asset:
            if not _is_number(asset["stroke_width"]) \
                    or asset["stroke_width"] <= 0:
                raise ValueError(f"vector asset {label!r}: 'stroke_width' "
                                 "must be a number above 0")


def validate_scene(scene, base_dir=None, library=None):
    """Validate a scene dict; raises ValueError with a clear message."""
    if not isinstance(scene, dict):
        raise ValueError(f"scene must be an object, got {scene!r}")
    for f in ("width", "height"):
        if f not in scene:
            raise ValueError(f"scene is missing '{f}'")
        if not _is_number(scene[f]) or scene[f] <= 0:
            raise ValueError(f"scene '{f}' must be a number above 0")
    if "t" in scene and (not _is_number(scene["t"]) or scene["t"] < 0):
        raise ValueError("scene 't' must be a number >= 0")
    assets = scene.get("assets")
    if not isinstance(assets, list):
        raise ValueError("scene 'assets' must be a list")
    seen_labels = set()
    for asset in assets:
        validate_asset(asset, base_dir, library)
        if is_vector(asset):
            if asset["label"] in seen_labels:
                raise ValueError(
                    f"duplicate vector label {asset['label']!r} in one scene")
            seen_labels.add(asset["label"])


# ----------------------------------------------------------------------
# Loading / saving
# ----------------------------------------------------------------------

def load_manifest(path, library=None):
    """Load and validate one manifest file -> scene dict."""
    path = Path(path)
    try:
        scene = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from None
    validate_scene(scene, base_dir=path.parent, library=library)
    return scene


def save_manifest(scene, path):
    """Write a scene dict to a manifest file (validates first)."""
    validate_scene(scene)
    Path(path).write_text(json.dumps(scene, indent=2) + "\n")


def frame_filename(t, fps=12):
    """Manifest file name for the keyframe at time `t`."""
    return "frame_%04d.json" % (int(round(t * fps)) + 1)


def _time_from_filename(name, fps=12):
    m = re.search(r"(\d+)", Path(name).stem)
    if not m:
        return None
    return (int(m.group(1)) - 1) / fps


def load_keyframes(folder, fps=12, library=None):
    """Load a folder of keyframe files -> list of scene dicts by time.

    Files must be *.json; each needs a "t" (seconds), or the time is
    derived from the number in the file name (frame_0001 -> t=0 at
    12fps). Returns the scenes sorted by "t".
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"keyframe folder not found: {folder}")
    files = sorted(folder.glob("*.json"))
    if not files:
        raise ValueError(f"no *.json keyframe files in {folder}")
    scenes = []
    for path in files:
        scene = load_manifest(path, library=library)
        if "t" not in scene:
            t = _time_from_filename(path.name, fps)
            if t is None:
                raise ValueError(f"{path}: keyframe file needs a 't' "
                                 "(seconds)")
            scene["t"] = t
        scenes.append(scene)
    scenes.sort(key=lambda s: s["t"])
    return scenes


def save_keyframe(scene, folder, fps=12):
    """Write one keyframe scene into `folder`, named by its time."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / frame_filename(scene["t"], fps)
    save_manifest(scene, path)
    return path


# ----------------------------------------------------------------------
# Interpolation
# ----------------------------------------------------------------------

def _vector_interp_points(asset, library):
    """(64, 2) points in [0, 1] (y = 0 at top) a vector asset morphs as.

    Returns (points, closed): library shapes resample their closed
    loop (single-line shapes as a flattened loop); inline brush
    strokes resample as open polylines.
    """
    if "shape" in asset:
        if library is None:
            raise ValueError(
                f"vector asset {asset['label']!r}: a library is needed to "
                "resolve its shape")
        entry = library[asset["shape"]]
        loop = _library_loop(entry)
        return _resample_closed(loop, N_POINTS), True
    pts = np.array(asset["points_01"], dtype=float)
    closed = bool(asset.get("closed", False))
    if closed:
        return _resample_closed(pts, N_POINTS), True
    return _resample_open(pts, N_POINTS), False


def interpolate_vector(a, b, e, library=None):
    """One in-between vector asset between `a` and `b` at ease `e`.

    Points blend point-by-point (both resampled to 64 first);
    width/height, x, y and scale lerp; z and anchor hold from `a`.
    The result is an inline "points_01" asset.
    """
    pa, closed_a = _vector_interp_points(a, library)
    pb, closed_b = _vector_interp_points(b, library)
    pts = (1.0 - e) * pa + e * pb
    wa, ha = vector_size(a, library)
    wb, hb = vector_size(b, library)
    out = {
        "label": a["label"],
        "points_01": [[round(float(x), 4), round(float(y), 4)]
                      for x, y in pts],
        "closed": closed_a and closed_b,
        "x": _lerp(a["x"], b["x"], e),
        "y": _lerp(a["y"], b["y"], e),
        "z": a.get("z", 0),
        "scale": _lerp(a.get("scale", 1.0), b.get("scale", 1.0), e),
        "anchor": a.get("anchor", "bottom-left"),
        "width": _lerp(wa, wb, e),
        "height": _lerp(ha, hb, e),
        "color": a.get("color", "black"),
        "stroke_width": a.get("stroke_width", 3),
    }
    if out["anchor"] == "custom":
        out["anchor_fx"] = a.get("anchor_fx", 0.0)
        out["anchor_fy"] = a.get("anchor_fy", 0.0)
    return out


def _raster_key(asset):
    if "image" in asset:
        return ("image", asset["image"])
    return ("image_data", hash(asset.get("image_data", "")))


def _asset_key(asset):
    if is_raster(asset):
        return _raster_key(asset)
    return ("vector", asset["label"])


def interpolate_asset(a, b, e, library=None):
    """One in-between asset between `a` and `b` at ease `e` in [0, 1].

    Raster assets (same image) lerp x, y, scale; vector assets also
    morph their shape. z and anchor hold from `a`. Mismatched asset
    kinds hold `a` unchanged.
    """
    if is_raster(a) and is_raster(b) and _raster_key(a) == _raster_key(b):
        out = copy.deepcopy(a)
        out["x"] = _lerp(a["x"], b["x"], e)
        out["y"] = _lerp(a["y"], b["y"], e)
        out["scale"] = _lerp(a.get("scale", 1.0), b.get("scale", 1.0), e)
        return out
    if is_vector(a) and is_vector(b):
        return interpolate_vector(a, b, e, library)
    return copy.deepcopy(a)


def interpolate_scenes(scene_a, scene_b, u, library=None):
    """The scene `u` of the way from `scene_a` to `scene_b` (u in [0, 1]).

    Assets are matched by image path (raster) or label (vector);
    an asset in A but not B holds its A values, an asset only in B
    appears with its B values. z and anchor hold from the earlier
    keyframe.
    """
    e = smoothstep(u)
    out = copy.deepcopy(scene_a)
    out["t"] = _lerp(scene_a.get("t", 0.0), scene_b.get("t", 0.0), u)
    by_key = {_asset_key(a): a for a in scene_b.get("assets", [])}
    new_assets = []
    for asset in scene_a.get("assets", []):
        other = by_key.get(_asset_key(asset))
        if other is None:
            new_assets.append(copy.deepcopy(asset))
        else:
            new_assets.append(interpolate_asset(asset, other, e, library))
    seen = {_asset_key(a) for a in scene_a.get("assets", [])}
    for asset in scene_b.get("assets", []):
        if _asset_key(asset) not in seen:
            new_assets.append(copy.deepcopy(asset))
    out["assets"] = new_assets
    return out


def frame_at_time(keyframes, t, library=None):
    """The scene at time `t`, interpolated between the keyframes.

    Before the first keyframe the first scene holds; after the last,
    the last scene holds.
    """
    if not keyframes:
        raise ValueError("no keyframes to interpolate from")
    ordered = sorted(keyframes, key=lambda s: s["t"])
    if t <= ordered[0]["t"]:
        return copy.deepcopy(ordered[0])
    if t >= ordered[-1]["t"]:
        return copy.deepcopy(ordered[-1])
    for a, b in zip(ordered, ordered[1:]):
        if a["t"] <= t <= b["t"]:
            u = (t - a["t"]) / (b["t"] - a["t"]) if b["t"] > a["t"] else 0.0
            return interpolate_scenes(a, b, u, library)
    return copy.deepcopy(ordered[-1])


def generate_in_betweens(keyframes, src_t, tgt_t, fps=12, library=None):
    """Intermediate scenes strictly between two keyframes.

    One scene at each 1/fps step between `src_t` and `tgt_t`.
    A step that lands on an existing keyframe is never overwritten —
    it is skipped. Returns [(t, scene), ...] in time order.
    """
    ordered = sorted(keyframes, key=lambda s: s["t"])
    times = [s["t"] for s in ordered]
    if not any(abs(t - src_t) < 1e-6 for t in times):
        raise ValueError(f"no keyframe at t={src_t}")
    if not any(abs(t - tgt_t) < 1e-6 for t in times):
        raise ValueError(f"no keyframe at t={tgt_t}")
    if tgt_t <= src_t:
        raise ValueError("target keyframe must be after the source")
    out = []
    n = int(round((tgt_t - src_t) * fps))
    for i in range(1, n):
        t = src_t + i / fps
        if any(abs(t - existing) < 1e-6 for existing in times):
            continue  # never overwrite a user keyframe
        out.append((t, frame_at_time(ordered, t, library)))
    return out


def next_stroke_label(scene):
    """Next free brush-stroke label ("stroke_1", "stroke_2", ...) in a scene."""
    n = 0
    for asset in scene.get("assets", []):
        label = asset.get("label", "")
        m = re.fullmatch(r"stroke_(\d+)", label)
        if m:
            n = max(n, int(m.group(1)))
    return f"stroke_{n + 1}"
