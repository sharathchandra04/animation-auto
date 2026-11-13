#!/usr/bin/env python3
"""
Lip-outline extractor — slow-motion output video with ONLY the lip outline.

Usage:
    python lip_outline.py --input video.mp4 --output lips_slow.mp4 --speed 0.5 --color green --thickness 2
    python lip_outline.py video.mp4 --speed 0.25 --output check_025.mp4
    python lip_outline.py --input video.mp4 --output lips_red.mp4 --speed 0.5 --color red --thickness 3 --lips outer
    python lip_outline.py --input video.mp4 --output wuhan_noodles_lips_slow.mp4 --speed 0.5 --style ghibli

Styles (--style, case-insensitive):
  traditional (default)  trace the real lip loops, outer and/or inner.
  ghibli                 one single Ghibli-style line only, overlaid on the
                         original video — no lip tracing, no skin patch.
                         Closed mouth: one curved line. Open mouth: one
                         single oval/circle outline. The shape is measured
                         from the lip landmarks (width, height, centre) each
                         frame and smoothed, so it opens and closes as the
                         person talks/eats. Default colour black, thickness 3.
  lipsync                real lip-sync animation from the audio. phonemes.py
                         transcribes the audio (Whisper), breaks words into
                         phonemes, and maps each phoneme to one of the 12
                         mouth shapes in the Lip Sync chart library
                         (mouth_library_v2/manifest.json). The shape is NOT
                         just swapped per phoneme: every shape is a traced
                         point list, resampled to the same 64 points, and
                         the points MORPH from the old shape to the new one
                         over a short transition (~0.05s, eased), so the
                         mouth travels between keyframes. Audio decides
                         WHICH shape and WHEN; the face landmarks decide
                         WHERE and HOW BIG (scaled to the real mouth
                         corners each frame). Default colour black,
                         thickness 3. Needs the phoneme timeline JSON
                         (built automatically from the video's audio if
                         it is not already there).

Output naming: the style is always added to the output file name, just
before the extension, e.g. --output wuhan_noodles_lips_slow.mp4 with
--style traditional gives wuhan_noodles_lips_slow_traditional.mp4, and
with --style ghibli gives wuhan_noodles_lips_slow_ghibli.mp4. The same
applies to auto-named outputs.

What it does, frame by frame:
  1. MediaPipe Face Mesh finds the face (max 1 face).
  2. It takes ONLY the lip loops — outer and/or inner, as chosen with
     --lips — no other face landmarks, no eyes/nose/jaw.
  3. It draws those loops as a thin polyline exactly on the mouth,
     overlaid on the original video (default --mode overlay).
  4. It writes the result at fps * --speed, so the same frames play
     slower (0.5x = twice as long, 0.25x = four times as long).
     No frames are dropped and no interpolation is invented.

Parameters (all changeable on the command line):
  --input / positional input   input video file
  --output                     output file name, e.g. lips_slow.mp4
                               (the style is appended: _traditional/_ghibli)
  --style / --mouth-style      traditional (default), ghibli, or lipsync
  --phonemes                   phoneme timeline JSON for --style lipsync
                               (default: <input>_phonemes.json next to the
                               input video; built automatically if missing)
  --morph                      transition time in seconds between mouth
                               shapes for --style lipsync (default 0.05)
  --speed / --slow             playback speed, 0.5 = half speed (default),
                               0.25 = quarter speed. Must be > 0 and <= 1.
  --color                      green (default for traditional), black
                               (default for ghibli), red, blue, yellow,
                               white, cyan, magenta
  --thickness                  outline thickness in pixels (default 2 for
                               traditional, 3 for ghibli)
  --lips                       outer, inner, or both (default both;
                               traditional style only, ignored for ghibli)
  --start / --end              start/end time in seconds (optional trim)
  --scale                      resize output, e.g. 0.5 = half size (default 1.0)
  --mode                       overlay (default, outline on original video),
                               black (outline on black background only),
                               both (write both)
  --no-audio                   drop audio. By default audio is kept and
                               slowed with ffmpeg when ffmpeg is available;
                               if ffmpeg is not available the output is
                               silent and a warning is printed.

Notes:
  - Frames where no face is found are written unchanged in structure
    (overlay: original frame; black: black frame) and counted in the
    summary, so timing never drifts.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# MediaPipe lip landmark loops (standard Face Mesh indices).
OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
              291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
              308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# BGR colours (OpenCV order).
COLORS = {
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "black": (0, 0, 0),
}

# Ghibli mouth: corner / top / bottom landmark indices (Face Mesh).
GHIBLI_LEFT = 61    # left mouth corner
GHIBLI_RIGHT = 291  # right mouth corner
GHIBLI_TOP = 0      # top of outer lip, centre
GHIBLI_BOTTOM = 17  # bottom of outer lip, centre
# Below this height/width ratio the mouth counts as closed.
GHIBLI_CLOSED_RATIO = 0.18


def draw_lips(frame, landmarks, color, thickness, lips):
    """Draw only the chosen lip outline(s) on `frame`, in place."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
    loops = []
    if lips in ("outer", "both"):
        loops.append(OUTER_LIPS)
    if lips in ("inner", "both"):
        loops.append(INNER_LIPS)
    for loop in loops:
        loop_pts = np.array([pts[i] for i in loop], dtype=np.int32)
        cv2.polylines(frame, [loop_pts], isClosed=True, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)
    return True


def draw_ghibli(frame, landmarks, color, thickness, state):
    """Draw ONE Ghibli-style mouth line on `frame`, in place.

    Measures only the mouth width (corner to corner), height (top to
    bottom) and centre from the landmarks — it never traces the lip
    shape. Closed mouth: a single curved line. Open mouth: a single
    oval outline. `state` is a dict holding the smoothed measurements
    from previous frames (or None on the first face frame); it is
    updated in place so the shape does not jitter frame to frame.
    Returns True if drawn.
    """
    h, w = frame.shape[:2]
    pts = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]
    left = pts[GHIBLI_LEFT]
    right = pts[GHIBLI_RIGHT]
    top = pts[GHIBLI_TOP]
    bottom = pts[GHIBLI_BOTTOM]
    width = max(1.0, abs(right[0] - left[0]))
    height = max(0.0, abs(bottom[1] - top[1]))
    cx = (left[0] + right[0]) / 2.0
    cy = (top[1] + bottom[1]) / 2.0
    # Exponential smoothing: 60% new measurement, 40% previous.
    if state.get("smooth") is None:
        state["smooth"] = (cx, cy, width, height)
    else:
        pcx, pcy, pw, ph = state["smooth"]
        state["smooth"] = (0.6 * cx + 0.4 * pcx,
                           0.6 * cy + 0.4 * pcy,
                           0.6 * width + 0.4 * pw,
                           0.6 * height + 0.4 * ph)
    cx, cy, width, height = state["smooth"]
    ratio = height / width if width else 0.0
    centre = (int(cx), int(cy))
    half_w = max(2, int(width / 2))
    if ratio < GHIBLI_CLOSED_RATIO:
        # Closed: one curved line — the bottom half of a shallow ellipse,
        # from the left corner to the right corner.
        half_h = max(2, int(width * 0.08))
        cv2.ellipse(frame, centre, (half_w, half_h), 0, 20, 160,
                    color, thickness, lineType=cv2.LINE_AA)
    else:
        # Open: one single oval outline (never a second/inner loop).
        half_h = max(2, int(height / 2))
        cv2.ellipse(frame, centre, (half_w, half_h), 0, 0, 360,
                    color, thickness, lineType=cv2.LINE_AA)
    return True


LIPSYNC_N_POINTS = 64
LIPSYNC_MORPH_DEFAULT = 0.05


def _resample_closed(pts, n):
    """Resample a closed loop of points to exactly `n` points.

    Points are ordered around the loop, starting at the leftmost
    point, all with the same orientation (signed area < 0 in image
    coordinates), so two shapes resampled this way can be morphed
    point-by-point without twisting.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return np.tile(pts[0], (n, 1))
    # Enforce one orientation: signed area < 0.
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1)
                        - np.roll(pts[:, 0], -1) * pts[:, 1])
    if area > 0:
        pts = pts[::-1]
    # Start at the leftmost point.
    start = int(np.argmin(pts[:, 0]))
    pts = np.vstack([pts[start:], pts[:start]])
    closed = np.vstack([pts, pts[:1]])
    d = np.sqrt(((np.diff(closed, axis=0)) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, closed[:, 0]),
                            np.interp(t, s, closed[:, 1])])


def load_mouth_library(manifest_path):
    """Load the traced mouth shapes.

    Returns {shape_name: (points, aspect)} where points is a
    (64, 2) array in 'mouth-width units': x in [0, 1] across the
    mouth, y in [0, aspect] down the mouth, so shapes with different
    proportions morph correctly. aspect = chart height / width.
    Line shapes (closed, m/b/p) are stored as a flattened loop:
    along the centreline and back, so they morph like any loop.
    """
    import json
    data = json.loads(Path(manifest_path).read_text())
    lib = {}
    for entry in data:
        pts = np.array(entry["points_01"], dtype=float)
        w, h = entry["chart_box"][2], entry["chart_box"][3]
        aspect = h / w if w else 0.1
        if entry["drawn_as"] == "single_line":
            loop = np.vstack([pts, pts[::-1]])
        else:
            loop = pts
        resampled = _resample_closed(loop, LIPSYNC_N_POINTS)
        # Scale y into mouth-width units (x stays 0..1).
        resampled = resampled * np.array([1.0, aspect])
        lib[entry["label"]] = (resampled, aspect)
    return lib


def _smoothstep(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def shape_at_time(library, segments, t, morph_s):
    """Morphed mouth shape (points, aspect, shape_name) at time `t`.

    `segments` is the phoneme timeline: [{shape, start, end}].
    Inside a segment (after its first `morph_s` seconds) the shape
    is held exactly. During the first `morph_s` seconds after a
    segment starts, the points morph from the previous shape to the
    new one with smoothstep easing — that is the in-between
    keyframe interpolation, not a hard PNG swap.
    """
    current = None
    previous = None
    for i, seg in enumerate(segments):
        if seg["start"] <= t < seg["end"]:
            current = seg
            previous = segments[i - 1] if i > 0 else None
            break
        if t < seg["start"]:
            current = seg
            previous = segments[i - 1] if i > 0 else None
            break
        previous = seg
    if current is None:
        return (*library["CLOSED"], "CLOSED")
    pts, aspect = library[current["shape"]]
    if previous is None or previous["shape"] == current["shape"]:
        return pts, aspect, current["shape"]
    elapsed = t - current["start"]
    if elapsed >= morph_s:
        return pts, aspect, current["shape"]
    ppts, paspect = library[previous["shape"]]
    u = _smoothstep(elapsed / morph_s)
    return (1.0 - u) * ppts + u * pts, \
        (1.0 - u) * paspect + u * aspect, current["shape"]


def draw_lipsync(frame, landmarks, color, thickness, state, shape_pts, aspect):
    """Draw the morphed mouth shape on `frame`, in place.

    `shape_pts` are in mouth-width units (x 0..1, y 0..aspect).
    The shape is scaled to the real mouth width (landmarks 61-291),
    centred on the real mouth centre, and smoothed like the ghibli
    measurements so it does not jitter.
    """
    h, w = frame.shape[:2]
    pts = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]
    left = pts[GHIBLI_LEFT]
    right = pts[GHIBLI_RIGHT]
    top = pts[GHIBLI_TOP]
    bottom = pts[GHIBLI_BOTTOM]
    width = max(1.0, abs(right[0] - left[0]))
    cx = (left[0] + right[0]) / 2.0
    cy = (top[1] + bottom[1]) / 2.0
    if state.get("smooth") is None:
        state["smooth"] = (cx, cy, width)
    else:
        pcx, pcy, pw = state["smooth"]
        state["smooth"] = (0.6 * cx + 0.4 * pcx,
                           0.6 * cy + 0.4 * pcy,
                           0.6 * width + 0.4 * pw)
    cx, cy, width = state["smooth"]
    xs = cx - width / 2.0 + shape_pts[:, 0] * width
    ys = cy - (aspect / 2.0) * width + shape_pts[:, 1] * width
    poly = np.column_stack([xs, ys]).astype(np.int32)
    cv2.polylines(frame, [poly], isClosed=True, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)
    return True


def styled_output(path, style):
    """Return `path` with _<style> added before the extension.

    wuhan_noodles_lips_slow.mp4 + ghibli ->
    wuhan_noodles_lips_slow_ghibli.mp4. If the name already ends with
    _traditional or _ghibli, that suffix is replaced, not doubled.
    """
    stem = path.stem
    for old in ("_traditional", "_ghibli", "_lipsync"):
        if stem.endswith(old):
            stem = stem[: -len(old)]
            break
    return path.with_name(f"{stem}_{style}{path.suffix}")


def atempo_chain(speed):
    """ffmpeg atempo filters (each 0.5-2.0) that multiply to `speed`."""
    if speed <= 0:
        return None
    chain = []
    remaining = speed
    while remaining < 0.5:
        chain.append("atempo=0.5")
        remaining /= 0.5
    chain.append(f"atempo={remaining:.4f}")
    return ",".join(chain)


def add_slowed_audio(video_path, input_path, speed):
    """Mux the input audio, slowed to `speed`, onto the video-only file.
    Returns True on success, False if ffmpeg is missing or there is no audio."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("Warning: ffmpeg not found, output will be silent.")
        return False
    # Does the input have an audio stream?
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(input_path)],
        capture_output=True, text=True)
    if not probe.stdout.strip():
        print("Note: input has no audio stream, output is silent.")
        return False
    tmp = video_path.with_name(video_path.stem + "_with_audio.mp4")
    filt = atempo_chain(speed)
    cmd = [ffmpeg, "-y", "-v", "error",
           "-i", str(video_path), "-i", str(input_path),
           "-filter:a", filt, "-map", "0:v", "-map", "1:a",
           "-c:v", "copy", "-shortest", str(tmp)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Warning: could not add slowed audio ({res.stderr.strip()[:200]}), "
              "keeping silent output.")
        return False
    tmp.replace(video_path)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_pos", nargs="?", default=None,
                    help="Input video file (or use --input)")
    ap.add_argument("--input", dest="input", default=None,
                    help="Input video file (e.g. video.mp4)")
    ap.add_argument("--output", default=None,
                    help="Output file name, e.g. lips_slow.mp4 "
                         "(the style is appended: _traditional/_ghibli)")
    ap.add_argument("--style", dest="style", type=str.lower, default="traditional",
                    choices=["traditional", "ghibli", "lipsync"],
                    help="traditional: trace the lip loops (default); "
                         "ghibli: one single Ghibli-style mouth line; "
                         "lipsync: audio phonemes mapped to the chart "
                         "mouth shapes, morphed between keyframes")
    ap.add_argument("--mouth-style", dest="style", type=str.lower, default=None,
                    choices=["traditional", "ghibli", "lipsync"],
                    help="Alias for --style")
    ap.add_argument("--phonemes", default=None,
                    help="Phoneme timeline JSON for --style lipsync "
                         "(default: <input>_phonemes.json; built "
                         "automatically from the audio if missing)")
    ap.add_argument("--morph", type=float, default=LIPSYNC_MORPH_DEFAULT,
                    help="Transition time in seconds between mouth shapes "
                         "for --style lipsync (default 0.05)")
    ap.add_argument("--speed", dest="speed", type=float, default=0.5,
                    help="Playback speed, 0.5 = half speed (default), "
                         "0.25 = quarter speed. Must be > 0 and <= 1.")
    ap.add_argument("--slow", dest="speed", type=float, default=None,
                    help="Alias for --speed")
    ap.add_argument("--color", choices=list(COLORS), default=None,
                    help="Outline colour (default: green for traditional, "
                         "black for ghibli)")
    ap.add_argument("--thickness", type=int, default=None,
                    help="Outline thickness in pixels (default: 2 for "
                         "traditional, 3 for ghibli)")
    ap.add_argument("--lips", choices=["outer", "inner", "both"], default="both",
                    help="Which lip outline to draw (default: both)")
    ap.add_argument("--start", type=float, default=None,
                    help="Start time in seconds (optional trim)")
    ap.add_argument("--end", type=float, default=None,
                    help="End time in seconds (optional trim)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Resize output, e.g. 0.5 = half size (default: 1.0)")
    ap.add_argument("--mode", choices=["overlay", "black", "both"], default="overlay",
                    help="overlay: outline on the original video (default); "
                         "black: outline on a black background only; "
                         "both: write both")
    ap.add_argument("--no-audio", action="store_true",
                    help="Drop audio from the output")
    args = ap.parse_args()

    src_name = args.input or args.input_pos
    if not src_name:
        ap.error("give an input video, e.g. --input video.mp4")
    if not 0 < args.speed <= 1:
        ap.error("--speed must be between 0 (exclusive) and 1")
    if not 0 < args.scale <= 1:
        ap.error("--scale must be between 0 (exclusive) and 1")
    if args.output and args.mode == "both":
        ap.error("--output can only be used with --mode overlay or --mode black")
    if args.style == "ghibli" and args.lips != "both":
        print("Note: --lips is ignored with --style ghibli "
              "(one single mouth line, no lip loops).")

    # Per-style defaults: traditional = green, 2px;
    # ghibli / lipsync = black, 3px.
    color_name = args.color or \
        ("black" if args.style in ("ghibli", "lipsync") else "green")
    thickness = args.thickness if args.thickness is not None \
        else (3 if args.style in ("ghibli", "lipsync") else 2)
    print(f"Style: {args.style}  colour: {color_name}  thickness: {thickness}")

    src = Path(src_name)
    if not src.is_file():
        print(f"ERROR: input video not found: {src}", file=sys.stderr)
        return 2

    library = None
    segments = None
    if args.style == "lipsync":
        manifest = Path(__file__).with_name("mouth_library_v2") / "manifest.json"
        if not manifest.is_file():
            print(f"ERROR: mouth library not found: {manifest}", file=sys.stderr)
            return 2
        library = load_mouth_library(manifest)
        print(f"Mouth library: {len(library)} shapes from {manifest}")
        timeline_path = Path(args.phonemes) if args.phonemes else \
            src.with_name(src.stem + "_phonemes.json")
        if not timeline_path.is_file():
            print(f"No phoneme timeline at {timeline_path} — "
                  "building it from the audio now (Whisper)...")
            import phonemes as phonemes_mod
            timeline = phonemes_mod.timeline_for_video(src)
            timeline_path.write_text(json.dumps(timeline, indent=2,
                                                ensure_ascii=False))
            print(f"Wrote {timeline_path}")
        else:
            import json as _json
            timeline = _json.loads(timeline_path.read_text())
        segments = timeline["segments"]
        print(f"Phoneme timeline: {len(timeline['phonemes'])} phonemes, "
              f"{len(segments)} shape segments, "
              f"language {timeline.get('language')}")
        print(f"Transcript: {timeline.get('transcript', '')[:200]}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"ERROR: could not open video: {src}", file=sys.stderr)
        return 2

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps_out = fps_in * args.speed
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = int(width * args.scale), int(height * args.scale)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int((args.start or 0) * fps_in)
    end_frame = int(args.end * fps_in) if args.end else total
    print(f"Input: {src}  {width}x{height}  {fps_in:.2f} fps  "
          f"{total if total else '?'} frames")
    print(f"Output: {out_w}x{out_h}  {fps_out:.2f} fps ({args.speed}x speed, "
          f"{1 / args.speed:.0f}x longer than the original)")

    modes = ["overlay", "black"] if args.mode == "both" else [args.mode]
    writers = {}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for mode in modes:
        if args.output:
            out_path = styled_output(Path(args.output), args.style)
        else:
            suffix = "lips_overlay" if mode == "overlay" else "lips_only"
            out_path = src.with_name(f"{src.stem}_{suffix}_{args.style}.mp4")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (out_w, out_h))
        if not writer.isOpened():
            print(f"ERROR: could not create output: {out_path}", file=sys.stderr)
            return 2
        writers[mode] = (out_path, writer)
        print(f"Writing {mode}: {out_path}")

    color = COLORS[color_name]
    drawn = 0
    no_face = 0
    frame_no = 0
    written = 0
    ghibli_state = {"smooth": None}

    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            if frame_no <= start_frame:
                continue
            if frame_no > end_frame:
                break

            if args.scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True
            face = results.multi_face_landmarks[0] \
                if results.multi_face_landmarks else None

            if "overlay" in writers:
                overlay = frame.copy()
                if face:
                    if args.style == "ghibli":
                        draw_ghibli(overlay, face, color, thickness,
                                    ghibli_state)
                    elif args.style == "lipsync":
                        t = (frame_no - 1) / fps_in
                        pts, aspect, _ = shape_at_time(
                            library, segments, t, args.morph)
                        draw_lipsync(overlay, face, color, thickness,
                                     ghibli_state, pts, aspect)
                    else:
                        draw_lips(overlay, face, color, thickness, args.lips)
                writers["overlay"][1].write(overlay)

            if "black" in writers:
                black = np.zeros_like(frame)
                if face:
                    if args.style == "ghibli":
                        draw_ghibli(black, face, color, thickness,
                                    ghibli_state)
                    elif args.style == "lipsync":
                        t = (frame_no - 1) / fps_in
                        pts, aspect, _ = shape_at_time(
                            library, segments, t, args.morph)
                        draw_lipsync(black, face, color, thickness,
                                     ghibli_state, pts, aspect)
                    else:
                        draw_lips(black, face, color, thickness, args.lips)
                writers["black"][1].write(black)

            written += 1
            if face:
                drawn += 1
            else:
                no_face += 1
                ghibli_state["smooth"] = None
            if written % 100 == 0:
                print(f"  ... {written} frames written "
                      f"({drawn} with lips, {no_face} without a face)")

    cap.release()
    for _, writer in writers.values():
        writer.release()

    print(f"Done: {written} frames — {args.style} mouth drawn on {drawn}, "
          f"no face found on {no_face}.")
    for mode, (path, _) in writers.items():
        print(f"  {mode}: {path}")
        if not args.no_audio and mode == "overlay":
            add_slowed_audio(path, src, args.speed)
    if no_face:
        print("Tip: if many frames have no face, try a closer / better-lit "
              "video, or lower the detection confidence in the code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
