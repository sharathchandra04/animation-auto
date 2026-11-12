#!/usr/bin/env python3
"""
Lip-outline extractor — slow-motion output video with ONLY the lip outline.

Usage:
    python lip_outline.py --input video.mp4 --output lips_slow.mp4 --speed 0.5 --color green --thickness 2
    python lip_outline.py video.mp4 --speed 0.25 --output check_025.mp4
    python lip_outline.py --input video.mp4 --output lips_red.mp4 --speed 0.5 --color red --thickness 3 --lips outer

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
  --speed / --slow             playback speed, 0.5 = half speed (default),
                               0.25 = quarter speed. Must be > 0 and <= 1.
  --color                      green (default), red, blue, yellow, white,
                               cyan, magenta
  --thickness                  outline thickness in pixels (default 2)
  --lips                       outer, inner, or both (default both)
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
}


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
                    help="Output file name, e.g. lips_slow.mp4")
    ap.add_argument("--speed", dest="speed", type=float, default=0.5,
                    help="Playback speed, 0.5 = half speed (default), "
                         "0.25 = quarter speed. Must be > 0 and <= 1.")
    ap.add_argument("--slow", dest="speed", type=float, default=None,
                    help="Alias for --speed")
    ap.add_argument("--color", choices=list(COLORS), default="green",
                    help="Outline colour (default: green)")
    ap.add_argument("--thickness", type=int, default=2,
                    help="Outline thickness in pixels (default: 2)")
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

    src = Path(src_name)
    if not src.is_file():
        print(f"ERROR: input video not found: {src}", file=sys.stderr)
        return 2

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
            out_path = Path(args.output)
        else:
            suffix = "lips_overlay" if mode == "overlay" else "lips_only"
            out_path = src.with_name(f"{src.stem}_{suffix}.mp4")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (out_w, out_h))
        if not writer.isOpened():
            print(f"ERROR: could not create output: {out_path}", file=sys.stderr)
            return 2
        writers[mode] = (out_path, writer)
        print(f"Writing {mode}: {out_path}")

    color = COLORS[args.color]
    drawn = 0
    no_face = 0
    frame_no = 0
    written = 0

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
                    draw_lips(overlay, face, color, args.thickness, args.lips)
                writers["overlay"][1].write(overlay)

            if "black" in writers:
                black = np.zeros_like(frame)
                if face:
                    draw_lips(black, face, color, args.thickness, args.lips)
                writers["black"][1].write(black)

            written += 1
            if face:
                drawn += 1
            else:
                no_face += 1
            if written % 100 == 0:
                print(f"  ... {written} frames written "
                      f"({drawn} with lips, {no_face} without a face)")

    cap.release()
    for _, writer in writers.values():
        writer.release()

    print(f"Done: {written} frames — lips drawn on {drawn}, "
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
