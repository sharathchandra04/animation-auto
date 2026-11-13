#!/usr/bin/env python3
"""Speak — audio in, speaking-mouth video out.

You record the audio (your own voice, any format: wav, mp3, m4a...),
this generates a video of a black-outline mouth speaking on a white
background, in lip-sync with your audio.

Usage:
    python speak.py --input myvoice.m4a --output speaking_mouth.mp4
    python speak.py --input myvoice.wav --output speaking.mp4 --fps 30 --size 720

How it works:
  1. phonemes.py transcribes the audio (Whisper), breaks words into
     phonemes, and maps each phoneme to one of the 12 mouth shapes in
     the Lip Sync chart library (mouth_library_v2/manifest.json).
  2. Every shape is a traced point list, resampled to 64 points, and
     the points MORPH from the old shape to the new one over --morph
     seconds (default 0.1, eased) — in-between keyframes, not swaps.
  3. There is no video input and no face to track, so the mouth is
     drawn in the centre of a white canvas at a fixed size
     (--mouth-width fraction of the canvas), and your original audio
     is muxed onto the video at normal speed.

Parameters:
  --input      audio file you recorded (wav, mp3, m4a, ...)
  --output     output video file, e.g. speaking_mouth.mp4
  --fps        video frame rate (default 30)
  --size       square canvas size in pixels (default 720)
  --mouth-width  mouth width as a fraction of the canvas (default 0.62)
  --thickness  line thickness in pixels (default 6)
  --morph      transition time in seconds between mouth shapes (default 0.1)
  --phonemes   phoneme timeline JSON (default: <input>_phonemes.json;
               built automatically from the audio if missing)
  --model      Whisper model for transcription (default base)
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

import lip_outline as lo
import phonemes as phonemes_mod


def audio_duration(audio_path):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(audio_path)],
        capture_output=True, text=True)
    return float(probe.stdout.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Audio file you recorded")
    ap.add_argument("--output", default=None,
                    help="Output video, e.g. speaking_mouth.mp4")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", type=int, default=720,
                    help="Square canvas size in pixels (default 720)")
    ap.add_argument("--mouth-width", type=float, default=0.62,
                    help="Mouth width as a fraction of the canvas (default 0.62)")
    ap.add_argument("--thickness", type=int, default=6)
    ap.add_argument("--morph", type=float, default=lo.LIPSYNC_MORPH_DEFAULT)
    ap.add_argument("--phonemes", default=None)
    ap.add_argument("--model", default="base")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_file():
        print(f"ERROR: audio file not found: {src}", file=sys.stderr)
        return 2
    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg is required (audio + muxing)", file=sys.stderr)
        return 2

    manifest = Path(__file__).with_name("mouth_library_v2") / "manifest.json"
    library = lo.load_mouth_library(manifest)
    print(f"Mouth library: {len(library)} shapes from {manifest}")

    timeline_path = Path(args.phonemes) if args.phonemes else \
        src.with_name(src.stem + "_phonemes.json")
    if not timeline_path.is_file():
        print(f"No phoneme timeline at {timeline_path} — "
              "building it from the audio now (Whisper)...")
        timeline = phonemes_mod.timeline_for_audio(src, args.model)
        timeline_path.write_text(json.dumps(timeline, indent=2,
                                            ensure_ascii=False))
        print(f"Wrote {timeline_path}")
    else:
        timeline = json.loads(timeline_path.read_text())
    segments = timeline["segments"]
    print(f"Phoneme timeline: {len(timeline['phonemes'])} phonemes, "
          f"{len(segments)} shape segments, "
          f"language {timeline.get('language')}")
    print(f"Transcript: {timeline.get('transcript', '')[:200]}")

    duration = audio_duration(src)
    n_frames = int(round(duration * args.fps))
    print(f"Audio: {duration:.2f}s -> {n_frames} frames at {args.fps} fps")

    out_path = Path(args.output) if args.output else \
        src.with_name(f"{src.stem}_speaking_mouth.mp4")
    tmp_path = out_path.with_name(out_path.stem + "_video_only.mp4")
    size = args.size
    mouth_w = size * args.mouth_width
    cx = cy = size / 2.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, args.fps, (size, size))
    if not writer.isOpened():
        print(f"ERROR: could not create output: {tmp_path}", file=sys.stderr)
        return 2

    for i in range(n_frames):
        t = i / args.fps
        pts, aspect, _ = lo.shape_at_time(library, segments, t, args.morph)
        frame = np.full((size, size, 3), 255, np.uint8)  # white background
        xs = cx - mouth_w / 2.0 + pts[:, 0] * mouth_w
        ys = cy - (aspect / 2.0) * mouth_w + pts[:, 1] * mouth_w
        poly = np.column_stack([xs, ys]).astype(np.int32)
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 0, 0),
                      thickness=args.thickness, lineType=cv2.LINE_AA)
        writer.write(frame)
        if (i + 1) % 300 == 0:
            print(f"  ... {i + 1}/{n_frames} frames")
    writer.release()

    # Mux the original audio at normal speed, H.264 video.
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-i", str(tmp_path), "-i", str(src),
           "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-crf", "23", "-preset", "medium",
           "-c:a", "aac", "-shortest", "-movflags", "+faststart",
           str(out_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    tmp_path.unlink(missing_ok=True)
    if res.returncode != 0:
        print(f"ERROR: could not mux audio ({res.stderr.strip()[:300]})",
              file=sys.stderr)
        return 2
    print(f"Done: {n_frames} frames, {duration:.2f}s -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
