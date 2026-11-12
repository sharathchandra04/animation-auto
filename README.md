# Lip outline — slow-motion video

Extracts **only the lip outline** (outer and/or inner lip loops, from
MediaPipe Face Mesh) and draws it exactly on top of the mouth, frame by
frame, overlaid on the original video. No other face landmarks are drawn.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Already installed here in `.venv`: opencv-python-headless 4.11.0,
mediapipe 0.10.21, numpy 1.26.4. Run with `.venv/bin/python`.

## Run

```bash
# Default: half speed (0.5x), green outline, overlaid on the original video
.venv/bin/python lip_outline.py --input video.mp4 --output lips_slow.mp4

# Every run is parameterised — change the flags to get a different output:
.venv/bin/python lip_outline.py --input video.mp4 --output check_025.mp4 --speed 0.25
.venv/bin/python lip_outline.py --input video.mp4 --output check_red.mp4 --speed 0.5 --color red --thickness 3 --lips outer
.venv/bin/python lip_outline.py --input video.mp4 --output lips_only_025.mp4 --speed 0.25 --mode black
```

## Parameters

- `--input` — input video file (a positional filename works too)
- `--output` — output file name, e.g. `lips_slow.mp4`
- `--speed` (alias `--slow`) — 0.5 = half speed (default), 0.25 = quarter speed
- `--color` — green (default), red, blue, yellow, white, cyan, magenta
- `--thickness` — outline thickness in pixels (default 2)
- `--lips` — outer, inner, or both (default both)
- `--start` / `--end` — start/end time in seconds (optional trim)
- `--scale` — resize output, e.g. 0.5 = half size (default 1.0)
- `--mode` — overlay (default, outline on original video), black (outline on
  a black background only), both (write both)
- `--no-audio` — drop audio. By default audio is kept and slowed with
  ffmpeg when ffmpeg is available; otherwise the output is silent.

## Outputs

- Default: `<input>_lips_overlay.mp4` — original video underneath, lip
  outline on top
- With `--mode black`: `<input>_lips_only.mp4` — black background, only
  the lip outline
- With `--mode both`: both of the above in one pass
- With `--output name.mp4`: exactly that file (only with a single `--mode`)

Outputs play at the slowed fps, so they are 2x as long at 0.5x (4x at 0.25x).

## How it works

MediaPipe Face Mesh finds the face each frame. The script draws only the
lip landmark loops (outer: 20 points, inner: 20 points) with OpenCV.
Frames with no face detected are still written (original frame in overlay
mode, black frame in black mode) so the timing never drifts; the final
summary counts them.

## Tested

- Test video: 60 frames, 640x360, 30 fps, 2.0 s
- Run 1 (`--speed 0.5 --color red --thickness 3 --lips outer`):
  output 60 frames, 15 fps, 4.0 s — verified with ffprobe
- Run 2 (`--speed 0.25 --scale 0.5 --start 0 --end 1 --lips both`):
  output 30 frames, 320x180, 7.5 fps, 4.0 s — verified with ffprobe
- `draw_lips` unit test (`test_draw.py`): 5,759 red pixels drawn — passed
- Note: the synthetic test video has no real face, so the lip drawing
  itself was verified by the unit test, not on a real face yet. Run it
  on a real video to check the outline quality.
