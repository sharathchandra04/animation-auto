# Lip outline — slow-motion video

Extracts **only the lip outline** (outer and/or inner lip loops, from
MediaPipe Face Mesh) and draws it exactly on top of the mouth, frame by
frame, overlaid on the original video. No other face landmarks are drawn.

Or, with `--style ghibli`, it draws **one single Ghibli-style mouth
line** instead: a curved line when the mouth is closed, a single oval
outline when it is open. No lip tracing, no skin patch — overlay only.

Or, with `--style lipsync`, it does **real lip-sync animation from the
audio**: `phonemes.py` transcribes the audio with Whisper, breaks words
into phonemes, and maps each phoneme to one of the 12 mouth shapes in
the Lip Sync chart library (`mouth_library_v2/manifest.json`). Shapes are
traced point lists, resampled to 64 points, and **morphed** between
keyframes over `--morph` seconds (default 0.1, eased) — not just swapped.
The timeline JSON is built automatically from the video's audio if
`<input>_phonemes.json` is not already there.

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
# Default: half speed (0.5x), traditional style, green outline
.venv/bin/python lip_outline.py --input video.mp4 --output lips_slow.mp4

# Ghibli style: one single black mouth line, overlay only
.venv/bin/python lip_outline.py --input video.mp4 --output wuhan_noodles_lips_slow.mp4 --speed 0.5 --style ghibli

# Lip-sync style: phonemes from the audio mapped to the 12 chart mouths,
# morphed between keyframes (timeline built automatically if missing)
.venv/bin/python lip_outline.py --input video.mp4 --output wuhan_noodles_lips_slow.mp4 --speed 0.5 --style lipsync

# Every run is parameterised — change the flags to get a different output:
.venv/bin/python lip_outline.py --input video.mp4 --output check_025.mp4 --speed 0.25
.venv/bin/python lip_outline.py --input video.mp4 --output check_red.mp4 --speed 0.5 --color red --thickness 3 --lips outer
.venv/bin/python lip_outline.py --input video.mp4 --output lips_only_025.mp4 --speed 0.25 --mode black
```

## Speak — audio in, speaking-mouth video out (`speak.py`)

Record your own voice, get a video of a black-outline mouth speaking on
a white background, in lip-sync with your audio:

```bash
.venv/bin/python speak.py --input myvoice.m4a --output speaking_mouth.mp4
```

It transcribes the audio with Whisper (see `phonemes.py`), maps phonemes
to the same 12 chart mouth shapes, and morphs between them over
`--morph` seconds (default 0.1). No video input, no face tracking — the
mouth is drawn in the centre of a white 720x720 canvas at a fixed size
(`--size`, `--mouth-width` 0.62, `--thickness` 6), and your original
audio is muxed on at normal speed. Any audio format works (wav, mp3,
m4a). Video is 12 fps by default (`--fps`) — the classic animation
rate: a frame lasts 0.083 s, close to the median phoneme (0.073 s), so
roughly one mouth shape lands on each frame. Only the very shortest
phonemes are skipped — shapes under 0.04 s merge into their neighbours
(`MIN_HOLD = 0.04` in `phonemes.py`), 47 of 429 segments (11%), mostly
fleeting consonants. The morph is 0.05 s. (60 fps and a 0.1 s minimum
hold were both tried and reverted: 60 fps alone did not fix the
flicker, and the 0.1 s hold cut 429 segments to 109, deleting whole
sounds.) Tested on the Wuhan audio: 57.05 s, 685 frames at 12 fps,
382 shape segments.

## Parameters

- `--input` — input video file (a positional filename works too)
- `--output` — output file name, e.g. `lips_slow.mp4`
- `--style` (alias `--mouth-style`) — `traditional` (default), `ghibli`,
  or `lipsync`, case-insensitive. Say `traditional` for the lip loops,
  `ghibli` for the single Ghibli mouth line, `lipsync` for the audio
  phoneme animation.
- `--phonemes` — phoneme timeline JSON for `--style lipsync` (default:
  `<input>_phonemes.json` next to the input video; built automatically
  from the audio if missing)
- `--morph` — transition time in seconds between mouth shapes for
  `--style lipsync` (default 0.1)
- `--speed` (alias `--slow`) — 0.5 = half speed (default), 0.25 = quarter speed
- `--color` — green (default for traditional), black (default for ghibli),
  red, blue, yellow, white, cyan, magenta
- `--thickness` — outline thickness in pixels (default 2 for traditional,
  3 for ghibli)
- `--lips` — outer, inner, or both (default both; traditional only,
  ignored for ghibli)
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
- With `--output name.mp4`: that name **plus the style**, e.g.
  `wuhan_noodles_lips_slow.mp4` + `--style traditional` →
  `wuhan_noodles_lips_slow_traditional.mp4`, and + `--style ghibli` →
  `wuhan_noodles_lips_slow_ghibli.mp4`. An existing `_traditional` /
  `_ghibli` suffix is replaced, not doubled.

Outputs play at the slowed fps, so they are 2x as long at 0.5x (4x at 0.25x).

## How it works

MediaPipe Face Mesh finds the face each frame.

- **Traditional:** the script draws only the lip landmark loops (outer:
  20 points, inner: 20 points) with OpenCV.
- **Lipsync:** `phonemes.py` transcribes the audio (Whisper, word
  timestamps), breaks each word into phonemes (CMU dictionary for
  English, pinyin mapping for Chinese), and maps each phoneme to one
  of the 12 chart mouth shapes: silence CLOSED; M/B/P M_B_P; F/V F_V;
  S/C/Z/SH/CH/D/T/K/G/N S_C_Z_SH_CH; AH/A AH_I_2; I AH_I_1; EE/UH/EN
  EE_UH_EN; OH/O OH; OO/U/W/R OO_U_W_R; L/TH L_TH. Every shape is a
  traced point list resampled to 64 points in the same order, so the
  animation morphs point-by-point from one shape to the next over
  `--morph` seconds (smoothstep eased) instead of swapping images.
  Audio decides which shape and when; the face landmarks decide where
  and how big (scaled to the real mouth corners each frame).
- **Ghibli:** the script measures only the mouth width (corner to
  corner), height (top to bottom) and centre, smooths those numbers over
  frames, then draws one single line: a curved line when the mouth is
  closed (height/width < 0.18), a single oval outline when it is open.
  Overlay only — the real lips stay visible underneath the line.

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
- Ghibli (`test_ghibli.py`): closed mouth drew 667 black pixels (curved
  line), open mouth drew 1,520 (single oval); output naming verified —
  `wuhan_noodles_lips_slow.mp4` → `…_ghibli.mp4` / `…_traditional.mp4`.
  Both styles also ran end-to-end on the synthetic test video (60 frames,
  15 fps, 4.0 s each).
- Real video (Wuhan noodles, 57.05 s): traditional style, 0.5x — lips
  drawn on 1,257 of 1,710 frames, output 114.0 s. Ghibli on that video
  has not been generated yet — say the word and it will be run with
  `--style ghibli`.


## Interpolate — in-between shapes between any two mouths (`interpolate.py`)

A standalone feature: given two mouth shapes, generate the
in-betweens. Both shapes are resampled to the same 64 points in the
same order, then blended point-by-point with smoothstep easing
(the same math as the lipsync morphing) or linear easing.

    from interpolate import interpolate_shapes
    frames = interpolate_shapes(lib, "CLOSED", "AH_I_2", steps=4)
    # -> 5 shapes: CLOSED, 0.25, 0.5, 0.75, AH_I_2

    # One in-between at any position t in [0, 1]:
    from interpolate import interpolate_at
    pts, aspect = interpolate_at(lib, "CLOSED", "AH_I_2", 0.5)

Shapes can be library names or raw (N, 2) point arrays. From the
command line it draws a contact sheet of the in-betweens:

    python interpolate.py CLOSED AH_I_2 --steps 6 --out demo.png

Tests: test_interpolate.py (endpoints exact, linear midpoint,
smoothstep easing, step counts, all 144 shape pairs).

## Remove background — solid colour out, transparent PNG (`remove_background.py`)

Step 1 of the cut-out feature for Auto Animate. You generate a
picture with AI on a solid background (an insect on magenta); this
cuts that colour out and saves a PNG with a transparent background.

    python remove_background.py insect.png --color magenta --out insect_cut.png
    python remove_background.py photo.png --color "#00FF00" --tolerance 60

- `--color`: the background colour to remove — a name (magenta,
  green, blue, white, black, red, yellow, cyan) or a hex code
  (#FF00FF). Default: magenta.
- `--tolerance` (default 40): how close a pixel must be to the key
  colour to count as background. AI "solid" backgrounds are never
  perfectly flat, so raise it if background is left behind, lower
  it if the subject is being eaten.
- `--feather` (default 2): softens the edge so it does not look
  jagged. Isolated specks outside the subject are dropped
  (largest connected piece kept).
- Output is always a PNG with an alpha channel — JPEG cannot hold
  transparency. Pick a background colour the subject does not
  contain: a magenta insect on magenta cannot be separated.

Tests: test_remove_background.py (magenta/green/hex removal,
tolerance behaviour, speck cleanup) — all passed on a synthetic
insect-on-magenta image: subject 21.9% of the image, corners
fully transparent, body fully opaque.

## Add outline — cut-out asset + outline of your colour (`add_outline.py`)

Step 2 of the cut-out feature for Auto Animate: the whole pipeline
in one command. It removes the background, saves the cut-out as an
asset/icon (transparent PNG), then draws an outline of a colour of
your choice around it — outside the subject, so no part of the
picture is ever covered.

    python add_outline.py insect.png --key-color magenta \
        --outline-color black --thickness 6
    # -> insect_cut.png (the asset) + insect_outlined.png (outlined)

    # From an already cut-out PNG (background removal is skipped):
    python add_outline.py insect_cut.png --outline-color "#FF0000"
    # Flatten onto a solid background instead of transparency:
    python add_outline.py insect.png --background white

- `--outline-color`: a name (black, red, white, ...) or a hex code
  (#FF0000). Default: black.
- `--thickness` (default 6): outline thickness in pixels. About
  1.5% of the image width is a good starting point.
- `--key-color`, `--tolerance`, `--feather` are passed to background
  removal (see above).

Tests: test_add_outline.py — ring is black and outside the subject,
thickness reaches exactly 3/6/10 px as asked, outline colour
respected (#FF0000 gives a red ring), pipeline works from a magenta
image and from an already-cut PNG, flatten works. All passed on
the synthetic insect: ring pixel (0,0,0,255), body untouched,
corner transparent.

## Scissor — cut an image into two pieces along a line (`scissor.py`)

The cut feature for Auto Animate. Place an image on a grid, give
two points on the grid (or a line equation), and the image is cut
into two pieces along the line through them — each piece saved as
its own transparent asset.

    python scissor.py insect_outlined.png --p1 100,50 --p2 300,350
    python scissor.py insect_outlined.png --equation "y = 0.5x + 10"
    python scissor.py insect_outlined.png --equation "x = 200"
    # Draw the grid preview first, to read coordinates off it:
    python scissor.py insect_outlined.png --show-grid --grid-step 50

- The line is the infinite line through your two points, so the cut
  runs across the whole image. The length of the segment between
  the points is reported (e.g. 360.6 px). Coordinates: x right,
  y down, origin top-left.
- The two pieces always rebuild the original exactly: every pixel
  goes to exactly one side, nothing is lost or doubled.
- Assets go to `--out-dir` (default `assets/cuts/`) and are named
  with a random ID and a timestamp, shared by both pieces of one
  cut so they stay paired:
  `insect_outlined_A_b7d9ab_20261113_194349.png`.

Tests: test_scissor.py — vertical/horizontal/diagonal cuts rebuild
the original exactly, equation matches points, identical points
rejected, naming pattern checked, pipeline saves two paired assets,
grid drawing verified. All passed; demo cut of the outlined insect
verified visually (head + antennae in one piece, body + legs in
the other, outline kept on both).

## Rotate — rotate an image by a given angle (`rotate.py`)

The rotate feature for Auto Animate. Give an image and an angle in
degrees; it rotates the image by that angle.

    python rotate.py insect.png --angle 45
    python rotate.py insect.png --angle -30 --out insect_r-30.png
    python rotate.py insect.png --angle 90 --keep-size
    python rotate.py insect.png --angle 45 --background white

- `--angle` is in degrees, any number. Positive is
  counter-clockwise, negative is clockwise (the standard
  convention, same as OpenCV and PIL).
- By default the canvas grows so nothing is cropped (a 45-degree
  rotation of a 400x400 image becomes 566x566). `--keep-size`
  keeps the original canvas and crops the corners instead.
- Empty corners are transparent (PNG with alpha) unless
  `--background` fills them with a solid colour.
- `--center x,y` rotates around a point other than the image
  centre; `--interpolation` is nearest, linear (default) or cubic.

Tests: test_rotate.py — 0 and 360 degrees are the identity, 90
degrees swaps dimensions, four quarter turns return the original,
direction verified with a marker dot (+90 moves top to left),
expand vs keep-size, transparency preserved (subject 34,975 px ->
34,998 px), background fill. All passed.

## Tilt — tilt an image on an axis, like a door (`tilt.py`)

The tilt feature for Auto Animate. Pick one edge as the hinge —
left, right, top or bottom — and an angle, and the image swings
forward around that edge in 3D, like pulling a door towards you.

    python tilt.py insect.png --axis left --angle 30
    python tilt.py insect.png --axis top --angle 45
    python tilt.py insect.png --axis right --angle -20
    python tilt.py insect.png --axis bottom --angle 30 --background white

- This is real perspective, not a flat squash: the image plane is
  rotated in 3D around the hinge edge and projected with a pinhole
  camera. The hinge edge does not move at all; the opposite (free)
  edge swings toward you, so it looks bigger, and the top and
  bottom edges converge toward the hinge.
- `--angle` is in degrees (-89 to 89). Positive pulls the free
  edge toward you; negative pushes it away, so it looks smaller.
- `--focal` is the camera distance (default twice the image size):
  larger is flatter, smaller is more dramatic.
- The canvas grows to fit; empty corners are transparent (PNG
  with alpha) unless `--background` fills them.

Tests: test_tilt.py — 0 degrees is the identity, the hinge keeps
its full length and stays on its edge, the free edge is magnified
(canvas taller than the original), a negative angle narrows the
image, all four axes work (left/right mirror each other, as do
top/bottom), transparency preserved, angles of 90+ rejected. All
passed; 45-degree demo on the insect verified visually on all
four axes.

## Render — display a list of assets on a screen (`render.py`)

The asset renderer for Auto Animate. You have a list of assets
(images), each with a position and a z-factor, and this displays
them all on one big screen.

A scene is a JSON file:

    {
      "width": 1000, "height": 800,
      "assets": [
        {"image": "insect.png", "x": 50, "y": 50, "z": 0},
        {"image": "insect_rot45.png", "x": 350, "y": 150, "z": 2}
      ]
    }

    python render.py scene.json --out screen.png
    python render.py scene.json --width 1920 --height 1080 --background white

- The screen is a grid with the origin at the BOTTOM-LEFT corner:
  x to the right, y going UP (the positive-positive Cartesian
  quadrant). The renderer flips it to image pixels for you.
- Each asset's hook is its BOTTOM-LEFT corner: that corner sits
  at (x, y), so the image covers [x, x+width] by [y, y+height].
- z-factor is the draw order: lowest z first, highest z last, so
  the highest z appears on top. Equal z keeps list order (later
  in the list is on top).
- Assets are clipped to the screen, and transparent PNGs blend
  correctly where they overlap. From Python the scene is just a
  list: `render_assets([Asset("a.png", x=50, y=50, z=0)], 1000, 800)`.

Tests: test_render.py — origin at bottom-left, y grows up, hook
is the bottom-left corner, highest z on top regardless of list
order, equal z keeps list order, clipping (half off-screen and
fully off-screen), transparent blending. All passed; demo scene
(demo_scene.json, 3 insect assets on a 1000x800 screen) verified
visually.

## Scale — zoom an image in or out (`scale.py`)

The scale feature for Auto Animate. Give an image and a zoom
factor (or an exact size) and it returns the image scaled.

    python scale.py insect.png --factor 2        # zoom in 2x
    python scale.py insect.png --factor 0.5      # zoom out to half
    python scale.py insect.png --width 800       # width 800, height follows
    python scale.py insect.png --width 800 --height 600
    python scale.py insect.png --fx 2 --fy 0.5  # stretch x, squash y

- `--factor` is a uniform zoom: above 1 zooms in, below 1 zooms
  out. `--fx` / `--fy` scale x and y separately.
- `--width` and/or `--height` set an exact size; one alone keeps
  the aspect ratio, both together set it exactly.
- `--interpolation` is nearest, linear (default), cubic or
  lanczos. Linear is smooth for drawings; nearest keeps hard
  pixel edges and is exact for whole-number zooms.
- Transparency is preserved: the alpha channel is scaled with
  the colours, so a cut-out stays a cut-out.

Tests: test_scale.py — factor doubles/halves, factor 1 is the
identity, 2x then 0.5x with nearest returns the original exactly,
exact sizes, non-uniform fx/fy, transparency preserved, bad
inputs rejected. All passed.
