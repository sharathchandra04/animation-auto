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
