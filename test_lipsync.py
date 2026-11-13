"""Unit test for the lipsync morphing (no video needed)."""
import sys
sys.path.insert(0, '.')
import numpy as np
import lip_outline as lo

lib = lo.load_mouth_library("mouth_library_v2/manifest.json")
print("shapes:", sorted(lib))
for name, (pts, aspect) in lib.items():
    assert pts.shape == (64, 2), (name, pts.shape)
    assert 0.0 <= pts[:, 0].min() and pts[:, 0].max() <= 1.0, name
print("all shapes resampled to 64 points, x in [0,1]")

# Line shapes must be (near-)flat in y: a single line, not a loop.
for name in ("CLOSED", "M_B_P"):
    pts, aspect = lib[name]
    print(f"{name}: aspect {aspect:.3f}, y range {pts[:,1].min():.3f}-{pts[:,1].max():.3f}")
    assert aspect < 0.2, name
    assert pts[:, 1].max() - pts[:, 1].min() < 0.10, name  # a curved line, not a loop

# Morphing: closed -> AH_I_2 at u=0, 0.5, 1 via shape_at_time.
segments = [{"shape": "CLOSED", "start": 0.0, "end": 1.0},
            {"shape": "AH_I_2", "start": 1.0, "end": 2.0},
            {"shape": "CLOSED", "start": 2.0, "end": 3.0}]
pts0, _, s0 = lo.shape_at_time(lib, segments, 0.5, 0.1)
pts1, _, s1 = lo.shape_at_time(lib, segments, 1.05, 0.1)  # mid-morph
pts2, _, s2 = lo.shape_at_time(lib, segments, 1.5, 0.1)
assert s0 == "CLOSED" and s2 == "AH_I_2"
# mid-morph height must sit between closed (flat) and open (tall)
h0 = pts0[:, 1].max() - pts0[:, 1].min()
h1 = pts1[:, 1].max() - pts1[:, 1].min()
h2 = pts2[:, 1].max() - pts2[:, 1].min()
print(f"morph heights: closed {h0:.3f} -> mid {h1:.3f} -> open {h2:.3f}")
assert h0 < h1 < h2
# Before the first segment and after the last: CLOSED, no crash.
assert lo.shape_at_time(lib, segments, 5.0, 0.1)[2] == "CLOSED"
print("lipsync morph test passed")
