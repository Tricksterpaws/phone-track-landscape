#!/usr/bin/env python3
"""Compare a solved landscape track to the fixture quads, and report motion.

Usage:
  python compare_track.py [path/to/landscape.json]

Fixture check (exit code):
  Exit 0 if every fixture corner is within --tol (default 1.5px) of
  fixtures/landscape/expected.json. Exit 1 if any fixture corner moves
  more than the tolerance, or a fixture frame is missing. This is the
  check that catches a wrong-but-smooth track.

Motion check (informational only):
  Reports the worst frame-to-frame corner jump and flags jumps at or
  above --jump-flag (default 5px). A lower max jump is not a pass and
  does not affect the exit code.

  The shipped landscape.json jumps 22.819px from frame 205 to 206
  (8.54-8.58s), inside the pickup gap around 7.8-8.6s (frames ~188-206)
  where the hand covers the phone on the bed. Do not treat a smoother
  re-solve as correct on that basis. A red alternate track on this plate
  is smoother (worst jump 6.5px) and still leaves the glass around
  frames 212-252. The shipped tracks/landscape.json is the regression
  oracle; do not "fix" it just to lower the pickup jump.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLATE = "landscape"
DEFAULT_TRACK = ROOT / "tracks" / "landscape.json"


def quads(track):
    out = []
    for flat in track["frames"]:
        if flat is None:
            out.append(None)
            continue
        out.append([[flat[k], flat[k + 1]] for k in range(0, 8, 2)])
    return out


def corner_err(a, b):
    return max(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(a, b))


def fixture_check(expected, got, tol):
    worst = 0.0
    failed = False
    print("== fixture check ==")
    for item in expected["frames"]:
        i = item["index"]
        if i >= len(got) or got[i] is None:
            print(f"f{i:04d}  MISSING")
            failed = True
            continue
        err = corner_err(item["quad"], got[i])
        worst = max(worst, err)
        flag = "  FAIL" if err > tol else ""
        print(f"f{i:04d}  {err:7.3f}px{flag}")
        if err > tol:
            failed = True
    print(f"worst {worst:.3f}px  tol {tol}px  frames {len(expected['frames'])}")
    return failed, worst


def motion_check(got, jump_flag):
    print("== motion check (informational; does not affect exit code) ==")
    worst = 0.0
    pair = None
    flagged = []
    for i in range(1, len(got)):
        a, b = got[i - 1], got[i]
        if a is None or b is None:
            continue
        jump = corner_err(a, b)
        if jump >= jump_flag:
            flagged.append((jump, i - 1, i))
        if jump > worst:
            worst, pair = jump, (i - 1, i)
    if pair is None:
        print("no consecutive frames to compare")
        return 0.0, None
    t0, t1 = pair[0] / 24.0, pair[1] / 24.0
    print(
        f"max jump {worst:.3f}px  frames {pair[0]}->{pair[1]}  "
        f"t={t0:.2f}-{t1:.2f}s  flagged>={jump_flag:g}px: {len(flagged)}"
    )
    flagged.sort(reverse=True)
    show = flagged[:15] if flagged else []
    for jump, a, b in show:
        print(f"  {a:4d}->{b:4d}  {jump:7.3f}px  t={a/24:.2f}-{b/24:.2f}s")
    if len(flagged) > 15:
        print(f"  ... {len(flagged) - 15} more")
    print(
        "A lower max jump than the shipped oracle is not proof the track is right."
    )
    return worst, pair


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("track", nargs="?", default=str(DEFAULT_TRACK))
    parser.add_argument("--tol", type=float, default=1.5)
    parser.add_argument("--jump-flag", type=float, default=5.0)
    args = parser.parse_args()

    expected = json.loads((ROOT / "fixtures" / PLATE / "expected.json").read_text())
    solved = json.loads(Path(args.track).read_text())
    got = quads(solved)
    failed, _ = fixture_check(expected, got, args.tol)
    print()
    motion_check(got, args.jump_flag)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
