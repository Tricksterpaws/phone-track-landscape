#!/usr/bin/env python3
"""Compare a solved landscape track to the fixture quads, and report motion.

Usage:
  python compare_track.py [path/to/landscape.json]
  python compare_track.py landscape [path/to/landscape.json]

The plate name is optional; this repo is landscape only. The existing
command `python compare_track.py tracks/landscape.json` still works.

Fixture check (exit code):
  Exit 0 if every fixture corner is within --tol (default 1.5px) of
  fixtures/landscape/expected.json. Exit 1 if any fixture corner moves
  more than the tolerance, or a fixture frame is missing. This is the
  only pass/fail gate, and the check that catches a wrong-but-smooth track.

Motion check (informational only; does not affect the exit code):
  1. Frame-to-frame corner jumps (px). Flags jumps at or above --jump-flag
     (default 5px). The shipped track jumps 22.819px from frame 205 to 206
     (8.54-8.58s), inside the pickup gap around 7.8-8.6s (frames ~188-206).
  2. Corner acceleration (px/frame^2): the change in per-frame movement,
     where a jump shows up as a spike. The shipped baseline in expected.json
     is 22.48 at frames 206, 205, 187, 188, 230. That 22.48 is not the same
     number as the 22.819px jump: it is round(acceleration at frame 206, 2),
     i.e. how much the 205->206 step differs from 206->207. A value more
     than --jump-margin (default 1.0) above the baseline is flagged, not
     failed.

  A lower max jump or spike is not a pass. A red alternate track on this
  plate is smoother (worst jump 6.5px) and still leaves the glass around
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


def jump_report(got, jump_flag, fps):
    print("-- frame-to-frame jumps --")
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
    t0, t1 = pair[0] / fps, pair[1] / fps
    print(
        f"max jump {worst:.3f}px  frames {pair[0]}->{pair[1]}  "
        f"t={t0:.2f}-{t1:.2f}s  flagged>={jump_flag:g}px: {len(flagged)}"
    )
    flagged.sort(reverse=True)
    show = flagged[:15]
    for jump, a, b in show:
        print(f"  {a:4d}->{b:4d}  {jump:7.3f}px  t={a/fps:.2f}-{b/fps:.2f}s")
    if len(flagged) > 15:
        print(f"  ... {len(flagged) - 15} more")
    return worst, pair


def accelerations(got, key_from=0):
    """(frame, px/frame^2) for every frame with a neighbour on each side.

    The frame where the screen turns green is skipped: the solve changes
    method there (bed flow -> keyed glass).
    """
    out = []
    for i in range(1, len(got) - 1):
        a, b, c = got[i - 1], got[i], got[i + 1]
        if a is None or b is None or c is None or abs(i - key_from) <= 1:
            continue
        acc = max(
            math.hypot(a[k][0] - 2 * b[k][0] + c[k][0], a[k][1] - 2 * b[k][1] + c[k][1])
            for k in range(4)
        )
        out.append((i, acc))
    return out


def acceleration_report(got, expected, solved, jump_margin):
    print("-- acceleration --")
    fps = solved.get("fps", expected.get("fps", 24.0))
    key_from = solved.get("keyFrom", expected.get("keyFrom", 0))
    js = sorted(accelerations(got, key_from), key=lambda t: -t[1])
    top = js[:5]
    peak = top[0][1] if top else 0.0
    spikes = ", ".join(f"f{i:04d} ({i / fps:.2f}s) {a:.1f}" for i, a in top)
    print(f"worst corner acceleration {peak:.2f} px/frame^2  [{spikes}]")
    flagged = False
    if "motion" in expected:
        baseline = expected["motion"]["max_jump"]
        limit = baseline + jump_margin
        if peak > limit:
            print(f"acceleration FLAG: {peak:.2f} > baseline {baseline} + {jump_margin}")
            flagged = True
        else:
            print(f"acceleration ok: baseline {baseline} + {jump_margin}")
    return peak, flagged


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="optional plate name (landscape) and/or path to a track JSON",
    )
    parser.add_argument("--tol", type=float, default=1.5)
    parser.add_argument("--jump-flag", type=float, default=5.0)
    parser.add_argument(
        "--jump-margin",
        type=float,
        default=1.0,
        help="informational flag if acceleration exceeds baseline by more than this (px/frame^2)",
    )
    ns = parser.parse_args(argv)
    positional = list(ns.args)
    if positional and positional[0] in ("landscape", "portrait"):
        positional = positional[1:]
    ns.track = positional[0] if positional else str(DEFAULT_TRACK)
    if len(positional) > 1:
        parser.error("too many positional arguments")
    return ns


def main(argv=None):
    args = parse_args(argv)
    expected = json.loads((ROOT / "fixtures" / PLATE / "expected.json").read_text())
    solved = json.loads(Path(args.track).read_text())
    got = quads(solved)
    fps = solved.get("fps", expected.get("fps", 24.0))
    failed, _ = fixture_check(expected, got, args.tol)
    print()
    print("== motion check (informational; does not affect exit code) ==")
    jump_report(got, args.jump_flag, fps)
    print()
    acceleration_report(got, expected, solved, args.jump_margin)
    print(
        "A lower max jump or acceleration than the shipped oracle is not proof the track is right."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
