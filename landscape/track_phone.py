#!/usr/bin/env python3
"""Planar corner track of the phone glass in the Trick intro plate.

Clean green rectangles are treated as ground truth. Gaps where a hand
covers the glass are filled with optical flow, then pinned so the gap
ends exactly on the next true rectangle. After the glass leaves the
frame, the corners keep moving on a short quadratic.
"""
import json
import math
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Paths are relative to this repository root only.
ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "1000030200.mp4")
# Write a re-solve beside the shipped oracle; do not default to tracks/landscape.json.
OUT_JSON = str(ROOT / "work" / "landscape.json")
QA_DIR = str(ROOT / "work" / "comp")
W, H = 1080, 608
FPS = 24.0


def green_mask(im):
    b, g, r = cv2.split(im)
    gi = g.astype(np.int16)
    ri = r.astype(np.int16)
    bi = b.astype(np.int16)
    m = (gi > 150) & (ri < 120) & (bi < 130) & (gi > ri + 60) & (gi > bi + 60)
    return m.astype(np.uint8) * 255


def order_quad(pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    i = int(np.argmin(pts[:, 1] + pts[:, 0] * 0.02))
    return np.roll(pts, -i, axis=0)


def align_quad(quad, prev):
    best, best_d = None, 1e18
    for k in range(4):
        rolled = np.roll(quad, -k, axis=0)
        d = float(np.sum((rolled - prev) ** 2))
        if d < best_d:
            best_d, best = d, rolled
    return best


def quad_area(quad):
    return float(abs(cv2.contourArea(quad.astype(np.float32))))


def side_aspect(q):
    w = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    h = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    if h < 1:
        return 99.0
    return float(w / h)


def positive_order(q):
    # TL->TR x TL->BL should be clockwise-positive in y-down (cross > 0).
    a = q[1] - q[0]
    b = q[3] - q[0]
    return float(a[0] * b[1] - a[1] * b[0]) > 0


def touches_border(quad, margin=4):
    return bool(
        quad[:, 0].min() <= margin
        or quad[:, 1].min() <= margin
        or quad[:, 0].max() >= W - 1 - margin
        or quad[:, 1].max() >= H - 1 - margin
    )


def measure(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 500:
        return None
    hull = cv2.convexHull(cnt)
    box = order_quad(cv2.boxPoints(cv2.minAreaRect(hull)))
    if not positive_order(box):
        box = box[[0, 3, 2, 1]]
    ba = quad_area(box)
    if ba < 500 or touches_border(box, 4):
        return None
    aspect = side_aspect(box)
    if not (1.35 <= aspect <= 6.2):
        return None
    poly = np.zeros_like(mask)
    cv2.fillConvexPoly(poly, np.int32(np.round(box)), 255)
    inside = np.count_nonzero((poly > 0) & (mask > 0))
    fill = inside / ba
    if fill < 0.68:
        return None
    return box


def bilinear(quad, u, v):
    tl, tr, br, bl = quad
    top = tl * (1 - u) + tr * u
    bot = bl * (1 - u) + br * u
    return top * (1 - v) + bot * v


def sample_points(quad):
    pts = []
    for u in np.linspace(-0.22, 1.22, 9):
        for v in (-0.28, -0.1, 1.1, 1.28):
            pts.append(bilinear(quad, float(u), float(v)))
    for v in np.linspace(0.05, 0.95, 5):
        pts.append(bilinear(quad, -0.16, float(v)))
        pts.append(bilinear(quad, 1.16, float(v)))
    pts = np.asarray(pts, dtype=np.float32)
    ok = (pts[:, 0] > 6) & (pts[:, 0] < W - 6) & (pts[:, 1] > 6) & (pts[:, 1] < H - 6)
    return pts[ok].reshape(-1, 1, 2)


def flow_step(prev_gray, gray, quad):
    pts = sample_points(quad)
    if len(pts) < 10:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, pts, None,
        winSize=(25, 25), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.03),
    )
    if nxt is None:
        return None
    st = st.reshape(-1).astype(bool)
    p0 = pts.reshape(-1, 2)[st]
    p1 = nxt.reshape(-1, 2)[st]
    if len(p0) < 10:
        return None
    aff, inliers = cv2.estimateAffinePartial2D(
        p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.5, maxIters=80, confidence=0.95
    )
    if aff is None or inliers is None or int(inliers.sum()) < 8:
        return None
    scale = math.hypot(float(aff[0, 0]), float(aff[1, 0]))
    if not (0.82 <= scale <= 1.22):
        return None
    if abs(float(aff[0, 2])) > 70 or abs(float(aff[1, 2])) > 70:
        return None
    hom = np.concatenate([quad, np.ones((4, 1))], axis=1)
    return (aff @ hom.T).T


def lerp(a, b, t):
    return a * (1 - t) + b * t


def expand_quad(quad, scale):
    c = quad.mean(axis=0)
    return c + (quad - c) * scale


def load_grays():
    cap = cv2.VideoCapture(SRC)
    grays, raw = [], []
    last = None
    while True:
        ok, im = cap.read()
        if not ok:
            break
        grays.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        box = measure(green_mask(im))
        if box is not None and last is not None:
            box = align_quad(box, last)
        if box is not None:
            last = box
        raw.append(box)
    return grays, raw


def mark_good(raw):
    good = [False] * len(raw)
    last_area = None
    for i, q in enumerate(raw):
        if q is None:
            continue
        a = quad_area(q)
        if last_area is None or a >= last_area * 0.88:
            good[i] = True
            last_area = a
    return good


def fill_gaps(grays, raw, good):
    n = len(raw)
    track = [None] * n
    for i, ok in enumerate(good):
        if ok:
            track[i] = raw[i].copy()

    # Centered 3-median on contiguous good runs to kill single-pixel speckle.
    for i in range(1, n - 1):
        if good[i] and good[i - 1] and good[i + 1]:
            stack = np.stack([raw[i - 1], raw[i], raw[i + 1]], axis=0)
            track[i] = np.median(stack, axis=0)

    goods = [i for i, ok in enumerate(good) if ok]
    if not goods:
        raise SystemExit("no good measurements")

    # Bed lock: the opening hold. Median of the first run of good frames
    # whose centers stay within 10px of the first one.
    first = goods[0]
    base = track[first]
    run = [base]
    last_stable = first
    for i in goods:
        if i > first + 160:
            break
        if np.linalg.norm(track[i].mean(0) - base.mean(0)) < 12:
            run.append(track[i])
            last_stable = i
        else:
            break
    med = np.median(np.stack(run, 0), axis=0)
    for i in range(0, last_stable + 1):
        track[i] = med.copy()
    print("bed lock", 0, last_stable, "from", len(run), "samples")

    # Gaps between locked/good samples.
    known = [i for i, q in enumerate(track) if q is not None and i <= goods[-1]]
    # rebuild known after bed lock
    known = [i for i in range(goods[-1] + 1) if track[i] is not None]
    gaps = []
    for a, b in zip(known, known[1:]):
        if b > a + 1:
            gaps.append((a, b))
    print("gaps", [(a, b, b - a - 1) for a, b in gaps])

    for a, b in gaps:
        steps = b - a
        pred = [track[a].copy()]
        cur = track[a].copy()
        flow_ok = True
        for i in range(a, b):
            nxt = flow_step(grays[i], grays[i + 1], cur)
            if nxt is None or quad_area(nxt) < quad_area(cur) * 0.7:
                flow_ok = False
                break
            # keep corner identity
            nxt = align_quad(nxt, cur)
            if not positive_order(nxt):
                flow_ok = False
                break
            cur = nxt
            pred.append(cur.copy())
        use_flow = flow_ok and len(pred) == steps + 1
        if use_flow:
            drift = pred[-1] - track[b]
            drift_n = float(np.linalg.norm(drift) / 4)
            if drift_n > 180:
                use_flow = False
        for k in range(1, steps):
            if use_flow:
                track[a + k] = pred[k] - (pred[-1] - track[b]) * (k / steps)
            else:
                track[a + k] = lerp(track[a], track[b], k / steps)
        print(f"  gap {a}->{b} flow={use_flow}")

    # Tail: glass reaches the frame edge. Quadratic on the last in-frame samples.
    last = max(i for i, q in enumerate(track) if q is not None)
    window = list(range(max(0, last - 18), last + 1))
    window = [i for i in window if track[i] is not None]
    print("tail from", last, "window", window[0], window[-1])
    ts = np.array(window, dtype=np.float64)
    coefs = []
    for c in range(8):
        ys = np.array([track[i].reshape(-1)[c] for i in window])
        # degree 2 if we have the points, else 1
        deg = 2 if len(window) >= 8 else 1
        coefs.append(np.polyfit(ts, ys, deg))
    for i in range(last + 1, n):
        vals = [float(np.polyval(coefs[c], i)) for c in range(8)]
        q = np.array(vals, dtype=np.float64).reshape(4, 2)
        # Don't let a wild quadratic fold the quad. Fall back to constant
        # velocity from the previous frame if the area shrinks or flips.
        prev = track[i - 1]
        if prev is not None and (not positive_order(q) or quad_area(q) < quad_area(prev) * 0.98):
            if i >= 2 and track[i - 2] is not None:
                q = prev + (prev - track[i - 2])
            else:
                q = prev
        track[i] = q
    return track, last_stable, last


def screen_metrics(track, raw, good):
    aspects = []
    for i, q in enumerate(track):
        if q is None or not good[i]:
            continue
        h = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
        w = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
        if h < 180:
            continue
        top = q[1] - q[0]
        left = q[3] - q[0]
        tilt = abs(top[1]) / (np.linalg.norm(top) + 1e-6) + abs(left[0]) / (np.linalg.norm(left) + 1e-6)
        if tilt < 0.18:
            aspects.append(w / h)
    aspect = float(np.median(aspects)) if aspects else 2.08
    print("aspect samples", len(aspects), "aspect", round(aspect, 4))

    # Corner radius on the largest frontal good frame.
    radius = 0.11
    candidates = []
    cap = cv2.VideoCapture(SRC)
    frames = {}
    idx = 0
    want = set()
    for i, ok in enumerate(good):
        if not ok:
            continue
        h = 0.5 * (np.linalg.norm(track[i][3] - track[i][0]) + np.linalg.norm(track[i][2] - track[i][1]))
        if h > 280:
            want.add(i)
    if not want:
        return aspect, radius
    target = sorted(want)[len(want) // 2]
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if idx == target:
            frames[idx] = im
            break
        idx += 1
    if target in frames:
        m = green_mask(frames[target]) > 0
        q = track[target]
        h = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
        ys, xs = np.where(m)
        sample = np.stack([xs, ys], 1).astype(np.float64)
        step = max(1, len(sample) // 12000)
        sample = sample[::step]
        ds = [float(np.min(np.linalg.norm(sample - cnr, axis=1))) for cnr in q]
        r_px = float(np.median(ds)) / 0.414
        radius = float(np.clip(r_px / h, 0.04, 0.22))
        print("radius frame", target, "ds", [round(d, 1) for d in ds], "frac", round(radius, 3))
    return aspect, radius


def render_qa(track):
    aspect = 2.08
    cw, ch = 960, int(round(960 / aspect))
    card = np.zeros((ch, cw, 3), np.uint8)
    card[:] = (32, 24, 20)
    for x in range(0, cw, 40):
        cv2.line(card, (x, 0), (x, ch), (80, 60, 40), 1)
    for y in range(0, ch, 40):
        cv2.line(card, (0, y), (cw, y), (80, 60, 40), 1)
    cv2.rectangle(card, (0, 0), (cw - 1, 40), (30, 30, 200), -1)
    cv2.rectangle(card, (0, ch - 40), (cw - 1, ch - 1), (180, 80, 20), -1)
    cv2.putText(card, "TOP", (cw // 2 - 36, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(card, "BOTTOM", (cw // 2 - 70, ch - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(card, "YOUR CLIP", (cw // 2 - 150, ch // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (240, 240, 240), 3, cv2.LINE_AA)
    for label, org in (("TL", (18, 78)), ("TR", (cw - 64, 78)), ("BL", (18, ch - 56)), ("BR", (cw - 70, ch - 56))):
        cv2.putText(card, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 80), 2, cv2.LINE_AA)
    src_pts = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], dtype=np.float32)

    want = [0, 72, 150, 186, 200, 220, 240, 258, 288, 336, 384, 432, 470, 500, 520, 534, 548, 562, 575]
    cap = cv2.VideoCapture(SRC)
    tiles = []
    i = 0
    os.makedirs(QA_DIR, exist_ok=True)
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i in want and track[i] is not None:
            q = track[i].astype(np.float32)
            M = cv2.getPerspectiveTransform(src_pts, q)
            warped = cv2.warpPerspective(card, M, (W, H))
            b, g, r = cv2.split(im)
            gi = g.astype(np.int16); ri = r.astype(np.int16); bi = b.astype(np.int16)
            key = (gi > 150) & (ri < 120) & (bi < 130) & (gi > ri + 60) & (gi > bi + 60)
            if i < 72:
                poly = np.zeros((H, W), np.uint8)
                cv2.fillConvexPoly(poly, np.int32(np.round(q)), 255)
                dark = (ri < 48) & (gi < 48) & (bi < 55)
                key = (poly > 0) & dark
            comp = im.copy()
            sel = key
            comp[sel] = warped[sel]
            cv2.polylines(comp, [np.int32(np.round(q))], True, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(comp, f"f{i}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(comp, f"f{i}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(cv2.resize(comp, (480, 270), interpolation=cv2.INTER_AREA))
            x0 = int(np.clip(q[:, 0].min() - 24, 0, W - 2))
            y0 = int(np.clip(q[:, 1].min() - 24, 0, H - 2))
            x1 = int(np.clip(q[:, 0].max() + 24, x0 + 2, W))
            y1 = int(np.clip(q[:, 1].max() + 24, y0 + 2, H))
            crop = comp[y0:y1, x0:x1]
            zw = 420
            zh = max(1, int(zw * crop.shape[0] / max(1, crop.shape[1])))
            cv2.imwrite(os.path.join(QA_DIR, f"z{i:04d}.jpg"), cv2.resize(crop, (zw, zh), interpolation=cv2.INTER_AREA))
        i += 1
    cols = 4
    rows = math.ceil(len(tiles) / cols)
    sheet = np.zeros((rows * 270, cols * 480, 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * 270:(r + 1) * 270, c * 480:(c + 1) * 480] = t
    cv2.imwrite(os.path.join(QA_DIR, "sheet.jpg"), sheet)
    print("qa tiles", len(tiles))


def main():
    grays, raw = load_grays()
    n = len(raw)
    good = mark_good(raw)
    print("frames", n, "raw", sum(q is not None for q in raw), "good", sum(good))
    # Where good drops out
    runs = []
    start = None
    for i, ok in enumerate(good):
        if ok and start is None:
            start = i
        if not ok and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, n - 1))
    print("good runs", runs)

    track, lock_end, last_in = fill_gaps(grays, raw, good)
    aspect, radius = screen_metrics(track, raw, good)

    # Slight outset so the pin swallows the green edge. Chroma still cuts the bezel.
    for i, q in enumerate(track):
        if q is not None:
            track[i] = expand_quad(q, 1.018)

    print("--- samples ---")
    for i in range(0, n, 12):
        q = track[i]
        tag = "good" if good[i] else ("raw" if raw[i] is not None else "gap")
        if q is None:
            print(f"{i:4d} {tag:5s} NONE")
        else:
            w = np.linalg.norm(q[1] - q[0])
            h = np.linalg.norm(q[3] - q[0])
            print(f"{i:4d} {tag:5s} TL({q[0,0]:7.1f},{q[0,1]:7.1f}) TR({q[1,0]:7.1f},{q[1,1]:7.1f}) w={w:6.1f} h={h:6.1f}")

    frames = []
    for q in track:
        if q is None:
            frames.append(None)
        else:
            frames.append([round(float(v), 2) for v in q.reshape(-1)])

    payload = {
        "fps": FPS,
        "width": W,
        "height": H,
        "frameCount": n,
        "screenAspect": round(aspect, 4),
        "cornerRadius": round(radius, 4),
        "keyFrom": 72,
        "frames": frames,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print("wrote", OUT_JSON, os.path.getsize(OUT_JSON))
    render_qa(track)


if __name__ == "__main__":
    main()
