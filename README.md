# phone-track-landscape

Planar corner track of the phone glass in the landscape intro plate.

This repository is landscape only. The track JSON is the only contract with a compositor. This repo does not include compositor, WebGL, or export code.

## Pinned plate

File: [`1000030200.mp4`](1000030200.mp4)

```
sha256  aff49a40c336123560646054b8224bddee7c81e9a648431aefdd21b36b5bf49c
```

Verify:

```bash
sha256sum 1000030200.mp4
```

Do not substitute another encode of this plate.

## JSON contract

`tracks/landscape.json` is the shipped regression oracle.

| Field | Value |
| --- | --- |
| `width` × `height` | 1080 × 608 pixels |
| `fps` | 24 |
| `frameCount` | 583 |
| `frames` | 583 entries, each 8 numbers: corners **TL, TR, BR, BL** in pixels, `[x0,y0,x1,y1,x2,y2,x3,y3]` |
| `keyFrom` | 72 |
| `screenAspect` | glass width / height (shipped: 2.0781) |
| `cornerRadius` | corner radius as a fraction of glass height (shipped: 0.1247) |

This track has **no `slack` field**. Do not add one to match the portrait contract.

`keyFrom` is the first frame where the green screen is treated as a keyable fill. Before that, the glass is still dark on the bed.

## Reproduce the shipped JSON

From the repository root, with the pinned plate in place:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Writes work/landscape.json and work/comp QA crops.
# Does not overwrite tracks/landscape.json.
python3 landscape/track_phone.py

python3 compare_track.py work/landscape.json
```

`landscape/track_phone.py` reads `1000030200.mp4` and writes `work/landscape.json`, both relative to the repo root.

A re-solve that stays within 1.5px of every fixture corner is a fixture pass. Exact byte-for-byte equality with `tracks/landscape.json` is not required for that pass, and a fixture-passing re-solve must still not replace the shipped oracle unless that is an explicit goal.

To test the shipped oracle itself:

```bash
python3 compare_track.py tracks/landscape.json
```


## Reproduction result (this import)

A dry run of `python3 landscape/track_phone.py` from this repo on the pinned plate finished in **5.285 seconds** (Australia/Brisbane). Metadata (`fps`, size, `frameCount`, `screenAspect`, `cornerRadius`, `keyFrom`) matched the shipped JSON. The re-solve was **not** an exact match and **did not** pass the fixture check:

- f0200: 69.074px off
- f0240: 129.694px off
- the other nine fixture frames: 0.000px
- worst deviation versus the full shipped track: 171.0605px at frame 232
- motion: max jump 10.207px at frames 581→582 — smoother than the shipped 22.819px pickup jump, and still wrong in the 200–240 region

That is why the fixture check exists. The re-solve was left in `work/` and was **not** written over `tracks/landscape.json`.

## Checks

`compare_track.py` always runs both checks. A smooth track can still be wrong.

### Fixture check (exit code)

Compares the solved track to `fixtures/landscape/expected.json`. Exit 1 if any fixture corner moves by more than **1.5px**, or if a fixture frame is missing. This is the check that catches a wrong-but-smooth track.

Fixture frames: 0, 72, 150, 200, 240, 258, 336, 432, 500, 548, 575.

### Motion check (informational)

Flags frame-to-frame corner jumps (default flag threshold 5px). This check does **not** affect the exit code.

The shipped `tracks/landscape.json` jumps **22.819px** from frame 205 to 206 (8.54–8.58s), inside the pickup gap at about 7.8–8.6s (frames ~188–206), when the hand covers the phone on the bed. Do not treat a lower max jump as proof the track is right. A red alternate track on this plate is smoother (worst jump 6.5px) and still leaves the glass around frames 212–252.

Do not "fix" the shipped track just to lower the pickup jump unless a fixture-passing re-solve is the explicit goal. `tracks/landscape.json` is the regression oracle.

## Layout

```
1000030200.mp4            pinned landscape plate
landscape/track_phone.py  solver
tracks/landscape.json     shipped oracle
fixtures/landscape/       expected quads + source frames
dumps/comp/               QA crops and contact sheet from the shipped solve
dumps/dbg/                earlier green-mask and frame peeks
compare_track.py
requirements.txt
```

`work/` is gitignored. Re-runs write there so shipped dumps and the oracle stay untouched.
