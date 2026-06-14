# pyLACE

A Python port of LACE — **L**imbless **A**nimal tra**C**k**E**r — a
marker-free pose estimator and identity tracker for small animals in
top-down arena video (adult *Drosophila*, larval flies, zebrafish),
originally published as:

> Garg, S., et al. (2022). LACE — Limbless Animal traCkEr: a
> marker-free pose estimator. *Frontiers in Behavioral Neuroscience*
> **16**:819146.
> <https://www.frontiersin.org/articles/10.3389/fnbeh.2022.819146/full>

The earlier MATLAB implementation is archived at
<https://github.com/zerotonin/LACE>.

## What's here

pyLACE is a single PyQt6 + numpy/OpenCV package covering the whole
tracking-to-review loop:

1. **Calibrate** the arena (per-recording sidecar JSON: pix/mm, ROIs,
   trim, background fractions). Consumes calibration written by the
   sister `arena_annotator` tool, or builds its own from a single
   annotated frame.
2. **Detect** flies frame-by-frame (background subtraction → connected
   components → watershed-on-distance-transform splitter → Hough
   rescue for under-counted frames). The greedy chainer assigns
   stable track ids.
3. **Clean** raw detections into per-frame trajectories — gap fill,
   outlier rejection, Savitzky-Golay smoothing on (x, y), heading
   disambiguation, yaw rate.
4. **Audit** identity swaps with a 4-state Kalman filter and
   Mahalanobis cost, combined with a per-track appearance fingerprint
   (pose-normalised, head-abdomen-canonicalised median patches).
   Optional verdict file overrides the cost gates.
5. **Review** the audit interactively — candidate detector
   (contact + jump + NaN spans) feeds a Qt dock in `pylace-explore`
   where every event gets one click of *accept / reject / mount /
   unknown* plus an N-spinbox permutation table for 3-way cycles.
6. **Measure** kinematics — speed, occupancy, thigmotaxis, walk/stop
   bouts, nearest-neighbour distance, polarisation, distance to wall.

## Install

```bash
pip install -e ".[dev]"            # editable + pytest, ruff
```

The package requires Python ≥ 3.10. Headless tests run on
`QT_QPA_PLATFORM=offscreen`.

## Command-line tools

| CLI                  | What it does                                                            |
|----------------------|-------------------------------------------------------------------------|
| `pylace-annotate`    | Annotate a reference frame → arena sidecar JSON (calibration, ROIs).    |
| `pylace-tune`        | Live preview to tune background + detector params before a batch run.   |
| `pylace-roi`         | Standalone ROI editor for an existing sidecar.                          |
| `pylace-detect`      | Run the detection pipeline; emit per-detection CSV.                     |
| `pylace-clean`       | Detections CSV → cleaned per-frame trajectory CSV.                      |
| `pylace-fingerprint` | Build per-track appearance medians from confident frames.               |
| `pylace-audit`       | Cost + appearance + verdict-aware identity-swap audit.                  |
| `pylace-candidates`  | Audit-independent candidate detector (contact, jump, NaN spans).        |
| `pylace-explore`     | Interactive PyQt6 inspector; `--review` opens the swap-review dock.     |
| `pylace-metrics`     | Summarise an audited trajectory (kinematics + multi-fly metrics).       |
| `pylace-inspect`     | Lightweight viewer for raw detections (no trajectory required).         |

## Three-pass review workflow

```bash
# 1. find candidate frames where identity is at risk
pylace-candidates  video.pylace_trajectory.csv
#                  → video.pylace_candidates.csv

# 2. run the cost-based audit (writes audited trajectory + swap log)
pylace-audit       video.pylace_trajectory.csv
#                  → video.pylace_audited.csv
#                  → video.pylace_audit_swaps.csv

# 3. open the GUI with the review dock
pylace-explore     video.mp4 video.pylace_audited.csv --review
#                  → video.pylace_verdicts.csv  (autosaved per keystroke)

# 4. re-run the audit; the sidecar verdicts are auto-detected and honoured
pylace-audit       video.pylace_trajectory.csv
```

Verdicts: `a` accept-swap (use the spinbox permutation — pair-swap or
3-way cycle), `r` reject-swap, `m` mount (rows tagged `event_type=mount`
in the audited CSV and dropped by downstream readers unless
`--include-mount`), `u` unknown.

## Layout

```
src/pylace/
├── annotator/   # PyQt6 calibration / ROI editor
├── detect/      # background, connected-components, watershed, Hough, chainer
├── inspect/     # raw-detection viewer
├── posthoc/     # clean, audit, fingerprint, metrics, explorer
├── review/      # candidates, verdicts, merge, review_panel, review_cli
├── roi/         # ROI editor (shared with annotator)
├── tracking/    # core trackers
├── tune/        # live parameter-tuning GUI
└── widgets/     # shared Qt widgets

tests/   # pytest, synthetic fixtures, headless Qt
docs/    # algorithm notes, schema, calibration recipe
figures/ # gitignored; PNG + SVG + CSV triplets
scripts/ # ops scripts that run inside the repo
```

## License

GPL-3.0-or-later.
