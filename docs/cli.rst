Command-line tools
==================

pyLACE ships eleven ``console_scripts`` entry points covering the full
calibrate → detect → clean → audit → review pipeline.

==========================  ===========================================================
CLI                         What it does
==========================  ===========================================================
``pylace-annotate``         Annotate a reference frame; emit the arena sidecar JSON.
``pylace-tune``             Live preview to tune background + detector params.
``pylace-roi``              Standalone ROI editor for an existing sidecar.
``pylace-detect``           Run the detection pipeline; emit per-detection CSV.
``pylace-clean``            Detections CSV → cleaned per-frame trajectory CSV.
``pylace-fingerprint``      Build per-track appearance medians from confident frames.
``pylace-audit``            Cost + appearance + verdict-aware identity-swap audit.
``pylace-candidates``       Audit-independent candidate detector (contact, jump, NaN).
``pylace-explore``          Interactive inspector; ``--review`` opens the review dock.
``pylace-metrics``          Summarise an audited trajectory.
``pylace-inspect``          Lightweight viewer for raw detections.
==========================  ===========================================================

Three-pass review workflow
--------------------------

.. code-block:: bash

   # 1. find candidate frames where identity is at risk
   pylace-candidates  video.pylace_trajectory.csv

   # 2. run the cost-based audit
   pylace-audit       video.pylace_trajectory.csv

   # 3. open the GUI with the review dock
   pylace-explore     video.mp4 video.pylace_audited.csv --review

   # 4. re-run the audit; the verdicts sidecar is auto-detected
   pylace-audit       video.pylace_trajectory.csv

Verdict keys in the GUI: ``a`` accept-swap (uses the spinbox
permutation — pair-swap or 3-way cycle), ``r`` reject-swap, ``m``
mount (rows tagged ``event_type=mount`` in the audited CSV and dropped
by downstream readers unless ``--include-mount``), ``u`` unknown.

Per-CLI ``--help``
------------------

Each entry point honours ``--help``; the smoke-test suite asserts that
``main(["--help"])`` exits with status 0 for every entry point.
