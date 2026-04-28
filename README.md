# pyLACE

A Python port of LACE — Limbless Animal traCkEr — a marker-free pose
estimator for limbless animals (larval *Drosophila*, adult zebrafish),
originally published as:

> Geurten BRH (2022). LACE — Limbless Animal traCkEr: a marker-free
> pose estimator. *Frontiers in Behavioral Neuroscience*.
> https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2022.819146/full

The earlier MATLAB implementation is archived at
https://github.com/zerotonin/LACE. The previous Python analysis layer
(database + plotting + post-hoc analysis) has moved to
https://github.com/zerotonin/pyLACEpostHoc.

## Status

Under active rewrite. Phase 0 (repo scaffolding) is in place; the
detection core, midline derivation, and PyQt6 GUI are next.

## Layout

```
src/pylace/   # importable package
tests/        # pytest, synthetic fixtures, mandatory CLI end-to-end test
docs/         # algorithm notes, schema, calibration recipe
figures/      # generated; gitignored; PNG + SVG + CSV triplets
scripts/      # ops scripts that run inside the repo
```

## License

GPL-3.0-or-later.
