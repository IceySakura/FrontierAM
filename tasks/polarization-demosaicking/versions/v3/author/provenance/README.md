# Polarization demosaicking vNext workspace

This directory is the staging area for the Wen extension. It is deliberately separate from the released v1.2 packages.

Current target:

- retain the frozen Qiu32 set;
- split the Wen source set into 5 public development scenes and 100 hidden evaluation scenes;
- compute one per-scene four-channel mean RMSE, average all 132 formal scenes, and apply one global 0/5/10 mapping;
- keep Wen5 diagnostics separate from the formal score.

Raw Wen source data and hidden references remain outside the repository in the private staging area. This directory now contains only conversion metadata, non-image calibration receipts, and staging scripts; the anchors are provisional until the scorer, audit, and data-rights release gates pass.

The released v1.2 packages remain outside this directory and must not be edited or rebuilt as part of staging work.

Staging documents are [problem.md](problem.md) and [evaluation_plan.md](evaluation_plan.md). The private calibration receipts and the global score audit are under [calibration/](calibration/); they contain no Wen image pixels.

The authorized internal source copy is currently staged outside the repository at `/private/tmp/wen-vnext/source/OL`. Its structure and file hashes are recorded in [wen_source_inventory.json](wen_source_inventory.json) and [wen_source_record.json](wen_source_record.json). It has not been copied into a release package.
