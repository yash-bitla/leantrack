# 8. Do not ship weights that come from MOT17

## Context

The failure predictor learns from MOT17 sequences, and the INT8 model uses MOT17 frames
for calibration. The MOT17 data has the CC BY-NC-SA 3.0 license, which does not permit
commercial use. This repository has the MIT license.

## Decision

The repository contains no weights of the failure predictor and no quantized model. The
scripts `bench/failure.py` and `bench/quantize.py` write them to `runs/` and `models/`,
which git ignores.

## Consequences

- A user must run a script to get these files. The README gives the commands.
- The failure predictor is weak on a new scene type in any case (calibration error
  0.095), so a shipped model would be of little use.
- The two demo GIFs do contain MOT17 frames. The README states the license and the
  source.
