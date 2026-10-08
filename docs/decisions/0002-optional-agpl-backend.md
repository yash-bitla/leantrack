# 2. Keep the AGPL detector backend optional

## Context

Ultralytics YOLO models are accurate and easy to use. The package and its weights have
the AGPL-3.0 license. This repository has the MIT license.

## Decision

The default detector backend is YOLOX through ONNX Runtime (Apache-2.0). The Ultralytics
backend is an optional extra. Only `src/leantrack/detect/ultralytics.py` imports the
package, and the import is inside the constructor.

## Consequences

- The core install has no AGPL dependency, and the container image uses YOLOX.
- A person who installs the extra accepts the AGPL-3.0 terms for that combination. The
  README states this.
- The Ultralytics backend has one test, for the error of a missing package. No
  experiment in this repository uses it.
