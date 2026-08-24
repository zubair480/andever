"""Freeze the mechanism axis directions into loopcore/precomputed/axes.npz.

Run this whenever biolearn or the reference epigenome changes, alongside
``tools.build_fastpanel``:

    python -m tools.build_axes

``ResponseModel`` resolves ``panels.AXES`` onto CpGs from biolearn's coefficient
tables and the reference age regression, which is a fixed computation over
static inputs. Freezing it is what lets the deployed service build the model
with numpy and pandas only. The npz stores the post-normalisation vectors, so
the runtime reloads the exact arrays this script produced rather than
re-deriving them.
"""

import os
import sys

from loopcore import compat  # noqa: F401  (torch must import before biolearn)

from loopcore import reference
from loopcore.intervention import AXES_NPZ, ResponseModel


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else AXES_NPZ

    ref, info = reference.build(progress=lambda m: print(f"  {m}"))
    print(f"Reference epigenome: {len(ref)} CpGs from {info.get('cohort')}")

    # precomputed=False so a stale npz cannot seed the one that replaces it.
    response = ResponseModel(ref, precomputed=False)
    response.save_axes(path)

    size = os.path.getsize(path)
    print(f"Wrote {path}")
    print(f"  {size:,} bytes ({size / 1024 / 1024:.2f} MB)")
    print(f"  {len(response.directions)} axes captured over "
          f"{len(ref)} indexed CpGs")
    total = 0
    for name, direction in response.directions.items():
        total += len(direction)
        print(f"    {name:28s} {len(direction):6,d} sites  "
              f"range [{direction.min():+.3f}, {direction.max():+.3f}]")
    print(f"  {total:,} CpG assignments in total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
