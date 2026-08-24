"""Freeze the biolearn eval panel into loopcore/precomputed/fastpanel.npz.

Run this whenever biolearn or the reference epigenome changes:

    python -m tools.build_fastpanel

The npz it writes is what the deployed service reads, so this is the only place
biolearn is needed. ``tests`` and the parity harness compare the result against
``loopcore.bioevals`` model by model.
"""

import os
import sys

from loopcore import compat  # noqa: F401  (torch must import before biolearn)

from loopcore import reference
from loopcore.fastpanel import PANEL_NPZ, FastPanel


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else PANEL_NPZ

    ref, info = reference.build(progress=lambda m: print(f"  {m}"))
    print(f"Reference epigenome: {len(ref)} CpGs from {info.get('cohort')}")

    panel = FastPanel.build(ref)
    panel.save(path)

    models = len(panel.lin_names) + 3          # + DunedinPACE, EpiTOC2, GrimAge
    size = os.path.getsize(path)
    print(f"Wrote {path}")
    print(f"  {size:,} bytes ({size / 1024 / 1024:.2f} MB)")
    print(f"  {models} panel models captured "
          f"({len(panel.grim_names)} GrimAge sub-models)")
    print(f"  {len(panel.cpgs)} CpGs in the fixed index")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
