"""Import-order shim.

On Windows, biolearn imports torch *after* numpy/scipy/sklearn have already
pulled in their own OpenMP runtime, which makes ``c10.dll`` fail to initialise
and hard-crashes the interpreter (access violation). Importing torch first
fixes it. Every module in this package imports ``loopcore.compat`` before it
touches biolearn.
"""

import os
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

SLIM = os.environ.get("LONGEVITY_LOOP_SLIM") == "1"

if SLIM:
    # Size-capped host: register stand-ins for torch/cvxpy/scipy/sklearn before
    # biolearn can import the real ones. See loopcore/slim.py.
    from . import slim as _slim

    _slim.install()
    TORCH_OK = False
else:
    try:  # torch must win the race for the OpenMP runtime
        import torch  # noqa: F401

        TORCH_OK = True
    except Exception as exc:  # pragma: no cover - depends on local install
        TORCH_OK = False
        warnings.warn(f"torch unavailable ({exc}); AltumAge will be skipped")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
