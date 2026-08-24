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

# biolearn imports torch at module load, and on Windows it must win the race
# for the OpenMP runtime or the interpreter dies with an access violation. The
# deployed service never imports biolearn at all, so torch being absent there is
# expected rather than a problem.
try:
    import torch  # noqa: F401

    TORCH_OK = True
except Exception:
    TORCH_OK = False


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
