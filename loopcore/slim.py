"""Slim import shim for size-capped hosts.

A Vercel serverless bundle may not exceed 250 MB unzipped. The honest
dependency set for this app is about 780 MB, and 657 MB of that is four
packages that biolearn imports at module load but barely uses on our code path:

    torch    529 MB   only AltumAge, which is not in our panel
    scipy     95 MB   only ``minimize_scalar`` (MiAge, unused) and
                      ``rankdata`` (DunedinPACE preprocessing, which we DO use)
    sklearn   28 MB   only ``LinearRegression`` inside the PC-clock path
    cvxpy      4 MB   imported, never called on our path

So this module registers minimal stand-ins for all four *before* biolearn is
imported. The one function we genuinely need, ``scipy.stats.rankdata``, is
reimplemented in numpy rather than stubbed out, and ``tests/parity`` checks that
every panel model returns bit-identical output with and without the real
packages installed.

Anything stubbed that actually gets called raises rather than returning a wrong
number. Import this before biolearn; ``loopcore.compat`` does it automatically
when ``LONGEVITY_LOOP_SLIM=1``.
"""

import sys
import types

import numpy as np

INSTALLED = False


def _unavailable(name):
    def raiser(*args, **kwargs):
        raise RuntimeError(
            f"{name} was called, but this build ships a stub for it. "
            f"Either the code path changed or the full dependency set is needed."
        )
    return raiser


def rankdata(a, method="average"):
    """Average-rank of each element, matching ``scipy.stats.rankdata``.

    DunedinPACE quantile-normalises against its gold-standard means and needs
    ties averaged exactly the way scipy does it, so this is the one stub that
    is a real implementation rather than a raiser.
    """
    arr = np.asarray(a).ravel()
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    sorted_arr = arr[order]

    i = 0
    n = len(arr)
    while i < n:
        j = i
        while j + 1 < n and sorted_arr[j + 1] == sorted_arr[i]:
            j += 1
        # Ranks are 1-based; ties all take the mean of the span they occupy.
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _torch_stub():
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")

    class Module:
        """Enough of nn.Module for AltumAge's class body to define itself."""

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            raise RuntimeError("torch is not available in the slim build")

        forward = __call__

        def eval(self):
            return self

        def load_state_dict(self, *args, **kwargs):
            raise RuntimeError("torch is not available in the slim build")

    nn.Module = Module
    for layer in ("Linear", "BatchNorm1d", "Dropout", "ReLU", "Sequential",
                  "Identity", "LayerNorm", "Softmax", "Sigmoid", "Tanh",
                  "GELU", "SiLU", "ELU", "ModuleList"):
        setattr(nn, layer, type(layer, (Module,), {}))
    for fn in ("relu", "silu", "gelu", "elu", "softmax", "sigmoid", "tanh",
               "dropout"):
        setattr(functional, fn, _unavailable(f"torch.nn.functional.{fn}"))
    nn.functional = functional
    torch.nn = nn
    for fn in ("load", "tensor", "no_grad", "from_numpy", "device", "Tensor",
               "float32", "float64", "cat", "stack"):
        setattr(torch, fn, _unavailable(f"torch.{fn}"))
    return {"torch": torch, "torch.nn": nn, "torch.nn.functional": functional}


def _scipy_stub():
    scipy = types.ModuleType("scipy")
    stats = types.ModuleType("scipy.stats")
    optimize = types.ModuleType("scipy.optimize")
    special = types.ModuleType("scipy.special")

    stats.rankdata = rankdata
    stats.__getattr__ = lambda name: _unavailable(f"scipy.stats.{name}")
    optimize.__getattr__ = lambda name: _unavailable(f"scipy.optimize.{name}")
    special.__getattr__ = lambda name: _unavailable(f"scipy.special.{name}")

    scipy.stats = stats
    scipy.optimize = optimize
    scipy.special = special
    return {"scipy": scipy, "scipy.stats": stats, "scipy.optimize": optimize,
            "scipy.special": special}


class LinearRegression:
    """Ordinary least squares with an intercept, in numpy.

    GrimAge uses this to regress DNAmGrimAge on chronological age across the
    samples it was handed, and reports the residual as AgeAccelGrim. Only fit
    and predict are needed, so this is a real implementation rather than a
    raiser; the parity test checks it against the sklearn result.
    """

    def __init__(self, *args, **kwargs):
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float).reshape(len(y), -1)
        y = np.asarray(y, dtype=float)
        design = np.column_stack([np.ones(len(X)), X])
        solution, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.intercept_ = float(solution[0])
        self.coef_ = solution[1:]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        return X @ self.coef_ + self.intercept_


def _sklearn_stub():
    sklearn = types.ModuleType("sklearn")
    linear_model = types.ModuleType("sklearn.linear_model")

    linear_model.LinearRegression = LinearRegression
    linear_model.__getattr__ = lambda name: _unavailable(
        f"sklearn.linear_model.{name}")
    sklearn.linear_model = linear_model
    return {"sklearn": sklearn, "sklearn.linear_model": linear_model}


def _cvxpy_stub():
    cvxpy = types.ModuleType("cvxpy")
    cvxpy.__getattr__ = lambda name: _unavailable(f"cvxpy.{name}")
    return {"cvxpy": cvxpy}


def install(force=False):
    """Register the stubs. Real packages already imported are left alone."""
    global INSTALLED
    if INSTALLED:
        return

    modules = {}
    modules.update(_torch_stub())
    modules.update(_scipy_stub())
    modules.update(_sklearn_stub())
    modules.update(_cvxpy_stub())

    for name, module in modules.items():
        root = name.split(".")[0]
        if not force and root in sys.modules and not getattr(
                sys.modules[root], "_longevity_stub", False):
            continue  # the real package is already loaded; do not shadow it
        module._longevity_stub = True
        sys.modules[name] = module

    INSTALLED = True
