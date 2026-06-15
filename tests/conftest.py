"""Shared pytest configuration for the test suite.

The models are small, so multi-threaded BLAS provides no benefit and actively
hurts: per-op threading overhead makes single fits slower, and under parallel
test execution (pytest-xdist) the worker processes oversubscribe the CPU and
contend badly. Pinning every numerical backend to a single thread makes each
test faster on its own and lets xdist scale cleanly across cores.

These must be set before NumPy / PyTensor are imported, which is why they live
at module import time in conftest.py (loaded by pytest, and by each xdist
worker, before any test module imports NumPy).
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
