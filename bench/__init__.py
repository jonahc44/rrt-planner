"""Benchmark harness for rrt_planner (see bench/run.py)."""

import os

# numpy's OpenBLAS pool spin-waits after any BLAS call and steals cores from
# the planner threads; the harness does no heavy linear algebra.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
