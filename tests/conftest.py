"""Test bootstrap for the Windows Anaconda MKL runtime."""
# Importing Torch before NumPy aborts this environment inside blas_fpe_check.
# Loading NumPy once at collection start makes the order deterministic.
import numpy  # noqa: F401
