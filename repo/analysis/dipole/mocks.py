"""Poisson mock skies (prototype Step 4)."""
import numpy as np


def expected_counts(cos_theta, abs_elat, mask, N0, s, D, dipole=True, ecliptic=True):
    """
    Expected count per pixel,
        lambda = N0 * (1 + s |elat|) * (1 + D cos theta),
    with the ecliptic and dipole factors switchable. Both effects change the
    *fraction* of sources observed, so they multiply. Masked pixels are 0.
    """
    lam = np.full(cos_theta.shape, float(N0))
    if ecliptic:
        lam *= 1 + s * abs_elat
    if dipole:
        lam *= 1 + D * cos_theta
    lam[mask] = 0.0
    return lam


def draw_mock(lam, rng):
    """One mock sky: an independent Poisson count in every pixel with mean
    lam. Masked pixels (lam = 0) always get 0."""
    return rng.poisson(lam)
